from __future__ import annotations

import logging
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from legal_robustness.config.schema import AppConfig
from legal_robustness.data.label_inventory import _label_key, _sort_key_for_label
from legal_robustness.data.normalized_types import (
    NormalizedRRCase,
    RRLabelInventoryReport,
    RRSectionMappingEntry,
    RRSectionMappingReport,
)
from legal_robustness.utils.exceptions import ConfigurationError, DatasetReconstructionError

RR_LABEL_BLOCK_PATTERN = re.compile(
    r"<summary>\s*List of RR labels\s*</summary>\s*(?P<body>.*?)</details>",
    re.IGNORECASE | re.DOTALL,
)
QUOTED_STRING_PATTERN = re.compile(r'"([^"]+)"')
VALID_UNMAPPED_BEHAVIORS = {
    "fail",
    "warn_and_route_to_other",
    "warn_and_keep_unmapped_bucket",
    "skip_sentences",
}


def discover_rr_label_names(dataset_root: Path, logger: logging.Logger | None = None) -> tuple[dict[str, str], str | None, list[str]]:
    logger = logger or logging.getLogger(__name__)
    warnings: list[str] = []
    readme_path = dataset_root / "README.md"
    if not readme_path.exists():
        warnings.append(f"RR label-name discovery skipped because dataset README was not found at {readme_path}.")
        return {}, None, warnings

    content = readme_path.read_text(encoding="utf-8")
    match = RR_LABEL_BLOCK_PATTERN.search(content)
    if not match:
        warnings.append(f"RR label-name discovery skipped because no RR label block was found in {readme_path}.")
        return {}, None, warnings

    label_names = QUOTED_STRING_PATTERN.findall(match.group("body"))
    if not label_names:
        warnings.append(f"RR label-name discovery skipped because no quoted RR labels were found in {readme_path}.")
        return {}, None, warnings

    mapping = {str(index): label_name for index, label_name in enumerate(label_names)}
    logger.info("Discovered %s RR label names from %s", len(mapping), readme_path)
    return mapping, "dataset_readme", warnings


def validate_rr_section_mapping(
    cases: list[NormalizedRRCase],
    label_inventory: RRLabelInventoryReport,
    config: AppConfig,
    logger: logging.Logger | None = None,
) -> RRSectionMappingReport:
    logger = logger or logging.getLogger(__name__)
    section_config = config.sections
    warnings: list[str] = []
    discovered_names, discovered_source, discovery_warnings = discover_rr_label_names(
        config.data.dataset_root or config.project_root,
        logger=logger,
    )
    warnings.extend(discovery_warnings)
    label_names = {**discovered_names, **dict(section_config.rr_label_names)}
    if section_config.rr_label_names and discovered_names:
        label_name_source = "dataset_readme+config_override"
    elif section_config.rr_label_names:
        label_name_source = "config"
    else:
        label_name_source = discovered_source or "unknown"

    if section_config.unmapped_label_behavior not in VALID_UNMAPPED_BEHAVIORS:
        raise ConfigurationError(
            "Invalid sections.unmapped_label_behavior. "
            f"Expected one of {sorted(VALID_UNMAPPED_BEHAVIORS)}, got {section_config.unmapped_label_behavior!r}."
        )
    if not section_config.rr_section_mapping:
        raise ConfigurationError("sections.rr_section_mapping must define at least one broad section.")
    if section_config.unmapped_label_behavior == "warn_and_route_to_other" and "other" not in section_config.rr_section_mapping:
        raise ConfigurationError(
            "sections.rr_section_mapping must define an 'other' bucket when unmapped_label_behavior is 'warn_and_route_to_other'."
        )

    mapping_lookup = _build_mapping_lookup(section_config.rr_section_mapping)
    mapped_counts_by_section: Counter[str] = Counter()
    entries: list[RRSectionMappingEntry] = []
    mapped_label_keys: set[str] = set()
    unmapped_label_keys: set[str] = set()

    for entry in label_inventory.entries:
        label_name = label_names.get(entry.label_key)
        mapped_section, resolution_strategy = _resolve_mapped_section(
            label_key=entry.label_key,
            raw_value=entry.raw_value,
            label_name=label_name,
            mapping_lookup=mapping_lookup,
        )
        if mapped_section is None:
            unmapped_label_keys.add(entry.label_key)
            resolution_strategy = "unmapped"
        else:
            mapped_label_keys.add(entry.label_key)
            mapped_counts_by_section[mapped_section] += entry.count

        entries.append(
            RRSectionMappingEntry(
                label_key=entry.label_key,
                raw_value=entry.raw_value,
                raw_type=entry.raw_type,
                label_name=label_name,
                mapped_section=mapped_section,
                resolution_strategy=resolution_strategy,
                count=entry.count,
                counts_by_split=dict(entry.counts_by_split),
                cases_with_label=entry.cases_with_label,
            )
        )

    cases_affected_by_unmapped_labels = 0
    sentences_affected_by_unmapped_labels = 0
    counts_by_output_section: Counter[str] = Counter(mapped_counts_by_section)
    output_sections = list(section_config.rr_section_mapping.keys())
    if section_config.unmapped_label_behavior == "warn_and_keep_unmapped_bucket":
        output_sections = [*output_sections, "unmapped"]
    for section in output_sections:
        counts_by_output_section.setdefault(section, 0)

    for case in cases:
        case_has_unmapped = False
        for label in case.rr_labels:
            label_key = _label_key(label)
            if label_key not in unmapped_label_keys:
                continue
            case_has_unmapped = True
            sentences_affected_by_unmapped_labels += 1
            if section_config.unmapped_label_behavior == "warn_and_route_to_other":
                counts_by_output_section["other"] += 1
            elif section_config.unmapped_label_behavior == "warn_and_keep_unmapped_bucket":
                counts_by_output_section["unmapped"] += 1
        if case_has_unmapped:
            cases_affected_by_unmapped_labels += 1

    total_sentence_labels = sum(entry.count for entry in label_inventory.entries)
    mapped_sentence_labels = total_sentence_labels - sentences_affected_by_unmapped_labels
    coverage_percent = round((mapped_sentence_labels / total_sentence_labels) * 100, 3) if total_sentence_labels else 100.0

    if unmapped_label_keys:
        warning = (
            f"RR section mapping left {len(unmapped_label_keys)} labels unmapped under behavior "
            f"{section_config.unmapped_label_behavior!r}."
        )
        warnings.append(warning)
        logger.warning("%s", warning)
        if not section_config.allow_partial_mapping:
            partial_warning = "RR section mapping is partial while sections.allow_partial_mapping is false."
            warnings.append(partial_warning)
            logger.warning("%s", partial_warning)
        if section_config.fail_on_unmapped_labels or section_config.unmapped_label_behavior == "fail":
            raise DatasetReconstructionError("RR section mapping failed because unmapped labels were detected.")
    elif not section_config.allow_partial_mapping:
        warning = "RR section mapping achieved full coverage while partial mapping was disabled."
        logger.info("%s", warning)

    summary = {
        "total_unique_labels": label_inventory.summary["total_unique_labels"],
        "mapped_label_count": len(mapped_label_keys),
        "unmapped_label_count": len(unmapped_label_keys),
        "mapped_labels": sorted(mapped_label_keys, key=_sort_key_for_label),
        "unmapped_labels": sorted(unmapped_label_keys, key=_sort_key_for_label),
        "label_name_source": label_name_source,
        "coverage_percent": coverage_percent,
        "cases_affected_by_unmapped_labels": cases_affected_by_unmapped_labels,
        "sentences_affected_by_unmapped_labels": sentences_affected_by_unmapped_labels,
        "counts_by_output_section": dict(sorted(counts_by_output_section.items())),
        "output_sections": output_sections,
        "fallback_behavior": section_config.unmapped_label_behavior,
        "allow_partial_mapping": section_config.allow_partial_mapping,
    }
    applied_config = {
        "rr_label_names": label_names,
        "rr_section_mapping": {
            section: [str(value) for value in values]
            for section, values in section_config.rr_section_mapping.items()
        },
        "unmapped_label_behavior": section_config.unmapped_label_behavior,
        "allow_partial_mapping": section_config.allow_partial_mapping,
        "fail_on_unmapped_labels": section_config.fail_on_unmapped_labels,
    }
    return RRSectionMappingReport(entries=entries, summary=summary, applied_config=applied_config, warnings=warnings)


def render_rr_section_mapping_report(report: RRSectionMappingReport) -> str:
    lines = [
        "# RR Section Mapping Report",
        "",
        f"- Label name source: `{report.summary['label_name_source']}`",
        f"- Coverage percent: `{report.summary['coverage_percent']}`",
        f"- Mapped labels: `{report.summary['mapped_label_count']}`",
        f"- Unmapped labels: `{report.summary['unmapped_label_count']}`",
        f"- Cases affected by unmapped labels: `{report.summary['cases_affected_by_unmapped_labels']}`",
        f"- Sentences affected by unmapped labels: `{report.summary['sentences_affected_by_unmapped_labels']}`",
        f"- Output sections: `{report.summary['output_sections']}`",
        f"- Fallback behavior: `{report.summary['fallback_behavior']}`",
        "",
        "## Applied Mapping",
        "",
    ]
    for section, labels in report.applied_config["rr_section_mapping"].items():
        lines.append(f"- `{section}` <- `{labels}`")

    lines.extend(
        [
            "",
            "| Label Key | Label Name | Raw Value | Section | Resolution | Count | Cases | Counts By Split |",
            "| --- | --- | --- | --- | --- | ---: | ---: | --- |",
        ]
    )
    for entry in report.entries:
        lines.append(
            f"| `{entry.label_key}` | `{entry.label_name}` | `{entry.raw_value}` | "
            f"`{entry.mapped_section}` | `{entry.resolution_strategy}` | {entry.count} | "
            f"{entry.cases_with_label} | `{entry.counts_by_split}` |"
        )
    if report.warnings:
        lines.extend(["", "## Warnings", ""])
        for warning in report.warnings:
            lines.append(f"- {warning}")
    return "\n".join(lines).strip() + "\n"


def resolve_section_for_label(label: Any, report: RRSectionMappingReport) -> tuple[str | None, str | None]:
    lookup = {
        entry.label_key: (entry.mapped_section, entry.label_name)
        for entry in report.entries
    }
    return lookup.get(_label_key(label), (None, None))


def _build_mapping_lookup(section_mapping: dict[str, tuple[Any, ...]]) -> dict[str, str]:
    lookup: dict[str, str] = {}
    for section, identifiers in section_mapping.items():
        for identifier in identifiers:
            canonical = _canonical_identifier(identifier)
            if canonical in lookup and lookup[canonical] != section:
                raise ConfigurationError(
                    f"RR section mapping identifier {identifier!r} is assigned to both {lookup[canonical]!r} and {section!r}."
                )
            lookup[canonical] = section
    return lookup


def _resolve_mapped_section(
    label_key: str,
    raw_value: Any,
    label_name: str | None,
    mapping_lookup: dict[str, str],
) -> tuple[str | None, str]:
    candidates = [
        ("label_name", label_name),
        ("label_key", label_key),
        ("raw_value", raw_value),
    ]
    for strategy, candidate in candidates:
        if candidate is None:
            continue
        section = mapping_lookup.get(_canonical_identifier(candidate))
        if section is not None:
            return section, strategy
    return None, "unmapped"


def _canonical_identifier(value: Any) -> str:
    return str(value).strip().casefold()
