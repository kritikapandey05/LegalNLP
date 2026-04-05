from __future__ import annotations

import logging
from collections import Counter, defaultdict
from statistics import mean, median
from typing import Any

from legal_robustness.config.schema import AppConfig
from legal_robustness.data.normalized_types import (
    NormalizedRRCase,
    RRReconstructionResult,
    RRSectionMappingReport,
    ReconstructedRRCase,
)
from legal_robustness.data.label_inventory import _label_key
from legal_robustness.utils.exceptions import DatasetReconstructionError


def reconstruct_rr_sections(
    cases: list[NormalizedRRCase],
    mapping_report: RRSectionMappingReport,
    config: AppConfig,
    logger: logging.Logger | None = None,
) -> RRReconstructionResult:
    logger = logger or logging.getLogger(__name__)
    warnings = list(mapping_report.warnings)
    output_sections = list(mapping_report.summary["output_sections"])
    behavior = config.sections.unmapped_label_behavior
    mapping_lookup = {
        entry.label_key: (entry.mapped_section, entry.label_name)
        for entry in mapping_report.entries
    }
    records: list[ReconstructedRRCase] = []
    counts_by_split: Counter[str] = Counter()
    empty_section_counts: Counter[str] = Counter()
    section_char_lengths: dict[str, list[int]] = defaultdict(list)
    section_sentence_lengths: dict[str, list[int]] = defaultdict(list)
    cases_all_content_in_other = 0
    cases_with_unmapped_labels = 0
    section_patterns: Counter[str] = Counter()
    non_empty_section_counts: list[int] = []
    non_empty_sections_by_split: dict[str, Counter[str]] = defaultdict(Counter)

    for case in cases:
        counts_by_split[case.split] += 1
        grouped_sentences: dict[str, list[str]] = {section: [] for section in output_sections}
        section_sentence_map: dict[str, list[int]] = {section: [] for section in output_sections}
        rr_label_names: list[str | None] = []
        unmapped_labels_in_case: list[Any] = []

        for index, (sentence, label) in enumerate(zip(case.sentences, case.rr_labels, strict=False)):
            mapped_section, label_name = mapping_lookup.get(_label_key(label), (None, None))
            rr_label_names.append(label_name)

            if mapped_section is None:
                unmapped_labels_in_case.append(label)
                if behavior == "fail":
                    raise DatasetReconstructionError(
                        f"RR reconstruction failed for case {case.case_id} because label {label!r} was unmapped."
                    )
                if behavior == "warn_and_route_to_other":
                    mapped_section = "other"
                elif behavior == "warn_and_keep_unmapped_bucket":
                    mapped_section = "unmapped"
                elif behavior == "skip_sentences":
                    continue
                else:
                    raise DatasetReconstructionError(f"Unsupported unmapped label behavior: {behavior!r}")

            grouped_sentences.setdefault(mapped_section, [])
            section_sentence_map.setdefault(mapped_section, [])
            grouped_sentences[mapped_section].append(sentence)
            section_sentence_map[mapped_section].append(index)

        grouped_sections = {
            section: "\n".join(grouped_sentences.get(section, []))
            for section in output_sections
        }
        section_lengths_chars = {section: len(text) for section, text in grouped_sections.items()}
        section_lengths_sentences = {section: len(section_sentence_map.get(section, [])) for section in output_sections}

        for section in output_sections:
            if section_lengths_sentences[section] == 0:
                empty_section_counts[section] += 1
            else:
                non_empty_sections_by_split[case.split][section] += 1
            section_char_lengths[section].append(section_lengths_chars[section])
            section_sentence_lengths[section].append(section_lengths_sentences[section])

        non_empty_sections = [section for section in output_sections if section_lengths_sentences[section] > 0]
        non_empty_section_counts.append(len(non_empty_sections))
        section_patterns["|".join(non_empty_sections) if non_empty_sections else "<empty>"] += 1
        if non_empty_sections == ["other"]:
            cases_all_content_in_other += 1
        if unmapped_labels_in_case:
            cases_with_unmapped_labels += 1

        records.append(
            ReconstructedRRCase(
                case_id=case.case_id,
                split=case.split,
                subset=case.subset,
                sentences=list(case.sentences),
                rr_labels=list(case.rr_labels),
                rr_label_names=rr_label_names,
                grouped_sections=grouped_sections,
                section_sentence_map=section_sentence_map,
                section_lengths_chars=section_lengths_chars,
                section_lengths_sentences=section_lengths_sentences,
                unmapped_labels_present=bool(unmapped_labels_in_case),
                unmapped_labels=_unique_unmapped_labels(unmapped_labels_in_case),
                source_file=case.source_file,
                source_task=case.source_task,
                source_metadata=dict(case.source_metadata),
            )
        )

    report = {
        "task": "rr_reconstruction",
        "total_cases": len(records),
        "counts_by_split": dict(sorted(counts_by_split.items())),
        "output_sections": output_sections,
        "empty_section_counts": dict(sorted(empty_section_counts.items())),
        "section_length_chars": {
            section: _describe_numeric_series(values)
            for section, values in sorted(section_char_lengths.items())
        },
        "section_length_sentences": {
            section: _describe_numeric_series(values)
            for section, values in sorted(section_sentence_lengths.items())
        },
        "cases_all_content_in_other": cases_all_content_in_other,
        "cases_with_unmapped_labels": cases_with_unmapped_labels,
        "average_sections_present": round(mean(non_empty_section_counts), 3) if non_empty_section_counts else 0.0,
        "unique_section_pattern_count": len(section_patterns),
        "section_patterns": dict(sorted(section_patterns.items())),
        "non_empty_sections_by_split": {
            split: dict(sorted(counter.items()))
            for split, counter in sorted(non_empty_sections_by_split.items())
        },
        "mapping_coverage_percent": mapping_report.summary["coverage_percent"],
        "mapping_fallback_behavior": mapping_report.summary["fallback_behavior"],
    }
    logger.info(
        "RR reconstruction completed for %s cases with %s cases affected by unmapped labels.",
        len(records),
        cases_with_unmapped_labels,
    )
    return RRReconstructionResult(task="rr_reconstruction", records=records, warnings=warnings, report=report)


def render_rr_reconstruction_report(report: dict[str, Any]) -> str:
    lines = [
        "# RR Reconstruction Report",
        "",
        f"- Total cases: `{report['total_cases']}`",
        f"- Counts by split: `{report['counts_by_split']}`",
        f"- Output sections: `{report['output_sections']}`",
        f"- Mapping coverage percent: `{report['mapping_coverage_percent']}`",
        f"- Mapping fallback behavior: `{report['mapping_fallback_behavior']}`",
        f"- Cases with unmapped labels: `{report['cases_with_unmapped_labels']}`",
        f"- Cases with all content in other: `{report['cases_all_content_in_other']}`",
        f"- Average sections present: `{report['average_sections_present']}`",
        f"- Unique section pattern count: `{report['unique_section_pattern_count']}`",
        "",
        "## Empty Sections",
        "",
        f"- Empty section counts: `{report['empty_section_counts']}`",
        "",
        "## Sentence-Length Stats",
        "",
    ]
    for section, stats in report["section_length_sentences"].items():
        lines.append(f"- `{section}`: `{stats}`")
    lines.extend(["", "## Character-Length Stats", ""])
    for section, stats in report["section_length_chars"].items():
        lines.append(f"- `{section}`: `{stats}`")
    return "\n".join(lines).strip() + "\n"


def _unique_unmapped_labels(values: list[Any]) -> list[Any]:
    unique: list[Any] = []
    for value in values:
        if value not in unique:
            unique.append(value)
    return unique


def _describe_numeric_series(values: list[int]) -> dict[str, int | float | None]:
    if not values:
        return {"count": 0, "min": None, "max": None, "mean": None, "median": None}
    return {
        "count": len(values),
        "min": min(values),
        "max": max(values),
        "mean": round(mean(values), 3),
        "median": median(values),
    }
