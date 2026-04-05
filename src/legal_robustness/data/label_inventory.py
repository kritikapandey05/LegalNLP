from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any

from legal_robustness.config.schema import AppConfig
from legal_robustness.data.normalized_types import NormalizedRRCase, RRLabelInventoryEntry, RRLabelInventoryReport


def generate_rr_label_inventory(cases: list[NormalizedRRCase], config: AppConfig) -> RRLabelInventoryReport:
    del config  # Reserved for future label-canonicalization and mapping policies.
    label_counts: Counter[str] = Counter()
    raw_values: dict[str, Any] = {}
    raw_types: dict[str, str] = {}
    split_counts: dict[str, Counter[str]] = defaultdict(Counter)
    cases_with_label: Counter[str] = Counter()
    null_label_count = 0
    blank_label_count = 0
    rare_labels: list[str] = []
    warnings: list[str] = []
    string_case_groups: dict[str, set[str]] = defaultdict(set)

    for case in cases:
        labels_seen_in_case: set[str] = set()
        for label in case.rr_labels:
            key = _label_key(label)
            raw_values.setdefault(key, label)
            raw_types.setdefault(key, type(label).__name__)
            label_counts[key] += 1
            split_counts[case.split][key] += 1
            labels_seen_in_case.add(key)
            if label is None:
                null_label_count += 1
            if isinstance(label, str):
                if not label.strip():
                    blank_label_count += 1
                string_case_groups[label.casefold()].add(label)
        for key in labels_seen_in_case:
            cases_with_label[key] += 1

    for key, count in sorted(label_counts.items()):
        if count <= 1:
            rare_labels.append(key)

    case_inconsistencies = {
        canonical: sorted(values)
        for canonical, values in sorted(string_case_groups.items())
        if len(values) > 1
    }
    if case_inconsistencies:
        warnings.append(f"Detected {len(case_inconsistencies)} case-inconsistent string label groups.")

    entries = [
        RRLabelInventoryEntry(
            label_key=key,
            raw_value=raw_values[key],
            raw_type=raw_types[key],
            count=label_counts[key],
            counts_by_split={
                split: split_counter.get(key, 0)
                for split, split_counter in sorted(split_counts.items())
                if split_counter.get(key, 0)
            },
            cases_with_label=cases_with_label[key],
        )
        for key in sorted(label_counts, key=_sort_key_for_label)
    ]
    summary = {
        "total_unique_labels": len(entries),
        "label_types": dict(sorted(Counter(entry.raw_type for entry in entries).items())),
        "null_label_count": null_label_count,
        "blank_label_count": blank_label_count,
        "rare_labels": rare_labels,
        "rare_label_count": len(rare_labels),
        "case_inconsistencies": case_inconsistencies,
    }
    return RRLabelInventoryReport(entries=entries, summary=summary, warnings=warnings)


def render_rr_label_inventory_report(report: RRLabelInventoryReport) -> str:
    lines = [
        "# RR Label Inventory",
        "",
        f"- Total unique labels: `{report.summary['total_unique_labels']}`",
        f"- Label types: `{report.summary['label_types']}`",
        f"- Null label count: `{report.summary['null_label_count']}`",
        f"- Blank label count: `{report.summary['blank_label_count']}`",
        f"- Rare label count: `{report.summary['rare_label_count']}`",
        "",
        "| Label Key | Raw Value | Type | Count | Cases With Label | Counts By Split |",
        "| --- | --- | --- | ---: | ---: | --- |",
    ]
    for entry in report.entries:
        lines.append(
            f"| `{entry.label_key}` | `{entry.raw_value}` | `{entry.raw_type}` | "
            f"{entry.count} | {entry.cases_with_label} | `{entry.counts_by_split}` |"
        )
    if report.summary["case_inconsistencies"]:
        lines.extend(["", "## Case Inconsistencies", ""])
        for canonical, variants in report.summary["case_inconsistencies"].items():
            lines.append(f"- `{canonical}` -> `{variants}`")
    if report.warnings:
        lines.extend(["", "## Warnings", ""])
        for warning in report.warnings:
            lines.append(f"- {warning}")
    return "\n".join(lines).strip() + "\n"


def _label_key(label: Any) -> str:
    if label is None:
        return "<null>"
    if isinstance(label, str):
        stripped = label.strip()
        return stripped if stripped else "<blank>"
    return str(label)


def _sort_key_for_label(label_key: str) -> tuple[int, Any]:
    try:
        return (0, int(label_key))
    except ValueError:
        return (1, label_key)
