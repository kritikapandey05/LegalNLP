from __future__ import annotations

from typing import Any

from legal_robustness.data.normalized_types import (
    NormalizedTaskResult,
    RRLabelInventoryReport,
    RRReconstructionResult,
    RRSectionMappingReport,
)
from legal_robustness.data.raw_types import DiscoveredDatasetManifest, RawTaskLoadResult


def summarize_raw_loading(
    manifest: DiscoveredDatasetManifest,
    cjpe_result: RawTaskLoadResult,
    rr_result: RawTaskLoadResult,
) -> dict[str, Any]:
    file_counts_by_task: dict[str, int] = {}
    declared_rows_by_task: dict[str, int] = {}
    for file_info in manifest.discovered_files:
        file_counts_by_task[file_info.task] = file_counts_by_task.get(file_info.task, 0) + 1
        declared_rows_by_task[file_info.task] = declared_rows_by_task.get(file_info.task, 0) + int(file_info.row_count or 0)

    task_summaries = {
        "cjpe": _summarize_task_result(cjpe_result),
        "rr": _summarize_task_result(rr_result),
    }
    task_summaries["rr"]["alignment_ok_records"] = rr_result.valid_records
    task_summaries["rr"]["alignment_error_records"] = rr_result.records_emitted - rr_result.valid_records

    return {
        "dataset_root": manifest.dataset_root,
        "tasks_requested": manifest.tasks_requested,
        "manifest": {
            "file_counts_by_task": file_counts_by_task,
            "declared_rows_by_task": declared_rows_by_task,
            "warnings": manifest.warnings,
        },
        "tasks": task_summaries,
        "overall_warnings": [
            *manifest.warnings,
            *cjpe_result.warnings,
            *rr_result.warnings,
        ],
    }


def _summarize_task_result(result: RawTaskLoadResult) -> dict[str, Any]:
    return {
        "files_discovered": len(result.files),
        "file_paths": [file_info.relative_path for file_info in result.files],
        "total_rows_seen": result.total_rows_seen,
        "records_emitted": result.records_emitted,
        "valid_records": result.valid_records,
        "malformed_rows": len(result.malformed_rows),
        "counts_by_split": result.counts_by_split,
        "warnings": result.warnings,
    }


def render_raw_loading_report(summary: dict[str, Any]) -> str:
    lines = [
        "# Raw Data Build Summary",
        "",
        f"- Dataset root: `{summary['dataset_root']}`",
        f"- Tasks requested: `{', '.join(summary['tasks_requested'])}`",
        "",
        "## Manifest",
        "",
    ]

    manifest = summary["manifest"]
    lines.append(f"- File counts by task: `{manifest['file_counts_by_task']}`")
    lines.append(f"- Declared rows by task: `{manifest['declared_rows_by_task']}`")
    if manifest["warnings"]:
        lines.append(f"- Manifest warnings: `{'; '.join(manifest['warnings'])}`")
    lines.append("")

    for task_name, task_summary in summary["tasks"].items():
        lines.extend(
            [
                f"## {task_name.upper()}",
                "",
                f"- Files discovered: `{task_summary['files_discovered']}`",
                f"- Total rows seen: `{task_summary['total_rows_seen']}`",
                f"- Records emitted: `{task_summary['records_emitted']}`",
                f"- Valid records: `{task_summary['valid_records']}`",
                f"- Malformed rows: `{task_summary['malformed_rows']}`",
                f"- Counts by split: `{task_summary['counts_by_split']}`",
            ]
        )
        if task_name == "rr":
            lines.append(f"- Alignment-ok records: `{task_summary['alignment_ok_records']}`")
            lines.append(f"- Alignment-error records: `{task_summary['alignment_error_records']}`")
        if task_summary["warnings"]:
            lines.append(f"- Warnings: `{'; '.join(task_summary['warnings'])}`")
        lines.append("")

    if summary["overall_warnings"]:
        lines.extend(["## Warnings", ""])
        for warning in summary["overall_warnings"]:
            lines.append(f"- {warning}")
        lines.append("")

    return "\n".join(lines).strip() + "\n"


def summarize_normalization(
    cjpe_result: NormalizedTaskResult,
    rr_result: NormalizedTaskResult,
    rr_label_inventory: RRLabelInventoryReport,
) -> dict[str, Any]:
    return {
        "tasks": {
            "cjpe": cjpe_result.report,
            "rr": rr_result.report,
        },
        "rr_label_inventory": {
            "summary": rr_label_inventory.summary,
            "warnings": rr_label_inventory.warnings,
        },
        "overall_warnings": [
            *cjpe_result.warnings,
            *rr_result.warnings,
            *rr_label_inventory.warnings,
        ],
    }


def render_normalization_summary(summary: dict[str, Any]) -> str:
    lines = [
        "# Normalized Data Summary",
        "",
        f"- CJPE total cases: `{summary['tasks']['cjpe']['total_cases']}`",
        f"- RR total cases: `{summary['tasks']['rr']['total_cases']}`",
        f"- RR unique labels: `{summary['rr_label_inventory']['summary']['total_unique_labels']}`",
        f"- RR label types: `{summary['rr_label_inventory']['summary']['label_types']}`",
        "",
        "## CJPE",
        "",
        f"- Counts by split: `{summary['tasks']['cjpe']['counts_by_split']}`",
        f"- Duplicate case ids: `{summary['tasks']['cjpe']['duplicate_case_id_count']}`",
        f"- Text length chars: `{summary['tasks']['cjpe']['text_length_chars']}`",
        "",
        "## RR",
        "",
        f"- Counts by split: `{summary['tasks']['rr']['counts_by_split']}`",
        f"- Duplicate case ids: `{summary['tasks']['rr']['duplicate_case_id_count']}`",
        f"- Sentence count stats: `{summary['tasks']['rr']['sentence_count']}`",
        f"- Label count stats: `{summary['tasks']['rr']['label_count']}`",
        f"- Non-aligned cases: `{summary['tasks']['rr']['non_aligned_cases']}`",
        "",
        "## RR Label Inventory",
        "",
        f"- Null label count: `{summary['rr_label_inventory']['summary']['null_label_count']}`",
        f"- Blank label count: `{summary['rr_label_inventory']['summary']['blank_label_count']}`",
        f"- Rare label count: `{summary['rr_label_inventory']['summary']['rare_label_count']}`",
        "",
    ]
    if summary["overall_warnings"]:
        lines.extend(["## Warnings", ""])
        for warning in summary["overall_warnings"]:
            lines.append(f"- {warning}")
        lines.append("")
    return "\n".join(lines).strip() + "\n"


def summarize_reconstruction(
    mapping_report: RRSectionMappingReport,
    reconstruction_result: RRReconstructionResult,
) -> dict[str, Any]:
    return {
        "mapping": {
            "summary": mapping_report.summary,
            "warnings": mapping_report.warnings,
        },
        "reconstruction": reconstruction_result.report,
        "overall_warnings": [
            *mapping_report.warnings,
            *reconstruction_result.warnings,
        ],
    }


def render_reconstruction_summary(summary: dict[str, Any]) -> str:
    lines = [
        "# RR Reconstruction Summary",
        "",
        f"- Coverage percent: `{summary['mapping']['summary']['coverage_percent']}`",
        f"- Unmapped labels: `{summary['mapping']['summary']['unmapped_labels']}`",
        f"- Output sections: `{summary['mapping']['summary']['output_sections']}`",
        f"- Total reconstructed cases: `{summary['reconstruction']['total_cases']}`",
        f"- Counts by split: `{summary['reconstruction']['counts_by_split']}`",
        f"- Cases with unmapped labels: `{summary['reconstruction']['cases_with_unmapped_labels']}`",
        f"- Cases with all content in other: `{summary['reconstruction']['cases_all_content_in_other']}`",
        f"- Unique section pattern count: `{summary['reconstruction']['unique_section_pattern_count']}`",
        "",
    ]
    if summary["overall_warnings"]:
        lines.extend(["## Warnings", ""])
        for warning in summary["overall_warnings"]:
            lines.append(f"- {warning}")
        lines.append("")
    return "\n".join(lines).strip() + "\n"
