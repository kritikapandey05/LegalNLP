from __future__ import annotations

import logging
from collections import Counter
from typing import Any

from legal_robustness.config.schema import AppConfig
from legal_robustness.data.loaders.common import (
    append_malformed_row,
    collect_expert_annotations,
    coerce_case_id,
    enforce_malformed_row_threshold,
    iter_parquet_rows,
    validate_required_columns,
)
from legal_robustness.data.raw_types import DatasetFileShard, RawRRCase, RawTaskLoadResult

REQUIRED_RR_COLUMNS = {"id", "text", "labels"}


def _is_list_like(value: Any) -> bool:
    return isinstance(value, (list, tuple))


def _normalize_sentences(value: Any) -> tuple[list[str], list[str]]:
    issues: list[str] = []
    if not _is_list_like(value):
        return [], ["Text field is not list-like."]
    sentences: list[str] = []
    for index, item in enumerate(value):
        if item is None:
            issues.append(f"Sentence at position {index} is null.")
            sentences.append("")
            continue
        if not isinstance(item, str):
            issues.append(f"Sentence at position {index} is not a string.")
        sentences.append(str(item))
    return sentences, issues


def _normalize_labels(value: Any) -> tuple[list[Any], list[str]]:
    if not _is_list_like(value):
        return [], ["Labels field is not list-like."]
    return list(value), []


def load_rr_raw_cases(
    files: list[DatasetFileShard],
    config: AppConfig,
    logger: logging.Logger | None = None,
) -> RawTaskLoadResult:
    logger = logger or logging.getLogger(__name__)
    warnings: list[str] = []
    records: list[RawRRCase] = []
    malformed_rows = []
    counts_by_split: Counter[str] = Counter()
    total_rows_seen = 0
    valid_records = 0

    if not files:
        warnings.append("No RR files were discovered for raw loading.")
        return RawTaskLoadResult(
            task="rr",
            files=[],
            records=[],
            malformed_rows=[],
            warnings=warnings,
            counts_by_split={},
            total_rows_seen=0,
            records_emitted=0,
            valid_records=0,
        )

    for file_info in files:
        validate_required_columns(file_info, REQUIRED_RR_COLUMNS)
        logger.info("Loading RR raw file %s", file_info.relative_path)
        for row_index, row in iter_parquet_rows(file_info.absolute_path, batch_size=config.data.loader_batch_size):
            total_rows_seen += 1
            case_id = coerce_case_id(row.get("id"))
            sentences, sentence_issues = _normalize_sentences(row.get("text"))
            rr_labels, label_issues = _normalize_labels(row.get("labels"))
            issues = [*sentence_issues, *label_issues]

            if case_id is None:
                issues.append("Missing or empty id field.")
            if sentences and rr_labels and len(sentences) != len(rr_labels):
                issues.append(
                    f"Text/label length mismatch: {len(sentences)} sentences vs {len(rr_labels)} labels."
                )

            if case_id is None or not _is_list_like(row.get("text")) or not _is_list_like(row.get("labels")):
                append_malformed_row(
                    diagnostics=malformed_rows,
                    task="rr",
                    split=file_info.split,
                    source_file=file_info.relative_path,
                    row_index=row_index,
                    case_id=case_id,
                    issue=" ".join(issues) if issues else "Row failed RR structural validation.",
                    raw_preview={"id": row.get("id"), "text": row.get("text"), "labels": row.get("labels")},
                )
                continue

            alignment_ok = len(issues) == 0
            if not alignment_ok:
                append_malformed_row(
                    diagnostics=malformed_rows,
                    task="rr",
                    split=file_info.split,
                    source_file=file_info.relative_path,
                    row_index=row_index,
                    case_id=case_id,
                    issue=" ".join(issues),
                    raw_preview={"id": row.get("id"), "text": row.get("text"), "labels": row.get("labels")},
                )

            records.append(
                RawRRCase(
                    task="rr",
                    split=file_info.split,
                    dataset_name=file_info.dataset_name,
                    subset=file_info.subset,
                    source_file=file_info.relative_path,
                    case_id=case_id,
                    sentences=sentences,
                    rr_labels=rr_labels,
                    alignment_ok=alignment_ok,
                    expert_annotations_raw=collect_expert_annotations(row),
                    source_metadata={
                        "source_task": "rr",
                        "source_split": file_info.split,
                        "source_subset": file_info.subset,
                        "source_dataset_name": file_info.dataset_name,
                        "source_row_index": row_index,
                        "shard_index": file_info.shard_index,
                        "shard_count": file_info.shard_count,
                        "validation_issues": issues,
                    },
                )
            )
            counts_by_split[file_info.split] += 1
            if alignment_ok:
                valid_records += 1

    malformed_count = len(malformed_rows)
    if malformed_count:
        warnings.append(f"RR loader recorded {malformed_count} malformed or misaligned rows.")
    enforce_malformed_row_threshold(
        task="rr",
        total_rows_seen=total_rows_seen,
        malformed_rows=malformed_count,
        max_fraction=config.data.max_malformed_row_fraction,
    )

    return RawTaskLoadResult(
        task="rr",
        files=files,
        records=records,
        malformed_rows=malformed_rows,
        warnings=warnings,
        counts_by_split=dict(sorted(counts_by_split.items())),
        total_rows_seen=total_rows_seen,
        records_emitted=len(records),
        valid_records=valid_records,
    )
