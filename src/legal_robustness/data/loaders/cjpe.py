from __future__ import annotations

import logging
from collections import Counter

from legal_robustness.config.schema import AppConfig
from legal_robustness.data.loaders.common import (
    append_malformed_row,
    collect_expert_annotations,
    coerce_case_id,
    enforce_malformed_row_threshold,
    iter_parquet_rows,
    validate_required_columns,
)
from legal_robustness.data.raw_types import DatasetFileShard, RawCJPECase, RawTaskLoadResult

REQUIRED_CJPE_COLUMNS = {"id", "text", "label"}


def load_cjpe_raw_cases(
    files: list[DatasetFileShard],
    config: AppConfig,
    logger: logging.Logger | None = None,
) -> RawTaskLoadResult:
    logger = logger or logging.getLogger(__name__)
    warnings: list[str] = []
    records: list[RawCJPECase] = []
    malformed_rows = []
    counts_by_split: Counter[str] = Counter()
    total_rows_seen = 0

    if not files:
        warnings.append("No CJPE files were discovered for raw loading.")
        return RawTaskLoadResult(
            task="cjpe",
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
        validate_required_columns(file_info, REQUIRED_CJPE_COLUMNS)
        logger.info("Loading CJPE raw file %s", file_info.relative_path)
        for row_index, row in iter_parquet_rows(file_info.absolute_path, batch_size=config.data.loader_batch_size):
            total_rows_seen += 1
            case_id = coerce_case_id(row.get("id"))
            text = row.get("text")
            label = row.get("label")
            issues: list[str] = []

            if case_id is None:
                issues.append("Missing or empty id field.")
            if not isinstance(text, str) or not text.strip():
                issues.append("Text field is missing, empty, or not a string.")
            if isinstance(label, (list, dict)):
                issues.append("Label field must be scalar-like, not list/dict.")
            if label is None:
                issues.append("Label field is missing.")

            if issues:
                append_malformed_row(
                    diagnostics=malformed_rows,
                    task="cjpe",
                    split=file_info.split,
                    source_file=file_info.relative_path,
                    row_index=row_index,
                    case_id=case_id,
                    issue=" ".join(issues),
                    raw_preview={"id": row.get("id"), "text": row.get("text"), "label": row.get("label")},
                )
                continue

            records.append(
                RawCJPECase(
                    task="cjpe",
                    split=file_info.split,
                    dataset_name=file_info.dataset_name,
                    subset=file_info.subset,
                    source_file=file_info.relative_path,
                    case_id=case_id,
                    raw_text=text,
                    label=label,
                    expert_annotations_raw=collect_expert_annotations(row),
                    source_metadata={
                        "source_task": "cjpe",
                        "source_split": file_info.split,
                        "source_subset": file_info.subset,
                        "source_dataset_name": file_info.dataset_name,
                        "source_row_index": row_index,
                        "shard_index": file_info.shard_index,
                        "shard_count": file_info.shard_count,
                    },
                )
            )
            counts_by_split[file_info.split] += 1

    malformed_count = len(malformed_rows)
    if malformed_count:
        warnings.append(f"CJPE loader recorded {malformed_count} malformed rows.")
    enforce_malformed_row_threshold(
        task="cjpe",
        total_rows_seen=total_rows_seen,
        malformed_rows=malformed_count,
        max_fraction=config.data.max_malformed_row_fraction,
    )

    return RawTaskLoadResult(
        task="cjpe",
        files=files,
        records=records,
        malformed_rows=malformed_rows,
        warnings=warnings,
        counts_by_split=dict(sorted(counts_by_split.items())),
        total_rows_seen=total_rows_seen,
        records_emitted=len(records),
        valid_records=len(records),
    )
