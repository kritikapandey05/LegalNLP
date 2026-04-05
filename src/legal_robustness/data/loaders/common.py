from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any

from legal_robustness.data.raw_types import DatasetFileShard, MalformedRowDiagnostic
from legal_robustness.utils.exceptions import DatasetLoadError, DatasetSchemaError


def iter_parquet_rows(file_path: Path, batch_size: int) -> Iterable[tuple[int, dict[str, Any]]]:
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise DatasetLoadError(
            "pyarrow is required for raw dataset loading. Install project dependencies before running build_dataset."
        ) from exc

    parquet_file = pq.ParquetFile(file_path)
    row_index = 0
    for batch in parquet_file.iter_batches(batch_size=batch_size):
        for row in batch.to_pylist():
            yield row_index, row
            row_index += 1


def validate_required_columns(file_info: DatasetFileShard, required_columns: set[str]) -> None:
    missing_columns = sorted(column for column in required_columns if column not in file_info.schema)
    if missing_columns:
        raise DatasetSchemaError(
            f"Required columns {missing_columns} were not found in {file_info.relative_path}. "
            f"Observed columns: {sorted(file_info.schema)}"
        )


def collect_expert_annotations(row: dict[str, Any]) -> dict[str, Any]:
    return {key: row.get(key) for key in row if key.startswith("expert_")}


def coerce_case_id(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def make_preview(value: Any, max_text_chars: int = 200, max_list_items: int = 5) -> Any:
    if isinstance(value, str):
        return value if len(value) <= max_text_chars else f"{value[:max_text_chars]}...<truncated>"
    if isinstance(value, list):
        preview = [make_preview(item, max_text_chars=max_text_chars, max_list_items=max_list_items) for item in value[:max_list_items]]
        if len(value) > max_list_items:
            preview.append(f"...<{len(value) - max_list_items} additional items truncated>")
        return preview
    if isinstance(value, dict):
        return {str(key): make_preview(item, max_text_chars=max_text_chars, max_list_items=max_list_items) for key, item in value.items()}
    return value


def append_malformed_row(
    diagnostics: list[MalformedRowDiagnostic],
    task: str,
    split: str,
    source_file: str,
    row_index: int,
    case_id: str | None,
    issue: str,
    raw_preview: dict[str, Any],
) -> None:
    diagnostics.append(
        MalformedRowDiagnostic(
            task=task,
            split=split,
            source_file=source_file,
            row_index=row_index,
            case_id=case_id,
            issue=issue,
            raw_preview=make_preview(raw_preview),
        )
    )


def enforce_malformed_row_threshold(task: str, total_rows_seen: int, malformed_rows: int, max_fraction: float) -> None:
    if total_rows_seen == 0:
        return
    malformed_fraction = malformed_rows / total_rows_seen
    if malformed_fraction > max_fraction:
        raise DatasetLoadError(
            f"Loader for task '{task}' aborted because malformed row fraction {malformed_fraction:.3f} "
            f"exceeded configured threshold {max_fraction:.3f}."
        )
