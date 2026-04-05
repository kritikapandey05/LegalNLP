from __future__ import annotations

import csv
import json
import logging
from collections import Counter
from dataclasses import dataclass, field, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from legal_robustness.config.schema import AppConfig
from legal_robustness.utils.exceptions import InspectionError
from legal_robustness.utils.paths import validate_dataset_root

SUPPORTED_STRUCTURED_SUFFIXES = {".parquet", ".json", ".jsonl", ".csv"}


def _serialize(value: Any) -> Any:
    if is_dataclass(value):
        return {key: _serialize(getattr(value, key)) for key in value.__dataclass_fields__}
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _serialize(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_serialize(item) for item in value]
    return value


def _truncate_text(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return f"{text[:max_chars]}...<truncated>"


def _make_json_safe(value: Any, max_chars: int, max_list_items: int) -> Any:
    if isinstance(value, str):
        return _truncate_text(value, max_chars)
    if isinstance(value, list):
        safe_items = [_make_json_safe(item, max_chars, max_list_items) for item in value[:max_list_items]]
        if len(value) > max_list_items:
            safe_items.append(f"...<{len(value) - max_list_items} additional items truncated>")
        return safe_items
    if isinstance(value, dict):
        return {str(key): _make_json_safe(item, max_chars, max_list_items) for key, item in value.items()}
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value


@dataclass(frozen=True)
class DirectorySummary:
    relative_path: str
    depth: int
    file_count: int
    subdirectory_count: int
    suffix_counts: dict[str, int] = field(default_factory=dict)


@dataclass(frozen=True)
class FileInspection:
    relative_path: str
    file_format: str
    size_bytes: int
    row_count: int | None = None
    columns: list[str] = field(default_factory=list)
    schema: dict[str, str] = field(default_factory=dict)
    sample_records: list[dict[str, Any]] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class DatasetInspectionReport:
    dataset_root: str
    created_at_utc: str
    tasks_detected: list[str]
    total_directories: int
    total_files: int
    supported_file_count: int
    directory_summaries: list[DirectorySummary]
    file_inspections: list[FileInspection]
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return _serialize(self)


class DatasetInspector:
    def __init__(self, config: AppConfig, logger: logging.Logger | None = None) -> None:
        self.config = config
        self.logger = logger or logging.getLogger(__name__)

    def inspect(self) -> DatasetInspectionReport:
        dataset_root = validate_dataset_root(self.config.data.dataset_root)
        directory_summaries = self._build_directory_summaries(dataset_root)
        file_paths = sorted(path for path in dataset_root.rglob("*") if path.is_file())
        structured_files = [path for path in file_paths if path.suffix.lower() in SUPPORTED_STRUCTURED_SUFFIXES]
        tasks_detected = sorted(
            {
                relative.parts[0]
                for path in structured_files
                for relative in [path.relative_to(dataset_root)]
                if len(relative.parts) > 1 and not relative.parts[0].startswith(".")
            }
        )

        file_inspections: list[FileInspection] = []
        warnings: list[str] = []
        for file_path in structured_files:
            relative_path = str(file_path.relative_to(dataset_root))
            self.logger.info("Inspecting %s", relative_path)
            try:
                file_inspections.append(self._inspect_file(file_path, dataset_root))
            except InspectionError as exc:
                warnings.append(str(exc))
                self.logger.warning("%s", exc)

        return DatasetInspectionReport(
            dataset_root=str(dataset_root),
            created_at_utc=datetime.now(timezone.utc).isoformat(),
            tasks_detected=tasks_detected,
            total_directories=len(directory_summaries),
            total_files=len(file_paths),
            supported_file_count=len(structured_files),
            directory_summaries=directory_summaries,
            file_inspections=file_inspections,
            warnings=warnings,
        )

    def _build_directory_summaries(self, dataset_root: Path) -> list[DirectorySummary]:
        summaries: list[DirectorySummary] = []
        max_depth = self.config.data.inspection_max_tree_depth
        for directory in sorted(path for path in dataset_root.rglob("*") if path.is_dir()):
            relative = directory.relative_to(dataset_root)
            depth = len(relative.parts)
            if depth > max_depth:
                continue
            files = [path for path in directory.iterdir() if path.is_file()]
            subdirectories = [path for path in directory.iterdir() if path.is_dir()]
            suffix_counts = Counter(path.suffix.lower() or "<none>" for path in files)
            summaries.append(
                DirectorySummary(
                    relative_path=str(relative) if relative.parts else ".",
                    depth=depth,
                    file_count=len(files),
                    subdirectory_count=len(subdirectories),
                    suffix_counts=dict(sorted(suffix_counts.items())),
                )
            )

        root_files = [path for path in dataset_root.iterdir() if path.is_file()]
        root_subdirs = [path for path in dataset_root.iterdir() if path.is_dir()]
        root_suffix_counts = Counter(path.suffix.lower() or "<none>" for path in root_files)
        root_summary = DirectorySummary(
            relative_path=".",
            depth=0,
            file_count=len(root_files),
            subdirectory_count=len(root_subdirs),
            suffix_counts=dict(sorted(root_suffix_counts.items())),
        )
        return [root_summary, *summaries]

    def _inspect_file(self, file_path: Path, dataset_root: Path) -> FileInspection:
        suffix = file_path.suffix.lower()
        if suffix == ".parquet":
            return self._inspect_parquet(file_path, dataset_root)
        if suffix in {".json", ".jsonl"}:
            return self._inspect_json(file_path, dataset_root)
        if suffix == ".csv":
            return self._inspect_csv(file_path, dataset_root)
        raise InspectionError(f"Unsupported structured file format for inspection: {file_path}")

    def _inspect_parquet(self, file_path: Path, dataset_root: Path) -> FileInspection:
        try:
            import pyarrow.parquet as pq
        except ImportError as exc:
            raise InspectionError(
                "pyarrow is required to inspect Parquet files. Install project dependencies before running inspection."
            ) from exc

        try:
            parquet_file = pq.ParquetFile(file_path)
        except Exception as exc:
            raise InspectionError(f"Failed to open Parquet file {file_path}: {exc}") from exc

        schema = {field.name: str(field.type) for field in parquet_file.schema_arrow}
        row_count = parquet_file.metadata.num_rows if parquet_file.metadata is not None else None
        sample_records: list[dict[str, Any]] = []
        try:
            for batch in parquet_file.iter_batches(batch_size=self.config.data.inspection_sample_rows):
                sample_records = [
                    _make_json_safe(
                        record,
                        self.config.data.inspection_max_text_chars,
                        self.config.data.inspection_max_list_items,
                    )
                    for record in batch.to_pylist()
                ]
                break
        except Exception as exc:
            raise InspectionError(f"Failed to sample rows from Parquet file {file_path}: {exc}") from exc

        return FileInspection(
            relative_path=str(file_path.relative_to(dataset_root)),
            file_format="parquet",
            size_bytes=file_path.stat().st_size,
            row_count=row_count,
            columns=list(schema.keys()),
            schema=schema,
            sample_records=sample_records,
        )

    def _inspect_json(self, file_path: Path, dataset_root: Path) -> FileInspection:
        suffix = file_path.suffix.lower()
        sample_records: list[dict[str, Any]] = []
        warnings: list[str] = []
        try:
            with file_path.open("r", encoding="utf-8") as handle:
                if suffix == ".json":
                    loaded = json.load(handle)
                    if isinstance(loaded, dict):
                        sample_records = [
                            _make_json_safe(
                                loaded,
                                self.config.data.inspection_max_text_chars,
                                self.config.data.inspection_max_list_items,
                            )
                        ]
                    elif isinstance(loaded, list):
                        sample_records = [
                            _make_json_safe(
                                item,
                                self.config.data.inspection_max_text_chars,
                                self.config.data.inspection_max_list_items,
                            )
                            for item in loaded[: self.config.data.inspection_sample_rows]
                        ]
                    else:
                        warnings.append("JSON content is not a mapping or list.")
                else:
                    for index, line in enumerate(handle):
                        if index >= self.config.data.inspection_sample_rows:
                            break
                        text = line.strip()
                        if not text:
                            continue
                        sample_records.append(
                            _make_json_safe(
                                json.loads(text),
                                self.config.data.inspection_max_text_chars,
                                self.config.data.inspection_max_list_items,
                            )
                        )
        except Exception as exc:
            raise InspectionError(f"Failed to inspect JSON file {file_path}: {exc}") from exc

        columns = sorted(
            {
                key
                for record in sample_records
                if isinstance(record, dict)
                for key in record
            }
        )
        return FileInspection(
            relative_path=str(file_path.relative_to(dataset_root)),
            file_format=suffix.removeprefix("."),
            size_bytes=file_path.stat().st_size,
            row_count=None,
            columns=columns,
            schema={column: "unknown" for column in columns},
            sample_records=sample_records,
            warnings=warnings,
        )

    def _inspect_csv(self, file_path: Path, dataset_root: Path) -> FileInspection:
        sample_records: list[dict[str, Any]] = []
        fieldnames: list[str] = []
        try:
            with file_path.open("r", encoding="utf-8", newline="") as handle:
                reader = csv.DictReader(handle)
                fieldnames = list(reader.fieldnames or [])
                for index, row in enumerate(reader):
                    if index >= self.config.data.inspection_sample_rows:
                        break
                    sample_records.append(
                        _make_json_safe(
                            dict(row),
                            self.config.data.inspection_max_text_chars,
                            self.config.data.inspection_max_list_items,
                        )
                    )
        except Exception as exc:
            raise InspectionError(f"Failed to inspect CSV file {file_path}: {exc}") from exc

        return FileInspection(
            relative_path=str(file_path.relative_to(dataset_root)),
            file_format="csv",
            size_bytes=file_path.stat().st_size,
            row_count=None,
            columns=fieldnames,
            schema={column: "unknown" for column in fieldnames},
            sample_records=sample_records,
        )


def render_markdown_report(report: DatasetInspectionReport) -> str:
    lines = [
        "# Dataset Inspection Report",
        "",
        f"- Dataset root: `{report.dataset_root}`",
        f"- Created at (UTC): `{report.created_at_utc}`",
        f"- Tasks detected: `{', '.join(report.tasks_detected) if report.tasks_detected else 'none'}`",
        f"- Directories inspected: `{report.total_directories}`",
        f"- Total files: `{report.total_files}`",
        f"- Structured files inspected: `{report.supported_file_count}`",
        "",
        "## Directory Summary",
        "",
        "| Relative Path | Depth | Files | Subdirectories | Suffix Counts |",
        "| --- | ---: | ---: | ---: | --- |",
    ]
    for directory in report.directory_summaries:
        suffix_counts = ", ".join(f"{key}:{value}" for key, value in directory.suffix_counts.items()) or "-"
        lines.append(
            f"| `{directory.relative_path}` | {directory.depth} | {directory.file_count} | "
            f"{directory.subdirectory_count} | {suffix_counts} |"
        )

    lines.extend(["", "## Structured File Samples", ""])
    if not report.file_inspections:
        lines.append("No structured files were inspected.")
    else:
        for file_inspection in report.file_inspections:
            lines.extend(
                [
                    f"### `{file_inspection.relative_path}`",
                    "",
                    f"- Format: `{file_inspection.file_format}`",
                    f"- Size (bytes): `{file_inspection.size_bytes}`",
                    f"- Row count: `{file_inspection.row_count if file_inspection.row_count is not None else 'unknown'}`",
                    f"- Columns: `{', '.join(file_inspection.columns) if file_inspection.columns else 'none'}`",
                    "",
                    "Schema:",
                    "",
                    "```json",
                    json.dumps(file_inspection.schema, indent=2, ensure_ascii=False),
                    "```",
                    "",
                    "Sample records:",
                    "",
                    "```json",
                    json.dumps(file_inspection.sample_records, indent=2, ensure_ascii=False),
                    "```",
                    "",
                ]
            )
            if file_inspection.warnings:
                lines.append(f"- Warnings: `{'; '.join(file_inspection.warnings)}`")
                lines.append("")
    if report.warnings:
        lines.extend(["## Warnings", ""])
        for warning in report.warnings:
            lines.append(f"- {warning}")
    return "\n".join(lines).strip() + "\n"
