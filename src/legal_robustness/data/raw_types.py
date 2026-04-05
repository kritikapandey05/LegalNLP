from __future__ import annotations

from dataclasses import dataclass, field, is_dataclass
from pathlib import Path
from typing import Any


def _serialize(value: Any) -> Any:
    if is_dataclass(value):
        return {key: _serialize(getattr(value, key)) for key in value.__dataclass_fields__}
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, tuple):
        return [_serialize(item) for item in value]
    if isinstance(value, list):
        return [_serialize(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _serialize(item) for key, item in value.items()}
    return value


@dataclass(frozen=True)
class DatasetFileShard:
    task: str
    dataset_name: str
    split: str
    subset: str | None
    relative_path: str
    absolute_path: Path
    shard_index: int | None = None
    shard_count: int | None = None
    row_count: int | None = None
    schema: dict[str, str] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return _serialize(self)


@dataclass(frozen=True)
class DiscoveredDatasetManifest:
    dataset_root: str
    tasks_requested: list[str]
    discovered_files: list[DatasetFileShard]
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return _serialize(self)


@dataclass(frozen=True)
class MalformedRowDiagnostic:
    task: str
    split: str
    source_file: str
    row_index: int
    case_id: str | None
    issue: str
    raw_preview: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return _serialize(self)


@dataclass(frozen=True)
class RawCJPECase:
    task: str
    split: str
    dataset_name: str
    subset: str | None
    source_file: str
    case_id: str
    raw_text: str
    label: int | str | None
    expert_annotations_raw: dict[str, Any] = field(default_factory=dict)
    source_metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return _serialize(self)


@dataclass(frozen=True)
class RawRRCase:
    task: str
    split: str
    dataset_name: str
    subset: str | None
    source_file: str
    case_id: str
    sentences: list[str]
    rr_labels: list[Any]
    alignment_ok: bool
    expert_annotations_raw: dict[str, Any] = field(default_factory=dict)
    source_metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return _serialize(self)


@dataclass(frozen=True)
class RawTaskLoadResult:
    task: str
    files: list[DatasetFileShard]
    records: list[Any]
    malformed_rows: list[MalformedRowDiagnostic]
    warnings: list[str]
    counts_by_split: dict[str, int]
    total_rows_seen: int
    records_emitted: int
    valid_records: int

    def to_dict(self) -> dict[str, Any]:
        return _serialize(self)
