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
class NormalizedCJPECase:
    case_id: str
    split: str
    subset: str | None
    label: int | str | None
    raw_text: str
    text_length_chars: int
    text_length_tokens_approx: int
    expert_annotations: dict[str, Any] = field(default_factory=dict)
    source_file: str = ""
    source_task: str = "cjpe"
    source_metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return _serialize(self)


@dataclass(frozen=True)
class NormalizedRRCase:
    case_id: str
    split: str
    subset: str | None
    sentences: list[str]
    rr_labels: list[Any]
    num_sentences: int
    num_labels: int
    empty_sentence_count: int
    empty_label_count: int
    alignment_ok: bool
    expert_annotations: dict[str, Any] = field(default_factory=dict)
    source_file: str = ""
    source_task: str = "rr"
    source_metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return _serialize(self)


@dataclass(frozen=True)
class RRLabelInventoryEntry:
    label_key: str
    raw_value: Any
    raw_type: str
    count: int
    counts_by_split: dict[str, int] = field(default_factory=dict)
    cases_with_label: int = 0

    def to_dict(self) -> dict[str, Any]:
        return _serialize(self)


@dataclass(frozen=True)
class NormalizedTaskResult:
    task: str
    records: list[Any]
    warnings: list[str]
    duplicate_case_ids: dict[str, int]
    report: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return _serialize(self)


@dataclass(frozen=True)
class RRLabelInventoryReport:
    entries: list[RRLabelInventoryEntry]
    summary: dict[str, Any]
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return _serialize(self)


@dataclass(frozen=True)
class RRSectionMappingEntry:
    label_key: str
    raw_value: Any
    raw_type: str
    label_name: str | None
    mapped_section: str | None
    resolution_strategy: str
    count: int
    counts_by_split: dict[str, int] = field(default_factory=dict)
    cases_with_label: int = 0

    def to_dict(self) -> dict[str, Any]:
        return _serialize(self)


@dataclass(frozen=True)
class RRSectionMappingReport:
    entries: list[RRSectionMappingEntry]
    summary: dict[str, Any]
    applied_config: dict[str, Any]
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return _serialize(self)


@dataclass(frozen=True)
class ReconstructedRRCase:
    case_id: str
    split: str
    subset: str | None
    sentences: list[str]
    rr_labels: list[Any]
    rr_label_names: list[str | None]
    grouped_sections: dict[str, str]
    section_sentence_map: dict[str, list[int]]
    section_lengths_chars: dict[str, int]
    section_lengths_sentences: dict[str, int]
    unmapped_labels_present: bool
    unmapped_labels: list[Any] = field(default_factory=list)
    source_file: str = ""
    source_task: str = "rr"
    source_metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return _serialize(self)


@dataclass(frozen=True)
class RRReconstructionResult:
    task: str
    records: list[ReconstructedRRCase]
    warnings: list[str]
    report: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return _serialize(self)

