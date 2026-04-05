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
class RRSentenceSupervisionRecord:
    case_id: str
    split: str
    subset: str | None
    sentence_index: int
    sentence_count: int
    sentence_text: str
    fine_rr_label: Any
    fine_rr_label_name: str | None
    broad_section_label: str
    previous_context_text: str = ""
    next_context_text: str = ""
    sentence_length_chars: int = 0
    sentence_length_tokens_approx: int = 0
    normalized_sentence_position: float = 0.0
    document_position_bucket: str = "start"
    source_file: str = ""
    source_metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return _serialize(self)


@dataclass(frozen=True)
class RRSentenceSupervisionResult:
    records: list[RRSentenceSupervisionRecord]
    warnings: list[str]
    report: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return _serialize(self)


@dataclass(frozen=True)
class SentenceSpan:
    sentence_index: int
    text: str
    start_char: int
    end_char: int

    def to_dict(self) -> dict[str, Any]:
        return _serialize(self)


@dataclass(frozen=True)
class CJPESegmentedCase:
    case_id: str
    split: str
    subset: str | None
    label: int | str | None
    sentence_count: int
    sentences: list[str]
    sentence_start_chars: list[int]
    sentence_end_chars: list[int]
    short_sentence_count: int = 0
    long_sentence_count: int = 0
    raw_text_length_chars: int = 0
    source_file: str = ""
    source_metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return _serialize(self)


@dataclass(frozen=True)
class CJPESentenceSegmentationResult:
    records: list[CJPESegmentedCase]
    warnings: list[str]
    report: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return _serialize(self)


@dataclass(frozen=True)
class SectionTaggerTrainingResult:
    model_path: str
    metadata_path: str
    metrics: dict[str, Any]
    prediction_samples: list[dict[str, Any]] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return _serialize(self)


@dataclass(frozen=True)
class CJPESentencePredictionCase:
    case_id: str
    split: str
    subset: str | None
    label: int | str | None
    raw_text: str
    sentences: list[str]
    sentence_indices: list[int]
    sentence_start_chars: list[int]
    sentence_end_chars: list[int]
    predicted_broad_labels: list[str]
    predicted_label_scores: list[float] = field(default_factory=list)
    prediction_metadata: dict[str, Any] = field(default_factory=dict)
    source_file: str = ""
    source_metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return _serialize(self)


@dataclass(frozen=True)
class CJPESentencePredictionResult:
    records: list[CJPESentencePredictionCase]
    warnings: list[str]
    report: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return _serialize(self)


@dataclass(frozen=True)
class CJPEPseudoSectionedCase:
    case_id: str
    cjpe_label: int | str | None
    split: str
    subset: str | None
    raw_text: str
    sentences: list[str]
    sentence_indices: list[int]
    sentence_start_chars: list[int]
    sentence_end_chars: list[int]
    predicted_broad_labels: list[str]
    predicted_label_scores: list[float] = field(default_factory=list)
    grouped_sections: dict[str, str] = field(default_factory=dict)
    section_sentence_map: dict[str, list[int]] = field(default_factory=dict)
    section_lengths_sentences: dict[str, int] = field(default_factory=dict)
    section_lengths_chars: dict[str, int] = field(default_factory=dict)
    prediction_metadata: dict[str, Any] = field(default_factory=dict)
    source_file: str = ""
    source_metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return _serialize(self)


@dataclass(frozen=True)
class CJPEPseudoSectionedResult:
    records: list[CJPEPseudoSectionedCase]
    warnings: list[str]
    report: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return _serialize(self)
