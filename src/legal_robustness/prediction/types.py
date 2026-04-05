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
class PredictionInputExample:
    case_id: str
    split: str
    subset: str | None
    label: int | str | None
    input_variant: str
    input_text: str
    input_text_length_chars: int
    sections_used: list[str] = field(default_factory=list)
    sections_omitted: list[str] = field(default_factory=list)
    empty_selected_sections: list[str] = field(default_factory=list)
    variant_metadata: dict[str, Any] = field(default_factory=dict)
    source_file: str = ""
    source_metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return _serialize(self)


@dataclass(frozen=True)
class BaselinePredictionRecord:
    case_id: str
    split: str
    subset: str | None
    gold_label: str
    predicted_label: str
    input_variant: str
    model_name: str
    correct: bool
    predicted_score: float
    predicted_probabilities: dict[str, float] = field(default_factory=dict)
    input_text_length_chars: int = 0
    sections_used: list[str] = field(default_factory=list)
    sections_omitted: list[str] = field(default_factory=list)
    source_file: str = ""

    def to_dict(self) -> dict[str, Any]:
        return _serialize(self)


@dataclass(frozen=True)
class BaselineModelRunResult:
    model_name: str
    input_variant: str
    model_path: str
    metrics_by_split: dict[str, Any]
    training_summary: dict[str, Any]
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return _serialize(self)
