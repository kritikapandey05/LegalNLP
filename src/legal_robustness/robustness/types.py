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
class PerturbedPredictionRecord:
    case_id: str
    split: str
    subset: str | None
    gold_label: str
    model_name: str
    input_variant: str
    perturbation_recipe: str
    perturbation_family: str
    prediction: str
    prediction_score: float
    predicted_probabilities: dict[str, float] = field(default_factory=dict)
    target_section: str | None = None
    target_section_was_empty: bool = False
    reference_prediction: str | None = None
    reference_prediction_score: float | None = None
    prediction_flipped: bool = False
    effective_coverage_group: str = "high_coverage"
    source_file: str = ""
    source_metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return _serialize(self)
