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
class PerturbationSpec:
    name: str
    family: str
    target_section: str | None = None
    sections_to_keep: tuple[str, ...] = ()
    section_order: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return _serialize(self)


@dataclass(frozen=True)
class PerturbedCJPECase:
    case_id: str
    split: str
    subset: str | None
    cjpe_label: int | str | None
    perturbation_name: str
    perturbation_family: str
    base_input_variant: str
    perturbed_text: str
    target_section: str | None = None
    sections_kept: list[str] = field(default_factory=list)
    sections_dropped: list[str] = field(default_factory=list)
    sections_masked: list[str] = field(default_factory=list)
    section_order: list[str] = field(default_factory=list)
    target_section_was_empty: bool = False
    original_text_length_chars: int = 0
    perturbed_text_length_chars: int = 0
    grouped_sections: dict[str, str] = field(default_factory=dict)
    source_file: str = ""
    source_metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return _serialize(self)
