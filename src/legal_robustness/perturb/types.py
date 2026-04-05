from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal


ApprovalStatus = Literal["pending", "approved", "rejected", "not_run"]


@dataclass(frozen=True)
class ValidationMetadata:
    semantic_validation_score: float | None = None
    approval_status: ApprovalStatus = "not_run"
    validator_name: str | None = None
    notes: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PerturbationMetadata:
    perturbation_name: str
    target_section: str | None
    source_text: str
    generated_text: str
    validation: ValidationMetadata = field(default_factory=ValidationMetadata)
    notes: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PerturbedExample:
    perturbation_id: str
    base_case_id: str
    perturbed_text: str
    metadata: PerturbationMetadata
    source_metadata: dict[str, Any] = field(default_factory=dict)
