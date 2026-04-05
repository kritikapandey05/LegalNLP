from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class CanonicalCaseExample:
    case_id: str
    split: str | None
    label: str | None
    raw_text: str
    sentences: list[str] = field(default_factory=list)
    sentence_metadata: list[dict[str, Any]] = field(default_factory=list)
    rr_labels: list[str] = field(default_factory=list)
    grouped_sections: dict[str, str] = field(default_factory=dict)
    explanation_annotations: dict[str, Any] = field(default_factory=dict)
    source_metadata: dict[str, Any] = field(default_factory=dict)
