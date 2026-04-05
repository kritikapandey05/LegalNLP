from __future__ import annotations

from typing import Any

from legal_robustness.section_transfer.diagnostics import BROAD_SECTION_ORDER
from legal_robustness.section_transfer.types import CJPEPseudoSectionedCase
from legal_robustness.utils.exceptions import PredictionError


INPUT_VARIANT_ORDER = (
    "full_text",
    "pseudo_all_sections",
    "pseudo_facts_reasoning",
    "pseudo_reasoning_only",
    "pseudo_without_conclusion",
    "pseudo_without_precedents",
)

SECTION_MARKER_TEMPLATE = "[{SECTION}_SECTION]"


def build_prediction_input_text(
    case: CJPEPseudoSectionedCase,
    *,
    variant_name: str,
    include_section_markers: bool,
) -> tuple[str, dict[str, Any]]:
    if variant_name == "full_text":
        return case.raw_text, {
            "variant_name": variant_name,
            "used_predicted_sections": False,
            "sections_used": [],
            "sections_omitted": [],
            "empty_selected_sections": [],
            "section_order": [],
        }

    sections_used = _sections_for_variant(variant_name)
    sections_omitted = [section for section in BROAD_SECTION_ORDER if section not in sections_used]
    empty_selected_sections = [
        section for section in sections_used if not case.grouped_sections.get(section, "").strip()
    ]
    text = compose_sectioned_text(
        grouped_sections=case.grouped_sections,
        section_order=BROAD_SECTION_ORDER,
        sections_to_include=sections_used,
        include_section_markers=include_section_markers,
    )
    return text, {
        "variant_name": variant_name,
        "used_predicted_sections": True,
        "sections_used": list(sections_used),
        "sections_omitted": sections_omitted,
        "empty_selected_sections": empty_selected_sections,
        "section_order": list(BROAD_SECTION_ORDER),
    }


def compose_sectioned_text(
    *,
    grouped_sections: dict[str, str],
    section_order: tuple[str, ...] | list[str],
    sections_to_include: tuple[str, ...] | list[str] | None = None,
    include_section_markers: bool,
    masked_sections: dict[str, str] | None = None,
) -> str:
    selected_sections = set(sections_to_include or section_order)
    masked_sections = masked_sections or {}
    parts: list[str] = []
    for section in section_order:
        if section not in selected_sections:
            continue
        raw_text = masked_sections.get(section, grouped_sections.get(section, ""))
        text = raw_text.strip()
        if not text:
            continue
        if include_section_markers:
            parts.append(f"{section_marker(section)}\n{text}")
        else:
            parts.append(text)
    return "\n\n".join(parts).strip()


def section_marker(section: str) -> str:
    return SECTION_MARKER_TEMPLATE.format(SECTION=section.upper())


def _sections_for_variant(variant_name: str) -> tuple[str, ...]:
    if variant_name == "pseudo_all_sections":
        return tuple(BROAD_SECTION_ORDER)
    if variant_name == "pseudo_facts_reasoning":
        return ("facts", "reasoning")
    if variant_name == "pseudo_reasoning_only":
        return ("reasoning",)
    if variant_name == "pseudo_without_conclusion":
        return tuple(section for section in BROAD_SECTION_ORDER if section != "conclusion")
    if variant_name == "pseudo_without_precedents":
        return tuple(section for section in BROAD_SECTION_ORDER if section != "precedents")
    raise PredictionError(f"Unsupported prediction input variant: {variant_name}")
