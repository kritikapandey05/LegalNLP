from __future__ import annotations

from legal_robustness.config.schema import AppConfig
from legal_robustness.prediction.input_variants import compose_sectioned_text
from legal_robustness.section_transfer.diagnostics import BROAD_SECTION_ORDER
from legal_robustness.section_transfer.types import CJPEPseudoSectionedCase
from legal_robustness.utils.exceptions import PerturbationError

from legal_robustness.perturbations.types import PerturbationSpec, PerturbedCJPECase


def build_perturbation_specs(config: AppConfig) -> list[PerturbationSpec]:
    recipe_lookup = {
        "drop_conclusion": PerturbationSpec(
            name="drop_conclusion",
            family="drop_section",
            target_section="conclusion",
        ),
        "drop_precedents": PerturbationSpec(
            name="drop_precedents",
            family="drop_section",
            target_section="precedents",
        ),
        "keep_facts_reasoning": PerturbationSpec(
            name="keep_facts_reasoning",
            family="keep_only_section_set",
            sections_to_keep=("facts", "reasoning"),
        ),
        "keep_reasoning_only": PerturbationSpec(
            name="keep_reasoning_only",
            family="keep_only_section_set",
            sections_to_keep=("reasoning",),
        ),
        "mask_conclusion": PerturbationSpec(
            name="mask_conclusion",
            family="mask_section_content",
            target_section="conclusion",
        ),
        "reorder_conclusion_first": PerturbationSpec(
            name="reorder_conclusion_first",
            family="reorder_sections",
            section_order=("conclusion", "facts", "reasoning", "precedents", "other"),
        ),
    }
    specs: list[PerturbationSpec] = []
    for recipe_name in config.perturbation.enabled_recipes:
        spec = recipe_lookup.get(recipe_name)
        if spec is None:
            raise PerturbationError(f"Unsupported perturbation recipe requested: {recipe_name}")
        specs.append(spec)
    return specs


def apply_perturbation(
    case: CJPEPseudoSectionedCase,
    *,
    spec: PerturbationSpec,
    config: AppConfig,
) -> PerturbedCJPECase:
    grouped_sections = dict(case.grouped_sections)
    section_order = list(BROAD_SECTION_ORDER)
    sections_kept = list(BROAD_SECTION_ORDER)
    sections_dropped: list[str] = []
    sections_masked: list[str] = []
    target_section_was_empty = False
    masked_sections: dict[str, str] = {}

    if spec.family == "drop_section":
        if spec.target_section is None:
            raise PerturbationError(f"Perturbation {spec.name} requires a target section.")
        target_section_was_empty = not grouped_sections.get(spec.target_section, "").strip()
        grouped_sections[spec.target_section] = ""
        sections_dropped = [spec.target_section]
        sections_kept = [section for section in BROAD_SECTION_ORDER if section != spec.target_section]
    elif spec.family == "keep_only_section_set":
        if not spec.sections_to_keep:
            raise PerturbationError(f"Perturbation {spec.name} requires sections_to_keep.")
        kept = set(spec.sections_to_keep)
        sections_kept = [section for section in BROAD_SECTION_ORDER if section in kept]
        sections_dropped = [section for section in BROAD_SECTION_ORDER if section not in kept]
        for section in sections_dropped:
            grouped_sections[section] = ""
    elif spec.family == "mask_section_content":
        if spec.target_section is None:
            raise PerturbationError(f"Perturbation {spec.name} requires a target section.")
        target_section_was_empty = not grouped_sections.get(spec.target_section, "").strip()
        if not target_section_was_empty:
            placeholder = config.perturbation.mask_placeholder_template.format(
                SECTION=spec.target_section.upper()
            )
            masked_sections[spec.target_section] = placeholder
            grouped_sections[spec.target_section] = placeholder
            sections_masked = [spec.target_section]
    elif spec.family == "reorder_sections":
        if not spec.section_order:
            raise PerturbationError(f"Perturbation {spec.name} requires a section_order.")
        section_order = list(spec.section_order)
    else:
        raise PerturbationError(f"Unsupported perturbation family: {spec.family}")

    perturbed_text = compose_sectioned_text(
        grouped_sections=grouped_sections,
        section_order=section_order,
        sections_to_include=sections_kept,
        include_section_markers=config.perturbation.include_section_markers,
        masked_sections=masked_sections,
    )
    return PerturbedCJPECase(
        case_id=case.case_id,
        split=case.split,
        subset=case.subset,
        cjpe_label=case.cjpe_label,
        perturbation_name=spec.name,
        perturbation_family=spec.family,
        base_input_variant="pseudo_all_sections",
        perturbed_text=perturbed_text,
        target_section=spec.target_section,
        sections_kept=sections_kept,
        sections_dropped=sections_dropped,
        sections_masked=sections_masked,
        section_order=section_order,
        target_section_was_empty=target_section_was_empty,
        original_text_length_chars=len(case.raw_text),
        perturbed_text_length_chars=len(perturbed_text),
        grouped_sections={section: grouped_sections.get(section, "") for section in BROAD_SECTION_ORDER},
        source_file=case.source_file,
        source_metadata=dict(case.source_metadata),
    )
