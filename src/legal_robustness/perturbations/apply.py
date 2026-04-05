from __future__ import annotations

from collections import Counter, defaultdict
from statistics import mean
from typing import Any

from legal_robustness.config.schema import AppConfig
from legal_robustness.perturbations.primitives import apply_perturbation, build_perturbation_specs
from legal_robustness.perturbations.types import PerturbedCJPECase
from legal_robustness.section_transfer.types import CJPEPseudoSectionedCase


def generate_perturbation_sets(
    cases: list[CJPEPseudoSectionedCase],
    *,
    config: AppConfig,
) -> tuple[dict[str, list[PerturbedCJPECase]], dict[str, Any], list[dict[str, object]]]:
    selected_splits = set(config.perturbation.evaluation_splits)
    filtered_cases = [case for case in cases if case.split in selected_splits]
    specs = build_perturbation_specs(config)
    rows_by_recipe: dict[str, list[PerturbedCJPECase]] = defaultdict(list)
    manifest_rows: dict[str, Any] = {}
    sample_rows: list[dict[str, object]] = []

    for spec in specs:
        recipe_rows: list[PerturbedCJPECase] = []
        empty_target_count = 0
        perturbed_lengths: list[int] = []
        for case in filtered_cases:
            row = apply_perturbation(case, spec=spec, config=config)
            if row.target_section_was_empty:
                empty_target_count += 1
            recipe_rows.append(row)
            perturbed_lengths.append(row.perturbed_text_length_chars)
        rows_by_recipe[spec.name] = recipe_rows
        manifest_rows[spec.name] = {
            "family": spec.family,
            "case_count": len(recipe_rows),
            "empty_target_case_count": empty_target_count,
            "average_perturbed_text_length_chars": round(mean(perturbed_lengths), 3) if perturbed_lengths else 0.0,
            "target_section": spec.target_section,
            "sections_to_keep": list(spec.sections_to_keep),
            "section_order": list(spec.section_order),
        }
        sample_rows.extend(
            build_perturbation_samples(
                recipe_rows,
                sample_size=min(config.perturbation.sample_size, 2),
                preview_chars=config.perturbation.preview_chars,
            )
        )

    manifest = {
        "task": "cjpe_perturbation_sets",
        "evaluation_splits": sorted(selected_splits),
        "recipe_count": len(specs),
        "recipes": manifest_rows,
        "total_examples": sum(len(rows) for rows in rows_by_recipe.values()),
        "warnings": [],
    }
    return dict(rows_by_recipe), manifest, sample_rows


def build_perturbation_samples(
    rows: list[PerturbedCJPECase],
    *,
    sample_size: int,
    preview_chars: int,
) -> list[dict[str, object]]:
    samples: list[dict[str, object]] = []
    for row in rows[:sample_size]:
        samples.append(
            {
                "case_id": row.case_id,
                "split": row.split,
                "cjpe_label": row.cjpe_label,
                "perturbation_name": row.perturbation_name,
                "perturbation_family": row.perturbation_family,
                "target_section": row.target_section,
                "sections_kept": row.sections_kept,
                "sections_dropped": row.sections_dropped,
                "sections_masked": row.sections_masked,
                "target_section_was_empty": row.target_section_was_empty,
                "perturbed_text_preview": _truncate_text(row.perturbed_text, max_chars=preview_chars),
            }
        )
    return samples


def summarize_perturbation_examples(rows_by_recipe: dict[str, list[PerturbedCJPECase]]) -> dict[str, Any]:
    counts_by_split: Counter[str] = Counter()
    for rows in rows_by_recipe.values():
        for row in rows:
            counts_by_split[row.split] += 1
    return {
        "counts_by_split": dict(sorted(counts_by_split.items())),
    }


def _truncate_text(text: str, *, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip() + "..."
