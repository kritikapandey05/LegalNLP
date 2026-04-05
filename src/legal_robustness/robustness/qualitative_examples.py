from __future__ import annotations

from collections import Counter
from typing import Any


def build_paper_qualitative_examples(
    failure_analysis_cases: list[dict[str, Any]],
    *,
    focused_recipes: tuple[str, ...] | list[str],
    primary_model_variant: str,
    model_variants: tuple[str, ...] | list[str],
    count_per_recipe: int,
    preview_chars: int,
    include_model_variants: tuple[str, ...] | list[str] | None = None,
    examples_per_category: int = 1,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    included_variants = tuple(include_model_variants or model_variants)
    if primary_model_variant not in included_variants:
        primary_model_variant = next(
            (
                variant
                for variant in included_variants
                if "averaged_passive_aggressive" in variant
            ),
            included_variants[0] if included_variants else primary_model_variant,
        )

    report_rows: list[dict[str, Any]] = []
    selected_examples: list[dict[str, Any]] = []
    for recipe_name in focused_recipes:
        recipe_cases = [
            case for case in failure_analysis_cases
            if case.get("perturbation_recipe") == recipe_name
        ]
        selected_for_recipe = _select_examples_for_recipe(
            recipe_cases,
            recipe_name=recipe_name,
            primary_model_variant=primary_model_variant,
            included_variants=included_variants,
            count=count_per_recipe,
            preview_chars=preview_chars,
            examples_per_category=examples_per_category,
        )
        selected_examples.extend(selected_for_recipe)
        category_counts = Counter(example["selection_category"] for example in selected_for_recipe)
        report_rows.append(
            {
                "recipe_name": recipe_name,
                "selected_example_count": len(selected_for_recipe),
                "selection_category_counts": dict(sorted(category_counts.items())),
            }
        )

    return (
        {
            "task": "paper_qualitative_examples",
            "focused_recipes": list(focused_recipes),
            "primary_model_variant": primary_model_variant,
            "included_model_variants": list(included_variants),
            "per_recipe_summary": report_rows,
        },
        selected_examples,
    )


def build_case_bundles(
    failure_analysis_cases: list[dict[str, Any]],
    *,
    focused_recipes: tuple[str, ...] | list[str],
    primary_model_variant: str,
    model_variants: tuple[str, ...] | list[str],
    bundle_size: int,
    preview_chars: int,
    include_model_variants: tuple[str, ...] | list[str] | None = None,
) -> dict[str, list[dict[str, Any]]]:
    included_variants = tuple(include_model_variants or model_variants)
    bundles: dict[str, list[dict[str, Any]]] = {}
    for recipe_name in focused_recipes:
        recipe_cases = [
            case for case in failure_analysis_cases
            if case.get("perturbation_recipe") == recipe_name
        ]
        ranked = sorted(
            recipe_cases,
            key=lambda case: _bundle_priority(
                case,
                recipe_name=recipe_name,
                primary_model_variant=primary_model_variant,
                included_variants=included_variants,
            ),
            reverse=True,
        )
        bundles[recipe_name] = [
            _build_selected_example(
                candidate,
                selection_category=_best_category_for_case(
                    candidate,
                    recipe_name=recipe_name,
                    primary_model_variant=primary_model_variant,
                    included_variants=included_variants,
                ),
                selection_reason=_selection_reason(
                    _best_category_for_case(
                        candidate,
                        recipe_name=recipe_name,
                        primary_model_variant=primary_model_variant,
                        included_variants=included_variants,
                    )
                ),
                included_variants=included_variants,
                preview_chars=preview_chars,
                priority_score=_bundle_priority(
                    candidate,
                    recipe_name=recipe_name,
                    primary_model_variant=primary_model_variant,
                    included_variants=included_variants,
                ),
            )
            for candidate in ranked[:bundle_size]
        ]
    return bundles


def render_paper_qualitative_examples(
    report: dict[str, Any],
    examples: list[dict[str, Any]],
) -> str:
    lines = [
        "# Paper-Facing Qualitative Examples",
        "",
        f"- Focused recipes: `{report['focused_recipes']}`",
        f"- Primary model variant: `{report['primary_model_variant']}`",
        f"- Included model variants: `{report['included_model_variants']}`",
        "",
    ]
    for recipe_summary in report.get("per_recipe_summary", []):
        lines.extend([f"## {recipe_summary['recipe_name']}", ""])
        lines.append(f"- Selected examples: `{recipe_summary['selected_example_count']}`")
        lines.append(f"- Category counts: `{recipe_summary['selection_category_counts']}`")
        lines.append("")
        for example in [row for row in examples if row["perturbation_recipe"] == recipe_summary["recipe_name"]]:
            lines.append(f"### {example['case_id']} | {example['selection_category']}")
            lines.append("")
            lines.append(f"- Gold label: `{example['gold_label']}`")
            lines.append(f"- Selection rationale: {example['selection_reason']}")
            lines.append(f"- Target section: `{example['target_section']}`")
            lines.append(f"- Target section was empty: `{example['target_section_was_empty']}`")
            lines.append(f"- Section presence: `{example['section_presence_summary']}`")
            lines.append("- Predictions:")
            for model_variant, payload in example["per_model_predictions"].items():
                lines.append(
                    f"  `{model_variant}`: reference `{payload['reference_prediction']}` -> perturbed `{payload['perturbed_prediction']}`, "
                    f"flipped `{payload['prediction_flipped']}`, perturbed_correct `{payload['perturbed_correct']}`"
                )
            lines.append("- Section previews:")
            for section_name, preview in example["section_previews"].items():
                lines.append(f"  `{section_name}`: {preview}")
            lines.append("")
    return "\n".join(lines).strip() + "\n"


def _select_examples_for_recipe(
    recipe_cases: list[dict[str, Any]],
    *,
    recipe_name: str,
    primary_model_variant: str,
    included_variants: tuple[str, ...],
    count: int,
    preview_chars: int,
    examples_per_category: int,
) -> list[dict[str, Any]]:
    categories = _category_priority_for_recipe(recipe_name)
    selected: list[dict[str, Any]] = []
    selected_case_ids: set[str] = set()
    for category in categories:
        candidates = [
            case for case in recipe_cases
            if category in _candidate_categories(
                case,
                recipe_name=recipe_name,
                primary_model_variant=primary_model_variant,
                included_variants=included_variants,
            )
        ]
        candidates.sort(
            key=lambda case: _bundle_priority(
                case,
                recipe_name=recipe_name,
                primary_model_variant=primary_model_variant,
                included_variants=included_variants,
            ),
            reverse=True,
        )
        taken = 0
        for candidate in candidates:
            if candidate["case_id"] in selected_case_ids:
                continue
            selected.append(
                _build_selected_example(
                    candidate,
                    selection_category=category,
                    selection_reason=_selection_reason(category),
                    included_variants=included_variants,
                    preview_chars=preview_chars,
                    priority_score=_bundle_priority(
                        candidate,
                        recipe_name=recipe_name,
                        primary_model_variant=primary_model_variant,
                        included_variants=included_variants,
                    ),
                )
            )
            selected_case_ids.add(candidate["case_id"])
            taken += 1
            if taken >= examples_per_category or len(selected) >= count:
                break
        if len(selected) >= count:
            return selected

    remaining = sorted(
        recipe_cases,
        key=lambda case: _bundle_priority(
            case,
            recipe_name=recipe_name,
            primary_model_variant=primary_model_variant,
            included_variants=included_variants,
        ),
        reverse=True,
    )
    for candidate in remaining:
        if candidate["case_id"] in selected_case_ids:
            continue
        fallback_category = _best_category_for_case(
            candidate,
            recipe_name=recipe_name,
            primary_model_variant=primary_model_variant,
            included_variants=included_variants,
        )
        selected.append(
            _build_selected_example(
                candidate,
                selection_category=fallback_category,
                selection_reason=_selection_reason(fallback_category),
                included_variants=included_variants,
                preview_chars=preview_chars,
                priority_score=_bundle_priority(
                    candidate,
                    recipe_name=recipe_name,
                    primary_model_variant=primary_model_variant,
                    included_variants=included_variants,
                ),
            )
        )
        selected_case_ids.add(candidate["case_id"])
        if len(selected) >= count:
            break
    return selected


def _candidate_categories(
    case: dict[str, Any],
    *,
    recipe_name: str,
    primary_model_variant: str,
    included_variants: tuple[str, ...],
) -> list[str]:
    categories: list[str] = []
    per_model = {
        model_variant: payload
        for model_variant, payload in dict(case.get("per_model_predictions", {})).items()
        if model_variant in included_variants
    }
    primary = per_model.get(primary_model_variant)
    if primary is None:
        return ["high_interest_remainder"]

    other_payloads = [
        payload
        for model_variant, payload in per_model.items()
        if model_variant != primary_model_variant
    ]
    other_correct_count = sum(payload["perturbed_correct"] for payload in other_payloads)
    other_wrong_count = sum(not payload["perturbed_correct"] for payload in other_payloads)

    if primary["perturbed_correct"] and other_wrong_count >= max(1, len(other_payloads) - 1):
        categories.append("apa_unique_success")
    if (not primary["perturbed_correct"]) and other_correct_count >= 1:
        categories.append("apa_unique_failure")
    if recipe_name == "drop_precedents" and (not case.get("target_section_was_empty", False)) and primary["prediction_flipped"]:
        categories.append("precedent_sensitive_flip")
    if recipe_name == "keep_reasoning_only" and primary["perturbed_correct"] and (not primary["prediction_flipped"]):
        categories.append("reasoning_sufficient_stable_case")
    if recipe_name == "keep_reasoning_only" and primary["reference_correct"] and (not primary["perturbed_correct"]):
        categories.append("reasoning_insufficient_failure_case")
    if per_model and all(not payload["perturbed_correct"] for payload in per_model.values()):
        categories.append("consensus_failure")
    if per_model and all(payload["reference_correct"] and payload["perturbed_correct"] for payload in per_model.values()):
        categories.append("consensus_robustness")
    if not categories:
        categories.append("high_interest_remainder")
    return categories


def _build_selected_example(
    case: dict[str, Any],
    *,
    selection_category: str,
    selection_reason: str,
    included_variants: tuple[str, ...],
    preview_chars: int,
    priority_score: int,
) -> dict[str, Any]:
    section_previews = {
        section: _truncate_text(preview, preview_chars=preview_chars)
        for section, preview in dict(case.get("section_previews", {})).items()
    }
    per_model_predictions = {
        model_variant: payload
        for model_variant, payload in dict(case.get("per_model_predictions", {})).items()
        if model_variant in included_variants
    }
    return {
        "case_id": case["case_id"],
        "split": case["split"],
        "gold_label": case["gold_label"],
        "perturbation_recipe": case["perturbation_recipe"],
        "selection_category": selection_category,
        "selection_tag": selection_category,
        "selection_reason": selection_reason,
        "target_section": case["target_section"],
        "target_section_was_empty": case["target_section_was_empty"],
        "section_presence_summary": dict(case.get("section_presence_summary", {})),
        "section_previews": section_previews,
        "per_model_predictions": per_model_predictions,
        "interestingness_score": case.get("interestingness_score", 0),
        "priority_score": priority_score,
        "source_file": case.get("source_file", ""),
        "source_metadata": dict(case.get("source_metadata", {})),
    }


def _bundle_priority(
    case: dict[str, Any],
    *,
    recipe_name: str,
    primary_model_variant: str,
    included_variants: tuple[str, ...],
) -> int:
    categories = _candidate_categories(
        case,
        recipe_name=recipe_name,
        primary_model_variant=primary_model_variant,
        included_variants=included_variants,
    )
    weights = {
        "apa_unique_success": 8,
        "apa_unique_failure": 7,
        "precedent_sensitive_flip": 6,
        "reasoning_sufficient_stable_case": 5,
        "reasoning_insufficient_failure_case": 6,
        "consensus_failure": 4,
        "consensus_robustness": 3,
        "high_interest_remainder": 1,
    }
    completeness = sum(bool(value) for value in dict(case.get("section_presence_summary", {})).values())
    return int(case.get("interestingness_score", 0)) + completeness + sum(weights.get(category, 0) for category in categories)


def _best_category_for_case(
    case: dict[str, Any],
    *,
    recipe_name: str,
    primary_model_variant: str,
    included_variants: tuple[str, ...],
) -> str:
    available = _candidate_categories(
        case,
        recipe_name=recipe_name,
        primary_model_variant=primary_model_variant,
        included_variants=included_variants,
    )
    for category in _category_priority_for_recipe(recipe_name):
        if category in available:
            return category
    return available[0]


def _category_priority_for_recipe(recipe_name: str) -> list[str]:
    if recipe_name == "drop_precedents":
        return [
            "apa_unique_success",
            "apa_unique_failure",
            "precedent_sensitive_flip",
            "consensus_failure",
            "consensus_robustness",
            "high_interest_remainder",
        ]
    return [
        "apa_unique_success",
        "apa_unique_failure",
        "reasoning_sufficient_stable_case",
        "reasoning_insufficient_failure_case",
        "consensus_failure",
        "consensus_robustness",
        "high_interest_remainder",
    ]


def _selection_reason(selection_category: str) -> str:
    reasons = {
        "apa_unique_success": "APA stays correct while all or most competing models fail under the perturbation.",
        "apa_unique_failure": "APA fails while at least one competing model remains correct, making the case useful for counterevidence.",
        "precedent_sensitive_flip": "Dropping precedents triggers a meaningful APA prediction change on a case with non-empty precedent content.",
        "reasoning_sufficient_stable_case": "Reasoning-only input preserves a correct and stable APA decision.",
        "reasoning_insufficient_failure_case": "Reasoning-only input is not sufficient for APA to preserve the correct label.",
        "consensus_failure": "All models fail after perturbation, making the case a strong shared-error example.",
        "consensus_robustness": "All models remain correct after perturbation, making the case a good control example.",
        "high_interest_remainder": "High-interest disagreement case selected to round out the qualitative package.",
    }
    return reasons.get(selection_category, "Selected as a high-interest qualitative example.")


def _truncate_text(text: str, *, preview_chars: int) -> str:
    if len(text) <= preview_chars:
        return text
    return text[:preview_chars].rstrip() + "..."
