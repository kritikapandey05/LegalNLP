from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any

from legal_robustness.robustness.types import PerturbedPredictionRecord


def build_stability_vs_correctness_summary(
    prediction_rows: list[PerturbedPredictionRecord],
    *,
    primary_split: str,
    focused_recipes: tuple[str, ...] | list[str],
    primary_model_variant: str,
    comparison_model_variants: tuple[str, ...] | list[str],
) -> dict[str, Any]:
    selected_variants = [primary_model_variant]
    for model_variant in comparison_model_variants:
        if model_variant not in selected_variants:
            selected_variants.append(model_variant)

    grouped_rows: dict[tuple[str, str], dict[str, PerturbedPredictionRecord]] = defaultdict(dict)
    for row in prediction_rows:
        if row.split != primary_split:
            continue
        if row.perturbation_recipe not in focused_recipes:
            continue
        model_variant = f"{row.model_name}::{row.input_variant}"
        if model_variant not in selected_variants:
            continue
        grouped_rows[(row.perturbation_recipe, row.case_id)][model_variant] = row

    recipe_summaries: list[dict[str, Any]] = []
    summary_takeaways: list[str] = []
    for recipe_name in focused_recipes:
        recipe_rows = [
            row_map
            for (group_recipe, _), row_map in grouped_rows.items()
            if group_recipe == recipe_name and primary_model_variant in row_map
        ]
        if not recipe_rows:
            continue
        per_model_slices: dict[str, Counter[str]] = {
            model_variant: Counter()
            for model_variant in selected_variants
        }
        pairwise_rows: list[dict[str, Any]] = []
        for model_variant in selected_variants:
            for row_map in recipe_rows:
                row = row_map.get(model_variant)
                if row is None:
                    continue
                slice_name = _slice_name(
                    prediction_flipped=row.prediction_flipped,
                    correct=row.prediction == row.gold_label,
                )
                per_model_slices[model_variant][slice_name] += 1

        for comparator_variant in comparison_model_variants:
            primary_correct_comparator_wrong = 0
            comparator_correct_primary_wrong = 0
            primary_flipped_comparator_stable = 0
            comparator_flipped_primary_stable = 0
            both_correct = 0
            both_wrong = 0
            both_stable = 0
            both_flipped = 0
            jointly_observed = 0
            for row_map in recipe_rows:
                primary_row = row_map.get(primary_model_variant)
                comparator_row = row_map.get(comparator_variant)
                if primary_row is None or comparator_row is None:
                    continue
                jointly_observed += 1
                primary_correct = primary_row.prediction == primary_row.gold_label
                comparator_correct = comparator_row.prediction == comparator_row.gold_label
                if primary_correct and not comparator_correct:
                    primary_correct_comparator_wrong += 1
                if comparator_correct and not primary_correct:
                    comparator_correct_primary_wrong += 1
                if primary_row.prediction_flipped and not comparator_row.prediction_flipped:
                    primary_flipped_comparator_stable += 1
                if comparator_row.prediction_flipped and not primary_row.prediction_flipped:
                    comparator_flipped_primary_stable += 1
                if primary_correct and comparator_correct:
                    both_correct += 1
                if (not primary_correct) and (not comparator_correct):
                    both_wrong += 1
                if (not primary_row.prediction_flipped) and (not comparator_row.prediction_flipped):
                    both_stable += 1
                if primary_row.prediction_flipped and comparator_row.prediction_flipped:
                    both_flipped += 1
            pairwise_rows.append(
                {
                    "comparator_model_variant": comparator_variant,
                    "joint_case_count": jointly_observed,
                    "primary_correct_comparator_wrong_count": primary_correct_comparator_wrong,
                    "comparator_correct_primary_wrong_count": comparator_correct_primary_wrong,
                    "primary_flipped_comparator_stable_count": primary_flipped_comparator_stable,
                    "comparator_flipped_primary_stable_count": comparator_flipped_primary_stable,
                    "both_correct_count": both_correct,
                    "both_wrong_count": both_wrong,
                    "both_stable_count": both_stable,
                    "both_flipped_count": both_flipped,
                    "correctness_advantage_margin_for_primary": (
                        primary_correct_comparator_wrong - comparator_correct_primary_wrong
                    ),
                    "stability_advantage_margin_for_primary": (
                        comparator_flipped_primary_stable - primary_flipped_comparator_stable
                    ),
                }
            )

        takeaways = _build_recipe_takeaways(
            recipe_name=recipe_name,
            primary_model_variant=primary_model_variant,
            pairwise_rows=pairwise_rows,
        )
        if takeaways:
            summary_takeaways.append(takeaways[0])
        recipe_summaries.append(
            {
                "recipe_name": recipe_name,
                "case_count": len(recipe_rows),
                "primary_model_variant": primary_model_variant,
                "per_model_stability_slices": {
                    model_variant: dict(counter)
                    for model_variant, counter in per_model_slices.items()
                },
                "pairwise_primary_comparisons": pairwise_rows,
                "takeaways": takeaways,
            }
        )

    return {
        "task": "stability_vs_correctness_summary",
        "primary_split": primary_split,
        "primary_model_variant": primary_model_variant,
        "comparison_model_variants": list(comparison_model_variants),
        "focused_recipes": list(focused_recipes),
        "recipe_summaries": recipe_summaries,
        "summary_takeaways": summary_takeaways,
    }


def render_stability_vs_correctness_summary(report: dict[str, Any]) -> str:
    lines = [
        "# Stability vs Correctness Summary",
        "",
        f"- Primary split: `{report['primary_split']}`",
        f"- Primary model variant: `{report['primary_model_variant']}`",
        f"- Comparison model variants: `{report['comparison_model_variants']}`",
        f"- Focused recipes: `{report['focused_recipes']}`",
        "",
    ]
    for recipe in report.get("recipe_summaries", []):
        lines.extend([f"## {recipe['recipe_name']}", ""])
        lines.append(f"- Cases analyzed: `{recipe['case_count']}`")
        lines.append("")
        lines.append("### Stability Slices")
        lines.append("")
        for model_variant, slices in recipe.get("per_model_stability_slices", {}).items():
            lines.append(
                f"- `{model_variant}`: stable_and_correct `{slices.get('stable_and_correct', 0)}`, "
                f"stable_but_wrong `{slices.get('stable_but_wrong', 0)}`, "
                f"flipped_to_correct `{slices.get('flipped_to_correct', 0)}`, "
                f"flipped_to_wrong `{slices.get('flipped_to_wrong', 0)}`"
            )
        lines.append("")
        lines.append("### APA Pairwise Comparisons")
        lines.append("")
        for pairwise in recipe.get("pairwise_primary_comparisons", []):
            lines.append(
                f"- `{report['primary_model_variant']}` vs `{pairwise['comparator_model_variant']}`: "
                f"APA-correct/comparator-wrong `{pairwise['primary_correct_comparator_wrong_count']}`, "
                f"comparator-correct/APA-wrong `{pairwise['comparator_correct_primary_wrong_count']}`, "
                f"APA-flipped/comparator-stable `{pairwise['primary_flipped_comparator_stable_count']}`, "
                f"comparator-flipped/APA-stable `{pairwise['comparator_flipped_primary_stable_count']}`"
            )
        if recipe.get("takeaways"):
            lines.extend(["", "### Takeaways", ""])
            for takeaway in recipe["takeaways"]:
                lines.append(f"- {takeaway}")
        lines.append("")
    if report.get("summary_takeaways"):
        lines.extend(["## Summary Takeaways", ""])
        for takeaway in report["summary_takeaways"]:
            lines.append(f"- {takeaway}")
        lines.append("")
    return "\n".join(lines).strip() + "\n"


def _slice_name(*, prediction_flipped: bool, correct: bool) -> str:
    if not prediction_flipped and correct:
        return "stable_and_correct"
    if not prediction_flipped and not correct:
        return "stable_but_wrong"
    if prediction_flipped and correct:
        return "flipped_to_correct"
    return "flipped_to_wrong"


def _build_recipe_takeaways(
    *,
    recipe_name: str,
    primary_model_variant: str,
    pairwise_rows: list[dict[str, Any]],
) -> list[str]:
    takeaways: list[str] = []
    for pairwise in pairwise_rows:
        comparator = pairwise["comparator_model_variant"]
        if pairwise["correctness_advantage_margin_for_primary"] > 0:
            takeaways.append(
                f"For `{recipe_name}`, `{primary_model_variant}` stays correct more often than `{comparator}` on the cases where the two disagree on correctness."
            )
        elif pairwise["correctness_advantage_margin_for_primary"] < 0:
            takeaways.append(
                f"For `{recipe_name}`, `{comparator}` stays correct more often than `{primary_model_variant}` on the hardest disagreement cases."
            )
        if pairwise["stability_advantage_margin_for_primary"] > 0:
            takeaways.append(
                f"`{primary_model_variant}` is more stable than `{comparator}` on `{recipe_name}` when only one of the two flips."
            )
        elif pairwise["stability_advantage_margin_for_primary"] < 0:
            takeaways.append(
                f"`{comparator}` is more stable than `{primary_model_variant}` on `{recipe_name}` when only one of the two flips."
            )
    return takeaways
