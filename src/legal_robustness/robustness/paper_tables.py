from __future__ import annotations

from typing import Any


def build_apa_focused_robustness_table(
    unperturbed_comparison: dict[str, Any],
    comparative_metrics: dict[str, Any],
    *,
    primary_model_variant: str,
    comparison_model_variants: tuple[str, ...] | list[str],
    focused_recipes: tuple[str, ...] | list[str],
) -> dict[str, Any]:
    unperturbed_lookup = {
        row["model_variant"]: row
        for row in unperturbed_comparison.get("model_variants", [])
    }
    recipe_lookup = {
        row["recipe_name"]: row
        for row in comparative_metrics.get("recipes", [])
    }
    rows: list[dict[str, Any]] = []
    comparison_variant_order = [
        model_variant
        for model_variant in comparison_model_variants
        if model_variant != primary_model_variant
    ]
    for recipe_name in focused_recipes:
        recipe = recipe_lookup.get(recipe_name)
        primary_reference = unperturbed_lookup.get(primary_model_variant)
        if recipe is None or primary_reference is None:
            continue
        model_metric_lookup = {
            row["model_variant"]: row
            for row in recipe.get("model_metrics", [])
        }
        primary_metrics = model_metric_lookup.get(primary_model_variant)
        if primary_metrics is None:
            continue
        comparison_rows = [
            model_metric_lookup[model_variant]
            for model_variant in comparison_variant_order
            if model_variant in model_metric_lookup
        ]
        rows.append(
            {
                "recipe_name": recipe_name,
                "target_section": recipe["target_section"],
                "effective_coverage": recipe["effective_non_empty_coverage"],
                "coverage_band": recipe["coverage_band"],
                "primary_model_variant": primary_model_variant,
                "unperturbed_macro_f1": primary_reference["macro_f1"],
                "perturbed_macro_f1": primary_metrics["macro_f1"],
                "delta_macro_f1": primary_metrics["macro_f1_delta_vs_reference"],
                "unperturbed_accuracy": primary_reference["accuracy"],
                "perturbed_accuracy": primary_metrics["accuracy"],
                "delta_accuracy": primary_metrics["accuracy_delta_vs_reference"],
                "flip_rate": primary_metrics["flip_rate"],
                "relative_retention": primary_metrics.get("macro_f1_retention"),
                "short_interpretation": _build_short_interpretation(
                    primary_model_variant=primary_model_variant,
                    primary_metrics=primary_metrics,
                    recipe_model_metrics=list(recipe.get("model_metrics", [])),
                ),
                "comparison_models": [
                    {
                        "model_variant": row["model_variant"],
                        "perturbed_macro_f1": row["macro_f1"],
                        "delta_macro_f1": row["macro_f1_delta_vs_reference"],
                        "perturbed_accuracy": row["accuracy"],
                        "delta_accuracy": row["accuracy_delta_vs_reference"],
                        "flip_rate": row["flip_rate"],
                    }
                    for row in comparison_rows
                ],
            }
        )
    return {
        "task": "apa_focused_robustness_table",
        "primary_split": comparative_metrics.get("primary_split"),
        "primary_model_variant": primary_model_variant,
        "focused_recipes": list(focused_recipes),
        "comparison_model_variants": comparison_variant_order,
        "rows": rows,
    }


def render_apa_focused_robustness_table(report: dict[str, Any]) -> str:
    lines = [
        "# APA-Focused Robustness Table",
        "",
        f"- Primary split: `{report['primary_split']}`",
        f"- Primary model variant: `{report['primary_model_variant']}`",
        f"- Focused recipes: `{report['focused_recipes']}`",
        "",
        "| Perturbation | Coverage | APA unpert. macro F1 | APA pert. macro F1 | Delta macro F1 | APA unpert. acc. | APA pert. acc. | Delta acc. | Flip rate | Interpretation |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in report.get("rows", []):
        lines.append(
            f"| `{row['recipe_name']}` | `{row['effective_coverage']}` | `{row['unperturbed_macro_f1']}` | "
            f"`{row['perturbed_macro_f1']}` | `{row['delta_macro_f1']}` | `{row['unperturbed_accuracy']}` | "
            f"`{row['perturbed_accuracy']}` | `{row['delta_accuracy']}` | `{row['flip_rate']}` | {row['short_interpretation']} |"
        )
    if report.get("rows"):
        lines.extend(["", "## Comparator Context", ""])
        for row in report["rows"]:
            lines.append(f"### {row['recipe_name']}")
            lines.append("")
            for comparator in row.get("comparison_models", []):
                lines.append(
                    f"- `{comparator['model_variant']}`: perturbed macro F1 `{comparator['perturbed_macro_f1']}`, "
                    f"delta `{comparator['delta_macro_f1']}`, perturbed accuracy `{comparator['perturbed_accuracy']}`, "
                    f"delta accuracy `{comparator['delta_accuracy']}`, flip rate `{comparator['flip_rate']}`"
                )
            lines.append("")
    return "\n".join(lines).strip() + "\n"


def _build_short_interpretation(
    *,
    primary_model_variant: str,
    primary_metrics: dict[str, Any],
    recipe_model_metrics: list[dict[str, Any]],
) -> str:
    strongest = max(
        recipe_model_metrics,
        key=lambda row: row["macro_f1"],
        default=None,
    )
    most_robust = max(
        recipe_model_metrics,
        key=lambda row: row["macro_f1_delta_vs_reference"],
        default=None,
    )
    most_stable = min(
        recipe_model_metrics,
        key=lambda row: row["flip_rate"],
        default=None,
    )
    strongest_variant = strongest["model_variant"] if strongest else None
    most_robust_variant = most_robust["model_variant"] if most_robust else None
    most_stable_variant = most_stable["model_variant"] if most_stable else None
    if strongest_variant == primary_model_variant and most_robust_variant == primary_model_variant:
        return "APA is strongest both in absolute performance and retention on this probe."
    if strongest_variant == primary_model_variant and most_stable_variant == primary_model_variant:
        return "APA stays strongest and most stable here, even if another model retains metric slightly better."
    if strongest_variant == primary_model_variant:
        return "APA keeps the best absolute score, but another model retains performance or stability better."
    if most_robust_variant == primary_model_variant:
        return "APA is not strongest in absolute score, but it loses less performance than its peers."
    if most_stable_variant == primary_model_variant:
        return "APA is not strongest overall, but it offers the cleanest stability profile under perturbation."
    if primary_metrics["macro_f1"] >= 0.5:
        return "APA remains competitive in absolute score, but at least one simpler comparator is stronger and/or more robust here."
    return "APA trails at least one simpler comparator on both absolute score and retention for this probe."
