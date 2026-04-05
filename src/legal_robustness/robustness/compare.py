from __future__ import annotations

from collections import Counter
from itertools import combinations
from typing import Any


def build_perturbation_coverage_report(evaluation_report: dict[str, Any]) -> dict[str, Any]:
    recipes: list[dict[str, Any]] = []
    for model_variant_key, payload in evaluation_report["model_variant_results"].items():
        for recipe_name, recipe_payload in payload["recipes"].items():
            coverage = recipe_payload["coverage"]
            recipes.append(
                {
                    "model_variant": model_variant_key,
                    "recipe_name": recipe_name,
                    "target_section": coverage["target_section"],
                    "coverage_band": coverage["coverage_band"],
                    "effective_non_empty_coverage": coverage["effective_non_empty_coverage"],
                    "empty_target_count": coverage["empty_target_count"],
                    "non_empty_target_count": coverage["non_empty_target_count"],
                    "recommended_for_future_experiments": coverage["recommended_for_future_experiments"],
                    "note": coverage["note"],
                }
            )
    return {
        "task": "perturbation_coverage",
        "baseline_run_dir": evaluation_report["baseline_run_dir"],
        "recipes": recipes,
    }


def build_section_aware_robustness_report(
    evaluation_report: dict[str, Any],
    *,
    primary_split: str,
    isolate_low_coverage: bool,
) -> dict[str, Any]:
    informative: list[dict[str, Any]] = []
    weak_probes: list[dict[str, Any]] = []
    for model_variant_key, payload in evaluation_report["model_variant_results"].items():
        for recipe_name, recipe_payload in payload["recipes"].items():
            split_metrics = recipe_payload["metrics_by_split"].get(primary_split)
            if not split_metrics:
                continue
            row = _recipe_summary_row(
                model_variant_key=model_variant_key,
                recipe_name=recipe_name,
                recipe_payload=recipe_payload,
                split_name=primary_split,
            )
            if recipe_payload["coverage"]["coverage_band"] == "low_coverage" and isolate_low_coverage:
                weak_probes.append(row)
            else:
                informative.append(row)

    informative.sort(key=lambda item: item["macro_f1_delta_vs_reference"])
    weak_probes.sort(key=lambda item: item["effective_non_empty_coverage"])
    return {
        "task": "section_aware_robustness",
        "primary_split": primary_split,
        "informative_perturbations": informative,
        "weak_probe_perturbations": weak_probes,
        "key_findings": _build_key_findings(informative, weak_probes),
        "reference_context_variants": evaluation_report.get("reference_context_variants", {}),
    }


def build_first_robustness_phase_readiness_summary(
    evaluation_report: dict[str, Any],
    robustness_report: dict[str, Any],
) -> dict[str, Any]:
    informative = robustness_report["informative_perturbations"]
    weak = robustness_report["weak_probe_perturbations"]
    recommendations: list[str] = []
    if informative:
        recommendations.append(
            "The pipeline is ready for a first pilot section-aware robustness write-up centered on higher-coverage perturbations."
        )
    if any(row["recipe_name"] == "drop_precedents" for row in informative):
        recommendations.append(
            "Precedent-focused perturbations are currently the strongest candidate for expansion because coverage is materially better than conclusion-focused probes."
        )
    if any(row["recipe_name"] == "keep_reasoning_only" for row in informative):
        recommendations.append(
            "Reasoning-only comparisons are now usable as a first section-ablation signal, although they should still be framed as pilot results because the underlying classifier is weak."
        )
    if weak:
        recommendations.append(
            "Low-coverage conclusion perturbations should remain in the appendix or caveat-heavy diagnostics rather than being treated as primary evidence."
        )
    recommendations.append(
        "The next best experiment is to score the same perturbation families with at least one stronger baseline while keeping pseudo_all_sections as the primary reference input."
    )
    return {
        "task": "first_robustness_phase_readiness",
        "baseline_run_dir": evaluation_report["baseline_run_dir"],
        "informative_perturbation_count": len(informative),
        "weak_probe_count": len(weak),
        "recommendations": recommendations,
        "ready_for_pilot_writeup": bool(informative),
    }


def build_comparative_robustness_metrics(
    evaluation_report: dict[str, Any],
    *,
    primary_split: str,
    include_relative_retention: bool = True,
) -> dict[str, Any]:
    model_variant_order = list(evaluation_report.get("selected_model_variants", ()))
    primary_input_variant = None
    if model_variant_order and all("::" in value for value in model_variant_order):
        input_variants = {value.split("::", maxsplit=1)[1] for value in model_variant_order}
        if len(input_variants) == 1:
            primary_input_variant = next(iter(input_variants))
    recipes: list[dict[str, Any]] = []
    recipe_index: dict[str, dict[str, Any]] = {}
    for model_variant_key in model_variant_order:
        payload = evaluation_report["model_variant_results"].get(model_variant_key, {})
        for recipe_name, recipe_payload in payload.get("recipes", {}).items():
            split_metrics = recipe_payload["metrics_by_split"].get(primary_split)
            if not split_metrics:
                continue
            recipe_entry = recipe_index.setdefault(
                recipe_name,
                {
                    "recipe_name": recipe_name,
                    "coverage_band": recipe_payload["coverage"]["coverage_band"],
                    "effective_non_empty_coverage": recipe_payload["coverage"]["effective_non_empty_coverage"],
                    "target_section": recipe_payload["coverage"]["target_section"],
                    "empty_target_count": recipe_payload["coverage"]["empty_target_count"],
                    "non_empty_target_count": recipe_payload["coverage"]["non_empty_target_count"],
                    "model_metrics": [],
                    "pairwise_comparisons": [],
                },
            )
            recipe_entry["model_metrics"].append(
                _comparative_model_metric_row(
                    model_variant_key=model_variant_key,
                    split_metrics=split_metrics,
                    include_relative_retention=include_relative_retention,
                )
            )

    for recipe_name in evaluation_report.get("selected_perturbation_recipes", []):
        if recipe_name not in recipe_index:
            continue
        recipe_entry = recipe_index[recipe_name]
        recipe_entry["model_metrics"] = sorted(
            recipe_entry["model_metrics"],
            key=lambda row: model_variant_order.index(row["model_variant"])
            if row["model_variant"] in model_variant_order
            else len(model_variant_order),
        )
        recipe_entry["pairwise_comparisons"] = _build_pairwise_comparisons(
            recipe_entry["model_metrics"]
        )
        recipes.append(recipe_entry)

    return {
        "task": "comparative_robustness_metrics",
        "baseline_run_dir": evaluation_report["baseline_run_dir"],
        "primary_split": primary_split,
        "model_variants": model_variant_order,
        "primary_input_variant": primary_input_variant,
        "recipes": recipes,
    }


def build_comparative_section_aware_robustness_report(
    unperturbed_comparison: dict[str, Any],
    comparative_metrics: dict[str, Any],
    *,
    isolate_low_coverage: bool,
    failure_analysis_summary: dict[str, Any] | None = None,
) -> dict[str, Any]:
    primary_input_variant = comparative_metrics.get("primary_input_variant")
    primary_reference = (
        unperturbed_comparison.get("strongest_by_macro_f1_by_input_variant", {}).get(primary_input_variant)
        if primary_input_variant
        else None
    ) or unperturbed_comparison.get("strongest_by_macro_f1")
    informative: list[dict[str, Any]] = []
    weak_probes: list[dict[str, Any]] = []
    for recipe in comparative_metrics["recipes"]:
        row = _comparative_recipe_summary_row(recipe)
        if recipe["coverage_band"] == "low_coverage" and isolate_low_coverage:
            weak_probes.append(row)
        else:
            informative.append(row)

    informative.sort(key=lambda item: item["robustness_margin_by_macro_f1_delta"], reverse=True)
    weak_probes.sort(key=lambda item: item["effective_non_empty_coverage"])
    main_takeaways = _build_comparative_takeaways(
        unperturbed_comparison=unperturbed_comparison,
        primary_reference=primary_reference,
        informative=informative,
        weak_probes=weak_probes,
        failure_analysis_summary=failure_analysis_summary,
    )
    recommended_next_direction = _build_next_direction_recommendations(
        unperturbed_comparison=unperturbed_comparison,
        primary_reference=primary_reference,
        informative=informative,
        weak_probes=weak_probes,
        failure_analysis_summary=failure_analysis_summary,
    )
    return {
        "task": "comparative_section_aware_robustness",
        "primary_split": comparative_metrics["primary_split"],
        "primary_input_variant": primary_input_variant,
        "primary_reference_model_variant": primary_reference["model_variant"] if primary_reference else None,
        "unperturbed_comparison": unperturbed_comparison,
        "informative_perturbation_comparison": informative,
        "weak_probe_perturbation_comparison": weak_probes,
        "failure_analysis_summary": failure_analysis_summary,
        "main_takeaways": main_takeaways,
        "recommended_next_direction": recommended_next_direction,
    }


def build_comparative_robustness_next_step_summary(
    unperturbed_comparison: dict[str, Any],
    comparative_report: dict[str, Any],
    *,
    failure_analysis_summary: dict[str, Any] | None = None,
) -> dict[str, Any]:
    primary_input_variant = comparative_report.get("primary_input_variant")
    strongest_unperturbed = (
        unperturbed_comparison.get("strongest_by_macro_f1_by_input_variant", {}).get(primary_input_variant)
        if primary_input_variant
        else None
    ) or unperturbed_comparison.get("strongest_by_macro_f1")
    informative = comparative_report["informative_perturbation_comparison"]
    robustness_wins = Counter(
        row["more_robust_model_by_macro_f1_delta"]
        for row in informative
        if row.get("more_robust_model_by_macro_f1_delta")
    )
    most_robust = robustness_wins.most_common(1)[0][0] if robustness_wins else None
    stability_wins = Counter(
        row["most_stable_model_by_flip_rate"]
        for row in informative
        if row.get("most_stable_model_by_flip_rate")
    )
    most_stable = stability_wins.most_common(1)[0][0] if stability_wins else None
    stability_vs_robustness_tension_detected = any(
        not row.get("stability_vs_robustness_aligned", True)
        for row in informative
    )
    contextual_model_variant = next(
        (
            row["model_variant"]
            for row in unperturbed_comparison.get("model_variants", [])
            if "section_contextual_logistic_regression" in row["model_variant"]
        ),
        None,
    )

    recommended_primary_model = strongest_unperturbed["model_variant"] if strongest_unperturbed else None
    if recommended_primary_model is None:
        recommended_primary_model = most_robust or most_stable
    elif most_robust is not None and recommended_primary_model != most_robust and most_robust == most_stable:
        recommended_primary_model = most_robust

    primary_writeup_perturbation = None
    if informative:
        primary_writeup_perturbation = max(
            informative,
            key=lambda row: (
                row["effective_non_empty_coverage"],
                row["robustness_margin_by_macro_f1_delta"],
            ),
        )["recipe_name"]
    secondary_writeup_perturbation = None
    if len(informative) > 1:
        ranked = sorted(
            informative,
            key=lambda row: (
                row["robustness_margin_by_macro_f1_delta"],
                row["effective_non_empty_coverage"],
            ),
            reverse=True,
        )
        secondary_writeup_perturbation = next(
            (
                row["recipe_name"]
                for row in ranked
                if row["recipe_name"] != primary_writeup_perturbation
            ),
            None,
        )

    ready_for_pilot_results_section = bool(informative) and (
        primary_writeup_perturbation in {"drop_precedents", "keep_reasoning_only"}
    )

    recommendations: list[str] = []
    if recommended_primary_model is not None:
        recommendations.append(
            f"Use `{recommended_primary_model}` as the main baseline for the next perturbation phase."
        )
    if contextual_model_variant is not None:
        if recommended_primary_model == contextual_model_variant:
            recommendations.append(
                "The contextual model is now strong enough to carry the main pilot comparison, with the linear baselines retained as ablation-oriented reference points."
            )
        else:
            recommendations.append(
                "Keep the contextual model in the comparison stack, but do not promote it to the sole main baseline yet because the current robustness/stability tradeoff is still mixed."
            )
    if any(row["recipe_name"] == "drop_precedents" for row in informative):
        recommendations.append(
            "Keep `drop_precedents` in the core perturbation set because it has strong coverage and produces a stable cross-model comparison signal."
        )
    if any(row["recipe_name"] == "keep_reasoning_only" for row in informative):
        recommendations.append(
            "Keep `keep_reasoning_only` as the main ablation-style probe because it currently produces the largest robustness stress test."
        )
    if comparative_report["weak_probe_perturbation_comparison"]:
        recommendations.append(
            "Deprioritize conclusion-target perturbations for the main analysis until transferred conclusion coverage improves materially."
        )
    if stability_vs_robustness_tension_detected:
        recommendations.append(
            "Keep reporting both macro-F1 retention and flip-rate stability because the more accurate model is not always the most stable under perturbation."
        )
    if failure_analysis_summary and failure_analysis_summary.get("focused_recipe_summaries"):
        recommendations.append(
            "Use the focused failure-analysis slices to choose paper examples, especially for `drop_precedents` and `keep_reasoning_only`."
        )
    if ready_for_pilot_results_section:
        recommendations.append(
            "The project now has enough focused evidence for a pilot paper-facing robustness results subsection, as long as the claims stay anchored to predicted pseudo-sections and the two high-value perturbation probes."
        )
    recommended_next_major_step = (
        "paper_results_packaging"
        if ready_for_pilot_results_section
        else "one_last_targeted_experiment"
    )
    if recommended_next_major_step == "paper_results_packaging":
        recommendations.append(
            "The next best major step is to package the current APA-centered evidence for a pilot results section, while reserving section-transfer refinement as a secondary follow-up rather than the immediate priority."
        )
    else:
        recommendations.append(
            "The next best major step is one last targeted experiment on the current focused probes before shifting fully into paper-facing packaging."
        )

    return {
        "task": "comparative_robustness_next_step_summary",
        "primary_split": comparative_report["primary_split"],
        "recommended_primary_model_variant": recommended_primary_model,
        "contextual_model_variant": contextual_model_variant,
        "unperturbed_strongest_model_variant": strongest_unperturbed["model_variant"] if strongest_unperturbed else None,
        "most_robust_model_variant_by_macro_f1_delta": most_robust,
        "most_stable_model_variant_by_flip_rate": most_stable,
        "stability_vs_robustness_tension_detected": stability_vs_robustness_tension_detected,
        "informative_perturbation_count": len(informative),
        "weak_probe_count": len(comparative_report["weak_probe_perturbation_comparison"]),
        "primary_writeup_perturbation": primary_writeup_perturbation,
        "secondary_writeup_perturbation": secondary_writeup_perturbation,
        "ready_for_pilot_results_section": ready_for_pilot_results_section,
        "recommended_next_major_step": recommended_next_major_step,
        "recommendations": recommendations,
    }


def _recipe_summary_row(
    *,
    model_variant_key: str,
    recipe_name: str,
    recipe_payload: dict[str, Any],
    split_name: str,
) -> dict[str, Any]:
    overall = recipe_payload["metrics_by_split"][split_name]["overall_metrics"]
    coverage = recipe_payload["coverage"]
    row = {
        "model_variant": model_variant_key,
        "recipe_name": recipe_name,
        "coverage_band": coverage["coverage_band"],
        "effective_non_empty_coverage": coverage["effective_non_empty_coverage"],
        "target_section": coverage["target_section"],
        "accuracy": overall["accuracy"],
        "macro_f1": overall["macro_f1"],
        "accuracy_delta_vs_reference": overall["accuracy_delta_vs_reference"],
        "macro_f1_delta_vs_reference": overall["macro_f1_delta_vs_reference"],
        "flip_rate": overall["flip_rate"],
        "non_empty_target_flip_rate": overall["non_empty_target_flip_rate"],
        "reference_accuracy": overall["reference_accuracy"],
        "reference_macro_f1": overall["reference_macro_f1"],
    }
    if "non_empty_target_metrics" in recipe_payload["metrics_by_split"][split_name]:
        non_empty = recipe_payload["metrics_by_split"][split_name]["non_empty_target_metrics"]
        row["non_empty_target_accuracy"] = non_empty["accuracy"]
        row["non_empty_target_macro_f1"] = non_empty["macro_f1"]
        row["non_empty_target_accuracy_delta_vs_reference"] = non_empty["accuracy_delta_vs_reference"]
        row["non_empty_target_macro_f1_delta_vs_reference"] = non_empty["macro_f1_delta_vs_reference"]
    return row


def _build_key_findings(
    informative: list[dict[str, Any]],
    weak_probes: list[dict[str, Any]],
) -> list[str]:
    findings: list[str] = []
    if informative:
        biggest_drop = min(informative, key=lambda item: item["macro_f1_delta_vs_reference"])
        biggest_flip = max(informative, key=lambda item: item["flip_rate"])
        findings.append(
            f"Largest informative macro-F1 drop came from `{biggest_drop['recipe_name']}` with delta `{biggest_drop['macro_f1_delta_vs_reference']}`."
        )
        findings.append(
            f"Highest informative prediction flip rate came from `{biggest_flip['recipe_name']}` at `{biggest_flip['flip_rate']}`."
        )
        drop_precedents = next((row for row in informative if row["recipe_name"] == "drop_precedents"), None)
        reasoning_only = next((row for row in informative if row["recipe_name"] == "keep_reasoning_only"), None)
        reorder = next((row for row in informative if row["recipe_name"] == "reorder_conclusion_first"), None)
        if drop_precedents is not None:
            findings.append(
                f"`drop_precedents` changed test macro F1 by `{drop_precedents['macro_f1_delta_vs_reference']}` relative to unperturbed `pseudo_all_sections`."
            )
        if reasoning_only is not None:
            findings.append(
                f"`keep_reasoning_only` retained test macro F1 `{reasoning_only['macro_f1']}` with delta `{reasoning_only['macro_f1_delta_vs_reference']}`."
            )
        if reorder is not None:
            findings.append(
                f"`reorder_conclusion_first` produced macro-F1 delta `{reorder['macro_f1_delta_vs_reference']}`, which quantifies the first ordering-sensitivity probe."
            )
    if weak_probes:
        low = min(weak_probes, key=lambda item: item["effective_non_empty_coverage"])
        findings.append(
            f"`{low['recipe_name']}` is currently a weak probe because effective non-empty coverage is only `{low['effective_non_empty_coverage']}`."
        )
    return findings


def _comparative_model_metric_row(
    *,
    model_variant_key: str,
    split_metrics: dict[str, Any],
    include_relative_retention: bool,
) -> dict[str, Any]:
    overall = split_metrics["overall_metrics"]
    accuracy_retention = _safe_ratio(overall["accuracy"], overall["reference_accuracy"]) if include_relative_retention else None
    macro_f1_retention = _safe_ratio(overall["macro_f1"], overall["reference_macro_f1"]) if include_relative_retention else None
    return {
        "model_variant": model_variant_key,
        "accuracy": overall["accuracy"],
        "macro_f1": overall["macro_f1"],
        "accuracy_delta_vs_reference": overall["accuracy_delta_vs_reference"],
        "macro_f1_delta_vs_reference": overall["macro_f1_delta_vs_reference"],
        "accuracy_retention": accuracy_retention,
        "macro_f1_retention": macro_f1_retention,
        "flip_rate": overall["flip_rate"],
        "non_empty_target_flip_rate": overall["non_empty_target_flip_rate"],
        "reference_accuracy": overall["reference_accuracy"],
        "reference_macro_f1": overall["reference_macro_f1"],
        "non_empty_target_metrics": split_metrics.get("non_empty_target_metrics"),
    }


def _build_pairwise_comparisons(model_metrics: list[dict[str, Any]]) -> list[dict[str, Any]]:
    comparisons: list[dict[str, Any]] = []
    for left, right in combinations(model_metrics, 2):
        comparisons.append(
            {
                "left_model_variant": left["model_variant"],
                "right_model_variant": right["model_variant"],
                "accuracy_difference_right_minus_left": round(right["accuracy"] - left["accuracy"], 6),
                "macro_f1_difference_right_minus_left": round(right["macro_f1"] - left["macro_f1"], 6),
                "reference_accuracy_difference_right_minus_left": round(
                    right["reference_accuracy"] - left["reference_accuracy"],
                    6,
                ),
                "reference_macro_f1_difference_right_minus_left": round(
                    right["reference_macro_f1"] - left["reference_macro_f1"],
                    6,
                ),
                "accuracy_delta_difference_right_minus_left": round(
                    right["accuracy_delta_vs_reference"] - left["accuracy_delta_vs_reference"],
                    6,
                ),
                "macro_f1_delta_difference_right_minus_left": round(
                    right["macro_f1_delta_vs_reference"] - left["macro_f1_delta_vs_reference"],
                    6,
                ),
                "accuracy_gap_change_right_minus_left": round(
                    (right["accuracy"] - left["accuracy"]) - (right["reference_accuracy"] - left["reference_accuracy"]),
                    6,
                ),
                "macro_f1_gap_change_right_minus_left": round(
                    (right["macro_f1"] - left["macro_f1"]) - (right["reference_macro_f1"] - left["reference_macro_f1"]),
                    6,
                ),
                "accuracy_retention_difference_right_minus_left": round(
                    (right["accuracy_retention"] or 0.0) - (left["accuracy_retention"] or 0.0),
                    6,
                ),
                "macro_f1_retention_difference_right_minus_left": round(
                    (right["macro_f1_retention"] or 0.0) - (left["macro_f1_retention"] or 0.0),
                    6,
                ),
                "more_robust_model_by_accuracy_delta": _winner_by_higher_metric(
                    left["model_variant"],
                    right["model_variant"],
                    left["accuracy_delta_vs_reference"],
                    right["accuracy_delta_vs_reference"],
                ),
                "more_robust_model_by_macro_f1_delta": _winner_by_higher_metric(
                    left["model_variant"],
                    right["model_variant"],
                    left["macro_f1_delta_vs_reference"],
                    right["macro_f1_delta_vs_reference"],
                ),
                "more_robust_model_by_macro_f1_retention": _winner_by_higher_metric(
                    left["model_variant"],
                    right["model_variant"],
                    left["macro_f1_retention"] or 0.0,
                    right["macro_f1_retention"] or 0.0,
                ),
                "stronger_model_under_perturbation_by_macro_f1": _winner_by_higher_metric(
                    left["model_variant"],
                    right["model_variant"],
                    left["macro_f1"],
                    right["macro_f1"],
                ),
                "stronger_model_unperturbed_by_macro_f1": _winner_by_higher_metric(
                    left["model_variant"],
                    right["model_variant"],
                    left["reference_macro_f1"],
                    right["reference_macro_f1"],
                ),
            }
        )
    return comparisons


def _comparative_recipe_summary_row(recipe: dict[str, Any]) -> dict[str, Any]:
    model_metrics = list(recipe.get("model_metrics", []))
    pairwise = list(recipe.get("pairwise_comparisons", []))
    strongest_model = max(model_metrics, key=lambda row: row["macro_f1"], default=None)
    most_robust_model = max(
        model_metrics,
        key=lambda row: row["macro_f1_delta_vs_reference"],
        default=None,
    )
    most_stable_model = min(
        model_metrics,
        key=lambda row: row["flip_rate"],
        default=None,
    )
    most_robust_by_retention = max(
        model_metrics,
        key=lambda row: row["macro_f1_retention"] if row["macro_f1_retention"] is not None else float("-inf"),
        default=None,
    )
    macro_f1_deltas = [row["macro_f1_delta_vs_reference"] for row in model_metrics]
    flip_rates = [row["flip_rate"] for row in model_metrics]
    macro_f1_values = [row["macro_f1"] for row in model_metrics]
    retention_values = [
        row["macro_f1_retention"]
        for row in model_metrics
        if row["macro_f1_retention"] is not None
    ]
    return {
        "recipe_name": recipe["recipe_name"],
        "coverage_band": recipe["coverage_band"],
        "effective_non_empty_coverage": recipe["effective_non_empty_coverage"],
        "target_section": recipe["target_section"],
        "model_metrics": model_metrics,
        "pairwise_comparisons": pairwise,
        "more_robust_model_by_macro_f1_delta": most_robust_model["model_variant"] if most_robust_model else None,
        "more_robust_model_by_macro_f1_retention": most_robust_by_retention["model_variant"] if most_robust_by_retention else None,
        "strongest_model_under_perturbation_by_macro_f1": strongest_model["model_variant"] if strongest_model else None,
        "most_stable_model_by_flip_rate": most_stable_model["model_variant"] if most_stable_model else None,
        "robustness_margin_by_macro_f1_delta": round(max(macro_f1_deltas) - min(macro_f1_deltas), 6) if macro_f1_deltas else 0.0,
        "stability_margin_by_flip_rate": round(max(flip_rates) - min(flip_rates), 6) if flip_rates else 0.0,
        "macro_f1_gap_under_perturbation": round(max(macro_f1_values) - min(macro_f1_values), 6) if macro_f1_values else 0.0,
        "macro_f1_retention_margin": round(max(retention_values) - min(retention_values), 6) if retention_values else None,
        "stability_vs_robustness_aligned": (
            most_stable_model["model_variant"] == most_robust_model["model_variant"]
            if most_stable_model and most_robust_model
            else True
        ),
    }


def _build_comparative_takeaways(
    *,
    unperturbed_comparison: dict[str, Any],
    primary_reference: dict[str, Any] | None,
    informative: list[dict[str, Any]],
    weak_probes: list[dict[str, Any]],
    failure_analysis_summary: dict[str, Any] | None = None,
) -> list[str]:
    takeaways: list[str] = []
    strongest = primary_reference or unperturbed_comparison.get("strongest_by_macro_f1")
    if strongest is not None:
        takeaways.append(
            f"Strongest unperturbed model variant on `{unperturbed_comparison['primary_split']}` was `{strongest['model_variant']}` with macro F1 `{strongest['macro_f1']}`."
        )
    if informative:
        largest_stress = max(informative, key=lambda row: row["robustness_margin_by_macro_f1_delta"])
        takeaways.append(
            f"Largest cross-model robustness separation came from `{largest_stress['recipe_name']}` with macro-F1-delta margin `{largest_stress['robustness_margin_by_macro_f1_delta']}`."
        )
        robustness_wins = Counter(
            row["more_robust_model_by_macro_f1_delta"]
            for row in informative
            if row.get("more_robust_model_by_macro_f1_delta")
        )
        if robustness_wins:
            winner, count = robustness_wins.most_common(1)[0]
            takeaways.append(
                f"`{winner}` was more robust by macro-F1 delta on `{count}` informative perturbation(s)."
            )
        stability_wins = Counter(
            row["most_stable_model_by_flip_rate"]
            for row in informative
            if row.get("most_stable_model_by_flip_rate")
        )
        if stability_wins:
            winner, count = stability_wins.most_common(1)[0]
            takeaways.append(
                f"`{winner}` was the most stable by flip rate on `{count}` informative perturbation(s)."
            )
        if any(not row.get("stability_vs_robustness_aligned", True) for row in informative):
            takeaways.append(
                "Stability and robustness are not perfectly aligned: the model with the best macro-F1 retention is not always the model with the lowest flip rate."
            )
        drop_precedents = next((row for row in informative if row["recipe_name"] == "drop_precedents"), None)
        reasoning_only = next((row for row in informative if row["recipe_name"] == "keep_reasoning_only"), None)
        if drop_precedents is not None:
            takeaways.append(
                f"`drop_precedents` remains a centerpiece perturbation because coverage is `{drop_precedents['effective_non_empty_coverage']}` and the robustness margin is `{drop_precedents['robustness_margin_by_macro_f1_delta']}`."
            )
        if reasoning_only is not None:
            takeaways.append(
                f"`keep_reasoning_only` remains the strongest ablation-style stress test, with macro-F1-delta margin `{reasoning_only['robustness_margin_by_macro_f1_delta']}` and stability gap `{reasoning_only['stability_margin_by_flip_rate']}`."
            )
    if weak_probes:
        low = min(weak_probes, key=lambda item: item["effective_non_empty_coverage"])
        takeaways.append(
            f"`{low['recipe_name']}` remains a weak probe because effective non-empty coverage is only `{low['effective_non_empty_coverage']}`."
        )
    if failure_analysis_summary and failure_analysis_summary.get("summary_takeaways"):
        takeaways.extend(failure_analysis_summary["summary_takeaways"][:2])
    return takeaways


def _build_next_direction_recommendations(
    *,
    unperturbed_comparison: dict[str, Any],
    primary_reference: dict[str, Any] | None,
    informative: list[dict[str, Any]],
    weak_probes: list[dict[str, Any]],
    failure_analysis_summary: dict[str, Any] | None = None,
) -> list[str]:
    recommendations: list[str] = []
    strongest = primary_reference or unperturbed_comparison.get("strongest_by_macro_f1")
    if strongest is not None:
        recommendations.append(
            f"Use `{strongest['model_variant']}` as the reference point for the next write-up unless a weaker unperturbed model proves consistently more robust across expanded perturbation families."
        )
    if any(row["recipe_name"] == "drop_precedents" for row in informative):
        recommendations.append(
            "Keep `drop_precedents` in the core benchmark because it provides both high coverage and a clear model-comparison signal."
        )
    if any(row["recipe_name"] == "keep_reasoning_only" for row in informative):
        recommendations.append(
            "Keep `keep_reasoning_only` as the strongest section-ablation probe in the next phase."
        )
    if any(row["recipe_name"] == "reorder_conclusion_first" for row in informative):
        recommendations.append(
            "Retain `reorder_conclusion_first` as a lightweight ordering-sensitivity control, but treat it as secondary to precedent and reasoning perturbations."
        )
    if weak_probes:
        recommendations.append(
            "Do not center the next phase on conclusion-target perturbations until transferred conclusion coverage improves."
        )
    if failure_analysis_summary and failure_analysis_summary.get("focused_recipe_summaries"):
        recommendations.append(
            "Keep focused failure analysis on `drop_precedents` and `keep_reasoning_only`, because those two recipes now provide the clearest cross-model error patterns."
        )
    return recommendations


def _safe_ratio(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return 0.0
    return round(numerator / denominator, 6)


def _winner_by_higher_metric(
    left_model_variant: str,
    right_model_variant: str,
    left_value: float,
    right_value: float,
) -> str:
    return right_model_variant if right_value > left_value else left_model_variant
