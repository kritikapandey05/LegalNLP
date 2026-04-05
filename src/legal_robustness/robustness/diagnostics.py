from __future__ import annotations

from typing import Any


def render_perturbed_evaluation_metrics(report: dict[str, Any]) -> str:
    lines = [
        "# Perturbed Evaluation Metrics",
        "",
        f"- Baseline run dir: `{report['baseline_run_dir']}`",
        f"- Selected model variants: `{report['selected_model_variants']}`",
        f"- Selected perturbation recipes: `{report['selected_perturbation_recipes']}`",
        f"- Evaluation splits: `{report['evaluation_splits']}`",
        "",
    ]
    for model_variant_key, payload in report["model_variant_results"].items():
        lines.extend([f"## {model_variant_key}", ""])
        for recipe_name, recipe_payload in payload["recipes"].items():
            lines.extend([f"### {recipe_name}", ""])
            coverage = recipe_payload["coverage"]
            lines.append(
                "- Coverage: "
                f"band `{coverage['coverage_band']}`, "
                f"effective non-empty coverage `{coverage['effective_non_empty_coverage']}`, "
                f"empty-target `{coverage['empty_target_count']}` of `{coverage['total_examples']}`"
            )
            for split_name, split_metrics in recipe_payload["metrics_by_split"].items():
                overall = split_metrics["overall_metrics"]
                lines.append(f"- {split_name} accuracy: `{overall['accuracy']}`")
                lines.append(f"- {split_name} macro F1: `{overall['macro_f1']}`")
                lines.append(f"- {split_name} accuracy delta vs reference: `{overall['accuracy_delta_vs_reference']}`")
                lines.append(f"- {split_name} macro F1 delta vs reference: `{overall['macro_f1_delta_vs_reference']}`")
                lines.append(f"- {split_name} flip rate: `{overall['flip_rate']}`")
                if "non_empty_target_metrics" in split_metrics:
                    targeted = split_metrics["non_empty_target_metrics"]
                    lines.append(
                        f"- {split_name} non-empty-target macro F1: `{targeted['macro_f1']}`"
                    )
                    lines.append(
                        f"- {split_name} non-empty-target macro F1 delta vs reference: "
                        f"`{targeted['macro_f1_delta_vs_reference']}`"
                    )
            lines.append("")
    if report.get("warnings"):
        lines.extend(["## Warnings", ""])
        for warning in report["warnings"]:
            lines.append(f"- {warning}")
        lines.append("")
    return "\n".join(lines).strip() + "\n"


def render_perturbation_coverage_report(report: dict[str, Any]) -> str:
    lines = [
        "# Perturbation Coverage Report",
        "",
        f"- Baseline run dir: `{report['baseline_run_dir']}`",
        "",
    ]
    for recipe in report["recipes"]:
        lines.append(
            f"- `{recipe['model_variant']} / {recipe['recipe_name']}`: coverage `{recipe['effective_non_empty_coverage']}`, "
            f"band `{recipe['coverage_band']}`, empty-target `{recipe['empty_target_count']}`, "
            f"recommended `{recipe['recommended_for_future_experiments']}`"
        )
    return "\n".join(lines).strip() + "\n"


def render_section_aware_robustness_report(report: dict[str, Any]) -> str:
    lines = [
        "# Section-Aware Robustness Report",
        "",
        f"- Primary split: `{report['primary_split']}`",
        "",
        "## Informative Perturbations",
        "",
    ]
    for row in report["informative_perturbations"]:
        lines.append(
            f"- `{row['recipe_name']}`: macro F1 `{row['macro_f1']}`, "
            f"delta `{row['macro_f1_delta_vs_reference']}`, "
            f"flip rate `{row['flip_rate']}`, "
            f"coverage `{row['effective_non_empty_coverage']}`"
        )
    lines.extend(["", "## Weak-Probe Perturbations", ""])
    for row in report["weak_probe_perturbations"]:
        lines.append(
            f"- `{row['recipe_name']}`: macro F1 `{row['macro_f1']}`, "
            f"delta `{row['macro_f1_delta_vs_reference']}`, "
            f"coverage `{row['effective_non_empty_coverage']}`"
        )
    if report.get("key_findings"):
        lines.extend(["", "## Key Findings", ""])
        for finding in report["key_findings"]:
            lines.append(f"- {finding}")
    if report.get("reference_context_variants"):
        lines.extend(["", "## Unperturbed Context", ""])
        for model_variant_key, variants in report["reference_context_variants"].items():
            for variant_name, metrics_by_split in variants.items():
                test_metrics = metrics_by_split.get(report["primary_split"])
                if not test_metrics:
                    continue
                lines.append(
                    f"- `{model_variant_key} / {variant_name}`: accuracy `{test_metrics['accuracy']}`, "
                    f"macro F1 `{test_metrics['macro_f1']}`"
                )
    return "\n".join(lines).strip() + "\n"


def render_first_robustness_phase_readiness_summary(report: dict[str, Any]) -> str:
    lines = [
        "# First Robustness Phase Readiness Summary",
        "",
        f"- Baseline run dir: `{report['baseline_run_dir']}`",
        f"- Informative perturbation count: `{report['informative_perturbation_count']}`",
        f"- Weak-probe count: `{report['weak_probe_count']}`",
        f"- Ready for pilot write-up: `{report['ready_for_pilot_writeup']}`",
        "",
        "## Recommendations",
        "",
    ]
    for recommendation in report["recommendations"]:
        lines.append(f"- {recommendation}")
    return "\n".join(lines).strip() + "\n"


def render_comparative_robustness_metrics(report: dict[str, Any]) -> str:
    lines = [
        "# Comparative Robustness Metrics",
        "",
        f"- Baseline run dir: `{report['baseline_run_dir']}`",
        f"- Primary split: `{report['primary_split']}`",
        f"- Primary input variant: `{report.get('primary_input_variant')}`",
        f"- Model variants: `{report['model_variants']}`",
        "",
    ]
    for recipe in report["recipes"]:
        lines.extend([f"## {recipe['recipe_name']}", ""])
        lines.append(
            f"- Coverage band `{recipe['coverage_band']}`, effective non-empty coverage `{recipe['effective_non_empty_coverage']}`"
        )
        for model_row in recipe["model_metrics"]:
            lines.append(
                f"- `{model_row['model_variant']}`: macro F1 `{model_row['macro_f1']}`, "
                f"delta `{model_row['macro_f1_delta_vs_reference']}`, "
                f"retention `{model_row['macro_f1_retention']}`, "
                f"flip rate `{model_row['flip_rate']}`"
            )
        for pairwise in recipe.get("pairwise_comparisons", []):
            lines.append(
                f"- Pairwise `{pairwise['left_model_variant']}` vs `{pairwise['right_model_variant']}`: "
                f"macro-F1-delta diff (right-left) `{pairwise['macro_f1_delta_difference_right_minus_left']}`, "
                f"more robust `{pairwise['more_robust_model_by_macro_f1_delta']}`"
            )
        lines.append("")
    return "\n".join(lines).strip() + "\n"


def render_comparative_section_aware_robustness_report(report: dict[str, Any]) -> str:
    lines = [
        "# Comparative Section-Aware Robustness Report",
        "",
        f"- Primary split: `{report['primary_split']}`",
        f"- Primary input variant: `{report.get('primary_input_variant')}`",
        f"- Primary reference model variant: `{report.get('primary_reference_model_variant')}`",
        "",
        "## Unperturbed Comparison",
        "",
    ]
    strongest = (
        report["unperturbed_comparison"].get("strongest_by_macro_f1_by_input_variant", {}).get(
            report.get("primary_input_variant")
        )
        or report["unperturbed_comparison"].get("strongest_by_macro_f1")
    )
    if strongest is not None:
        lines.append(
            f"- Strongest unperturbed model variant: `{strongest['model_variant']}` with macro F1 `{strongest['macro_f1']}`"
        )
    for row in report["unperturbed_comparison"].get("model_variants", []):
        lines.append(
            f"- `{row['model_variant']}`: accuracy `{row['accuracy']}`, macro F1 `{row['macro_f1']}`"
        )
    lines.extend(["", "## Informative Perturbation Comparison", ""])
    for row in report["informative_perturbation_comparison"]:
        lines.append(
            f"- `{row['recipe_name']}`: strongest under perturbation `{row['strongest_model_under_perturbation_by_macro_f1']}`, "
            f"most robust by delta `{row['more_robust_model_by_macro_f1_delta']}`, "
            f"most stable by flip rate `{row['most_stable_model_by_flip_rate']}`, "
            f"robustness margin `{row['robustness_margin_by_macro_f1_delta']}`, "
            f"coverage `{row['effective_non_empty_coverage']}`"
        )
        for model_row in row.get("model_metrics", []):
            lines.append(
                f"  {model_row['model_variant']}: macro F1 `{model_row['macro_f1']}`, "
                f"delta `{model_row['macro_f1_delta_vs_reference']}`, "
                f"flip rate `{model_row['flip_rate']}`"
            )
    lines.extend(["", "## Weak-Probe Perturbation Comparison", ""])
    for row in report["weak_probe_perturbation_comparison"]:
        lines.append(
            f"- `{row['recipe_name']}`: more robust `{row['more_robust_model_by_macro_f1_delta']}`, "
            f"most stable `{row['most_stable_model_by_flip_rate']}`, "
            f"robustness margin `{row['robustness_margin_by_macro_f1_delta']}`, "
            f"coverage `{row['effective_non_empty_coverage']}`"
        )
    if report.get("failure_analysis_summary"):
        failure_summary = report["failure_analysis_summary"]
        lines.extend(["", "## Focused Failure Analysis", ""])
        for recipe in failure_summary.get("focused_recipe_summaries", []):
            lines.append(
                f"- `{recipe['recipe_name']}`: prediction disagreement rate `{recipe['prediction_disagreement_rate']}`, "
                f"flip disagreement rate `{recipe['flip_disagreement_rate']}`, "
                f"primary-model advantage cases `{recipe['primary_model_correct_while_others_fail_count']}`"
            )
    if report.get("apa_focused_summary_rows"):
        lines.extend(["", "## APA-Centered Focused Summary", ""])
        for row in report["apa_focused_summary_rows"]:
            lines.append(
                f"- `{row['recipe_name']}`: APA macro F1 `{row['unperturbed_macro_f1']}` -> `{row['perturbed_macro_f1']}`, "
                f"delta `{row['delta_macro_f1']}`, flip rate `{row['flip_rate']}`, coverage `{row['effective_coverage']}`"
            )
    if report.get("stability_vs_correctness_summary"):
        stability = report["stability_vs_correctness_summary"]
        lines.extend(["", "## Stability vs Correctness", ""])
        for takeaway in stability.get("summary_takeaways", []):
            lines.append(f"- {takeaway}")
    if report.get("qualitative_artifact_references"):
        lines.extend(["", "## Qualitative Artifact References", ""])
        for key, value in report["qualitative_artifact_references"].items():
            lines.append(f"- `{key}`: `{value}`")
    if report.get("main_takeaways"):
        lines.extend(["", "## Main Takeaways", ""])
        for takeaway in report["main_takeaways"]:
            lines.append(f"- {takeaway}")
    if report.get("recommended_next_direction"):
        lines.extend(["", "## Recommended Next Direction", ""])
        for line in report["recommended_next_direction"]:
            lines.append(f"- {line}")
    return "\n".join(lines).strip() + "\n"


def render_comparative_robustness_next_step_summary(report: dict[str, Any]) -> str:
    lines = [
        "# Comparative Robustness Next-Step Summary",
        "",
        f"- Primary split: `{report['primary_split']}`",
        f"- Recommended primary model variant: `{report['recommended_primary_model_variant']}`",
        f"- Contextual model variant: `{report.get('contextual_model_variant')}`",
        f"- Unperturbed strongest model variant: `{report['unperturbed_strongest_model_variant']}`",
        f"- Most robust model variant by macro-F1 delta: `{report['most_robust_model_variant_by_macro_f1_delta']}`",
        f"- Most stable model variant by flip rate: `{report['most_stable_model_variant_by_flip_rate']}`",
        f"- Stability-vs-robustness tension detected: `{report['stability_vs_robustness_tension_detected']}`",
        f"- Informative perturbation count: `{report['informative_perturbation_count']}`",
        f"- Weak-probe count: `{report['weak_probe_count']}`",
        f"- Primary write-up perturbation: `{report.get('primary_writeup_perturbation')}`",
        f"- Secondary write-up perturbation: `{report.get('secondary_writeup_perturbation')}`",
        f"- Ready for pilot results section: `{report.get('ready_for_pilot_results_section')}`",
        f"- Recommended next major step: `{report.get('recommended_next_major_step')}`",
        "",
        "## Recommendations",
        "",
    ]
    for recommendation in report["recommendations"]:
        lines.append(f"- {recommendation}")
    return "\n".join(lines).strip() + "\n"


def render_failure_analysis_summary(report: dict[str, Any]) -> str:
    lines = [
        "# Failure Analysis Summary",
        "",
        f"- Primary split: `{report['primary_split']}`",
        f"- Primary model variant: `{report['primary_model_variant']}`",
        f"- Contextual model variant: `{report.get('contextual_model_variant')}`",
        f"- Focused recipes: `{report['focused_recipes']}`",
        f"- Model variants: `{report['model_variants']}`",
        "",
    ]
    for recipe in report.get("focused_recipe_summaries", []):
        lines.extend([f"## {recipe['recipe_name']}", ""])
        lines.append(f"- Cases analyzed: `{recipe['case_count']}`")
        lines.append(f"- Non-empty target cases: `{recipe['non_empty_target_case_count']}`")
        lines.append(f"- Empty-target cases: `{recipe['empty_target_case_count']}`")
        lines.append(f"- Prediction disagreement rate: `{recipe['prediction_disagreement_rate']}`")
        lines.append(f"- Flip disagreement rate: `{recipe['flip_disagreement_rate']}`")
        lines.append(
            f"- Primary-model advantage cases: `{recipe['primary_model_correct_while_others_fail_count']}`"
        )
        lines.append(
            f"- Primary-model disadvantage cases: `{recipe['primary_model_wrong_while_others_correct_count']}`"
        )
        lines.append(
            f"- Primary-model flipped while others stable: `{recipe['primary_model_flipped_while_others_stable_count']}`"
        )
        lines.append(
            f"- Contextual unique successes: `{recipe.get('contextual_unique_success_count', 0)}`"
        )
        lines.append(
            f"- Contextual unique failures: `{recipe.get('contextual_unique_failure_count', 0)}`"
        )
        lines.append(
            f"- Linear-model agreement vs contextual difference: `{recipe.get('linear_models_agree_contextual_diff_count', 0)}`"
        )
        lines.append(
            f"- Contextual+APA agree while NB/logistic differ: `{recipe.get('contextual_and_apa_agree_nb_logistic_diff_count', 0)}`"
        )
        lines.append(f"- All-models fail count: `{recipe.get('all_models_fail_count', 0)}`")
        lines.append(f"- All-models flip count: `{recipe.get('all_models_flip_count', 0)}`")
        lines.append(f"- All-models stable count: `{recipe.get('all_models_stable_count', 0)}`")
        lines.append("")
        lines.append("### Stability vs Correctness Slices")
        lines.append("")
        for model_variant, slices in recipe.get("per_model_stability_slices", {}).items():
            lines.append(
                f"- `{model_variant}`: stable_and_correct `{slices.get('stable_and_correct', 0)}`, "
                f"stable_but_wrong `{slices.get('stable_but_wrong', 0)}`, "
                f"flipped_to_correct `{slices.get('flipped_to_correct', 0)}`, "
                f"flipped_to_wrong `{slices.get('flipped_to_wrong', 0)}`"
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
