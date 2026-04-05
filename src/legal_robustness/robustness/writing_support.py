from __future__ import annotations

from pathlib import Path
from typing import Any


def render_draft_support_introduction(
    *,
    primary_model_variant: str,
    primary_probe: str,
    secondary_probe: str,
) -> str:
    lines = [
        "# Draft Support Introduction",
        "",
        "Problem framing:",
        "Legal judgment prediction systems are often evaluated only on aggregate accuracy, even though legal documents are internally structured and different parts of a judgment may carry different kinds of decision signal.",
        "",
        "Why robustness matters:",
        "If a model changes behavior substantially when legally meaningful parts of a judgment are removed or isolated, then standard full-text performance can hide a brittle decision process. This matters especially in legal NLP, where precedent discussion, factual narratives, and judicial reasoning play distinct doctrinal roles.",
        "",
        "Why section-aware evaluation matters:",
        "A section-aware robustness setup asks not only whether a model predicts correctly, but also which inferred section types its prediction depends on. That creates a more interpretable stress test than generic noise or token-level corruption.",
        "",
        "Why RR to CJPE transfer is needed:",
        "The rhetorical-role corpus supplies reliable section supervision, while CJPE provides the scale needed for document-level prediction. The paper therefore uses RR to learn broad rhetorical structure and transfers that structure to CJPE to create predicted pseudo-sections for downstream robustness evaluation.",
        "",
        "Novelty framing:",
        f"The central contribution is a section-aware robustness evaluation pipeline for legal judgment prediction that centers `{primary_model_variant}` and uses `{primary_probe}` as the primary probe with `{secondary_probe}` as supporting evidence, while keeping the section-transfer caveat explicit throughout.",
        "",
    ]
    return "\n".join(lines)


def render_draft_support_method(
    *,
    section_transfer_run_dir: Path,
    baseline_run_dir: Path,
    robustness_run_dir: Path,
    primary_model_variant: str,
    primary_probe: str,
    secondary_probe: str,
    supporting_model_variants: tuple[str, ...] | list[str],
) -> str:
    lines = [
        "# Draft Support Method",
        "",
        "RR supervision pipeline:",
        f"The method begins from the frozen RR-to-CJPE section-transfer run at `{section_transfer_run_dir}`. RR provides broad rhetorical supervision, which is used to train a sentence-level section tagger and transfer predicted broad sections to CJPE.",
        "",
        "CJPE pseudo-sectioning:",
        "The transferred CJPE representation keeps predicted `facts`, `precedents`, `reasoning`, `conclusion`, and `other` segments as pseudo-sections. These are predicted/transferred structures rather than gold annotations and should be described that way in the manuscript.",
        "",
        "Prediction models:",
        f"The main reporting model is `{primary_model_variant}`. Supporting comparison models retained in the paper are `{list(supporting_model_variants)}`.",
        "",
        "Perturbation definitions:",
        f"`{primary_probe}` tests how much judgment signal survives when the input is restricted to predicted reasoning content alone. `{secondary_probe}` tests how sensitive predictions are to the removal of predicted precedent content while keeping the rest of the pseudo-sectioned document intact.",
        "",
        "Evaluation setup:",
        f"The canonical baseline artifacts come from `{baseline_run_dir}` and the canonical robustness artifacts come from `{robustness_run_dir}`. The paper should describe the evaluation as a focused pilot robustness setup over two section-aware probes rather than a full perturbation benchmark.",
        "",
    ]
    return "\n".join(lines)


def render_draft_support_results(
    *,
    table_main_results: dict[str, Any],
    table_model_comparison: dict[str, Any],
    stability_narrative: dict[str, Any],
    primary_bundle: dict[str, Any],
    secondary_bundle: dict[str, Any],
) -> str:
    main_rows = {row["condition_key"]: row for row in table_main_results.get("rows", [])}
    ranked = table_model_comparison.get("rows", [])
    ranked = sorted(ranked, key=lambda row: row["unperturbed_macro_f1"], reverse=True)
    top = ranked[0] if ranked else None
    nb_row = next((row for row in ranked if row["model_label"] == "NB"), None)
    logistic_row = next((row for row in ranked if row["model_label"] == "Logistic"), None)
    primary_examples = [example["case_id"] for example in primary_bundle.get("examples", [])[:3]]
    secondary_examples = [example["case_id"] for example in secondary_bundle.get("examples", [])[:3]]
    lines = [
        "# Draft Support Results",
        "",
    ]
    if top is not None:
        lines.append(
            f"Unperturbed comparison: `{top['model_label']}` is the strongest model on `pseudo_all_sections`, with accuracy `{_fmt(top['unperturbed_accuracy'])}` and macro F1 `{_fmt(top['unperturbed_macro_f1'])}`."
        )
        lines.append("")
    if nb_row is not None and logistic_row is not None:
        lines.append(
            f"Comparator context: NB remains the strongest simpler comparator by retained macro F1 on several probes, while logistic regression remains the most stable by flip rate. That distinction should be stated explicitly rather than folded into a single notion of robustness."
        )
        lines.append("")
    if "keep_reasoning_only" in main_rows:
        row = main_rows["keep_reasoning_only"]
        lines.append(
            f"Primary probe: under `keep_reasoning_only`, APA macro F1 changes from `{_fmt(main_rows['unperturbed']['macro_f1'])}` to `{_fmt(row['macro_f1'])}` (delta `{_fmt(row['delta_macro_f1'])}`) and accuracy changes from `{_fmt(main_rows['unperturbed']['accuracy'])}` to `{_fmt(row['accuracy'])}` (delta `{_fmt(row['delta_accuracy'])}`). This is the largest degradation in the focused package and should anchor the main results subsection."
        )
        lines.append("")
    if "drop_precedents" in main_rows:
        row = main_rows["drop_precedents"]
        lines.append(
            f"Secondary probe: under `drop_precedents`, APA macro F1 changes to `{_fmt(row['macro_f1'])}` with delta `{_fmt(row['delta_macro_f1'])}`, and accuracy changes to `{_fmt(row['accuracy'])}` with delta `{_fmt(row['delta_accuracy'])}`. The effect is smaller than reasoning-only ablation but still meaningful, especially given effective coverage `{_fmt(row['effective_coverage'])}`."
        )
        lines.append("")
    if stability_narrative.get("paragraphs"):
        lines.append(
            f"Stability-vs-correctness interpretation: {stability_narrative['paragraphs'][-1]}"
        )
        lines.append("")
    lines.append(
        f"Qualitative anchors: use `{primary_examples}` from the primary bundle and `{secondary_examples}` from the secondary bundle as the first candidate case references when drafting the results narrative."
    )
    lines.append("")
    return "\n".join(lines)


def render_draft_support_limitations() -> str:
    lines = [
        "# Draft Support Limitations",
        "",
        "- The CJPE sections used in the robustness analysis are transferred/predicted pseudo-sections derived from RR supervision rather than gold section annotations.",
        "- The contextual approximation remains weaker than the best simple baselines in the current environment and should not be overinterpreted.",
        "- The perturbation scope is intentionally focused on two high-value probes and should be described as a pilot robustness setup rather than an exhaustive benchmark.",
        "- The current results are strong enough for pilot paper drafting, but they are still pilot-scale evidence rather than a final large-scale benchmark claim.",
        "",
    ]
    return "\n".join(lines)


def render_draft_support_appendix(
    *,
    freeze_manifest: dict[str, Any],
    reproducibility_commands_path: Path,
    primary_bundle_path: Path,
    secondary_bundle_path: Path,
) -> str:
    lines = [
        "# Draft Support Appendix",
        "",
        "Artifact manifest summary:",
        f"The canonical runs are frozen in `{freeze_manifest['canonical_section_transfer_run_dir']}`, `{freeze_manifest['canonical_baseline_run_dir']}`, and `{freeze_manifest['canonical_robustness_run_dir']}`.",
        "",
        "Perturbation definitions:",
        "- `keep_reasoning_only` retains only predicted reasoning text to test how much decision signal survives when facts and precedents are removed.",
        "- `drop_precedents` removes predicted precedent content to test sensitivity to the loss of supporting doctrinal and citation-heavy material.",
        "",
        "Example bundle references:",
        f"- Primary qualitative bundle: `{primary_bundle_path}`",
        f"- Secondary qualitative bundle: `{secondary_bundle_path}`",
        "",
        "Reproducibility reference:",
        f"- Command sheet: `{reproducibility_commands_path}`",
        "",
    ]
    return "\n".join(lines)


def render_paper_table_selection(
    *,
    primary_bundle_path: Path,
    secondary_bundle_path: Path,
) -> str:
    lines = [
        "# Paper Table Selection",
        "",
        "## Main Paper",
        "",
        "- `table_main_results.md`: main APA-centered robustness table for the two focused probes.",
        "- `table_model_comparison.md`: compact four-model comparison showing why APA is central while NB and logistic remain important comparators.",
        "- `table_stability_vs_correctness.md`: use in the main paper only if space allows; otherwise move it to the appendix and summarize it in prose.",
        "",
        "## Appendix",
        "",
        "- `table_stability_vs_correctness.md`: preferred appendix placement if the main text needs to stay compact.",
        f"- `{primary_bundle_path.name}`: main-text-adjacent appendix support for reasoning-only cases.",
        f"- `{secondary_bundle_path.name}`: supporting appendix cases for precedent removal.",
        "- `paper_qualitative_examples.md`: fuller review sheet for additional examples beyond the small set cited in the text.",
        "",
    ]
    return "\n".join(lines)


def render_paper_figure_selection() -> str:
    lines = [
        "# Paper Figure Selection",
        "",
        "## Main Paper",
        "",
        "- Build one perturbation-delta bar chart from `chart_data_robustness_deltas.json`, centered on APA with NB and logistic as comparators.",
        "- Build one flip-rate / stability comparison chart from `chart_data_flip_rates.json` to support the stability-versus-correctness argument.",
        "",
        "## Appendix",
        "",
        "- Build one unperturbed model comparison bar chart from `chart_data_main_performance.json`.",
        "- Build one coverage-aware probe chart from `chart_data_coverage.json` if the appendix needs an explicit coverage panel.",
        "",
    ]
    return "\n".join(lines)


def render_paper_reproducibility_commands(
    *,
    section_transfer_run_dir: Path,
    baseline_run_dir: Path,
    robustness_run_dir: Path,
    results_package_dir: Path,
    drafting_package_dir: Path,
) -> str:
    section_transfer_suffix = _infer_run_suffix(section_transfer_run_dir.name, "section_transfer")
    baseline_suffix = _infer_run_suffix(baseline_run_dir.name, "prediction_baselines")
    robustness_suffix = _infer_run_suffix(robustness_run_dir.name, "robustness")
    drafting_suffix = drafting_package_dir.name
    lines = [
        "# Paper Reproducibility Commands",
        "",
        "Section-transfer artifacts:",
        f"`python scripts/run_section_transfer.py --dataset-root \"C:\\Users\\ashis\\IL-TUR\" --run-name \"{section_transfer_suffix}\"`",
        f"Expected output: `{section_transfer_run_dir}`",
        "",
        "Baseline artifacts:",
        f"`python scripts/train_baseline.py --section-transfer-dir \"{section_transfer_run_dir}\" --run-name \"{baseline_suffix}\"`",
        f"Expected output: `{baseline_run_dir}`",
        "",
        "Robustness and results-package artifacts:",
        f"`python scripts/evaluate_robustness.py --baseline-run-dir \"{baseline_run_dir}\" --run-name \"{robustness_suffix}\"`",
        f"Expected output: `{robustness_run_dir}`",
        f"Canonical results package: `{results_package_dir}`",
        "",
        "Paper drafting package:",
        f"`python scripts/export_paper_drafting_package.py --baseline-run-dir \"{baseline_run_dir}\" --robustness-run-dir \"{robustness_run_dir}\" --run-name \"{drafting_suffix}\"`",
        f"Expected output: `{drafting_package_dir}`",
        "",
    ]
    return "\n".join(lines)


def build_targeted_strengthening_check(
    *,
    comparative_metrics: dict[str, Any],
    primary_model_variant: str,
    target_recipe: str = "drop_precedents",
) -> dict[str, Any]:
    recipe = next(
        (row for row in comparative_metrics.get("recipes", []) if row["recipe_name"] == target_recipe),
        None,
    )
    if recipe is None:
        return {
            "task": "targeted_strengthening_check",
            "status": "not_run",
            "check_name": "missing_recipe",
            "summary": f"Recipe `{target_recipe}` was not found, so the targeted strengthening check could not be run.",
        }
    primary_metrics = next(
        (
            row
            for row in recipe.get("model_metrics", [])
            if row["model_variant"] == primary_model_variant
        ),
        None,
    )
    if primary_metrics is None:
        return {
            "task": "targeted_strengthening_check",
            "status": "not_run",
            "check_name": "missing_primary_model",
            "summary": f"Primary model `{primary_model_variant}` was not found inside recipe `{target_recipe}`.",
        }
    non_empty = primary_metrics.get("non_empty_target_metrics")
    ranking_rows: list[dict[str, Any]] = []
    for row in recipe.get("model_metrics", []):
        non_empty_metrics = row.get("non_empty_target_metrics")
        if non_empty_metrics is None:
            continue
        ranking_rows.append(
            {
                "model_variant": row["model_variant"],
                "macro_f1": non_empty_metrics["macro_f1"],
                "accuracy": non_empty_metrics["accuracy"],
                "delta_macro_f1": non_empty_metrics["macro_f1_delta_vs_reference"],
            }
        )
    ranking_rows.sort(key=lambda row: row["macro_f1"], reverse=True)
    if non_empty is None:
        summary = f"No non-empty-target slice was available for `{target_recipe}`, so no strengthening slice was produced."
        status = "not_run"
    else:
        summary = (
            f"For `{target_recipe}`, the APA non-empty-target slice still shows a meaningful but modest degradation: macro F1 `{_fmt(non_empty['reference_macro_f1'])}` -> "
            f"`{_fmt(non_empty['macro_f1'])}` with delta `{_fmt(non_empty['macro_f1_delta_vs_reference'])}`. "
            "This keeps the qualitative manuscript conclusion unchanged while removing ambiguity about empty-target cases."
        )
        status = "pass"
    return {
        "task": "targeted_strengthening_check",
        "status": status,
        "check_name": "non_empty_precedent_slice",
        "target_recipe": target_recipe,
        "primary_model_variant": primary_model_variant,
        "primary_model_overall_metrics": {
            "macro_f1": primary_metrics["macro_f1"],
            "accuracy": primary_metrics["accuracy"],
            "delta_macro_f1": primary_metrics["macro_f1_delta_vs_reference"],
            "flip_rate": primary_metrics["flip_rate"],
        },
        "primary_model_non_empty_target_metrics": non_empty,
        "non_empty_slice_ranking": ranking_rows,
        "summary": summary,
    }


def render_targeted_strengthening_check(report: dict[str, Any]) -> str:
    lines = [
        "# Targeted Strengthening Check",
        "",
        f"- Status: `{report['status']}`",
        f"- Check name: `{report['check_name']}`",
        f"- Target recipe: `{report.get('target_recipe')}`",
        f"- Primary model variant: `{report.get('primary_model_variant')}`",
        "",
        report["summary"],
        "",
    ]
    if report.get("primary_model_non_empty_target_metrics"):
        metrics = report["primary_model_non_empty_target_metrics"]
        lines.append(
            f"APA non-empty-target slice: accuracy `{_fmt(metrics['accuracy'])}`, macro F1 `{_fmt(metrics['macro_f1'])}`, delta macro F1 `{_fmt(metrics['macro_f1_delta_vs_reference'])}`."
        )
        lines.append("")
    if report.get("non_empty_slice_ranking"):
        lines.append("Non-empty-target ranking:")
        for row in report["non_empty_slice_ranking"]:
            lines.append(
                f"- `{row['model_variant']}`: macro F1 `{_fmt(row['macro_f1'])}`, accuracy `{_fmt(row['accuracy'])}`, delta `{_fmt(row['delta_macro_f1'])}`"
            )
        lines.append("")
    return "\n".join(lines)


def build_paper_readiness_summary(
    *,
    freeze_manifest: dict[str, Any],
    consistency_check: dict[str, Any],
    packaging_next_step_summary: dict[str, Any],
    targeted_strengthening_check: dict[str, Any] | None = None,
) -> dict[str, Any]:
    ready = (
        consistency_check.get("overall_status") == "pass"
        and packaging_next_step_summary.get("ready_to_begin_pilot_results_drafting", False)
    )
    caveats = list(packaging_next_step_summary.get("visible_caveats", []))
    exact_next_action = (
        "Begin drafting the pilot manuscript results section from draft_support_results.md, then use table_main_results.md and qualitative_examples_primary.md as the first citation anchors."
        if ready
        else "Resolve the remaining consistency or readiness issues before drafting the manuscript."
    )
    if targeted_strengthening_check and targeted_strengthening_check.get("status") == "pass":
        exact_next_action += " Keep the targeted strengthening-check note available as a reviewer-facing confidence addendum."
    return {
        "task": "paper_readiness_summary",
        "canonical_results_package_dir": freeze_manifest["canonical_results_package_dir"],
        "ready_for_pilot_paper_drafting": ready,
        "central_claim": packaging_next_step_summary["central_claim"],
        "primary_model_variant": freeze_manifest["primary_model_variant"],
        "primary_probe": freeze_manifest["primary_probe"],
        "secondary_probe": freeze_manifest["secondary_probe"],
        "consistency_status": consistency_check["overall_status"],
        "one_more_experiment_necessary": not ready,
        "visible_caveats": caveats,
        "exact_next_action": exact_next_action,
    }


def render_paper_readiness_summary(report: dict[str, Any]) -> str:
    lines = [
        "# Paper Readiness Summary",
        "",
        f"- Canonical results package dir: `{report['canonical_results_package_dir']}`",
        f"- Ready for pilot paper drafting: `{report['ready_for_pilot_paper_drafting']}`",
        f"- Primary model variant: `{report['primary_model_variant']}`",
        f"- Primary probe: `{report['primary_probe']}`",
        f"- Secondary probe: `{report['secondary_probe']}`",
        f"- Consistency status: `{report['consistency_status']}`",
        f"- One more experiment necessary: `{report['one_more_experiment_necessary']}`",
        "",
        "## Central Claim",
        "",
        report["central_claim"],
        "",
        "## Visible Caveats",
        "",
    ]
    for caveat in report.get("visible_caveats", []):
        lines.append(f"- {caveat}")
    lines.extend(["", "## Exact Next Action", ""])
    lines.append(report["exact_next_action"])
    lines.append("")
    return "\n".join(lines)


def _infer_run_suffix(directory_name: str, stage_name: str) -> str:
    marker = f"_{stage_name}_"
    if marker in directory_name:
        return directory_name.split(marker, maxsplit=1)[1]
    return directory_name


def _fmt(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        if value == int(value):
            return str(int(value))
        return f"{value:.6f}"
    return str(value)
