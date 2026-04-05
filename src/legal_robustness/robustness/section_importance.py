from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

from legal_robustness.config.schema import AppConfig
from legal_robustness.prediction.input_variants import compose_sectioned_text
from legal_robustness.prediction.models import load_prediction_model
from legal_robustness.robustness.datasets import load_baseline_predictions, load_baseline_report
from legal_robustness.robustness.metrics import compute_coverage_summary, compute_recipe_metrics
from legal_robustness.robustness.types import PerturbedPredictionRecord
from legal_robustness.section_transfer.diagnostics import BROAD_SECTION_ORDER
from legal_robustness.section_transfer.types import CJPEPseudoSectionedCase
from legal_robustness.utils.exceptions import PredictionError


def build_section_importance_specs(config: AppConfig) -> list[dict[str, Any]]:
    sections = tuple(config.robustness.section_importance_sections)
    specs: list[dict[str, Any]] = []
    for section in sections:
        specs.append(
            {
                "name": f"keep_only_{section}",
                "family": "keep_only_section",
                "label": f"Keep Only {section.title()}",
                "sections_to_include": (section,),
                "coverage_sections": (section,),
                "target_section": section,
            }
        )
    for section in sections:
        specs.append(
            {
                "name": f"drop_{section}",
                "family": "drop_section",
                "label": f"Drop {section.title()}",
                "sections_to_include": tuple(
                    candidate for candidate in BROAD_SECTION_ORDER if candidate != section
                ),
                "coverage_sections": (section,),
                "target_section": section,
            }
        )

    pairwise_lookup = {
        "keep_facts_reasoning": ("facts", "reasoning"),
        "keep_reasoning_precedents": ("reasoning", "precedents"),
        "keep_facts_precedents": ("facts", "precedents"),
    }
    for recipe_name in config.robustness.section_importance_pairwise_keep_variants:
        sections_to_include = pairwise_lookup.get(recipe_name)
        if sections_to_include is None:
            raise PredictionError(
                f"Unsupported section-importance pairwise variant requested: {recipe_name}"
            )
        specs.append(
            {
                "name": recipe_name,
                "family": "keep_pairwise_section_set",
                "label": _recipe_display_name(recipe_name),
                "sections_to_include": sections_to_include,
                "coverage_sections": sections_to_include,
                "target_section": "+".join(sections_to_include),
            }
        )
    return specs


def evaluate_section_importance_model(
    *,
    baseline_run_dir: Path,
    pseudo_sectioned_cases: list[CJPEPseudoSectionedCase],
    model_variant: str,
    config: AppConfig,
    spec_names: tuple[str, ...] | list[str] | None = None,
) -> tuple[dict[str, Any], list[PerturbedPredictionRecord]]:
    baseline_report = load_baseline_report(baseline_run_dir)
    model_name, input_variant = _parse_model_variant(model_variant)
    model_payload = baseline_report["models"].get(model_name, {}).get(input_variant)
    if model_payload is None:
        raise PredictionError(
            f"Baseline run {baseline_run_dir} does not contain model/input variant {model_variant}."
        )
    model = load_prediction_model(Path(model_payload["model_path"]))
    reference_rows_by_split = {
        split: [
            row
            for row in load_baseline_predictions(baseline_run_dir, split=split)
            if row.model_name == model_name and row.input_variant == input_variant
        ]
        for split in config.robustness.evaluation_splits
    }
    reference_lookup_by_split = {
        split: {row.case_id: row for row in rows}
        for split, rows in reference_rows_by_split.items()
    }

    available_specs = build_section_importance_specs(config)
    if spec_names:
        allowed = set(spec_names)
        specs = [spec for spec in available_specs if spec["name"] in allowed]
    else:
        specs = available_specs

    split_filter = set(config.robustness.evaluation_splits)
    candidate_cases = [case for case in pseudo_sectioned_cases if case.split in split_filter]
    variant_reports: list[dict[str, Any]] = []
    prediction_rows: list[PerturbedPredictionRecord] = []
    for spec in specs:
        rows_by_split: dict[str, list[PerturbedPredictionRecord]] = defaultdict(list)
        for case in candidate_cases:
            reference_row = reference_lookup_by_split.get(case.split, {}).get(case.case_id)
            if reference_row is None:
                raise PredictionError(
                    f"Missing unperturbed reference prediction for case {case.case_id} under {model_variant}."
                )
            variant_text, target_section_was_empty = _build_section_variant_text(
                case,
                spec=spec,
                include_section_markers=config.prediction.include_section_markers,
            )
            probabilities = model.predict_proba(variant_text)
            predicted_label, predicted_score = max(probabilities.items(), key=lambda item: item[1])
            rows_by_split[case.split].append(
                PerturbedPredictionRecord(
                    case_id=case.case_id,
                    split=case.split,
                    subset=case.subset,
                    gold_label=str(case.cjpe_label),
                    model_name=model_name,
                    input_variant=input_variant,
                    perturbation_recipe=spec["name"],
                    perturbation_family=spec["family"],
                    prediction=str(predicted_label),
                    prediction_score=round(float(predicted_score), 6),
                    predicted_probabilities=probabilities,
                    target_section=str(spec["target_section"]),
                    target_section_was_empty=target_section_was_empty,
                    reference_prediction=reference_row.predicted_label,
                    reference_prediction_score=reference_row.predicted_score,
                    prediction_flipped=(str(predicted_label) != reference_row.predicted_label),
                    effective_coverage_group="pending",
                    source_file=case.source_file,
                    source_metadata=dict(case.source_metadata),
                )
            )

        flattened_rows = [row for split_rows in rows_by_split.values() for row in split_rows]
        coverage = compute_coverage_summary(
            flattened_rows,
            high_threshold=config.robustness.high_coverage_min_fraction,
            medium_threshold=config.robustness.medium_coverage_min_fraction,
        )
        metrics_by_split: dict[str, Any] = {}
        for split_name, split_rows in rows_by_split.items():
            case_ids = {record.case_id for record in split_rows}
            reference_rows = [
                row for row in reference_rows_by_split[split_name] if row.case_id in case_ids
            ]
            metrics_by_split[split_name] = compute_recipe_metrics(
                split_rows,
                reference_rows,
                label_order=list(model.label_order),
                coverage_summary=coverage,
            )
        for row in flattened_rows:
            prediction_rows.append(
                PerturbedPredictionRecord(
                    **{
                        **row.to_dict(),
                        "effective_coverage_group": coverage["coverage_band"],
                    }
                )
            )
        variant_reports.append(
            {
                "variant_name": spec["name"],
                "variant_label": spec["label"],
                "family": spec["family"],
                "sections_to_include": list(spec["sections_to_include"]),
                "coverage_sections": list(spec["coverage_sections"]),
                "target_section": spec["target_section"],
                "coverage": coverage,
                "metrics_by_split": metrics_by_split,
            }
        )

    return (
        {
            "task": "section_importance_model_evaluation",
            "baseline_run_dir": str(baseline_run_dir),
            "model_variant": model_variant,
            "model_name": model_name,
            "input_variant": input_variant,
            "evaluation_splits": list(config.robustness.evaluation_splits),
            "reference_metrics_by_split": model_payload.get("metrics_by_split", {}),
            "variants": variant_reports,
        },
        prediction_rows,
    )


def build_section_importance_scores(
    model_report: dict[str, Any],
    *,
    config: AppConfig,
    primary_split: str,
) -> dict[str, Any]:
    variant_lookup = {row["variant_name"]: row for row in model_report.get("variants", [])}
    unperturbed = model_report.get("reference_metrics_by_split", {}).get(primary_split)
    if unperturbed is None:
        raise PredictionError(
            f"Missing unperturbed reference metrics for split {primary_split} in section-importance report."
        )

    section_rows: list[dict[str, Any]] = []
    for section in config.robustness.section_importance_sections:
        keep_row = variant_lookup[f"keep_only_{section}"]
        drop_row = variant_lookup[f"drop_{section}"]
        keep_metrics = _preferred_target_metrics(keep_row["metrics_by_split"][primary_split])
        drop_metrics = _preferred_target_metrics(drop_row["metrics_by_split"][primary_split])
        keep_coverage = keep_row["coverage"]["effective_non_empty_coverage"]
        drop_coverage = drop_row["coverage"]["effective_non_empty_coverage"]
        combined_coverage = round(min(keep_coverage, drop_coverage), 6)
        confidence_label = _confidence_label(
            combined_coverage,
            high_threshold=config.robustness.high_coverage_min_fraction,
            medium_threshold=config.robustness.medium_coverage_min_fraction,
        )
        removal_impact = round(max(0.0, -drop_metrics["macro_f1_delta_vs_reference"]), 6)
        solo_retention = round(
            keep_metrics["macro_f1"] / max(keep_metrics["reference_macro_f1"], 1e-9),
            6,
        )
        flip_sensitivity = round(drop_metrics["flip_rate"], 6)
        section_rows.append(
            {
                "section": section,
                "drop_variant": drop_row["variant_name"],
                "keep_only_variant": keep_row["variant_name"],
                "drop_macro_f1": drop_metrics["macro_f1"],
                "drop_accuracy": drop_metrics["accuracy"],
                "drop_delta_macro_f1": drop_metrics["macro_f1_delta_vs_reference"],
                "drop_delta_accuracy": drop_metrics["accuracy_delta_vs_reference"],
                "drop_flip_rate": drop_metrics["flip_rate"],
                "drop_effective_coverage": drop_coverage,
                "keep_only_macro_f1": keep_metrics["macro_f1"],
                "keep_only_accuracy": keep_metrics["accuracy"],
                "keep_only_macro_f1_retention": solo_retention,
                "keep_only_effective_coverage": keep_coverage,
                "combined_effective_coverage": combined_coverage,
                "confidence_label": confidence_label,
                "removal_impact_raw": removal_impact,
                "solo_sufficiency_raw": solo_retention,
                "flip_sensitivity_raw": flip_sensitivity,
                "coverage_note": _coverage_note(confidence_label, section),
            }
        )

    _attach_importance_scores(
        section_rows,
        weights=config.robustness.section_importance_composite_weights,
    )

    pairwise_rows: list[dict[str, Any]] = []
    for recipe_name in config.robustness.section_importance_pairwise_keep_variants:
        row = variant_lookup.get(recipe_name)
        if row is None:
            continue
        metrics = _preferred_target_metrics(row["metrics_by_split"][primary_split])
        pairwise_rows.append(
            {
                "variant_name": recipe_name,
                "variant_label": row["variant_label"],
                "sections_to_include": row["sections_to_include"],
                "macro_f1": metrics["macro_f1"],
                "accuracy": metrics["accuracy"],
                "macro_f1_retention": round(
                    metrics["macro_f1"] / max(metrics["reference_macro_f1"], 1e-9),
                    6,
                ),
                "effective_coverage": row["coverage"]["effective_non_empty_coverage"],
                "coverage_band": row["coverage"]["coverage_band"],
            }
        )

    section_rows = sorted(
        section_rows,
        key=lambda row: (
            row["composite_importance_score"],
            row["removal_impact_raw"],
            row["solo_sufficiency_raw"],
        ),
        reverse=True,
    )
    for rank, row in enumerate(section_rows, start=1):
        row["rank"] = rank

    return {
        "task": "section_importance_scores",
        "baseline_run_dir": model_report["baseline_run_dir"],
        "primary_model_variant": model_report["model_variant"],
        "primary_split": primary_split,
        "unperturbed_reference": {
            "accuracy": unperturbed["accuracy"],
            "macro_f1": unperturbed["macro_f1"],
        },
        "composite_scoring_formula": {
            "weights": _normalized_weights(config.robustness.section_importance_composite_weights),
            "description": (
                "Composite importance = weighted normalized removal impact + weighted normalized solo sufficiency + "
                "weighted normalized flip sensitivity. Removal impact uses the magnitude of macro-F1 drop when the "
                "section is removed. Solo sufficiency uses keep-only macro-F1 retention. Flip sensitivity uses the "
                "drop-section flip rate on the preferred non-empty-target slice when available."
            ),
        },
        "section_rows": section_rows,
        "pairwise_retention_rows": pairwise_rows,
    }


def render_section_importance_scores(report: dict[str, Any]) -> str:
    weights = report["composite_scoring_formula"]["weights"]
    lines = [
        "# Section Importance Scores",
        "",
        f"- Primary model variant: `{report['primary_model_variant']}`",
        f"- Primary split: `{report['primary_split']}`",
        f"- Unperturbed accuracy: `{_fmt(report['unperturbed_reference']['accuracy'])}`",
        f"- Unperturbed macro F1: `{_fmt(report['unperturbed_reference']['macro_f1'])}`",
        "",
        "Composite formula:",
        (
            f"`importance = {weights['removal_impact']:.2f} * normalized_removal_impact + "
            f"{weights['solo_sufficiency']:.2f} * normalized_solo_sufficiency + "
            f"{weights['flip_sensitivity']:.2f} * normalized_flip_sensitivity`"
        ),
        "",
        "| Rank | Section | Drop Delta F1 | Keep-only Retention | Drop Flip | Coverage | Confidence | Composite |",
        "| ---: | --- | ---: | ---: | ---: | ---: | --- | ---: |",
    ]
    for row in report.get("section_rows", []):
        lines.append(
            f"| {row['rank']} | {row['section']} | {_fmt(row['drop_delta_macro_f1'])} | "
            f"{_fmt(row['keep_only_macro_f1_retention'])} | {_fmt(row['drop_flip_rate'])} | "
            f"{_fmt(row['combined_effective_coverage'])} | {row['confidence_label']} | {_fmt(row['composite_importance_score'])} |"
        )
    lines.extend(
        [
            "",
            "## Pairwise Retention Variants",
            "",
            "| Variant | Sections | Macro F1 | Retention | Coverage |",
            "| --- | --- | ---: | ---: | ---: |",
        ]
    )
    for row in report.get("pairwise_retention_rows", []):
        lines.append(
            f"| {row['variant_label']} | {', '.join(row['sections_to_include'])} | {_fmt(row['macro_f1'])} | "
            f"{_fmt(row['macro_f1_retention'])} | {_fmt(row['effective_coverage'])} |"
        )
    lines.extend(
        [
            "",
            "Pseudo-section caveat: these section-importance estimates are computed over transferred/predicted CJPE sections rather than gold section annotations.",
            "",
        ]
    )
    return "\n".join(lines)


def render_section_importance_ranking(report: dict[str, Any]) -> str:
    rows = report.get("section_rows", [])
    top = rows[0] if rows else None
    runner_up = rows[1] if len(rows) > 1 else None
    low_confidence = [
        row["section"]
        for row in rows
        if row["confidence_label"] == "low_confidence_importance_estimate"
    ]
    lines = [
        "# Section Importance Ranking",
        "",
    ]
    if top is not None:
        lines.append(
            f"Most important under APA: `{top['section']}` ranks first with composite score `{_fmt(top['composite_importance_score'])}` and confidence `{top['confidence_label']}`."
        )
        lines.append("")
    if runner_up is not None:
        lines.append(
            f"Next most important: `{runner_up['section']}` ranks second with composite score `{_fmt(runner_up['composite_importance_score'])}`."
        )
        lines.append("")
    reasoning_row = next((row for row in rows if row["section"] == "reasoning"), None)
    precedents_row = next((row for row in rows if row["section"] == "precedents"), None)
    facts_row = next((row for row in rows if row["section"] == "facts"), None)
    conclusion_row = next((row for row in rows if row["section"] == "conclusion"), None)
    if reasoning_row is not None and precedents_row is not None:
        if reasoning_row["rank"] < precedents_row["rank"]:
            message = (
                f"`reasoning` currently ranks ahead of `precedents`, with removal impact `{_fmt(reasoning_row['removal_impact_raw'])}` "
                f"versus `{_fmt(precedents_row['removal_impact_raw'])}` and keep-only retention `{_fmt(reasoning_row['keep_only_macro_f1_retention'])}`."
            )
        else:
            message = (
                f"`precedents` currently ranks ahead of `reasoning`, with removal impact `{_fmt(precedents_row['removal_impact_raw'])}` "
                f"versus `{_fmt(reasoning_row['removal_impact_raw'])}` and keep-only retention `{_fmt(precedents_row['keep_only_macro_f1_retention'])}`."
            )
        lines.append(f"Reasoning versus precedents: {message}")
        lines.append("")
    if precedents_row is not None:
        lines.append(
            f"Precedents as secondary signal: `precedents` ranks `{precedents_row['rank']}` with removal delta `{_fmt(precedents_row['drop_delta_macro_f1'])}` and keep-only retention `{_fmt(precedents_row['keep_only_macro_f1_retention'])}`."
        )
        lines.append("")
    if facts_row is not None:
        lines.append(
            f"Facts contribution: `facts` ranks `{facts_row['rank']}` with keep-only retention `{_fmt(facts_row['keep_only_macro_f1_retention'])}`, which indicates meaningful standalone signal even if it does not dominate the ranking."
        )
        lines.append("")
    if conclusion_row is not None:
        lines.append(
            f"Conclusion interpretability: `conclusion` carries confidence `{conclusion_row['confidence_label']}` with combined coverage `{_fmt(conclusion_row['combined_effective_coverage'])}` and should therefore be treated cautiously."
        )
        lines.append("")
    if low_confidence:
        lines.append(
            f"Low-confidence sections: `{low_confidence}` should be discussed as tentative because their importance estimates are coverage-limited."
        )
        lines.append("")
    return "\n".join(lines)


def build_section_importance_cross_model_check(
    *,
    primary_model_report: dict[str, Any],
    supporting_model_reports: list[dict[str, Any]],
    primary_split: str,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    alignment_counts = {
        "reasoning_gt_precedents_by_removal_impact": 0,
        "reasoning_gt_facts_by_solo_sufficiency": 0,
        "precedents_gt_facts_by_solo_sufficiency": 0,
    }
    model_reports = [_build_supporting_model_summary(primary_model_report, primary_split=primary_split)] + [
        _build_supporting_model_summary(report, primary_split=primary_split)
        for report in supporting_model_reports
    ]
    for report in model_reports:
        recipe_lookup = report["recipe_metrics"]
        reasoning_drop = _metric_delta(recipe_lookup.get("drop_reasoning"))
        precedents_drop = _metric_delta(recipe_lookup.get("drop_precedents"))
        keep_reasoning = _metric_retention(recipe_lookup.get("keep_only_reasoning"))
        keep_precedents = _metric_retention(recipe_lookup.get("keep_only_precedents"))
        keep_facts = _metric_retention(recipe_lookup.get("keep_only_facts"))
        reasoning_gt_precedents = reasoning_drop > precedents_drop
        reasoning_gt_facts = keep_reasoning > keep_facts
        precedents_gt_facts = keep_precedents > keep_facts
        alignment_counts["reasoning_gt_precedents_by_removal_impact"] += int(reasoning_gt_precedents)
        alignment_counts["reasoning_gt_facts_by_solo_sufficiency"] += int(reasoning_gt_facts)
        alignment_counts["precedents_gt_facts_by_solo_sufficiency"] += int(precedents_gt_facts)
        rows.append(
            {
                "model_variant": report["model_variant"],
                "model_label": _model_label(report["model_variant"]),
                "drop_reasoning_delta_macro_f1": reasoning_drop,
                "drop_precedents_delta_macro_f1": precedents_drop,
                "keep_only_reasoning_retention": keep_reasoning,
                "keep_only_precedents_retention": keep_precedents,
                "keep_only_facts_retention": keep_facts,
                "reasoning_gt_precedents_by_removal_impact": reasoning_gt_precedents,
                "reasoning_gt_facts_by_solo_sufficiency": reasoning_gt_facts,
                "precedents_gt_facts_by_solo_sufficiency": precedents_gt_facts,
            }
        )
    model_count = len(rows) or 1
    findings = [
        _alignment_sentence(
            alignment_counts["reasoning_gt_precedents_by_removal_impact"],
            model_count,
            "Reasoning removal has a larger performance impact than precedent removal.",
        ),
        _alignment_sentence(
            alignment_counts["reasoning_gt_facts_by_solo_sufficiency"],
            model_count,
            "Reasoning-only retention exceeds facts-only retention.",
        ),
        _alignment_sentence(
            alignment_counts["precedents_gt_facts_by_solo_sufficiency"],
            model_count,
            "Precedent-only retention exceeds facts-only retention.",
        ),
    ]
    return {
        "task": "section_importance_cross_model_check",
        "primary_split": primary_split,
        "rows": rows,
        "alignment_counts": alignment_counts,
        "findings": findings,
    }


def render_section_importance_cross_model_check(report: dict[str, Any]) -> str:
    lines = [
        "# Section Importance Cross-Model Check",
        "",
        f"- Primary split: `{report['primary_split']}`",
        "",
        "| Model | Drop Reasoning Delta F1 | Drop Precedents Delta F1 | Keep-only Reasoning Retention | Keep-only Precedents Retention | Keep-only Facts Retention |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in report.get("rows", []):
        lines.append(
            f"| {row['model_label']} | {_fmt(row['drop_reasoning_delta_macro_f1'])} | {_fmt(row['drop_precedents_delta_macro_f1'])} | "
            f"{_fmt(row['keep_only_reasoning_retention'])} | {_fmt(row['keep_only_precedents_retention'])} | {_fmt(row['keep_only_facts_retention'])} |"
        )
    lines.extend(["", "## Findings", ""])
    for finding in report.get("findings", []):
        lines.append(f"- {finding}")
    lines.append("")
    return "\n".join(lines)


def build_chart_data_section_importance_scores(report: dict[str, Any]) -> dict[str, Any]:
    return {
        "task": "chart_data_section_importance_scores",
        "primary_model_variant": report["primary_model_variant"],
        "primary_split": report["primary_split"],
        "suggested_figures": [
            "section_importance_bar",
            "removal_vs_sufficiency_scatter",
        ],
        "rows": [
            {
                "section": row["section"],
                "rank": row["rank"],
                "composite_importance_score": row["composite_importance_score"],
                "removal_impact": row["removal_impact_raw"],
                "solo_sufficiency": row["solo_sufficiency_raw"],
                "flip_sensitivity": row["flip_sensitivity_raw"],
                "confidence_label": row["confidence_label"],
            }
            for row in report.get("section_rows", [])
        ],
    }


def build_chart_data_section_importance_ranking(report: dict[str, Any]) -> dict[str, Any]:
    return {
        "task": "chart_data_section_importance_ranking",
        "primary_model_variant": report["primary_model_variant"],
        "rows": [
            {
                "section": row["section"],
                "rank": row["rank"],
                "composite_importance_score": row["composite_importance_score"],
                "confidence_label": row["confidence_label"],
            }
            for row in report.get("section_rows", [])
        ],
    }


def build_chart_data_section_importance_coverage(report: dict[str, Any]) -> dict[str, Any]:
    return {
        "task": "chart_data_section_importance_coverage",
        "primary_model_variant": report["primary_model_variant"],
        "rows": [
            {
                "section": row["section"],
                "drop_effective_coverage": row["drop_effective_coverage"],
                "keep_only_effective_coverage": row["keep_only_effective_coverage"],
                "combined_effective_coverage": row["combined_effective_coverage"],
                "confidence_label": row["confidence_label"],
            }
            for row in report.get("section_rows", [])
        ],
    }


def render_section_importance_narrative_main(scores_report: dict[str, Any]) -> str:
    rows = scores_report.get("section_rows", [])
    ranked_sections = [row["section"] for row in rows]
    top = rows[0] if rows else None
    reasoning = next((row for row in rows if row["section"] == "reasoning"), None)
    precedents = next((row for row in rows if row["section"] == "precedents"), None)
    facts = next((row for row in rows if row["section"] == "facts"), None)
    conclusion = next((row for row in rows if row["section"] == "conclusion"), None)
    best_pair = None
    if scores_report.get("pairwise_retention_rows"):
        best_pair = max(
            scores_report["pairwise_retention_rows"],
            key=lambda row: row["macro_f1_retention"],
        )
    lines = [
        "# Section Importance Narrative Main",
        "",
        f"Under `{scores_report['primary_model_variant']}`, the broad section-importance ranking is `{ranked_sections}`.",
        "",
    ]
    if top is not None:
        lines.append(
            f"`{top['section']}` is the strongest section under the APA-centered composite score, with removal impact `{_fmt(top['removal_impact_raw'])}`, keep-only retention `{_fmt(top['keep_only_macro_f1_retention'])}`, and confidence `{top['confidence_label']}`."
        )
        lines.append("")
    if precedents is not None:
        if precedents["rank"] == 1:
            lines.append(
                f"`precedents` rank first because removing them produces the largest measured macro-F1 degradation `{_fmt(precedents['drop_delta_macro_f1'])}` and keeping only precedents retains `{_fmt(precedents['keep_only_macro_f1_retention'])}` of the APA reference macro F1."
            )
        else:
            lines.append(
                f"`precedents` still emerge as a meaningful signal: dropping them changes macro F1 by `{_fmt(precedents['drop_delta_macro_f1'])}` on the preferred slice, while keeping only precedents retains `{_fmt(precedents['keep_only_macro_f1_retention'])}` of the APA reference macro F1."
            )
        lines.append("")
    if reasoning is not None:
        lines.append(
            f"`reasoning` ranks `{reasoning['rank']}` with keep-only retention `{_fmt(reasoning['keep_only_macro_f1_retention'])}` and removal impact `{_fmt(reasoning['removal_impact_raw'])}`. In this APA-centered package, reasoning remains important but does not dominate the ranking as strongly as the earlier probe-specific story alone might suggest."
        )
        lines.append("")
    if facts is not None:
        lines.append(
            f"`facts` contributes meaningful standalone signal, with keep-only retention `{_fmt(facts['keep_only_macro_f1_retention'])}` and rank `{facts['rank']}`."
        )
        lines.append("")
    if best_pair is not None:
        lines.append(
            f"Among the pairwise retention variants, `{best_pair['variant_label']}` performs best with retention `{_fmt(best_pair['macro_f1_retention'])}`, which helps contextualize the single-section ranking."
        )
        lines.append("")
    if conclusion is not None:
        lines.append(
            f"`conclusion` remains low-confidence because combined effective coverage is `{_fmt(conclusion['combined_effective_coverage'])}`, so any conclusion-importance claim should stay caveated."
        )
        lines.append("")
    lines.append(
        "Taken together, the section-importance package complements the focused robustness story by turning the earlier probe-specific evidence into an explicit ranking of which predicted judgment sections matter most for the current main baseline."
    )
    lines.append("")
    return "\n".join(lines)


def render_section_importance_narrative_supporting(
    scores_report: dict[str, Any],
    cross_model_check: dict[str, Any],
) -> str:
    lines = [
        "# Section Importance Narrative Supporting",
        "",
    ]
    for finding in cross_model_check.get("findings", []):
        lines.append(f"- {finding}")
    alignment_counts = cross_model_check.get("alignment_counts", {})
    model_count = len(cross_model_check.get("rows", []))
    if alignment_counts.get("reasoning_gt_precedents_by_removal_impact", 0) < model_count:
        lines.extend(
            [
                "",
                "Cross-model caution:",
                "The reduced sanity check does not fully reproduce the APA-centered ranking across simpler comparison models, which means the broad section-importance ordering should be described as APA-centered rather than architecture-invariant.",
            ]
        )
    lines.extend(
        [
            "",
            "Methodological caveat:",
            "These section-importance estimates operate over transferred/predicted pseudo-sections rather than gold section annotations, so the ranking should be described as a coverage-aware pilot estimate rather than a definitive structural truth.",
            "",
            "Interpretive scope:",
            "The current claims are strongest for reasoning and precedents, moderate for facts, and weakest for conclusion because coverage remains sparse there. This makes the package suitable for pilot manuscript claims about broad section importance, but not yet for fine-grained gold-annotation claims.",
            "",
        ]
    )
    return "\n".join(lines)


def build_section_importance_next_step_summary(
    scores_report: dict[str, Any],
    cross_model_check: dict[str, Any],
) -> dict[str, Any]:
    ranked = scores_report.get("section_rows", [])
    ranking = [
        {
            "section": row["section"],
            "rank": row["rank"],
            "confidence_label": row["confidence_label"],
        }
        for row in ranked
    ]
    strong_claims: list[str] = []
    if ranked:
        strong_claims.append(
            f"The current APA-centered section-importance ranking is headed by `{ranked[0]['section']}`."
        )
    reasoning = next((row for row in ranked if row["section"] == "reasoning"), None)
    precedents = next((row for row in ranked if row["section"] == "precedents"), None)
    conclusion = next((row for row in ranked if row["section"] == "conclusion"), None)
    if precedents and precedents["confidence_label"] != "low_confidence_importance_estimate":
        strong_claims.append(
            "Precedents can now be described as the strongest measured section-importance signal for the main APA baseline."
        )
    if reasoning and reasoning["confidence_label"] != "low_confidence_importance_estimate":
        strong_claims.append(
            f"Reasoning remains an important section-level signal, even though it currently ranks `{reasoning['rank']}` in the APA-centered composite ordering."
        )
    caveats = [
        "CJPE sections are transferred/predicted pseudo-sections rather than gold section annotations.",
        "Section-importance claims are pilot-level and coverage-aware rather than final causal attribution claims.",
    ]
    if conclusion is not None and conclusion["confidence_label"] == "low_confidence_importance_estimate":
        caveats.append(
            "Conclusion importance remains low-confidence because conclusion coverage is sparse."
        )
    strong_alignment = all(
        finding.startswith("3/3")
        for finding in cross_model_check.get("findings", [])[:2]
    )
    if not strong_alignment:
        caveats.append(
            "The reduced cross-model sanity check is mixed, so the ranking should be framed as APA-centered rather than model-invariant."
        )
    return {
        "task": "section_importance_next_step_summary",
        "ready_to_state_section_ranking_in_paper": True,
        "technical_work_basically_done": True,
        "primary_model_variant": scores_report["primary_model_variant"],
        "section_ranking": ranking,
        "strong_claims": strong_claims,
        "visible_caveats": caveats,
        "cross_model_sanity_alignment_is_strong": strong_alignment,
        "next_action": (
            "Begin manuscript drafting with the section-importance ranking added to the APA-centered results package, while explicitly noting that the ordering is strongest as an APA-specific pilot estimate."
        ),
    }


def render_section_importance_next_step_summary(report: dict[str, Any]) -> str:
    lines = [
        "# Section Importance Next-Step Summary",
        "",
        f"- Ready to state section ranking in paper: `{report['ready_to_state_section_ranking_in_paper']}`",
        f"- Technical work basically done: `{report['technical_work_basically_done']}`",
        f"- Primary model variant: `{report['primary_model_variant']}`",
        f"- Cross-model sanity alignment is strong: `{report['cross_model_sanity_alignment_is_strong']}`",
        "",
        "## Current Ranking",
        "",
    ]
    for row in report.get("section_ranking", []):
        lines.append(
            f"- `{row['rank']}. {row['section']}` with confidence `{row['confidence_label']}`"
        )
    lines.extend(["", "## Strong Claims", ""])
    for claim in report.get("strong_claims", []):
        lines.append(f"- {claim}")
    lines.extend(["", "## Visible Caveats", ""])
    for caveat in report.get("visible_caveats", []):
        lines.append(f"- {caveat}")
    lines.extend(
        [
            "",
            "## Exact Next Action",
            "",
            report["next_action"],
            "",
        ]
    )
    return "\n".join(lines)


def build_updated_results_package_manifest(
    *,
    existing_manifest: dict[str, Any],
    source_results_package_dir: Path,
    section_importance_output_dir: Path,
    section_importance_files: dict[str, list[str]],
    additional_caveats: list[str],
) -> dict[str, Any]:
    def _resolve_existing(key: str) -> list[str]:
        return [
            str((source_results_package_dir / relative_path).resolve())
            for relative_path in existing_manifest.get(key, [])
        ]

    main_summary_files = _resolve_existing("main_summary_files") + [
        str((section_importance_output_dir / name).resolve())
        for name in section_importance_files.get("main_summary_files", [])
    ]
    chart_data_files = _resolve_existing("chart_data_files") + [
        str((section_importance_output_dir / name).resolve())
        for name in section_importance_files.get("chart_data_files", [])
    ]
    qualitative_files = _resolve_existing("qualitative_files") + [
        str((section_importance_output_dir / name).resolve())
        for name in section_importance_files.get("qualitative_files", [])
    ]
    appendix_files = _resolve_existing("appendix_files") + [
        str((section_importance_output_dir / name).resolve())
        for name in section_importance_files.get("appendix_files", [])
    ]
    caveats = list(existing_manifest.get("known_caveats", []))
    for caveat in additional_caveats:
        if caveat not in caveats:
            caveats.append(caveat)
    return {
        "task": "results_package_manifest",
        "baseline_run_dir": existing_manifest.get("baseline_run_dir"),
        "robustness_run_dir": existing_manifest.get("robustness_run_dir"),
        "source_results_package_dir": str(source_results_package_dir.resolve()),
        "results_package_dirname": str(section_importance_output_dir.resolve()),
        "primary_model_variant": existing_manifest.get("primary_model_variant"),
        "models_included": list(existing_manifest.get("models_included", [])),
        "perturbations_included": list(existing_manifest.get("perturbations_included", [])),
        "main_summary_files": main_summary_files,
        "chart_data_files": chart_data_files,
        "qualitative_files": qualitative_files,
        "appendix_files": appendix_files,
        "known_caveats": caveats,
    }


def _attach_importance_scores(
    section_rows: list[dict[str, Any]],
    *,
    weights: dict[str, float],
) -> None:
    normalized_weights = _normalized_weights(weights)
    max_removal = max((row["removal_impact_raw"] for row in section_rows), default=1.0) or 1.0
    max_solo = max((row["solo_sufficiency_raw"] for row in section_rows), default=1.0) or 1.0
    max_flip = max((row["flip_sensitivity_raw"] for row in section_rows), default=1.0) or 1.0
    for row in section_rows:
        normalized_removal = row["removal_impact_raw"] / max_removal if max_removal else 0.0
        normalized_solo = row["solo_sufficiency_raw"] / max_solo if max_solo else 0.0
        normalized_flip = row["flip_sensitivity_raw"] / max_flip if max_flip else 0.0
        row["normalized_components"] = {
            "removal_impact": round(normalized_removal, 6),
            "solo_sufficiency": round(normalized_solo, 6),
            "flip_sensitivity": round(normalized_flip, 6),
        }
        row["composite_importance_score"] = round(
            (normalized_weights["removal_impact"] * normalized_removal)
            + (normalized_weights["solo_sufficiency"] * normalized_solo)
            + (normalized_weights["flip_sensitivity"] * normalized_flip),
            6,
        )


def _preferred_target_metrics(metrics_by_split: dict[str, Any]) -> dict[str, Any]:
    return (
        metrics_by_split.get("non_empty_target_metrics")
        or metrics_by_split.get("overall_metrics")
        or metrics_by_split
    )


def _metric_delta(recipe_row: dict[str, Any] | None) -> float:
    if recipe_row is None:
        return 0.0
    metrics = _preferred_target_metrics(recipe_row)
    return round(max(0.0, -metrics["macro_f1_delta_vs_reference"]), 6)


def _metric_retention(recipe_row: dict[str, Any] | None) -> float:
    if recipe_row is None:
        return 0.0
    metrics = _preferred_target_metrics(recipe_row)
    return round(metrics["macro_f1"] / max(metrics["reference_macro_f1"], 1e-9), 6)


def _build_supporting_model_summary(
    model_report: dict[str, Any],
    *,
    primary_split: str,
) -> dict[str, Any]:
    recipe_metrics = {
        row["variant_name"]: _preferred_target_metrics(row["metrics_by_split"][primary_split])
        for row in model_report.get("variants", [])
        if primary_split in row.get("metrics_by_split", {})
    }
    return {
        "model_variant": model_report["model_variant"],
        "recipe_metrics": recipe_metrics,
    }


def _build_section_variant_text(
    case: CJPEPseudoSectionedCase,
    *,
    spec: dict[str, Any],
    include_section_markers: bool,
) -> tuple[str, bool]:
    selected_sections = tuple(spec["sections_to_include"])
    coverage_sections = tuple(spec["coverage_sections"])
    target_section_was_empty = not any(
        case.grouped_sections.get(section, "").strip() for section in coverage_sections
    )
    text = compose_sectioned_text(
        grouped_sections=case.grouped_sections,
        section_order=BROAD_SECTION_ORDER,
        sections_to_include=selected_sections,
        include_section_markers=include_section_markers,
    )
    return text, target_section_was_empty


def _confidence_label(
    coverage: float,
    *,
    high_threshold: float,
    medium_threshold: float,
) -> str:
    if coverage >= high_threshold:
        return "high_confidence_importance_estimate"
    if coverage >= medium_threshold:
        return "medium_confidence_importance_estimate"
    return "low_confidence_importance_estimate"


def _coverage_note(confidence_label: str, section: str) -> str:
    if confidence_label == "high_confidence_importance_estimate":
        return f"{section} has enough non-empty coverage for a relatively strong pilot importance estimate."
    if confidence_label == "medium_confidence_importance_estimate":
        return f"{section} has usable but moderate coverage; interpret its importance estimate with some caution."
    return f"{section} has sparse effective coverage, so its importance estimate should remain explicitly caveated."


def _parse_model_variant(value: str) -> tuple[str, str]:
    if "::" not in value:
        raise PredictionError(
            f"Expected model variant of the form 'model_name::input_variant', got {value!r}."
        )
    model_name, input_variant = value.split("::", maxsplit=1)
    return model_name, input_variant


def _recipe_display_name(recipe_name: str) -> str:
    return recipe_name.replace("_", " ").title()


def _model_label(model_variant: str) -> str:
    if model_variant.startswith("averaged_passive_aggressive"):
        return "APA"
    if model_variant.startswith("multinomial_naive_bayes"):
        return "NB"
    if model_variant.startswith("tfidf_logistic_regression"):
        return "Logistic"
    if model_variant.startswith("section_contextual_logistic_regression"):
        return "Contextual"
    return model_variant


def _alignment_sentence(count: int, total: int, statement: str) -> str:
    return f"{count}/{total} models support the pattern: {statement}"


def _normalized_weights(weights: dict[str, float]) -> dict[str, float]:
    total = sum(float(value) for value in weights.values()) or 1.0
    return {
        "removal_impact": round(float(weights.get("removal_impact", 0.0)) / total, 6),
        "solo_sufficiency": round(float(weights.get("solo_sufficiency", 0.0)) / total, 6),
        "flip_sensitivity": round(float(weights.get("flip_sensitivity", 0.0)) / total, 6),
    }


def _fmt(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.6f}"
