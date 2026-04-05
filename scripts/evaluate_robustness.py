from __future__ import annotations

import csv
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from legal_robustness.config.loader import load_app_config
from legal_robustness.robustness import (
    build_apa_focused_robustness_table,
    build_appendix_bundle,
    build_case_bundles,
    build_chart_data_coverage,
    build_chart_data_flip_rates,
    build_chart_data_main_performance,
    build_chart_data_robustness_deltas,
    build_comparative_robustness_metrics,
    build_comparative_robustness_next_step_summary,
    build_comparative_section_aware_robustness_report,
    build_failure_analysis,
    build_focused_perturbation_interpretation,
    build_first_robustness_phase_readiness_summary,
    build_paper_results_packaging_next_step_summary,
    build_paper_qualitative_examples,
    build_pilot_results_section_summary,
    build_perturbation_coverage_report,
    build_recipe_qualitative_bundle,
    build_results_narratives,
    build_results_package_manifest,
    build_section_aware_robustness_report,
    build_stability_vs_correctness_narrative,
    build_stability_vs_correctness_summary,
    build_table_main_results,
    build_table_model_comparison,
    build_table_stability_vs_correctness,
    evaluate_selected_perturbations,
    render_apa_focused_robustness_table,
    render_appendix_bundle,
    render_comparative_robustness_metrics,
    render_comparative_robustness_next_step_summary,
    render_comparative_section_aware_robustness_report,
    render_failure_analysis_summary,
    render_focused_perturbation_interpretation,
    render_first_robustness_phase_readiness_summary,
    render_paper_results_packaging_next_step_summary,
    render_paper_qualitative_examples,
    render_pilot_results_section_summary,
    render_perturbation_coverage_report,
    render_perturbed_evaluation_metrics,
    render_recipe_qualitative_bundle,
    render_results_narrative_main,
    render_results_narrative_supporting,
    render_results_package_manifest,
    render_section_aware_robustness_report,
    render_stability_vs_correctness_narrative,
    render_stability_vs_correctness_summary,
    render_table_main_results,
    render_table_model_comparison,
    render_table_stability_vs_correctness,
    resolve_baseline_run_dir,
)
from legal_robustness.prediction import (
    build_unperturbed_model_comparison,
    expand_unperturbed_model_variants,
    load_pseudo_sectioned_cases_from_run_dir,
    render_unperturbed_model_comparison,
)
from legal_robustness.utils.artifacts import (
    create_stage_output_dir,
    write_json,
    write_jsonl,
    write_parquet,
    write_text,
    write_yaml,
)
from legal_robustness.utils.cli import build_common_parser
from legal_robustness.utils.logging import configure_logging, get_logger
from legal_robustness.utils.paths import ensure_directory
from legal_robustness.utils.seeds import seed_everything


def parse_args() -> object:
    parser = build_common_parser(
        "Run the first section-aware robustness evaluation by scoring trained baselines on selected perturbation sets."
    )
    parser.add_argument(
        "--baseline-run-dir",
        default=None,
        help="Optional path to an existing prediction-baseline run directory. Defaults to the most recent run under outputs/reports/prediction_baselines.",
    )
    return parser.parse_args()


def _write_confusion_matrix_csv(path: Path, rows: list[dict[str, object]]) -> None:
    ensure_directory(path.parent)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _build_prediction_samples(rows, *, sample_size: int) -> list[dict[str, object]]:
    samples: list[dict[str, object]] = []
    for row in rows[:sample_size]:
        samples.append(
            {
                "case_id": row.case_id,
                "split": row.split,
                "gold_label": row.gold_label,
                "model_name": row.model_name,
                "input_variant": row.input_variant,
                "perturbation_recipe": row.perturbation_recipe,
                "perturbation_family": row.perturbation_family,
                "prediction": row.prediction,
                "prediction_score": row.prediction_score,
                "reference_prediction": row.reference_prediction,
                "prediction_flipped": row.prediction_flipped,
                "target_section": row.target_section,
                "target_section_was_empty": row.target_section_was_empty,
                "effective_coverage_group": row.effective_coverage_group,
            }
        )
    return samples


def _resolve_primary_model_variant(
    *,
    config,
    evaluation_report: dict[str, object],
    unperturbed_comparison: dict[str, object],
) -> str | None:
    selected_model_variants = tuple(evaluation_report.get("selected_model_variants", []))
    configured = getattr(config.robustness, "primary_model_variant", None)
    if (
        config.robustness.apa_centered_reporting
        and configured
        and configured in selected_model_variants
    ):
        return configured
    strongest_by_variant = (
        unperturbed_comparison.get("strongest_by_macro_f1_by_input_variant", {}).get("pseudo_all_sections", {})
        if unperturbed_comparison.get("strongest_by_macro_f1_by_input_variant")
        else {}
    )
    if strongest_by_variant.get("model_variant") in selected_model_variants:
        return strongest_by_variant["model_variant"]
    strongest = unperturbed_comparison.get("strongest_by_macro_f1")
    if strongest and strongest.get("model_variant") in selected_model_variants:
        return strongest["model_variant"]
    return selected_model_variants[0] if selected_model_variants else None


def main() -> int:
    args = parse_args()
    config = load_app_config(
        config_path=Path(args.config),
        dataset_root_override=Path(args.dataset_root) if args.dataset_root else None,
        run_name_override=args.run_name,
        log_level_override=args.log_level,
        project_root=PROJECT_ROOT,
    )
    output_dir = create_stage_output_dir(
        base_dir=config.output.reports_dir / "robustness",
        stage_name="robustness",
        run_name=config.runtime.run_name,
    )
    configure_logging(
        level=config.logging.level,
        log_file=output_dir / "evaluate_robustness.log" if config.logging.log_to_file else None,
    )
    logger = get_logger("legal_robustness.scripts.evaluate_robustness")
    seed_everything(config.runtime.seed, deterministic=config.runtime.deterministic)
    write_yaml(output_dir / "config_snapshot.yaml", config.to_dict())

    baseline_run_dir = resolve_baseline_run_dir(
        reports_dir=config.output.reports_dir,
        explicit_run_dir=Path(args.baseline_run_dir) if args.baseline_run_dir else None,
    )
    logger.info("Using baseline artifacts from %s", baseline_run_dir)

    evaluation_report, prediction_rows, confusion_rows = evaluate_selected_perturbations(
        baseline_run_dir=baseline_run_dir,
        config=config,
        logger=logger,
    )
    coverage_report = build_perturbation_coverage_report(evaluation_report)
    primary_split = config.robustness.evaluation_splits[0]
    baseline_report_path = baseline_run_dir / "baseline_prediction_metrics.json"
    baseline_report = None
    if baseline_report_path.exists():
        import json

        baseline_report = json.loads(baseline_report_path.read_text(encoding="utf-8"))
    else:
        baseline_report = {"models": {}}
    selected_model_variants = expand_unperturbed_model_variants(
        config.robustness.selected_model_variants,
        include_full_text=config.robustness.include_full_text_in_unperturbed_comparison,
    )
    unperturbed_comparison = build_unperturbed_model_comparison(
        baseline_report,
        primary_split=primary_split,
        selected_model_variants=selected_model_variants,
    )
    pseudo_sectioned_cases = load_pseudo_sectioned_cases_from_run_dir(
        Path(baseline_report["source_section_transfer_run_dir"])
    )
    robustness_report = build_section_aware_robustness_report(
        evaluation_report,
        primary_split=primary_split,
        isolate_low_coverage=config.robustness.isolate_low_coverage_recipes,
    )
    comparative_metrics = build_comparative_robustness_metrics(
        evaluation_report,
        primary_split=primary_split,
        include_relative_retention=config.robustness.compute_relative_retention,
    )
    selected_model_variants_runtime = tuple(evaluation_report.get("selected_model_variants", []))
    primary_model_variant = _resolve_primary_model_variant(
        config=config,
        evaluation_report=evaluation_report,
        unperturbed_comparison=unperturbed_comparison,
    )
    contextual_model_variant = next(
        (
            model_variant
            for model_variant in selected_model_variants_runtime
            if "section_contextual_logistic_regression" in model_variant
        ),
        None,
    )
    failure_analysis_summary, failure_analysis_cases = build_failure_analysis(
        prediction_rows,
        pseudo_sectioned_cases=pseudo_sectioned_cases,
        primary_split=primary_split,
        focused_recipes=config.robustness.failure_analysis_recipes,
        selected_model_variants=config.robustness.selected_model_variants,
        primary_model_variant=primary_model_variant,
        contextual_model_variant=contextual_model_variant,
        case_limit=config.robustness.failure_analysis_case_limit,
        preview_chars=config.robustness.failure_analysis_preview_chars,
        enable_disagreement_analysis=config.robustness.enable_failure_analysis_disagreement_analysis,
    )
    comparative_report = build_comparative_section_aware_robustness_report(
        unperturbed_comparison,
        comparative_metrics,
        isolate_low_coverage=config.robustness.isolate_low_coverage_recipes,
        failure_analysis_summary=failure_analysis_summary,
    )
    comparative_next_step_summary = build_comparative_robustness_next_step_summary(
        unperturbed_comparison,
        comparative_report,
        failure_analysis_summary=failure_analysis_summary,
    )
    apa_focused_table = build_apa_focused_robustness_table(
        unperturbed_comparison,
        comparative_metrics,
        primary_model_variant=primary_model_variant or "",
        comparison_model_variants=config.robustness.selected_model_variants,
        focused_recipes=config.robustness.failure_analysis_recipes,
    )
    stability_summary = build_stability_vs_correctness_summary(
        prediction_rows,
        primary_split=primary_split,
        focused_recipes=config.robustness.failure_analysis_recipes,
        primary_model_variant=primary_model_variant or "",
        comparison_model_variants=config.robustness.stability_comparison_model_variants,
    )
    focused_interpretation = build_focused_perturbation_interpretation(
        apa_focused_table,
    )
    readiness_summary = build_first_robustness_phase_readiness_summary(
        evaluation_report,
        robustness_report,
    )
    qualitative_report, qualitative_examples = build_paper_qualitative_examples(
        failure_analysis_cases,
        focused_recipes=config.robustness.failure_analysis_recipes,
        primary_model_variant=primary_model_variant or "",
        model_variants=selected_model_variants_runtime,
        count_per_recipe=config.robustness.qualitative_example_count_per_recipe,
        preview_chars=config.robustness.qualitative_preview_chars,
        include_model_variants=config.robustness.qualitative_include_model_variants,
        examples_per_category=config.robustness.qualitative_examples_per_category,
    )
    case_bundles = build_case_bundles(
        failure_analysis_cases,
        focused_recipes=config.robustness.failure_analysis_recipes,
        primary_model_variant=primary_model_variant or "",
        model_variants=selected_model_variants_runtime,
        bundle_size=config.robustness.case_bundle_size,
        preview_chars=config.robustness.qualitative_preview_chars,
        include_model_variants=config.robustness.qualitative_include_model_variants,
    )
    pilot_results_summary = build_pilot_results_section_summary(
        unperturbed_comparison,
        apa_focused_table,
        stability_summary,
        qualitative_report,
        comparative_next_step_summary,
    )
    primary_recipe = (
        comparative_next_step_summary.get("primary_writeup_perturbation")
        or "keep_reasoning_only"
    )
    secondary_recipe = (
        comparative_next_step_summary.get("secondary_writeup_perturbation")
        or "drop_precedents"
    )
    table_main_results = build_table_main_results(
        apa_focused_table,
        primary_recipe=primary_recipe,
        secondary_recipe=secondary_recipe,
    )
    table_model_comparison = build_table_model_comparison(
        unperturbed_comparison,
        comparative_metrics,
        primary_model_variant=primary_model_variant or "",
        focused_recipes=(primary_recipe, secondary_recipe),
    )
    chart_data_main_performance = build_chart_data_main_performance(
        unperturbed_comparison,
    )
    chart_data_robustness_deltas = build_chart_data_robustness_deltas(
        comparative_metrics,
        focused_recipes=(primary_recipe, secondary_recipe),
    )
    chart_data_flip_rates = build_chart_data_flip_rates(
        comparative_metrics,
        stability_summary,
        focused_recipes=(primary_recipe, secondary_recipe),
    )
    chart_data_coverage = build_chart_data_coverage(
        comparative_metrics,
        focused_recipes=(primary_recipe, secondary_recipe),
    )
    table_stability_vs_correctness = build_table_stability_vs_correctness(
        stability_summary,
    )
    stability_narrative = build_stability_vs_correctness_narrative(
        stability_summary,
    )
    primary_qualitative_bundle = build_recipe_qualitative_bundle(
        qualitative_examples,
        recipe_name=primary_recipe,
        count=config.robustness.primary_qualitative_example_count,
        preview_chars=config.robustness.narrative_preview_chars,
    )
    secondary_qualitative_bundle = build_recipe_qualitative_bundle(
        qualitative_examples,
        recipe_name=secondary_recipe,
        count=config.robustness.secondary_qualitative_example_count,
        preview_chars=config.robustness.narrative_preview_chars,
    )
    packaging_next_step_summary = build_paper_results_packaging_next_step_summary(
        comparative_next_step_summary,
    )
    results_narratives = build_results_narratives(
        table_main_results,
        table_model_comparison,
        packaging_next_step_summary,
    )
    appendix_keep_reasoning_only_bundle = build_appendix_bundle(
        recipe_name="keep_reasoning_only",
        recipe_summary_row=next(
            (
                row
                for row in table_main_results.get("rows", [])
                if row["condition_key"] == "keep_reasoning_only"
            ),
            None,
        ),
        model_comparison=table_model_comparison,
        qualitative_bundle=primary_qualitative_bundle if primary_recipe == "keep_reasoning_only" else secondary_qualitative_bundle,
        case_bundle_filename="keep_reasoning_only_case_bundle.jsonl",
    )
    appendix_drop_precedents_bundle = build_appendix_bundle(
        recipe_name="drop_precedents",
        recipe_summary_row=next(
            (
                row
                for row in table_main_results.get("rows", [])
                if row["condition_key"] == "drop_precedents"
            ),
            None,
        ),
        model_comparison=table_model_comparison,
        qualitative_bundle=primary_qualitative_bundle if primary_recipe == "drop_precedents" else secondary_qualitative_bundle,
        case_bundle_filename="drop_precedents_case_bundle.jsonl",
    )
    comparative_report["apa_focused_summary_rows"] = apa_focused_table.get("rows", [])
    comparative_report["stability_vs_correctness_summary"] = stability_summary
    comparative_report["qualitative_artifact_references"] = {
        "paper_qualitative_examples": "paper_qualitative_examples.md",
        "drop_precedents_case_bundle": "drop_precedents_case_bundle.jsonl",
        "keep_reasoning_only_case_bundle": "keep_reasoning_only_case_bundle.jsonl",
    }
    package_dir = output_dir / config.robustness.results_package_dirname
    ensure_directory(package_dir)

    write_json(output_dir / "perturbed_evaluation_metrics.json", evaluation_report)
    write_text(
        output_dir / "perturbed_evaluation_metrics.md",
        render_perturbed_evaluation_metrics(evaluation_report),
    )
    if config.robustness.export_per_example_predictions:
        write_parquet(
            output_dir / "perturbed_predictions.parquet",
            [row.to_dict() for row in prediction_rows],
        )
    if config.robustness.export_comparative_predictions:
        write_parquet(
            output_dir / "comparative_perturbed_predictions.parquet",
            [row.to_dict() for row in prediction_rows],
        )
        write_parquet(
            output_dir / "focused_comparative_perturbed_predictions.parquet",
            [row.to_dict() for row in prediction_rows],
        )
    write_jsonl(
        output_dir / "perturbed_predictions_sample.jsonl",
        _build_prediction_samples(prediction_rows, sample_size=config.robustness.sample_size),
    )
    if config.robustness.export_comparative_predictions:
        write_jsonl(
            output_dir / "comparative_perturbed_predictions_sample.jsonl",
            _build_prediction_samples(prediction_rows, sample_size=config.robustness.sample_size),
        )
        write_jsonl(
            output_dir / "focused_comparative_perturbed_predictions_sample.jsonl",
            _build_prediction_samples(prediction_rows, sample_size=config.robustness.sample_size),
        )
    _write_confusion_matrix_csv(
        output_dir / "perturbed_confusion_matrix.csv",
        confusion_rows,
    )
    write_json(output_dir / "unperturbed_model_comparison.json", unperturbed_comparison)
    write_text(
        output_dir / "unperturbed_model_comparison.md",
        render_unperturbed_model_comparison(unperturbed_comparison),
    )
    write_json(output_dir / "perturbation_coverage_report.json", coverage_report)
    write_text(
        output_dir / "perturbation_coverage_report.md",
        render_perturbation_coverage_report(coverage_report),
    )
    write_json(output_dir / "section_aware_robustness_report.json", robustness_report)
    write_text(
        output_dir / "section_aware_robustness_report.md",
        render_section_aware_robustness_report(robustness_report),
    )
    write_json(output_dir / "comparative_robustness_metrics.json", comparative_metrics)
    write_text(
        output_dir / "comparative_robustness_metrics.md",
        render_comparative_robustness_metrics(comparative_metrics),
    )
    write_json(output_dir / "focused_comparative_robustness_metrics.json", comparative_metrics)
    write_text(
        output_dir / "focused_comparative_robustness_metrics.md",
        render_comparative_robustness_metrics(comparative_metrics),
    )
    write_json(
        output_dir / "comparative_section_aware_robustness_report.json",
        comparative_report,
    )
    write_text(
        output_dir / "comparative_section_aware_robustness_report.md",
        render_comparative_section_aware_robustness_report(comparative_report),
    )
    write_json(
        output_dir / "focused_comparative_robustness_report.json",
        comparative_report,
    )
    write_text(
        output_dir / "focused_comparative_robustness_report.md",
        render_comparative_section_aware_robustness_report(comparative_report),
    )
    write_json(
        output_dir / "comparative_robustness_next_step_summary.json",
        comparative_next_step_summary,
    )
    write_text(
        output_dir / "comparative_robustness_next_step_summary.md",
        render_comparative_robustness_next_step_summary(comparative_next_step_summary),
    )
    write_json(output_dir / "failure_analysis_summary.json", failure_analysis_summary)
    write_text(
        output_dir / "failure_analysis_summary.md",
        render_failure_analysis_summary(failure_analysis_summary),
    )
    write_jsonl(
        output_dir / "failure_analysis_cases.jsonl",
        failure_analysis_cases,
    )
    write_json(output_dir / "paper_qualitative_examples_report.json", qualitative_report)
    write_jsonl(
        output_dir / "paper_qualitative_examples.jsonl",
        qualitative_examples,
    )
    write_text(
        output_dir / "paper_qualitative_examples.md",
        render_paper_qualitative_examples(qualitative_report, qualitative_examples),
    )
    write_json(output_dir / "apa_focused_robustness_table.json", apa_focused_table)
    write_text(
        output_dir / "apa_focused_robustness_table.md",
        render_apa_focused_robustness_table(apa_focused_table),
    )
    write_json(output_dir / "stability_vs_correctness_summary.json", stability_summary)
    write_text(
        output_dir / "stability_vs_correctness_summary.md",
        render_stability_vs_correctness_summary(stability_summary),
    )
    write_text(
        output_dir / "focused_perturbation_interpretation.md",
        render_focused_perturbation_interpretation(focused_interpretation),
    )
    write_json(output_dir / "pilot_results_section_summary.json", pilot_results_summary)
    write_text(
        output_dir / "pilot_results_section_summary.md",
        render_pilot_results_section_summary(pilot_results_summary),
    )
    write_jsonl(
        output_dir / "drop_precedents_case_bundle.jsonl",
        case_bundles.get("drop_precedents", []),
    )
    write_jsonl(
        output_dir / "keep_reasoning_only_case_bundle.jsonl",
        case_bundles.get("keep_reasoning_only", []),
    )
    write_json(output_dir / "first_robustness_phase_readiness_summary.json", readiness_summary)
    write_text(
        output_dir / "first_robustness_phase_readiness_summary.md",
        render_first_robustness_phase_readiness_summary(readiness_summary),
    )

    write_json(package_dir / "table_main_results.json", table_main_results)
    write_text(
        package_dir / "table_main_results.md",
        render_table_main_results(table_main_results),
    )
    write_json(package_dir / "table_model_comparison.json", table_model_comparison)
    write_text(
        package_dir / "table_model_comparison.md",
        render_table_model_comparison(table_model_comparison),
    )
    if config.robustness.export_chart_ready_data:
        write_json(package_dir / "chart_data_main_performance.json", chart_data_main_performance)
        write_json(package_dir / "chart_data_robustness_deltas.json", chart_data_robustness_deltas)
        write_json(package_dir / "chart_data_flip_rates.json", chart_data_flip_rates)
        write_json(package_dir / "chart_data_coverage.json", chart_data_coverage)
    write_json(
        package_dir / "table_stability_vs_correctness.json",
        table_stability_vs_correctness,
    )
    write_text(
        package_dir / "table_stability_vs_correctness.md",
        render_table_stability_vs_correctness(table_stability_vs_correctness),
    )
    write_text(
        package_dir / "stability_vs_correctness_narrative.md",
        render_stability_vs_correctness_narrative(stability_narrative),
    )
    write_text(
        package_dir / "results_narrative_main.md",
        render_results_narrative_main(results_narratives),
    )
    write_text(
        package_dir / "results_narrative_supporting.md",
        render_results_narrative_supporting(results_narratives),
    )
    write_text(
        package_dir / "focused_perturbation_interpretation.md",
        render_focused_perturbation_interpretation(focused_interpretation),
    )
    write_json(package_dir / "pilot_results_section_summary.json", pilot_results_summary)
    write_text(
        package_dir / "pilot_results_section_summary.md",
        render_pilot_results_section_summary(pilot_results_summary),
    )
    write_text(
        package_dir / "qualitative_examples_primary.md",
        render_recipe_qualitative_bundle(primary_qualitative_bundle),
    )
    write_text(
        package_dir / "qualitative_examples_secondary.md",
        render_recipe_qualitative_bundle(secondary_qualitative_bundle),
    )
    write_jsonl(
        package_dir / "paper_qualitative_examples.jsonl",
        qualitative_examples,
    )
    write_text(
        package_dir / "paper_qualitative_examples.md",
        render_paper_qualitative_examples(qualitative_report, qualitative_examples),
    )
    write_jsonl(
        package_dir / "drop_precedents_case_bundle.jsonl",
        case_bundles.get("drop_precedents", []),
    )
    write_jsonl(
        package_dir / "keep_reasoning_only_case_bundle.jsonl",
        case_bundles.get("keep_reasoning_only", []),
    )
    if config.robustness.export_appendix_bundles:
        write_text(
            package_dir / "appendix_keep_reasoning_only_bundle.md",
            render_appendix_bundle(appendix_keep_reasoning_only_bundle),
        )
        write_text(
            package_dir / "appendix_drop_precedents_bundle.md",
            render_appendix_bundle(appendix_drop_precedents_bundle),
        )
    write_json(
        package_dir / "paper_results_packaging_next_step_summary.json",
        packaging_next_step_summary,
    )
    write_text(
        package_dir / "paper_results_packaging_next_step_summary.md",
        render_paper_results_packaging_next_step_summary(packaging_next_step_summary),
    )
    main_summary_files = [
        "table_main_results.json",
        "table_main_results.md",
        "table_model_comparison.json",
        "table_model_comparison.md",
        "table_stability_vs_correctness.json",
        "table_stability_vs_correctness.md",
        "stability_vs_correctness_narrative.md",
        "results_narrative_main.md",
        "results_narrative_supporting.md",
        "focused_perturbation_interpretation.md",
        "pilot_results_section_summary.json",
        "pilot_results_section_summary.md",
        "paper_results_packaging_next_step_summary.json",
        "paper_results_packaging_next_step_summary.md",
    ]
    chart_data_files = [
        "chart_data_main_performance.json",
        "chart_data_robustness_deltas.json",
        "chart_data_flip_rates.json",
        "chart_data_coverage.json",
    ] if config.robustness.export_chart_ready_data else []
    qualitative_files = [
        "qualitative_examples_primary.md",
        "qualitative_examples_secondary.md",
        "paper_qualitative_examples.md",
        "drop_precedents_case_bundle.jsonl",
        "keep_reasoning_only_case_bundle.jsonl",
    ]
    appendix_files = [
        "appendix_keep_reasoning_only_bundle.md",
        "appendix_drop_precedents_bundle.md",
    ] if config.robustness.export_appendix_bundles else []
    results_package_manifest = build_results_package_manifest(
        baseline_run_dir=baseline_run_dir,
        robustness_run_dir=output_dir,
        package_dirname=config.robustness.results_package_dirname,
        primary_model_variant=primary_model_variant or "",
        models_included=list(selected_model_variants_runtime),
        perturbations_included=[primary_recipe, secondary_recipe],
        chart_data_files=chart_data_files,
        main_summary_files=main_summary_files,
        qualitative_files=qualitative_files,
        appendix_files=appendix_files,
        caveats=packaging_next_step_summary["visible_caveats"],
    )
    write_json(
        package_dir / "results_package_manifest.json",
        results_package_manifest,
    )
    write_text(
        package_dir / "results_package_manifest.md",
        render_results_package_manifest(results_package_manifest),
    )

    logger.info(
        "Perturbed evaluation complete for model_variants=%s recipes=%s",
        evaluation_report["selected_model_variants"],
        evaluation_report["selected_perturbation_recipes"],
    )
    logger.info(
        "Robustness summary: informative=%s weak=%s",
        len(robustness_report["informative_perturbations"]),
        len(robustness_report["weak_probe_perturbations"]),
    )
    logger.info(
        "Comparative robustness summary: informative=%s weak=%s",
        len(comparative_report["informative_perturbation_comparison"]),
        len(comparative_report["weak_probe_perturbation_comparison"]),
    )
    logger.info(
        "Qualitative example export complete: examples=%s recipes=%s contextual=%s",
        len(qualitative_examples),
        qualitative_report["focused_recipes"],
        contextual_model_variant,
    )
    logger.info(
        "APA-centered package complete: primary_model=%s bundles=%s pilot_ready=%s",
        primary_model_variant,
        {recipe: len(rows) for recipe, rows in case_bundles.items()},
        pilot_results_summary["ready_for_pilot_results_section"],
    )
    logger.info(
        "Results package complete: package_dir=%s main_files=%s chart_files=%s appendix_files=%s",
        package_dir,
        len(main_summary_files),
        len(chart_data_files),
        len(appendix_files),
    )
    logger.info("Robustness artifacts written to %s", output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
