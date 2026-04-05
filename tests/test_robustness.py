from __future__ import annotations

import json
from pathlib import Path

from legal_robustness.config.schema import (
    AppConfig,
    DataConfig,
    LoggingConfig,
    OutputConfig,
    PerturbationConfig,
    PredictionConfig,
    RobustnessConfig,
    RuntimeConfig,
    SectionConfig,
    SectionTransferConfig,
)
from legal_robustness.prediction.train_baseline import train_prediction_baselines
from legal_robustness.prediction import (
    build_unperturbed_model_comparison,
    expand_unperturbed_model_variants,
)
from legal_robustness.robustness import (
    build_appendix_layout_guide,
    build_apa_focused_robustness_table,
    build_appendix_bundle,
    build_case_bundles,
    build_chart_data_coverage,
    build_chart_data_flip_rates,
    build_chart_data_main_performance,
    build_chart_data_robustness_deltas,
    build_chart_data_section_importance_coverage,
    build_chart_data_section_importance_ranking,
    build_chart_data_section_importance_scores,
    build_claim_to_evidence_traceability,
    build_comparative_robustness_metrics,
    build_comparative_robustness_next_step_summary,
    build_comparative_section_aware_robustness_report,
    build_failure_analysis,
    build_figure_captions,
    build_final_package_consistency_check,
    build_focused_perturbation_interpretation,
    build_first_robustness_phase_readiness_summary,
    build_main_text_layout_guide,
    build_paper_consistency_check,
    build_paper_abstract_support,
    build_paper_conclusion_support,
    build_paper_freeze_manifest,
    build_paper_handoff_summary,
    build_paper_intro_support,
    build_paper_limitations_ethics_support,
    build_paper_method_support,
    build_paper_results_packaging_next_step_summary,
    build_paper_readiness_summary,
    build_paper_results_support,
    build_paper_qualitative_examples,
    build_pilot_results_section_summary,
    build_perturbation_coverage_report,
    build_recipe_qualitative_bundle,
    build_results_narratives,
    build_results_package_manifest,
    build_submission_package_manifest,
    build_section_importance_cross_model_check,
    build_section_importance_next_step_summary,
    build_section_importance_scores,
    build_section_aware_robustness_report,
    build_stability_vs_correctness_narrative,
    build_stability_vs_correctness_summary,
    build_table_captions,
    build_targeted_strengthening_check,
    build_table_main_results,
    build_table_model_comparison,
    build_table_stability_vs_correctness,
    build_updated_results_package_manifest,
    evaluate_section_importance_model,
    evaluate_selected_perturbations,
    render_apa_focused_robustness_table,
    render_appendix_bundle,
    render_comparative_robustness_metrics,
    render_comparative_robustness_next_step_summary,
    render_comparative_section_aware_robustness_report,
    render_draft_support_appendix,
    render_draft_support_introduction,
    render_draft_support_limitations,
    render_draft_support_method,
    render_draft_support_results,
    render_claim_to_evidence_traceability,
    render_failure_analysis_summary,
    render_figure_captions,
    render_figure_manifest,
    render_final_package_consistency_check,
    render_focused_perturbation_interpretation,
    render_first_robustness_phase_readiness_summary,
    render_main_text_layout_guide,
    render_paper_consistency_check,
    render_paper_abstract_support,
    render_paper_conclusion_support,
    render_paper_figure_selection,
    render_paper_freeze_manifest,
    render_paper_handoff_summary,
    render_paper_intro_support,
    render_paper_limitations_ethics_support,
    render_paper_method_support,
    render_paper_results_packaging_next_step_summary,
    render_paper_results_support,
    render_paper_readiness_summary,
    render_paper_reproducibility_commands,
    render_paper_table_selection,
    render_paper_qualitative_examples,
    render_pilot_results_section_summary,
    render_perturbation_coverage_report,
    render_perturbed_evaluation_metrics,
    render_recipe_qualitative_bundle,
    render_results_narrative_main,
    render_results_narrative_supporting,
    render_results_package_manifest,
    render_submission_figures,
    render_submission_package_manifest,
    render_section_importance_cross_model_check,
    render_section_importance_narrative_main,
    render_section_importance_narrative_supporting,
    render_section_importance_next_step_summary,
    render_section_importance_ranking,
    render_section_importance_scores,
    render_section_aware_robustness_report,
    render_stability_vs_correctness_narrative,
    render_stability_vs_correctness_summary,
    render_appendix_layout_guide,
    render_table_captions,
    render_targeted_strengthening_check,
    render_table_main_results,
    render_table_model_comparison,
    render_table_stability_vs_correctness,
)
from legal_robustness.section_transfer.types import CJPEPseudoSectionedCase
from legal_robustness.perturbations.apply import generate_perturbation_sets
from legal_robustness.utils.artifacts import write_json, write_jsonl, write_parquet, write_text


def build_test_config(project_root: Path) -> AppConfig:
    return AppConfig(
        project_name="robustness_test_project",
        project_root=project_root,
        data=DataConfig(dataset_root=project_root / "dataset"),
        output=OutputConfig(
            root_dir=project_root / "outputs",
            reports_dir=project_root / "outputs" / "reports",
            dataset_inspection_dir=project_root / "outputs" / "reports" / "dataset_inspection",
            caches_dir=project_root / "outputs" / "caches",
            models_dir=project_root / "outputs" / "models",
            perturbations_dir=project_root / "outputs" / "perturbations",
            evaluations_dir=project_root / "outputs" / "evaluations",
            analysis_dir=project_root / "outputs" / "analysis",
        ),
        logging=LoggingConfig(),
        runtime=RuntimeConfig(seed=7),
        sections=SectionConfig(),
        section_transfer=SectionTransferConfig(),
        prediction=PredictionConfig(
            baseline_models=(
                "tfidf_logistic_regression",
                "multinomial_naive_bayes",
                "averaged_passive_aggressive",
                "section_contextual_logistic_regression",
            ),
            input_variants=("pseudo_all_sections",),
            evaluation_splits=("dev", "test"),
            hashing_dimension=512,
            logistic_epochs=8,
            logistic_learning_rate=0.2,
            passive_aggressive_epochs=10,
            passive_aggressive_aggressiveness=0.8,
            contextual_input_variants=("pseudo_all_sections",),
            contextual_hashing_dimension=1024,
            contextual_learning_rate=0.15,
            contextual_epochs=10,
            contextual_l2_weight=0.000001,
            contextual_section_sentence_limit=2,
            contextual_sentence_token_limit=24,
            sample_size=3,
            prediction_preview_chars=100,
        ),
        perturbation=PerturbationConfig(
            enabled_recipes=(
                "drop_precedents",
                "keep_reasoning_only",
                "drop_conclusion",
            ),
            evaluation_splits=("test",),
            sample_size=3,
            preview_chars=100,
        ),
        robustness=RobustnessConfig(
            selected_model_variants=(
                "tfidf_logistic_regression::pseudo_all_sections",
                "multinomial_naive_bayes::pseudo_all_sections",
                "averaged_passive_aggressive::pseudo_all_sections",
                "section_contextual_logistic_regression::pseudo_all_sections",
            ),
            selected_perturbation_recipes=("drop_precedents", "keep_reasoning_only", "drop_conclusion"),
            evaluation_splits=("test",),
            high_coverage_min_fraction=0.7,
            medium_coverage_min_fraction=0.3,
            isolate_low_coverage_recipes=True,
            include_full_text_in_unperturbed_comparison=False,
            apa_centered_reporting=True,
            primary_model_variant="averaged_passive_aggressive::pseudo_all_sections",
            reference_context_variants=("pseudo_all_sections",),
            sample_size=3,
            failure_analysis_recipes=("drop_precedents", "keep_reasoning_only"),
            failure_analysis_case_limit=4,
            failure_analysis_preview_chars=80,
            enable_failure_analysis_disagreement_analysis=True,
            qualitative_example_count_per_recipe=2,
            qualitative_examples_per_category=1,
            qualitative_preview_chars=90,
            case_bundle_size=3,
            results_package_dirname="results_package",
            primary_qualitative_example_count=2,
            secondary_qualitative_example_count=2,
            narrative_preview_chars=70,
            export_chart_ready_data=True,
            export_appendix_bundles=True,
            canonical_section_transfer_run_dir=project_root / "section_transfer_run",
            canonical_baseline_run_dir=project_root / "baseline_run",
            canonical_robustness_run_dir=project_root / "robustness_run",
            paper_freeze_output_dirname="paper_drafting_package",
            export_writing_support_bundles=True,
            run_targeted_strengthening_check=True,
            export_paper_selection_guidance=True,
            stability_comparison_model_variants=(
                "multinomial_naive_bayes::pseudo_all_sections",
                "tfidf_logistic_regression::pseudo_all_sections",
            ),
            canonical_paper_drafting_package_dir=project_root / "paper_drafting_package",
            canonical_section_importance_run_dir=project_root / "section_importance_run",
            submission_package_output_dirname="submission_package",
            submission_figure_formats=("svg",),
            export_claim_traceability=True,
            export_caption_files=True,
            export_placement_guides=True,
        ),
    )


def make_case(
    case_id: str,
    split: str,
    label: int,
    *,
    precedent_token: str,
    reasoning_token: str,
    conclusion_text: str,
) -> CJPEPseudoSectionedCase:
    facts = "The material tax facts are recorded."
    precedents = f"The court discussed precedent {precedent_token} in detail."
    reasoning = f"The reasoning text contains signal {reasoning_token}."
    other = "Procedural narrative."
    grouped_sections = {
        "facts": facts,
        "precedents": precedents,
        "reasoning": reasoning,
        "conclusion": conclusion_text,
        "other": other,
    }
    sentences = [facts, precedents, reasoning, conclusion_text, other]
    return CJPEPseudoSectionedCase(
        case_id=case_id,
        cjpe_label=label,
        split=split,
        subset="single",
        raw_text=" ".join(sentence for sentence in sentences if sentence),
        sentences=sentences,
        sentence_indices=[0, 1, 2, 3, 4],
        sentence_start_chars=[0, 10, 20, 30, 40],
        sentence_end_chars=[9, 19, 29, 39, 49],
        predicted_broad_labels=["facts", "precedents", "reasoning", "conclusion", "other"],
        predicted_label_scores=[0.9, 0.91, 0.92, 0.93, 0.8],
        grouped_sections=grouped_sections,
        section_sentence_map={
            "facts": [0],
            "precedents": [1],
            "reasoning": [2],
            "conclusion": [3] if conclusion_text else [],
            "other": [4],
        },
        section_lengths_sentences={
            "facts": 1,
            "precedents": 1,
            "reasoning": 1,
            "conclusion": 1 if conclusion_text else 0,
            "other": 1,
        },
        section_lengths_chars={key: len(value) for key, value in grouped_sections.items()},
        prediction_metadata={"classifier_type": "naive_bayes"},
        source_file="cjpe/sample.parquet",
        source_metadata={"source_row_index": 0},
    )


def make_cases() -> list[CJPEPseudoSectionedCase]:
    train_cases = [
        make_case(f"train-pos-{idx}", "train", 1, precedent_token="ALLOW_PRECEDENT", reasoning_token="neutral", conclusion_text="The matter is disposed.")
        for idx in range(4)
    ] + [
        make_case(f"train-neg-{idx}", "train", 0, precedent_token="DISMISS_PRECEDENT", reasoning_token="neutral", conclusion_text="The matter is disposed.")
        for idx in range(4)
    ]
    dev_cases = [
        make_case("dev-pos", "dev", 1, precedent_token="ALLOW_PRECEDENT", reasoning_token="neutral", conclusion_text="The matter is disposed."),
        make_case("dev-neg", "dev", 0, precedent_token="DISMISS_PRECEDENT", reasoning_token="neutral", conclusion_text="The matter is disposed."),
    ]
    test_cases = [
        make_case("test-pos-a", "test", 1, precedent_token="ALLOW_PRECEDENT", reasoning_token="neutral", conclusion_text=""),
        make_case("test-pos-b", "test", 1, precedent_token="ALLOW_PRECEDENT", reasoning_token="neutral", conclusion_text=""),
        make_case("test-neg-a", "test", 0, precedent_token="DISMISS_PRECEDENT", reasoning_token="neutral", conclusion_text=""),
        make_case("test-neg-b", "test", 0, precedent_token="DISMISS_PRECEDENT", reasoning_token="neutral", conclusion_text="Conclusion only here."),
    ]
    return train_cases + dev_cases + test_cases


def prepare_baseline_run(tmp_path: Path) -> tuple[Path, AppConfig]:
    config = build_test_config(tmp_path)
    baseline_run_dir = tmp_path / "baseline_run"
    baseline_run_dir.mkdir(parents=True, exist_ok=True)
    cases = make_cases()

    baseline_report, predictions_by_split, _, _ = train_prediction_baselines(
        cases,
        config=config,
        output_dir=baseline_run_dir,
        source_section_transfer_run_dir=tmp_path / "section_transfer_run",
    )
    write_json(baseline_run_dir / "baseline_prediction_metrics.json", baseline_report)
    for split_name, rows in predictions_by_split.items():
        write_parquet(
            baseline_run_dir / f"baseline_prediction_predictions_{split_name}.parquet",
            [row.to_dict() for row in rows],
        )

    perturbation_rows_by_recipe, perturbation_manifest, _ = generate_perturbation_sets(
        cases,
        config=config,
    )
    perturbation_dir = baseline_run_dir / "cjpe_perturbation_sets"
    perturbation_dir.mkdir(parents=True, exist_ok=True)
    for recipe_name, rows in perturbation_rows_by_recipe.items():
        write_parquet(
            perturbation_dir / f"{recipe_name}.parquet",
            [row.to_dict() for row in rows],
        )
    write_json(baseline_run_dir / "perturbation_manifest.json", perturbation_manifest)
    return baseline_run_dir, config


def test_evaluate_selected_perturbations_computes_deltas_and_flips(tmp_path: Path) -> None:
    baseline_run_dir, config = prepare_baseline_run(tmp_path)

    evaluation_report, prediction_rows, confusion_rows = evaluate_selected_perturbations(
        baseline_run_dir=baseline_run_dir,
        config=config,
    )

    recipe_metrics = evaluation_report["model_variant_results"]["tfidf_logistic_regression::pseudo_all_sections"]["recipes"]["drop_precedents"]["metrics_by_split"]["test"]["overall_metrics"]
    other_recipe_metrics = evaluation_report["model_variant_results"]["multinomial_naive_bayes::pseudo_all_sections"]["recipes"]["drop_precedents"]["metrics_by_split"]["test"]["overall_metrics"]
    third_recipe_metrics = evaluation_report["model_variant_results"]["averaged_passive_aggressive::pseudo_all_sections"]["recipes"]["drop_precedents"]["metrics_by_split"]["test"]["overall_metrics"]
    assert prediction_rows
    assert confusion_rows
    assert "accuracy_delta_vs_reference" in recipe_metrics
    assert "macro_f1_delta_vs_reference" in recipe_metrics
    assert "flip_rate" in recipe_metrics
    assert "macro_f1_delta_vs_reference" in other_recipe_metrics
    assert "macro_f1_delta_vs_reference" in third_recipe_metrics
    assert prediction_rows[0].reference_prediction is not None
    assert isinstance(prediction_rows[0].prediction_flipped, bool)
    assert prediction_rows[0].to_dict()["perturbation_recipe"]


def test_coverage_reports_and_readiness_isolate_low_coverage_conclusion_probe(tmp_path: Path) -> None:
    baseline_run_dir, config = prepare_baseline_run(tmp_path)

    evaluation_report, _, _ = evaluate_selected_perturbations(
        baseline_run_dir=baseline_run_dir,
        config=config,
    )
    coverage_report = build_perturbation_coverage_report(evaluation_report)
    robustness_report = build_section_aware_robustness_report(
        evaluation_report,
        primary_split="test",
        isolate_low_coverage=True,
    )
    comparative_metrics = build_comparative_robustness_metrics(
        evaluation_report,
        primary_split="test",
    )
    comparative_report = build_comparative_section_aware_robustness_report(
        {
            "primary_split": "test",
            "model_variants": [
                {
                    "model_variant": "tfidf_logistic_regression::pseudo_all_sections",
                    "accuracy": 1.0,
                    "macro_f1": 1.0,
                },
                {
                    "model_variant": "multinomial_naive_bayes::pseudo_all_sections",
                    "accuracy": 1.0,
                    "macro_f1": 1.0,
                },
                {
                    "model_variant": "averaged_passive_aggressive::pseudo_all_sections",
                    "accuracy": 1.0,
                    "macro_f1": 1.0,
                },
                {
                    "model_variant": "section_contextual_logistic_regression::pseudo_all_sections",
                    "accuracy": 1.0,
                    "macro_f1": 1.0,
                },
            ],
            "strongest_by_macro_f1": {
                "model_variant": "averaged_passive_aggressive::pseudo_all_sections",
                "macro_f1": 1.0,
            },
        },
        comparative_metrics,
        isolate_low_coverage=True,
    )
    comparative_next_step = build_comparative_robustness_next_step_summary(
        comparative_report["unperturbed_comparison"],
        comparative_report,
    )
    readiness = build_first_robustness_phase_readiness_summary(
        evaluation_report,
        robustness_report,
    )

    coverage_lookup = {row["recipe_name"]: row for row in coverage_report["recipes"]}
    assert coverage_lookup["drop_precedents"]["coverage_band"] == "high_coverage"
    assert coverage_lookup["drop_conclusion"]["coverage_band"] == "low_coverage"
    assert any(row["recipe_name"] == "drop_precedents" for row in robustness_report["informative_perturbations"])
    assert any(row["recipe_name"] == "drop_conclusion" for row in robustness_report["weak_probe_perturbations"])
    assert comparative_metrics["recipes"]
    assert len(comparative_metrics["recipes"][0]["pairwise_comparisons"]) == 6
    pairwise = comparative_metrics["recipes"][0]["pairwise_comparisons"][0]
    assert "macro_f1_delta_difference_right_minus_left" in pairwise
    assert "macro_f1_retention_difference_right_minus_left" in pairwise
    assert comparative_report["informative_perturbation_comparison"]
    assert comparative_next_step["recommendations"]
    assert "most_stable_model_variant_by_flip_rate" in comparative_next_step
    assert "ready_for_pilot_results_section" in comparative_next_step
    assert "recommended_next_major_step" in comparative_next_step
    assert readiness["recommendations"]
    assert "Perturbed Evaluation Metrics" in render_perturbed_evaluation_metrics(evaluation_report)
    assert "Perturbation Coverage Report" in render_perturbation_coverage_report(coverage_report)
    assert "Section-Aware Robustness Report" in render_section_aware_robustness_report(robustness_report)
    assert "Comparative Robustness Metrics" in render_comparative_robustness_metrics(comparative_metrics)
    assert "Comparative Section-Aware Robustness Report" in render_comparative_section_aware_robustness_report(comparative_report)
    assert "Comparative Robustness Next-Step Summary" in render_comparative_robustness_next_step_summary(comparative_next_step)
    assert "First Robustness Phase Readiness Summary" in render_first_robustness_phase_readiness_summary(readiness)


def test_failure_analysis_builds_disagreement_and_stability_slices(tmp_path: Path) -> None:
    baseline_run_dir, config = prepare_baseline_run(tmp_path)
    evaluation_report, prediction_rows, _ = evaluate_selected_perturbations(
        baseline_run_dir=baseline_run_dir,
        config=config,
    )
    failure_summary, failure_cases = build_failure_analysis(
        prediction_rows,
        pseudo_sectioned_cases=make_cases(),
        primary_split="test",
        focused_recipes=config.robustness.failure_analysis_recipes,
        selected_model_variants=config.robustness.selected_model_variants,
        primary_model_variant="averaged_passive_aggressive::pseudo_all_sections",
        case_limit=config.robustness.failure_analysis_case_limit,
        preview_chars=config.robustness.failure_analysis_preview_chars,
        enable_disagreement_analysis=config.robustness.enable_failure_analysis_disagreement_analysis,
    )

    assert evaluation_report["model_variant_results"]
    assert failure_summary["focused_recipe_summaries"]
    assert failure_cases
    focused = {row["recipe_name"]: row for row in failure_summary["focused_recipe_summaries"]}
    assert "drop_precedents" in focused
    assert "prediction_disagreement_rate" in focused["drop_precedents"]
    assert "per_model_stability_slices" in focused["drop_precedents"]
    assert "pairwise_prediction_disagreement" in focused["drop_precedents"]
    assert "contextual_unique_success_count" in focused["drop_precedents"]
    assert "linear_models_agree_contextual_diff_count" in focused["drop_precedents"]
    assert any("per_model_predictions" in row for row in failure_cases)
    assert "Failure Analysis Summary" in render_failure_analysis_summary(failure_summary)


def test_qualitative_example_selection_renders_paper_facing_output(tmp_path: Path) -> None:
    baseline_run_dir, config = prepare_baseline_run(tmp_path)
    _, prediction_rows, _ = evaluate_selected_perturbations(
        baseline_run_dir=baseline_run_dir,
        config=config,
    )
    failure_summary, failure_cases = build_failure_analysis(
        prediction_rows,
        pseudo_sectioned_cases=make_cases(),
        primary_split="test",
        focused_recipes=config.robustness.failure_analysis_recipes,
        selected_model_variants=config.robustness.selected_model_variants,
        primary_model_variant="averaged_passive_aggressive::pseudo_all_sections",
        case_limit=config.robustness.failure_analysis_case_limit,
        preview_chars=config.robustness.failure_analysis_preview_chars,
        enable_disagreement_analysis=config.robustness.enable_failure_analysis_disagreement_analysis,
    )
    report, examples = build_paper_qualitative_examples(
        failure_cases,
        focused_recipes=config.robustness.failure_analysis_recipes,
        primary_model_variant=config.robustness.primary_model_variant,
        model_variants=config.robustness.selected_model_variants,
        count_per_recipe=config.robustness.qualitative_example_count_per_recipe,
        preview_chars=config.robustness.qualitative_preview_chars,
        include_model_variants=config.robustness.qualitative_include_model_variants,
        examples_per_category=config.robustness.qualitative_examples_per_category,
    )

    assert report["per_recipe_summary"]
    assert examples
    assert "selection_category" in examples[0]
    assert "per_model_predictions" in examples[0]
    assert "Paper-Facing Qualitative Examples" in render_paper_qualitative_examples(report, examples)


def test_apa_focused_tables_stability_summaries_and_case_bundles(tmp_path: Path) -> None:
    baseline_run_dir, config = prepare_baseline_run(tmp_path)
    evaluation_report, prediction_rows, _ = evaluate_selected_perturbations(
        baseline_run_dir=baseline_run_dir,
        config=config,
    )
    comparative_metrics = build_comparative_robustness_metrics(
        evaluation_report,
        primary_split="test",
    )
    unperturbed_comparison = {
        "primary_split": "test",
        "model_variants": [
            {
                "model_variant": "tfidf_logistic_regression::pseudo_all_sections",
                "model_name": "tfidf_logistic_regression",
                "input_variant": "pseudo_all_sections",
                "accuracy": 0.5,
                "macro_f1": 0.5,
            },
            {
                "model_variant": "multinomial_naive_bayes::pseudo_all_sections",
                "model_name": "multinomial_naive_bayes",
                "input_variant": "pseudo_all_sections",
                "accuracy": 0.7,
                "macro_f1": 0.7,
            },
            {
                "model_variant": "averaged_passive_aggressive::pseudo_all_sections",
                "model_name": "averaged_passive_aggressive",
                "input_variant": "pseudo_all_sections",
                "accuracy": 0.8,
                "macro_f1": 0.8,
            },
        ],
        "strongest_by_macro_f1": {
            "model_variant": "averaged_passive_aggressive::pseudo_all_sections",
            "macro_f1": 0.8,
        },
        "strongest_by_macro_f1_by_input_variant": {
            "pseudo_all_sections": {
                "model_variant": "averaged_passive_aggressive::pseudo_all_sections",
                "macro_f1": 0.8,
            }
        },
    }
    failure_summary, failure_cases = build_failure_analysis(
        prediction_rows,
        pseudo_sectioned_cases=make_cases(),
        primary_split="test",
        focused_recipes=config.robustness.failure_analysis_recipes,
        selected_model_variants=config.robustness.selected_model_variants,
        primary_model_variant=config.robustness.primary_model_variant,
        case_limit=config.robustness.failure_analysis_case_limit,
        preview_chars=config.robustness.failure_analysis_preview_chars,
        enable_disagreement_analysis=config.robustness.enable_failure_analysis_disagreement_analysis,
    )
    apa_table = build_apa_focused_robustness_table(
        unperturbed_comparison,
        comparative_metrics,
        primary_model_variant=config.robustness.primary_model_variant,
        comparison_model_variants=config.robustness.selected_model_variants,
        focused_recipes=config.robustness.failure_analysis_recipes,
    )
    stability_summary = build_stability_vs_correctness_summary(
        prediction_rows,
        primary_split="test",
        focused_recipes=config.robustness.failure_analysis_recipes,
        primary_model_variant=config.robustness.primary_model_variant,
        comparison_model_variants=config.robustness.stability_comparison_model_variants,
    )
    qualitative_report, qualitative_examples = build_paper_qualitative_examples(
        failure_cases,
        focused_recipes=config.robustness.failure_analysis_recipes,
        primary_model_variant=config.robustness.primary_model_variant,
        model_variants=config.robustness.selected_model_variants,
        count_per_recipe=config.robustness.qualitative_example_count_per_recipe,
        preview_chars=config.robustness.qualitative_preview_chars,
        include_model_variants=config.robustness.qualitative_include_model_variants,
        examples_per_category=config.robustness.qualitative_examples_per_category,
    )
    bundles = build_case_bundles(
        failure_cases,
        focused_recipes=config.robustness.failure_analysis_recipes,
        primary_model_variant=config.robustness.primary_model_variant,
        model_variants=config.robustness.selected_model_variants,
        bundle_size=config.robustness.case_bundle_size,
        preview_chars=config.robustness.qualitative_preview_chars,
        include_model_variants=config.robustness.qualitative_include_model_variants,
    )
    interpretation = build_focused_perturbation_interpretation(apa_table)
    next_step_summary = {
        "ready_for_pilot_results_section": True,
        "recommendations": ["Use APA as the main baseline."],
        "recommended_next_major_step": "paper_results_packaging",
    }
    pilot_summary = build_pilot_results_section_summary(
        unperturbed_comparison,
        apa_table,
        stability_summary,
        qualitative_report,
        next_step_summary,
    )

    assert apa_table["rows"]
    assert "APA-Focused Robustness Table" in render_apa_focused_robustness_table(apa_table)
    assert stability_summary["recipe_summaries"]
    assert "Stability vs Correctness Summary" in render_stability_vs_correctness_summary(stability_summary)
    assert bundles["drop_precedents"]
    assert bundles["keep_reasoning_only"]
    assert qualitative_examples
    assert interpretation["entries"]
    assert "Focused Perturbation Interpretation" in render_focused_perturbation_interpretation(interpretation)
    assert pilot_summary["focused_findings"]
    assert "Pilot Results Section Summary" in render_pilot_results_section_summary(pilot_summary)


def test_results_package_builders_render_manuscript_artifacts(tmp_path: Path) -> None:
    baseline_run_dir, config = prepare_baseline_run(tmp_path)
    evaluation_report, prediction_rows, _ = evaluate_selected_perturbations(
        baseline_run_dir=baseline_run_dir,
        config=config,
    )
    baseline_report = json.loads(
        (baseline_run_dir / "baseline_prediction_metrics.json").read_text(encoding="utf-8")
    )
    selected_model_variants = expand_unperturbed_model_variants(
        config.robustness.selected_model_variants,
        include_full_text=config.robustness.include_full_text_in_unperturbed_comparison,
    )
    unperturbed_comparison = build_unperturbed_model_comparison(
        baseline_report,
        primary_split="test",
        selected_model_variants=selected_model_variants,
    )
    comparative_metrics = build_comparative_robustness_metrics(
        evaluation_report,
        primary_split="test",
    )
    failure_summary, failure_cases = build_failure_analysis(
        prediction_rows,
        pseudo_sectioned_cases=make_cases(),
        primary_split="test",
        focused_recipes=config.robustness.failure_analysis_recipes,
        selected_model_variants=config.robustness.selected_model_variants,
        primary_model_variant=config.robustness.primary_model_variant,
        contextual_model_variant="section_contextual_logistic_regression::pseudo_all_sections",
        case_limit=config.robustness.failure_analysis_case_limit,
        preview_chars=config.robustness.failure_analysis_preview_chars,
        enable_disagreement_analysis=config.robustness.enable_failure_analysis_disagreement_analysis,
    )
    comparative_report = build_comparative_section_aware_robustness_report(
        unperturbed_comparison,
        comparative_metrics,
        isolate_low_coverage=config.robustness.isolate_low_coverage_recipes,
        failure_analysis_summary=failure_summary,
    )
    comparative_next_step_summary = build_comparative_robustness_next_step_summary(
        unperturbed_comparison,
        comparative_report,
        failure_analysis_summary=failure_summary,
    )
    apa_table = build_apa_focused_robustness_table(
        unperturbed_comparison,
        comparative_metrics,
        primary_model_variant=config.robustness.primary_model_variant,
        comparison_model_variants=config.robustness.selected_model_variants,
        focused_recipes=config.robustness.failure_analysis_recipes,
    )
    stability_summary = build_stability_vs_correctness_summary(
        prediction_rows,
        primary_split="test",
        focused_recipes=config.robustness.failure_analysis_recipes,
        primary_model_variant=config.robustness.primary_model_variant,
        comparison_model_variants=config.robustness.stability_comparison_model_variants,
    )
    qualitative_report, qualitative_examples = build_paper_qualitative_examples(
        failure_cases,
        focused_recipes=config.robustness.failure_analysis_recipes,
        primary_model_variant=config.robustness.primary_model_variant,
        model_variants=config.robustness.selected_model_variants,
        count_per_recipe=config.robustness.qualitative_example_count_per_recipe,
        preview_chars=config.robustness.qualitative_preview_chars,
        include_model_variants=config.robustness.qualitative_include_model_variants,
        examples_per_category=config.robustness.qualitative_examples_per_category,
    )
    case_bundles = build_case_bundles(
        failure_cases,
        focused_recipes=config.robustness.failure_analysis_recipes,
        primary_model_variant=config.robustness.primary_model_variant,
        model_variants=config.robustness.selected_model_variants,
        bundle_size=config.robustness.case_bundle_size,
        preview_chars=config.robustness.qualitative_preview_chars,
        include_model_variants=config.robustness.qualitative_include_model_variants,
    )
    pilot_summary = build_pilot_results_section_summary(
        unperturbed_comparison,
        apa_table,
        stability_summary,
        qualitative_report,
        comparative_next_step_summary,
    )

    primary_recipe = comparative_next_step_summary.get("primary_writeup_perturbation") or "keep_reasoning_only"
    secondary_recipe = comparative_next_step_summary.get("secondary_writeup_perturbation") or "drop_precedents"
    table_main = build_table_main_results(
        apa_table,
        primary_recipe=primary_recipe,
        secondary_recipe=secondary_recipe,
    )
    table_comparison = build_table_model_comparison(
        unperturbed_comparison,
        comparative_metrics,
        primary_model_variant=config.robustness.primary_model_variant,
        focused_recipes=(primary_recipe, secondary_recipe),
    )
    chart_main = build_chart_data_main_performance(unperturbed_comparison)
    chart_deltas = build_chart_data_robustness_deltas(
        comparative_metrics,
        focused_recipes=(primary_recipe, secondary_recipe),
    )
    chart_flips = build_chart_data_flip_rates(
        comparative_metrics,
        stability_summary,
        focused_recipes=(primary_recipe, secondary_recipe),
    )
    chart_coverage = build_chart_data_coverage(
        comparative_metrics,
        focused_recipes=(primary_recipe, secondary_recipe),
    )
    table_stability = build_table_stability_vs_correctness(stability_summary)
    stability_narrative = build_stability_vs_correctness_narrative(stability_summary)
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
    packaging_next_step = build_paper_results_packaging_next_step_summary(
        comparative_next_step_summary,
    )
    narratives = build_results_narratives(
        table_main,
        table_comparison,
        packaging_next_step,
    )
    appendix_keep = build_appendix_bundle(
        recipe_name="keep_reasoning_only",
        recipe_summary_row=next(
            (row for row in table_main["rows"] if row["condition_key"] == "keep_reasoning_only"),
            None,
        ),
        model_comparison=table_comparison,
        qualitative_bundle=primary_qualitative_bundle if primary_recipe == "keep_reasoning_only" else secondary_qualitative_bundle,
        case_bundle_filename="keep_reasoning_only_case_bundle.jsonl",
    )
    manifest = build_results_package_manifest(
        baseline_run_dir=baseline_run_dir,
        robustness_run_dir=tmp_path / "robustness_run",
        package_dirname=config.robustness.results_package_dirname,
        primary_model_variant=config.robustness.primary_model_variant,
        models_included=list(config.robustness.selected_model_variants),
        perturbations_included=[primary_recipe, secondary_recipe],
        chart_data_files=[
            "chart_data_main_performance.json",
            "chart_data_robustness_deltas.json",
        ],
        main_summary_files=["table_main_results.md", "results_narrative_main.md"],
        qualitative_files=["qualitative_examples_primary.md"],
        appendix_files=["appendix_keep_reasoning_only_bundle.md"],
        caveats=packaging_next_step["visible_caveats"],
    )

    assert pilot_summary["focused_findings"]
    assert table_main["rows"]
    assert table_comparison["rows"]
    assert chart_main["rows"]
    assert chart_deltas["rows"]
    assert chart_flips["flip_rate_rows"]
    assert chart_flips["stability_slice_rows"]
    assert chart_coverage["rows"]
    assert table_stability["rows"]
    assert primary_qualitative_bundle["examples"]
    assert secondary_qualitative_bundle["examples"]
    assert case_bundles["drop_precedents"]
    assert "Table Main Results" in render_table_main_results(table_main)
    assert "Table Model Comparison" in render_table_model_comparison(table_comparison)
    assert "Table Stability vs Correctness" in render_table_stability_vs_correctness(table_stability)
    assert "Stability vs Correctness Narrative" in render_stability_vs_correctness_narrative(stability_narrative)
    assert "Qualitative Examples" in render_recipe_qualitative_bundle(primary_qualitative_bundle)
    assert "Results Narrative Main" in render_results_narrative_main(narratives)
    assert "Results Narrative Supporting" in render_results_narrative_supporting(narratives)
    assert "Appendix Bundle" in render_appendix_bundle(appendix_keep)
    assert "Results Package Manifest" in render_results_package_manifest(manifest)
    assert "Paper Results Packaging Next-Step Summary" in render_paper_results_packaging_next_step_summary(packaging_next_step)


def test_paper_freeze_consistency_and_writing_support_package(tmp_path: Path) -> None:
    baseline_run_dir, config = prepare_baseline_run(tmp_path)
    section_transfer_run_dir = tmp_path / "section_transfer_run"
    robustness_run_dir = tmp_path / "robustness_run"
    results_package_dir = robustness_run_dir / "results_package"
    output_dir = tmp_path / "paper_drafting_package"
    section_transfer_run_dir.mkdir(parents=True, exist_ok=True)
    results_package_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    evaluation_report, prediction_rows, _ = evaluate_selected_perturbations(
        baseline_run_dir=baseline_run_dir,
        config=config,
    )
    baseline_report = json.loads(
        (baseline_run_dir / "baseline_prediction_metrics.json").read_text(encoding="utf-8")
    )
    selected_model_variants = expand_unperturbed_model_variants(
        config.robustness.selected_model_variants,
        include_full_text=config.robustness.include_full_text_in_unperturbed_comparison,
    )
    unperturbed_comparison = build_unperturbed_model_comparison(
        baseline_report,
        primary_split="test",
        selected_model_variants=selected_model_variants,
    )
    comparative_metrics = build_comparative_robustness_metrics(
        evaluation_report,
        primary_split="test",
    )
    failure_summary, failure_cases = build_failure_analysis(
        prediction_rows,
        pseudo_sectioned_cases=make_cases(),
        primary_split="test",
        focused_recipes=config.robustness.failure_analysis_recipes,
        selected_model_variants=config.robustness.selected_model_variants,
        primary_model_variant=config.robustness.primary_model_variant,
        contextual_model_variant="section_contextual_logistic_regression::pseudo_all_sections",
        case_limit=config.robustness.failure_analysis_case_limit,
        preview_chars=config.robustness.failure_analysis_preview_chars,
        enable_disagreement_analysis=config.robustness.enable_failure_analysis_disagreement_analysis,
    )
    comparative_report = build_comparative_section_aware_robustness_report(
        unperturbed_comparison,
        comparative_metrics,
        isolate_low_coverage=config.robustness.isolate_low_coverage_recipes,
        failure_analysis_summary=failure_summary,
    )
    comparative_next_step_summary = build_comparative_robustness_next_step_summary(
        unperturbed_comparison,
        comparative_report,
        failure_analysis_summary=failure_summary,
    )
    apa_table = build_apa_focused_robustness_table(
        unperturbed_comparison,
        comparative_metrics,
        primary_model_variant=config.robustness.primary_model_variant,
        comparison_model_variants=config.robustness.selected_model_variants,
        focused_recipes=config.robustness.failure_analysis_recipes,
    )
    stability_summary = build_stability_vs_correctness_summary(
        prediction_rows,
        primary_split="test",
        focused_recipes=config.robustness.failure_analysis_recipes,
        primary_model_variant=config.robustness.primary_model_variant,
        comparison_model_variants=config.robustness.stability_comparison_model_variants,
    )
    qualitative_report, qualitative_examples = build_paper_qualitative_examples(
        failure_cases,
        focused_recipes=config.robustness.failure_analysis_recipes,
        primary_model_variant=config.robustness.primary_model_variant,
        model_variants=config.robustness.selected_model_variants,
        count_per_recipe=config.robustness.qualitative_example_count_per_recipe,
        preview_chars=config.robustness.qualitative_preview_chars,
        include_model_variants=config.robustness.qualitative_include_model_variants,
        examples_per_category=config.robustness.qualitative_examples_per_category,
    )
    case_bundles = build_case_bundles(
        failure_cases,
        focused_recipes=config.robustness.failure_analysis_recipes,
        primary_model_variant=config.robustness.primary_model_variant,
        model_variants=config.robustness.selected_model_variants,
        bundle_size=config.robustness.case_bundle_size,
        preview_chars=config.robustness.qualitative_preview_chars,
        include_model_variants=config.robustness.qualitative_include_model_variants,
    )
    primary_recipe = comparative_next_step_summary["primary_writeup_perturbation"]
    secondary_recipe = comparative_next_step_summary["secondary_writeup_perturbation"]
    table_main = build_table_main_results(
        apa_table,
        primary_recipe=primary_recipe,
        secondary_recipe=secondary_recipe,
    )
    table_comparison = build_table_model_comparison(
        unperturbed_comparison,
        comparative_metrics,
        primary_model_variant=config.robustness.primary_model_variant,
        focused_recipes=(primary_recipe, secondary_recipe),
    )
    table_stability = build_table_stability_vs_correctness(stability_summary)
    stability_narrative = build_stability_vs_correctness_narrative(stability_summary)
    primary_bundle = build_recipe_qualitative_bundle(
        qualitative_examples,
        recipe_name=primary_recipe,
        count=config.robustness.primary_qualitative_example_count,
        preview_chars=config.robustness.narrative_preview_chars,
    )
    secondary_bundle = build_recipe_qualitative_bundle(
        qualitative_examples,
        recipe_name=secondary_recipe,
        count=config.robustness.secondary_qualitative_example_count,
        preview_chars=config.robustness.narrative_preview_chars,
    )
    packaging_next_step = build_paper_results_packaging_next_step_summary(
        comparative_next_step_summary,
    )
    results_manifest = build_results_package_manifest(
        baseline_run_dir=baseline_run_dir,
        robustness_run_dir=robustness_run_dir,
        package_dirname="results_package",
        primary_model_variant=config.robustness.primary_model_variant,
        models_included=list(config.robustness.selected_model_variants),
        perturbations_included=[primary_recipe, secondary_recipe],
        chart_data_files=["chart_data_main_performance.json"],
        main_summary_files=[
            "table_main_results.md",
            "table_model_comparison.md",
            "table_stability_vs_correctness.md",
        ],
        qualitative_files=[
            "qualitative_examples_primary.md",
            "qualitative_examples_secondary.md",
            "paper_qualitative_examples.md",
        ],
        appendix_files=[
            "appendix_keep_reasoning_only_bundle.md",
            "appendix_drop_precedents_bundle.md",
        ],
        caveats=packaging_next_step["visible_caveats"],
    )

    write_json(results_package_dir / "table_main_results.json", table_main)
    write_text(results_package_dir / "table_main_results.md", render_table_main_results(table_main))
    write_json(results_package_dir / "table_model_comparison.json", table_comparison)
    write_text(results_package_dir / "table_model_comparison.md", render_table_model_comparison(table_comparison))
    write_json(results_package_dir / "table_stability_vs_correctness.json", table_stability)
    write_text(results_package_dir / "table_stability_vs_correctness.md", render_table_stability_vs_correctness(table_stability))
    write_text(results_package_dir / "qualitative_examples_primary.md", render_recipe_qualitative_bundle(primary_bundle))
    write_text(results_package_dir / "qualitative_examples_secondary.md", render_recipe_qualitative_bundle(secondary_bundle))
    appendix_keep = build_appendix_bundle(
        recipe_name="keep_reasoning_only",
        recipe_summary_row=next((row for row in table_main["rows"] if row["condition_key"] == "keep_reasoning_only"), None),
        model_comparison=table_comparison,
        qualitative_bundle=primary_bundle if primary_recipe == "keep_reasoning_only" else secondary_bundle,
        case_bundle_filename="keep_reasoning_only_case_bundle.jsonl",
    )
    appendix_drop = build_appendix_bundle(
        recipe_name="drop_precedents",
        recipe_summary_row=next((row for row in table_main["rows"] if row["condition_key"] == "drop_precedents"), None),
        model_comparison=table_comparison,
        qualitative_bundle=primary_bundle if primary_recipe == "drop_precedents" else secondary_bundle,
        case_bundle_filename="drop_precedents_case_bundle.jsonl",
    )
    write_text(results_package_dir / "appendix_keep_reasoning_only_bundle.md", render_appendix_bundle(appendix_keep))
    write_text(results_package_dir / "appendix_drop_precedents_bundle.md", render_appendix_bundle(appendix_drop))
    write_text(results_package_dir / "paper_qualitative_examples.md", render_paper_qualitative_examples(qualitative_report, qualitative_examples))
    write_jsonl(results_package_dir / "paper_qualitative_examples.jsonl", qualitative_examples)
    write_json(results_package_dir / "chart_data_main_performance.json", {"series": []})
    write_json(results_package_dir / "results_package_manifest.json", results_manifest)
    write_text(results_package_dir / "results_package_manifest.md", render_results_package_manifest(results_manifest))

    write_json(robustness_run_dir / "comparative_robustness_metrics.json", comparative_metrics)
    write_json(robustness_run_dir / "unperturbed_model_comparison.json", unperturbed_comparison)
    write_json(robustness_run_dir / "stability_vs_correctness_summary.json", stability_summary)
    write_jsonl(robustness_run_dir / "failure_analysis_cases.jsonl", failure_cases)

    introduction_path = output_dir / "draft_support_introduction.md"
    method_path = output_dir / "draft_support_method.md"
    results_path = output_dir / "draft_support_results.md"
    limitations_path = output_dir / "draft_support_limitations.md"
    appendix_path = output_dir / "draft_support_appendix.md"
    reproducibility_path = output_dir / "paper_reproducibility_commands.md"
    table_selection_path = output_dir / "paper_table_selection.md"
    figure_selection_path = output_dir / "paper_figure_selection.md"

    write_text(
        introduction_path,
        render_draft_support_introduction(
            primary_model_variant=config.robustness.primary_model_variant,
            primary_probe=primary_recipe,
            secondary_probe=secondary_recipe,
        ),
    )
    write_text(
        method_path,
        render_draft_support_method(
            section_transfer_run_dir=section_transfer_run_dir,
            baseline_run_dir=baseline_run_dir,
            robustness_run_dir=robustness_run_dir,
            primary_model_variant=config.robustness.primary_model_variant,
            primary_probe=primary_recipe,
            secondary_probe=secondary_recipe,
            supporting_model_variants=tuple(config.robustness.selected_model_variants[1:]),
        ),
    )
    write_text(
        results_path,
        render_draft_support_results(
            table_main_results=table_main,
            table_model_comparison=table_comparison,
            stability_narrative=stability_narrative,
            primary_bundle=primary_bundle,
            secondary_bundle=secondary_bundle,
        ),
    )
    write_text(limitations_path, render_draft_support_limitations())
    write_text(
        reproducibility_path,
        render_paper_reproducibility_commands(
            section_transfer_run_dir=section_transfer_run_dir,
            baseline_run_dir=baseline_run_dir,
            robustness_run_dir=robustness_run_dir,
            results_package_dir=results_package_dir,
            drafting_package_dir=output_dir,
        ),
    )
    write_text(
        appendix_path,
        render_draft_support_appendix(
            freeze_manifest={
                "canonical_section_transfer_run_dir": str(section_transfer_run_dir),
                "canonical_baseline_run_dir": str(baseline_run_dir),
                "canonical_robustness_run_dir": str(robustness_run_dir),
            },
            reproducibility_commands_path=reproducibility_path,
            primary_bundle_path=results_package_dir / "qualitative_examples_primary.md",
            secondary_bundle_path=results_package_dir / "qualitative_examples_secondary.md",
        ),
    )
    write_text(
        table_selection_path,
        render_paper_table_selection(
            primary_bundle_path=results_package_dir / "qualitative_examples_primary.md",
            secondary_bundle_path=results_package_dir / "qualitative_examples_secondary.md",
        ),
    )
    write_text(figure_selection_path, render_paper_figure_selection())

    freeze_manifest = build_paper_freeze_manifest(
        section_transfer_run_dir=section_transfer_run_dir,
        baseline_run_dir=baseline_run_dir,
        robustness_run_dir=robustness_run_dir,
        results_package_dir=results_package_dir,
        drafting_package_dir=output_dir,
        primary_model_variant=config.robustness.primary_model_variant,
        primary_probe=primary_recipe,
        secondary_probe=secondary_recipe,
        supporting_model_variants=tuple(config.robustness.selected_model_variants[1:]),
        manuscript_artifacts={
            "introduction": [introduction_path],
            "method": [method_path, reproducibility_path],
            "results": [results_package_dir / "table_main_results.md", results_path],
            "limitations": [limitations_path],
            "appendix": [appendix_path, results_package_dir / "appendix_keep_reasoning_only_bundle.md"],
        },
    )
    consistency = build_paper_consistency_check(
        freeze_manifest=freeze_manifest,
        results_package_manifest=results_manifest,
        unperturbed_comparison=unperturbed_comparison,
        comparative_metrics=comparative_metrics,
        table_main_results=table_main,
        table_model_comparison=table_comparison,
        qualitative_examples=qualitative_examples,
        failure_analysis_cases=failure_cases,
        appendix_bundle_paths={
            "keep_reasoning_only": results_package_dir / "appendix_keep_reasoning_only_bundle.md",
            "drop_precedents": results_package_dir / "appendix_drop_precedents_bundle.md",
        },
    )
    strengthening = build_targeted_strengthening_check(
        comparative_metrics=comparative_metrics,
        primary_model_variant=config.robustness.primary_model_variant,
        target_recipe=secondary_recipe,
    )
    readiness = build_paper_readiness_summary(
        freeze_manifest=freeze_manifest,
        consistency_check=consistency,
        packaging_next_step_summary=packaging_next_step,
        targeted_strengthening_check=strengthening,
    )

    assert consistency["overall_status"] == "pass"
    assert "Paper Freeze Manifest" in render_paper_freeze_manifest(freeze_manifest)
    assert "Paper Consistency Check" in render_paper_consistency_check(consistency)
    assert "Draft Support Introduction" in introduction_path.read_text(encoding="utf-8")
    assert "Draft Support Method" in method_path.read_text(encoding="utf-8")
    assert "Draft Support Results" in results_path.read_text(encoding="utf-8")
    assert "Draft Support Limitations" in limitations_path.read_text(encoding="utf-8")
    assert "Draft Support Appendix" in appendix_path.read_text(encoding="utf-8")
    assert "Paper Table Selection" in table_selection_path.read_text(encoding="utf-8")
    assert "Paper Figure Selection" in figure_selection_path.read_text(encoding="utf-8")
    assert "Paper Reproducibility Commands" in reproducibility_path.read_text(encoding="utf-8")
    assert "Targeted Strengthening Check" in render_targeted_strengthening_check(strengthening)
    assert readiness["ready_for_pilot_paper_drafting"] is True
    assert "Paper Readiness Summary" in render_paper_readiness_summary(readiness)


def test_section_importance_package_builds_rankings_and_manifest(tmp_path: Path) -> None:
    baseline_run_dir, config = prepare_baseline_run(tmp_path)
    primary_report, _ = evaluate_section_importance_model(
        baseline_run_dir=baseline_run_dir,
        pseudo_sectioned_cases=make_cases(),
        model_variant=config.robustness.section_importance_primary_model_variant,
        config=config,
    )
    scores = build_section_importance_scores(
        primary_report,
        config=config,
        primary_split="test",
    )
    supporting_reports = []
    for model_variant in config.robustness.section_importance_supporting_model_variants:
        report, _ = evaluate_section_importance_model(
            baseline_run_dir=baseline_run_dir,
            pseudo_sectioned_cases=make_cases(),
            model_variant=model_variant,
            config=config,
            spec_names=config.robustness.section_importance_cross_model_recipes,
        )
        supporting_reports.append(report)
    cross_model = build_section_importance_cross_model_check(
        primary_model_report=primary_report,
        supporting_model_reports=supporting_reports,
        primary_split="test",
    )
    next_step = build_section_importance_next_step_summary(scores, cross_model)

    chart_scores = build_chart_data_section_importance_scores(scores)
    chart_ranking = build_chart_data_section_importance_ranking(scores)
    chart_coverage = build_chart_data_section_importance_coverage(scores)

    source_results_package_dir = tmp_path / "results_package"
    source_results_package_dir.mkdir(parents=True, exist_ok=True)
    for filename in (
        "table_main_results.md",
        "table_model_comparison.md",
        "chart_data_main_performance.json",
        "qualitative_examples_primary.md",
        "appendix_keep_reasoning_only_bundle.md",
    ):
        write_text(source_results_package_dir / filename, "placeholder")
    updated_manifest = build_updated_results_package_manifest(
        existing_manifest={
            "baseline_run_dir": str(baseline_run_dir),
            "robustness_run_dir": str(tmp_path / "robustness"),
            "primary_model_variant": config.robustness.primary_model_variant,
            "models_included": list(config.robustness.selected_model_variants),
            "perturbations_included": ["keep_reasoning_only", "drop_precedents"],
            "main_summary_files": ["table_main_results.md", "table_model_comparison.md"],
            "chart_data_files": ["chart_data_main_performance.json"],
            "qualitative_files": ["qualitative_examples_primary.md"],
            "appendix_files": ["appendix_keep_reasoning_only_bundle.md"],
            "known_caveats": ["Pseudo-sections are predicted, not gold."],
        },
        source_results_package_dir=source_results_package_dir,
        section_importance_output_dir=tmp_path / "section_importance",
        section_importance_files={
            "main_summary_files": ["section_importance_scores.md"],
            "chart_data_files": ["chart_data_section_importance_scores.json"],
            "qualitative_files": [],
            "appendix_files": ["section_importance_cross_model_check.md"],
        },
        additional_caveats=next_step["visible_caveats"],
    )

    assert len(scores["section_rows"]) == 5
    assert len(chart_scores["rows"]) == 5
    assert len(chart_ranking["rows"]) == 5
    assert len(chart_coverage["rows"]) == 5
    assert any(
        row["section"] == "conclusion"
        and row["confidence_label"] == "low_confidence_importance_estimate"
        for row in scores["section_rows"]
    )
    assert cross_model["rows"]
    assert next_step["ready_to_state_section_ranking_in_paper"] is True
    assert any(
        "section_importance_scores.md" in path
        for path in updated_manifest["main_summary_files"]
    )
    assert "Section Importance Scores" in render_section_importance_scores(scores)
    assert "Section Importance Ranking" in render_section_importance_ranking(scores)
    assert "Section Importance Cross-Model Check" in render_section_importance_cross_model_check(cross_model)
    assert "Section Importance Narrative Main" in render_section_importance_narrative_main(scores)
    assert "Section Importance Narrative Supporting" in render_section_importance_narrative_supporting(scores, cross_model)
    assert "Section Importance Next-Step Summary" in render_section_importance_next_step_summary(next_step)
    assert "Results Package Manifest" in render_results_package_manifest(updated_manifest)


def test_submission_package_builders_render_figures_traceability_and_handoff(tmp_path: Path) -> None:
    results_package_dir = tmp_path / "results_package"
    paper_drafting_dir = tmp_path / "paper_drafting_package"
    section_importance_dir = tmp_path / "section_importance"
    submission_dir = tmp_path / "submission_package"
    for path in (results_package_dir, paper_drafting_dir, section_importance_dir, submission_dir):
        path.mkdir(parents=True, exist_ok=True)

    chart_main = {
        "rows": [
            {"model_variant": "averaged_passive_aggressive::pseudo_all_sections", "model_label": "APA", "macro_f1": 0.56, "accuracy": 0.57},
            {"model_variant": "multinomial_naive_bayes::pseudo_all_sections", "model_label": "NB", "macro_f1": 0.54, "accuracy": 0.55},
            {"model_variant": "tfidf_logistic_regression::pseudo_all_sections", "model_label": "Logistic", "macro_f1": 0.39, "accuracy": 0.51},
            {"model_variant": "section_contextual_logistic_regression::pseudo_all_sections", "model_label": "Contextual approx.", "macro_f1": 0.38, "accuracy": 0.50},
        ]
    }
    chart_deltas = {
        "rows": [
            {"recipe_name": "keep_reasoning_only", "model_variant": "averaged_passive_aggressive::pseudo_all_sections", "model_label": "APA", "macro_f1_delta": -0.05},
            {"recipe_name": "keep_reasoning_only", "model_variant": "multinomial_naive_bayes::pseudo_all_sections", "model_label": "NB", "macro_f1_delta": -0.03},
            {"recipe_name": "keep_reasoning_only", "model_variant": "tfidf_logistic_regression::pseudo_all_sections", "model_label": "Logistic", "macro_f1_delta": -0.04},
            {"recipe_name": "keep_reasoning_only", "model_variant": "section_contextual_logistic_regression::pseudo_all_sections", "model_label": "Contextual approx.", "macro_f1_delta": -0.04},
            {"recipe_name": "drop_precedents", "model_variant": "averaged_passive_aggressive::pseudo_all_sections", "model_label": "APA", "macro_f1_delta": -0.02},
            {"recipe_name": "drop_precedents", "model_variant": "multinomial_naive_bayes::pseudo_all_sections", "model_label": "NB", "macro_f1_delta": 0.01},
            {"recipe_name": "drop_precedents", "model_variant": "tfidf_logistic_regression::pseudo_all_sections", "model_label": "Logistic", "macro_f1_delta": -0.01},
            {"recipe_name": "drop_precedents", "model_variant": "section_contextual_logistic_regression::pseudo_all_sections", "model_label": "Contextual approx.", "macro_f1_delta": -0.02},
        ]
    }
    chart_flips = {
        "flip_rate_rows": [
            {"recipe_name": "keep_reasoning_only", "model_label": "APA", "flip_rate": 0.33},
            {"recipe_name": "keep_reasoning_only", "model_label": "NB", "flip_rate": 0.22},
            {"recipe_name": "keep_reasoning_only", "model_label": "Logistic", "flip_rate": 0.05},
            {"recipe_name": "keep_reasoning_only", "model_label": "Contextual approx.", "flip_rate": 0.03},
            {"recipe_name": "drop_precedents", "model_label": "APA", "flip_rate": 0.14},
            {"recipe_name": "drop_precedents", "model_label": "NB", "flip_rate": 0.10},
            {"recipe_name": "drop_precedents", "model_label": "Logistic", "flip_rate": 0.02},
            {"recipe_name": "drop_precedents", "model_label": "Contextual approx.", "flip_rate": 0.03},
        ]
    }
    chart_scores = {
        "rows": [
            {"section": "precedents", "rank": 1, "composite_importance_score": 0.93, "confidence_label": "high_confidence_importance_estimate"},
            {"section": "facts", "rank": 2, "composite_importance_score": 0.53, "confidence_label": "high_confidence_importance_estimate"},
            {"section": "reasoning", "rank": 3, "composite_importance_score": 0.41, "confidence_label": "high_confidence_importance_estimate"},
            {"section": "other", "rank": 4, "composite_importance_score": 0.34, "confidence_label": "high_confidence_importance_estimate"},
            {"section": "conclusion", "rank": 5, "composite_importance_score": 0.20, "confidence_label": "low_confidence_importance_estimate"},
        ]
    }
    chart_coverage = {
        "rows": [
            {"section": "precedents", "combined_effective_coverage": 0.85, "confidence_label": "high_confidence_importance_estimate"},
            {"section": "facts", "combined_effective_coverage": 0.92, "confidence_label": "high_confidence_importance_estimate"},
            {"section": "reasoning", "combined_effective_coverage": 0.91, "confidence_label": "high_confidence_importance_estimate"},
            {"section": "other", "combined_effective_coverage": 0.71, "confidence_label": "high_confidence_importance_estimate"},
            {"section": "conclusion", "combined_effective_coverage": 0.03, "confidence_label": "low_confidence_importance_estimate"},
        ]
    }
    write_json(results_package_dir / "chart_data_main_performance.json", chart_main)
    write_json(results_package_dir / "chart_data_robustness_deltas.json", chart_deltas)
    write_json(results_package_dir / "chart_data_flip_rates.json", chart_flips)
    write_json(section_importance_dir / "chart_data_section_importance_scores.json", chart_scores)
    write_json(section_importance_dir / "chart_data_section_importance_coverage.json", chart_coverage)

    figure_manifest = render_submission_figures(
        output_dir=submission_dir,
        figure_formats=("png", "svg"),
        chart_data_main_performance=chart_main,
        chart_data_robustness_deltas=chart_deltas,
        chart_data_flip_rates=chart_flips,
        chart_data_section_importance_scores=chart_scores,
        chart_data_section_importance_coverage=chart_coverage,
        source_file_map={
            "chart_data_main_performance": results_package_dir / "chart_data_main_performance.json",
            "chart_data_robustness_deltas": results_package_dir / "chart_data_robustness_deltas.json",
            "chart_data_flip_rates": results_package_dir / "chart_data_flip_rates.json",
            "chart_data_section_importance_scores": section_importance_dir / "chart_data_section_importance_scores.json",
            "chart_data_section_importance_coverage": section_importance_dir / "chart_data_section_importance_coverage.json",
        },
    )
    write_json(submission_dir / "figure_manifest.json", figure_manifest)
    write_text(submission_dir / "figure_manifest.md", render_figure_manifest(figure_manifest))

    table_main_results = {
        "rows": [
            {"condition_key": "unperturbed", "accuracy": 0.57, "macro_f1": 0.56, "delta_accuracy": 0.0, "delta_macro_f1": 0.0, "flip_rate": 0.0, "effective_coverage": 1.0},
            {"condition_key": "keep_reasoning_only", "accuracy": 0.54, "macro_f1": 0.51, "delta_accuracy": -0.03, "delta_macro_f1": -0.05, "flip_rate": 0.33, "effective_coverage": 1.0},
            {"condition_key": "drop_precedents", "accuracy": 0.56, "macro_f1": 0.54, "delta_accuracy": -0.01, "delta_macro_f1": -0.02, "flip_rate": 0.14, "effective_coverage": 0.85},
        ]
    }
    table_model_comparison = {
        "rows": [
            {"model_label": "APA", "unperturbed_accuracy": 0.57, "unperturbed_macro_f1": 0.56, "keep_reasoning_only_flip_rate": 0.33, "drop_precedents_delta_macro_f1": -0.02},
            {"model_label": "NB", "unperturbed_accuracy": 0.55, "unperturbed_macro_f1": 0.54, "keep_reasoning_only_flip_rate": 0.22, "drop_precedents_delta_macro_f1": 0.01},
            {"model_label": "Logistic", "unperturbed_accuracy": 0.51, "unperturbed_macro_f1": 0.39, "keep_reasoning_only_flip_rate": 0.05, "drop_precedents_delta_macro_f1": -0.01},
        ]
    }
    stability_table = {"rows": []}
    section_importance_scores = {
        "section_rows": [
            {"section": "precedents", "rank": 1, "combined_effective_coverage": 0.85, "confidence_label": "high_confidence_importance_estimate"},
            {"section": "facts", "rank": 2, "combined_effective_coverage": 0.92, "confidence_label": "high_confidence_importance_estimate"},
            {"section": "reasoning", "rank": 3, "combined_effective_coverage": 0.91, "confidence_label": "high_confidence_importance_estimate"},
            {"section": "other", "rank": 4, "combined_effective_coverage": 0.71, "confidence_label": "high_confidence_importance_estimate"},
            {"section": "conclusion", "rank": 5, "combined_effective_coverage": 0.03, "confidence_label": "low_confidence_importance_estimate"},
        ]
    }
    section_importance_cross_model_check = {
        "alignment_counts": {
            "reasoning_gt_precedents_by_removal_impact": 1,
            "precedents_gt_facts_by_solo_sufficiency": 3,
        }
    }
    section_importance_next_step_summary = {
        "section_ranking": [
            {"section": "precedents"},
            {"section": "facts"},
            {"section": "reasoning"},
            {"section": "other"},
            {"section": "conclusion"},
        ],
        "next_action": "Begin manuscript drafting from the frozen submission package.",
    }

    for filename in (
        "table_main_results.md",
        "table_model_comparison.md",
        "table_stability_vs_correctness.md",
        "qualitative_examples_primary.md",
        "qualitative_examples_secondary.md",
        "paper_qualitative_examples.md",
        "appendix_keep_reasoning_only_bundle.md",
        "appendix_drop_precedents_bundle.md",
    ):
        write_text(results_package_dir / filename, filename)
    for filename in (
        "section_importance_scores.md",
        "section_importance_ranking.md",
        "section_importance_cross_model_check.md",
        "section_importance_next_step_summary.md",
    ):
        write_text(section_importance_dir / filename, filename)

    abstract = render_paper_abstract_support(
        build_paper_abstract_support(
            table_main_results=table_main_results,
            table_model_comparison=table_model_comparison,
            section_importance_scores=section_importance_scores,
        )
    )
    intro = render_paper_intro_support(
        build_paper_intro_support(
            primary_model_variant="averaged_passive_aggressive::pseudo_all_sections",
            primary_probe="keep_reasoning_only",
            secondary_probe="drop_precedents",
        )
    )
    method = render_paper_method_support(
        build_paper_method_support(
            freeze_manifest={
                "canonical_section_transfer_run_dir": str(tmp_path / "section_transfer"),
                "canonical_baseline_run_dir": str(tmp_path / "baseline"),
                "canonical_robustness_run_dir": str(tmp_path / "robustness"),
            },
            primary_model_variant="averaged_passive_aggressive::pseudo_all_sections",
            primary_probe="keep_reasoning_only",
            secondary_probe="drop_precedents",
            section_importance_summary={"composite_formula": "demo formula"},
        )
    )
    results = render_paper_results_support(
        build_paper_results_support(
            table_main_results=table_main_results,
            table_model_comparison=table_model_comparison,
            section_importance_scores=section_importance_scores,
            section_importance_cross_model_check=section_importance_cross_model_check,
        )
    )
    limitations = render_paper_limitations_ethics_support(build_paper_limitations_ethics_support())
    conclusion = render_paper_conclusion_support(
        build_paper_conclusion_support(
            section_importance_next_step_summary=section_importance_next_step_summary,
        )
    )
    writing_paths = [
        submission_dir / "paper_abstract_support.md",
        submission_dir / "paper_intro_support.md",
        submission_dir / "paper_method_support.md",
        submission_dir / "paper_results_support.md",
        submission_dir / "paper_limitations_ethics_support.md",
        submission_dir / "paper_conclusion_support.md",
    ]
    for path, content in zip(writing_paths, (abstract, intro, method, results, limitations, conclusion), strict=True):
        write_text(path, content)

    traceability = build_claim_to_evidence_traceability(
        results_package_dir=results_package_dir,
        section_importance_dir=section_importance_dir,
        submission_package_dir=submission_dir,
        table_main_results=table_main_results,
        table_model_comparison=table_model_comparison,
        stability_table=stability_table,
        section_importance_scores=section_importance_scores,
        section_importance_cross_model_check=section_importance_cross_model_check,
        figure_manifest=figure_manifest,
    )
    traceability_path = submission_dir / "claim_to_evidence_traceability.md"
    write_text(traceability_path, render_claim_to_evidence_traceability(traceability))

    main_layout = render_main_text_layout_guide(build_main_text_layout_guide())
    appendix_layout = render_appendix_layout_guide(build_appendix_layout_guide())
    main_layout_path = submission_dir / "main_text_layout_guide.md"
    appendix_layout_path = submission_dir / "appendix_layout_guide.md"
    write_text(main_layout_path, main_layout)
    write_text(appendix_layout_path, appendix_layout)

    reproducibility_path = submission_dir / "paper_reproducibility_commands.md"
    write_text(reproducibility_path, "repro commands")

    figure_captions = build_figure_captions(figure_manifest)
    table_captions = build_table_captions()
    figure_captions_path = submission_dir / "figure_captions.md"
    table_captions_path = submission_dir / "table_captions.md"
    write_text(figure_captions_path, render_figure_captions(figure_captions))
    write_text(table_captions_path, render_table_captions(table_captions))

    freeze_manifest = {
        "canonical_section_transfer_run_dir": str(tmp_path / "section_transfer"),
        "canonical_baseline_run_dir": str(tmp_path / "baseline"),
        "canonical_robustness_run_dir": str(tmp_path / "robustness"),
        "canonical_results_package_dir": str(results_package_dir),
        "paper_drafting_package_dir": str(paper_drafting_dir),
        "primary_model_variant": "averaged_passive_aggressive::pseudo_all_sections",
        "primary_probe": "keep_reasoning_only",
        "secondary_probe": "drop_precedents",
    }
    submission_manifest = build_submission_package_manifest(
        freeze_manifest=freeze_manifest,
        results_package_dir=results_package_dir,
        paper_drafting_package_dir=paper_drafting_dir,
        section_importance_dir=section_importance_dir,
        submission_package_dir=submission_dir,
        figure_manifest=figure_manifest,
        writing_files=writing_paths,
        qualitative_files=[
            results_package_dir / "qualitative_examples_primary.md",
            results_package_dir / "qualitative_examples_secondary.md",
        ],
        appendix_files=[
            results_package_dir / "appendix_keep_reasoning_only_bundle.md",
            results_package_dir / "appendix_drop_precedents_bundle.md",
        ],
        manifest_files=[
            submission_dir / "submission_package_manifest.json",
            submission_dir / "submission_package_manifest.md",
            submission_dir / "figure_manifest.json",
            figure_captions_path,
            table_captions_path,
        ],
        layout_files=[main_layout_path, appendix_layout_path],
        reproducibility_file=reproducibility_path,
        traceability_file=traceability_path,
        consistency_file=submission_dir / "final_package_consistency_check.md",
    )
    submission_manifest_json = submission_dir / "submission_package_manifest.json"
    submission_manifest_md = submission_dir / "submission_package_manifest.md"
    write_json(submission_manifest_json, submission_manifest)
    write_text(submission_manifest_md, render_submission_package_manifest(submission_manifest))

    final_check = build_final_package_consistency_check(
        prior_consistency_check={"overall_status": "pass"},
        submission_package_manifest=submission_manifest,
        figure_manifest=figure_manifest,
        claim_traceability=traceability,
        writing_file_paths=writing_paths,
        traceability_path=traceability_path,
        allowed_root_paths=[results_package_dir, paper_drafting_dir, section_importance_dir, submission_dir, tmp_path],
    )
    handoff = build_paper_handoff_summary(
        submission_package_manifest=submission_manifest,
        final_consistency_check=final_check,
        section_importance_next_step_summary=section_importance_next_step_summary,
    )

    assert figure_manifest["figures"]
    assert "Figure Manifest" in render_figure_manifest(figure_manifest)
    assert traceability["claims"]
    assert "Claim to Evidence Traceability" in render_claim_to_evidence_traceability(traceability)
    assert "Main Text Layout Guide" in main_layout
    assert "Appendix Layout Guide" in appendix_layout
    assert final_check["overall_status"] in {"pass", "warning"}
    assert "Final Package Consistency Check" in render_final_package_consistency_check(final_check)
    assert handoff["open_first"]
    assert "Paper Handoff Summary" in render_paper_handoff_summary(handoff)
