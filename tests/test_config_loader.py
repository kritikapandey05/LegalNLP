from __future__ import annotations

from pathlib import Path

from legal_robustness.config.loader import load_app_config


def test_load_app_config_expands_env_vars_and_resolves_paths(tmp_path: Path, monkeypatch) -> None:
    dataset_root = tmp_path / "dataset"
    dataset_root.mkdir()
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "\n".join(
            [
                "project_name: unit_test_project",
                "data:",
                "  dataset_root: ${TEST_DATASET_ROOT}",
                "output:",
                "  root_dir: outputs",
                "logging:",
                "  level: debug",
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("TEST_DATASET_ROOT", str(dataset_root))

    config = load_app_config(config_path=config_path, project_root=tmp_path)

    assert config.project_name == "unit_test_project"
    assert config.data.dataset_root == dataset_root.resolve()
    assert config.output.root_dir == (tmp_path / "outputs").resolve()
    assert config.logging.level == "DEBUG"


def test_cli_dataset_override_takes_precedence(tmp_path: Path) -> None:
    first_dataset = tmp_path / "dataset_a"
    second_dataset = tmp_path / "dataset_b"
    first_dataset.mkdir()
    second_dataset.mkdir()
    config_path = tmp_path / "config.yaml"
    config_path.write_text("data:\n  dataset_root: dataset_a\n", encoding="utf-8")

    config = load_app_config(
        config_path=config_path,
        dataset_root_override=second_dataset,
        project_root=tmp_path,
    )

    assert config.data.dataset_root == second_dataset.resolve()


def test_unset_env_placeholders_collapse_to_empty_values(tmp_path: Path, monkeypatch) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "\n".join(
            [
                "data:",
                "  dataset_root: ${UNSET_DATASET_ROOT}",
                "runtime:",
                "  run_name: ${UNSET_RUN_NAME}",
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.delenv("UNSET_DATASET_ROOT", raising=False)
    monkeypatch.delenv("UNSET_RUN_NAME", raising=False)

    config = load_app_config(config_path=config_path, project_root=tmp_path)

    assert config.data.dataset_root is None
    assert config.runtime.run_name is None


def test_load_app_config_parses_rr_section_mapping(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "\n".join(
            [
                "sections:",
                "  rr_label_names:",
                "    '0': Fact",
                "  rr_section_mapping:",
                "    facts:",
                "      - Fact",
                "      - 0",
                "    other:",
                "      - None",
                "  unmapped_label_behavior: warn_and_route_to_other",
                "  allow_partial_mapping: true",
            ]
        ),
        encoding="utf-8",
    )

    config = load_app_config(config_path=config_path, project_root=tmp_path)

    assert config.sections.rr_label_names == {"0": "Fact"}
    assert config.sections.rr_section_mapping["facts"] == ("Fact", 0)
    assert config.sections.unmapped_label_behavior == "warn_and_route_to_other"


def test_load_app_config_parses_section_transfer_settings(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "\n".join(
            [
                "section_transfer:",
                "  label_mode: broad",
                "  sentence_segmentation_method: heuristic_legal",
                "  sentence_segmentation_abbreviations:",
                "    - dr",
                "    - j",
                "  segmentation_short_sentence_chars: 12",
                "  segmentation_long_sentence_chars: 350",
                "  sample_size: 8",
                "  context_window_size: 2",
                "  use_position_features: false",
                "  use_context_features: true",
                "  use_token_bigrams: false",
                "  feature_min_count: 3",
                "  max_vocabulary_size: 1234",
                "  classifier_type: multinomial_naive_bayes",
                "  export_prediction_probabilities: false",
                "  dominant_section_ratio_threshold: 0.75",
                "  sample_sentence_preview_count: 7",
            ]
        ),
        encoding="utf-8",
    )

    config = load_app_config(config_path=config_path, project_root=tmp_path)

    assert config.section_transfer.label_mode == "broad"
    assert config.section_transfer.sentence_segmentation_method == "heuristic_legal"
    assert config.section_transfer.sentence_segmentation_abbreviations == ("dr", "j")
    assert config.section_transfer.segmentation_short_sentence_chars == 12
    assert config.section_transfer.segmentation_long_sentence_chars == 350
    assert config.section_transfer.sample_size == 8
    assert config.section_transfer.context_window_size == 2
    assert config.section_transfer.use_position_features is False
    assert config.section_transfer.use_context_features is True
    assert config.section_transfer.use_token_bigrams is False
    assert config.section_transfer.feature_min_count == 3
    assert config.section_transfer.max_vocabulary_size == 1234
    assert config.section_transfer.classifier_type == "multinomial_naive_bayes"
    assert config.section_transfer.export_prediction_probabilities is False
    assert config.section_transfer.dominant_section_ratio_threshold == 0.75
    assert config.section_transfer.sample_sentence_preview_count == 7
    assert not hasattr(config, "alignment")


def test_load_app_config_parses_prediction_and_perturbation_settings(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "\n".join(
            [
                "prediction:",
                "  baseline_models:",
                "    - tfidf_logistic_regression",
                "  input_variants:",
                "    - full_text",
                "    - pseudo_reasoning_only",
                "  evaluation_splits:",
                "    - test",
                "  include_section_markers: false",
                "  max_tokens_per_document: 256",
                "  min_token_chars: 3",
                "  use_token_bigrams: true",
                "  hashing_dimension: 2048",
                "  logistic_learning_rate: 0.1",
                "  logistic_epochs: 5",
                "  logistic_l2_weight: 0.0001",
                "  passive_aggressive_epochs: 6",
                "  passive_aggressive_aggressiveness: 0.8",
                "  contextual_input_variants:",
                "    - pseudo_all_sections",
                "  contextual_hashing_dimension: 12345",
                "  contextual_learning_rate: 0.12",
                "  contextual_epochs: 9",
                "  contextual_l2_weight: 0.00002",
                "  contextual_section_sentence_limit: 3",
                "  contextual_sentence_token_limit: 77",
                "  sample_size: 4",
                "  prediction_preview_chars: 160",
                "  model_artifact_prefix: demo",
                "perturbation:",
                "  enabled_recipes:",
                "    - drop_conclusion",
                "    - mask_conclusion",
                "  evaluation_splits:",
                "    - dev",
                "  include_section_markers: false",
                "  mask_placeholder_template: '[MASKED_{SECTION}]'",
                "  sample_size: 3",
                "  preview_chars: 111",
                "  run_perturbed_evaluation: true",
            ]
        ),
        encoding="utf-8",
    )

    config = load_app_config(config_path=config_path, project_root=tmp_path)

    assert config.prediction.baseline_models == ("tfidf_logistic_regression",)
    assert config.prediction.input_variants == ("full_text", "pseudo_reasoning_only")
    assert config.prediction.evaluation_splits == ("test",)
    assert config.prediction.include_section_markers is False
    assert config.prediction.max_tokens_per_document == 256
    assert config.prediction.min_token_chars == 3
    assert config.prediction.use_token_bigrams is True
    assert config.prediction.hashing_dimension == 2048
    assert config.prediction.logistic_learning_rate == 0.1
    assert config.prediction.logistic_epochs == 5
    assert config.prediction.logistic_l2_weight == 0.0001
    assert config.prediction.passive_aggressive_epochs == 6
    assert config.prediction.passive_aggressive_aggressiveness == 0.8
    assert config.prediction.contextual_input_variants == ("pseudo_all_sections",)
    assert config.prediction.contextual_hashing_dimension == 12345
    assert config.prediction.contextual_learning_rate == 0.12
    assert config.prediction.contextual_epochs == 9
    assert config.prediction.contextual_l2_weight == 0.00002
    assert config.prediction.contextual_section_sentence_limit == 3
    assert config.prediction.contextual_sentence_token_limit == 77
    assert config.prediction.sample_size == 4
    assert config.prediction.prediction_preview_chars == 160
    assert config.prediction.model_artifact_prefix == "demo"
    assert config.perturbation.enabled_recipes == ("drop_conclusion", "mask_conclusion")
    assert config.perturbation.evaluation_splits == ("dev",)
    assert config.perturbation.include_section_markers is False
    assert config.perturbation.mask_placeholder_template == "[MASKED_{SECTION}]"
    assert config.perturbation.sample_size == 3
    assert config.perturbation.preview_chars == 111
    assert config.perturbation.run_perturbed_evaluation is True


def test_load_app_config_parses_robustness_settings(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "\n".join(
            [
                "robustness:",
                "  selected_model_variants:",
                "    - tfidf_logistic_regression::pseudo_all_sections",
                "  selected_perturbation_recipes:",
                "    - drop_precedents",
                "    - mask_conclusion",
                "  evaluation_splits:",
                "    - test",
                "  export_per_example_predictions: false",
                "  export_comparative_predictions: false",
                "  export_prediction_probabilities: false",
                "  compute_flip_rate: false",
                "  compute_relative_retention: false",
                "  high_coverage_min_fraction: 0.8",
                "  medium_coverage_min_fraction: 0.4",
                "  isolate_low_coverage_recipes: false",
                "  include_full_text_in_unperturbed_comparison: false",
                "  apa_centered_reporting: false",
                "  primary_model_variant: averaged_passive_aggressive::pseudo_all_sections",
                "  reference_context_variants:",
                "    - pseudo_all_sections",
                "    - full_text",
                "  sample_size: 6",
                "  failure_analysis_recipes:",
                "    - drop_precedents",
                "    - keep_reasoning_only",
                "  failure_analysis_case_limit: 11",
                "  failure_analysis_preview_chars: 180",
                "  enable_failure_analysis_disagreement_analysis: false",
                "  qualitative_example_count_per_recipe: 4",
                "  qualitative_examples_per_category: 2",
                "  qualitative_preview_chars: 150",
                "  qualitative_include_model_variants:",
                "    - averaged_passive_aggressive::pseudo_all_sections",
                "    - section_contextual_logistic_regression::pseudo_all_sections",
                "  case_bundle_size: 9",
                "  results_package_dirname: paper_bundle",
                "  primary_qualitative_example_count: 5",
                "  secondary_qualitative_example_count: 3",
                "  narrative_preview_chars: 210",
                "  export_chart_ready_data: false",
                "  export_appendix_bundles: false",
                "  canonical_section_transfer_run_dir: outputs/reports/section_transfer/frozen-run",
                "  canonical_baseline_run_dir: outputs/reports/prediction_baselines/frozen-run",
                "  canonical_robustness_run_dir: outputs/reports/robustness/frozen-run",
                "  paper_freeze_output_dirname: paper_freeze_bundle",
                "  export_writing_support_bundles: false",
                "  run_targeted_strengthening_check: false",
                "  export_paper_selection_guidance: false",
                "  stability_comparison_model_variants:",
                "    - multinomial_naive_bayes::pseudo_all_sections",
                "    - tfidf_logistic_regression::pseudo_all_sections",
                "  canonical_paper_drafting_package_dir: outputs/reports/paper_drafting_package/frozen-run",
                "  section_importance_output_dirname: section_importance_bundle",
                "  canonical_section_importance_run_dir: outputs/reports/section_importance/frozen-run",
                "  section_importance_primary_model_variant: averaged_passive_aggressive::pseudo_all_sections",
                "  section_importance_supporting_model_variants:",
                "    - multinomial_naive_bayes::pseudo_all_sections",
                "    - tfidf_logistic_regression::pseudo_all_sections",
                "  section_importance_sections:",
                "    - facts",
                "    - precedents",
                "    - reasoning",
                "    - conclusion",
                "    - other",
                "  section_importance_pairwise_keep_variants:",
                "    - keep_facts_reasoning",
                "    - keep_reasoning_precedents",
                "  run_section_importance_cross_model_check: false",
                "  section_importance_cross_model_recipes:",
                "    - keep_only_reasoning",
                "    - drop_reasoning",
                "  section_importance_composite_weights:",
                "    removal_impact: 0.5",
                "    solo_sufficiency: 0.3",
                "    flip_sensitivity: 0.2",
                "  export_section_importance_chart_data: false",
                "  submission_package_output_dirname: final_submission_bundle",
                "  submission_figure_formats:",
                "    - png",
                "    - svg",
                "  export_claim_traceability: false",
                "  export_caption_files: false",
                "  export_placement_guides: false",
            ]
        ),
        encoding="utf-8",
    )

    config = load_app_config(config_path=config_path, project_root=tmp_path)

    assert config.robustness.selected_model_variants == (
        "tfidf_logistic_regression::pseudo_all_sections",
    )
    assert config.robustness.selected_perturbation_recipes == (
        "drop_precedents",
        "mask_conclusion",
    )
    assert config.robustness.evaluation_splits == ("test",)
    assert config.robustness.export_per_example_predictions is False
    assert config.robustness.export_comparative_predictions is False
    assert config.robustness.export_prediction_probabilities is False
    assert config.robustness.compute_flip_rate is False
    assert config.robustness.compute_relative_retention is False
    assert config.robustness.high_coverage_min_fraction == 0.8
    assert config.robustness.medium_coverage_min_fraction == 0.4
    assert config.robustness.isolate_low_coverage_recipes is False
    assert config.robustness.include_full_text_in_unperturbed_comparison is False
    assert config.robustness.apa_centered_reporting is False
    assert config.robustness.primary_model_variant == "averaged_passive_aggressive::pseudo_all_sections"
    assert config.robustness.reference_context_variants == (
        "pseudo_all_sections",
        "full_text",
    )
    assert config.robustness.sample_size == 6
    assert config.robustness.failure_analysis_recipes == (
        "drop_precedents",
        "keep_reasoning_only",
    )
    assert config.robustness.failure_analysis_case_limit == 11
    assert config.robustness.failure_analysis_preview_chars == 180
    assert config.robustness.enable_failure_analysis_disagreement_analysis is False
    assert config.robustness.qualitative_example_count_per_recipe == 4
    assert config.robustness.qualitative_examples_per_category == 2
    assert config.robustness.qualitative_preview_chars == 150
    assert config.robustness.qualitative_include_model_variants == (
        "averaged_passive_aggressive::pseudo_all_sections",
        "section_contextual_logistic_regression::pseudo_all_sections",
    )
    assert config.robustness.case_bundle_size == 9
    assert config.robustness.results_package_dirname == "paper_bundle"
    assert config.robustness.primary_qualitative_example_count == 5
    assert config.robustness.secondary_qualitative_example_count == 3
    assert config.robustness.narrative_preview_chars == 210
    assert config.robustness.export_chart_ready_data is False
    assert config.robustness.export_appendix_bundles is False
    assert config.robustness.canonical_section_transfer_run_dir == (
        tmp_path / "outputs" / "reports" / "section_transfer" / "frozen-run"
    )
    assert config.robustness.canonical_baseline_run_dir == (
        tmp_path / "outputs" / "reports" / "prediction_baselines" / "frozen-run"
    )
    assert config.robustness.canonical_robustness_run_dir == (
        tmp_path / "outputs" / "reports" / "robustness" / "frozen-run"
    )
    assert config.robustness.paper_freeze_output_dirname == "paper_freeze_bundle"
    assert config.robustness.export_writing_support_bundles is False
    assert config.robustness.run_targeted_strengthening_check is False
    assert config.robustness.export_paper_selection_guidance is False
    assert config.robustness.stability_comparison_model_variants == (
        "multinomial_naive_bayes::pseudo_all_sections",
        "tfidf_logistic_regression::pseudo_all_sections",
    )
    assert config.robustness.canonical_paper_drafting_package_dir == (
        tmp_path / "outputs" / "reports" / "paper_drafting_package" / "frozen-run"
    )
    assert config.robustness.section_importance_output_dirname == "section_importance_bundle"
    assert config.robustness.canonical_section_importance_run_dir == (
        tmp_path / "outputs" / "reports" / "section_importance" / "frozen-run"
    )
    assert config.robustness.section_importance_primary_model_variant == (
        "averaged_passive_aggressive::pseudo_all_sections"
    )
    assert config.robustness.section_importance_supporting_model_variants == (
        "multinomial_naive_bayes::pseudo_all_sections",
        "tfidf_logistic_regression::pseudo_all_sections",
    )
    assert config.robustness.section_importance_sections == (
        "facts",
        "precedents",
        "reasoning",
        "conclusion",
        "other",
    )
    assert config.robustness.section_importance_pairwise_keep_variants == (
        "keep_facts_reasoning",
        "keep_reasoning_precedents",
    )
    assert config.robustness.run_section_importance_cross_model_check is False
    assert config.robustness.section_importance_cross_model_recipes == (
        "keep_only_reasoning",
        "drop_reasoning",
    )
    assert config.robustness.section_importance_composite_weights == {
        "removal_impact": 0.5,
        "solo_sufficiency": 0.3,
        "flip_sensitivity": 0.2,
    }
    assert config.robustness.export_section_importance_chart_data is False
    assert config.robustness.submission_package_output_dirname == "final_submission_bundle"
    assert config.robustness.submission_figure_formats == ("png", "svg")
    assert config.robustness.export_claim_traceability is False
    assert config.robustness.export_caption_files is False
    assert config.robustness.export_placement_guides is False
