from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

import yaml

from legal_robustness.config.schema import (
    AppConfig,
    DataConfig,
    LoggingConfig,
    OutputConfig,
    PerturbationConfig,
    PredictionConfig,
    RobustnessConfig,
    RuntimeConfig,
    SectionTransferConfig,
    SectionConfig,
)
from legal_robustness.utils.exceptions import ConfigurationError
from legal_robustness.utils.paths import resolve_optional_path, resolve_path

ENV_VAR_PATTERN = re.compile(r"\$\{([^}]+)\}|\$([A-Za-z_][A-Za-z0-9_]*)|%([^%]+)%")


def _expand_env_vars(value: Any) -> Any:
    if isinstance(value, str):
        expanded = os.path.expandvars(value)
        return ENV_VAR_PATTERN.sub(lambda match: os.environ.get(next(group for group in match.groups() if group), ""), expanded)
    if isinstance(value, list):
        return [_expand_env_vars(item) for item in value]
    if isinstance(value, dict):
        return {key: _expand_env_vars(item) for key, item in value.items()}
    return value


def _read_yaml_config(config_path: Path) -> dict[str, Any]:
    if not config_path.exists():
        raise ConfigurationError(f"Configuration file does not exist: {config_path}")
    if not config_path.is_file():
        raise ConfigurationError(f"Configuration path is not a file: {config_path}")
    with config_path.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle) or {}
    if not isinstance(loaded, dict):
        raise ConfigurationError(f"Configuration file must contain a mapping at the top level: {config_path}")
    return loaded


def load_app_config(
    config_path: Path,
    dataset_root_override: Path | None = None,
    run_name_override: str | None = None,
    log_level_override: str | None = None,
    project_root: Path | None = None,
) -> AppConfig:
    project_root = (project_root or Path.cwd()).resolve()
    config_path = config_path.resolve()
    raw = _expand_env_vars(_read_yaml_config(config_path))

    data_raw = dict(raw.get("data", {}))
    if dataset_root_override is not None:
        data_raw["dataset_root"] = str(dataset_root_override)

    output_raw = dict(raw.get("output", {}))
    logging_raw = dict(raw.get("logging", {}))
    if log_level_override:
        logging_raw["level"] = log_level_override

    runtime_raw = dict(raw.get("runtime", {}))
    if run_name_override:
        runtime_raw["run_name"] = run_name_override

    data = DataConfig(
        dataset_root=resolve_optional_path(data_raw.get("dataset_root"), project_root),
        task_allowlist=tuple(data_raw.get("task_allowlist", ("cjpe", "rr"))),
        inspection_sample_rows=int(data_raw.get("inspection_sample_rows", 2)),
        inspection_max_text_chars=int(data_raw.get("inspection_max_text_chars", 400)),
        inspection_max_list_items=int(data_raw.get("inspection_max_list_items", 10)),
        inspection_max_tree_depth=int(data_raw.get("inspection_max_tree_depth", 5)),
        loader_batch_size=int(data_raw.get("loader_batch_size", 256)),
        max_malformed_row_fraction=float(data_raw.get("max_malformed_row_fraction", 0.5)),
        export_normalized_samples=bool(data_raw.get("export_normalized_samples", True)),
        normalized_sample_size=int(data_raw.get("normalized_sample_size", 5)),
        fail_on_duplicate_ids=bool(data_raw.get("fail_on_duplicate_ids", False)),
        collapse_text_whitespace=bool(data_raw.get("collapse_text_whitespace", False)),
        rr_label_normalization=str(data_raw.get("rr_label_normalization", "preserve")),
    )
    output = OutputConfig(
        root_dir=resolve_path(output_raw.get("root_dir", "outputs"), project_root),
        reports_dir=resolve_path(output_raw.get("reports_dir", "outputs/reports"), project_root),
        dataset_inspection_dir=resolve_path(
            output_raw.get("dataset_inspection_dir", "outputs/reports/dataset_inspection"),
            project_root,
        ),
        caches_dir=resolve_path(output_raw.get("caches_dir", "outputs/caches"), project_root),
        models_dir=resolve_path(output_raw.get("models_dir", "outputs/models"), project_root),
        perturbations_dir=resolve_path(output_raw.get("perturbations_dir", "outputs/perturbations"), project_root),
        evaluations_dir=resolve_path(output_raw.get("evaluations_dir", "outputs/evaluations"), project_root),
        analysis_dir=resolve_path(output_raw.get("analysis_dir", "outputs/analysis"), project_root),
    )
    logging_config = LoggingConfig(
        level=(logging_raw.get("level") or "INFO").upper(),
        log_to_file=bool(logging_raw.get("log_to_file", True)),
    )
    runtime = RuntimeConfig(
        seed=int(runtime_raw.get("seed", 42)),
        deterministic=bool(runtime_raw.get("deterministic", True)),
        run_name=runtime_raw.get("run_name") or None,
    )
    sections_raw = dict(raw.get("sections", {}))
    section_mapping_raw = dict(sections_raw.get("rr_section_mapping", {}))
    sections = SectionConfig(
        rr_label_names={str(key): str(value) for key, value in dict(sections_raw.get("rr_label_names", {})).items()},
        rr_section_mapping={
            str(section): tuple(values if isinstance(values, list) else [values])
            for section, values in section_mapping_raw.items()
        },
        unmapped_label_behavior=str(sections_raw.get("unmapped_label_behavior", "warn_and_route_to_other")),
        allow_partial_mapping=bool(sections_raw.get("allow_partial_mapping", True)),
        fail_on_unmapped_labels=bool(sections_raw.get("fail_on_unmapped_labels", False)),
        export_reconstruction_samples=bool(sections_raw.get("export_reconstruction_samples", True)),
        reconstruction_sample_size=int(sections_raw.get("reconstruction_sample_size", 5)),
    )
    section_transfer_raw = dict(raw.get("section_transfer", {}))
    section_transfer = SectionTransferConfig(
        label_mode=str(section_transfer_raw.get("label_mode", "broad")),
        sentence_segmentation_method=str(
            section_transfer_raw.get("sentence_segmentation_method", "heuristic_legal")
        ),
        sentence_segmentation_abbreviations=tuple(
            str(value) for value in section_transfer_raw.get("sentence_segmentation_abbreviations", ())
        )
        or SectionTransferConfig().sentence_segmentation_abbreviations,
        segmentation_short_sentence_chars=int(
            section_transfer_raw.get("segmentation_short_sentence_chars", 15)
        ),
        segmentation_long_sentence_chars=int(
            section_transfer_raw.get("segmentation_long_sentence_chars", 400)
        ),
        sample_size=int(section_transfer_raw.get("sample_size", 10)),
        context_window_size=int(section_transfer_raw.get("context_window_size", 1)),
        use_position_features=bool(section_transfer_raw.get("use_position_features", True)),
        use_context_features=bool(section_transfer_raw.get("use_context_features", True)),
        use_token_bigrams=bool(section_transfer_raw.get("use_token_bigrams", True)),
        feature_min_count=int(section_transfer_raw.get("feature_min_count", 2)),
        max_vocabulary_size=int(section_transfer_raw.get("max_vocabulary_size", 50000)),
        classifier_type=str(section_transfer_raw.get("classifier_type", "multinomial_naive_bayes")),
        export_prediction_probabilities=bool(
            section_transfer_raw.get("export_prediction_probabilities", True)
        ),
        dominant_section_ratio_threshold=float(
            section_transfer_raw.get("dominant_section_ratio_threshold", 0.8)
        ),
        sample_sentence_preview_count=int(
            section_transfer_raw.get("sample_sentence_preview_count", 5)
        ),
    )
    prediction_raw = dict(raw.get("prediction", {}))
    prediction = PredictionConfig(
        baseline_models=tuple(
            str(value) for value in prediction_raw.get("baseline_models", ())
        ) or PredictionConfig().baseline_models,
        input_variants=tuple(
            str(value) for value in prediction_raw.get("input_variants", ())
        ) or PredictionConfig().input_variants,
        evaluation_splits=tuple(
            str(value) for value in prediction_raw.get("evaluation_splits", ())
        ) or PredictionConfig().evaluation_splits,
        include_section_markers=bool(
            prediction_raw.get("include_section_markers", True)
        ),
        max_tokens_per_document=int(
            prediction_raw.get("max_tokens_per_document", 512)
        ),
        min_token_chars=int(prediction_raw.get("min_token_chars", 2)),
        use_token_bigrams=bool(prediction_raw.get("use_token_bigrams", False)),
        hashing_dimension=int(prediction_raw.get("hashing_dimension", 16384)),
        logistic_learning_rate=float(
            prediction_raw.get("logistic_learning_rate", 0.05)
        ),
        logistic_epochs=int(prediction_raw.get("logistic_epochs", 3)),
        logistic_l2_weight=float(
            prediction_raw.get("logistic_l2_weight", 0.000001)
        ),
        passive_aggressive_epochs=int(
            prediction_raw.get("passive_aggressive_epochs", 5)
        ),
        passive_aggressive_aggressiveness=float(
            prediction_raw.get("passive_aggressive_aggressiveness", 1.0)
        ),
        contextual_input_variants=tuple(
            str(value) for value in prediction_raw.get("contextual_input_variants", ())
        ) or PredictionConfig().contextual_input_variants,
        contextual_hashing_dimension=int(
            prediction_raw.get("contextual_hashing_dimension", 32768)
        ),
        contextual_learning_rate=float(
            prediction_raw.get("contextual_learning_rate", 0.08)
        ),
        contextual_epochs=int(prediction_raw.get("contextual_epochs", 4)),
        contextual_l2_weight=float(
            prediction_raw.get("contextual_l2_weight", 0.0000005)
        ),
        contextual_section_sentence_limit=int(
            prediction_raw.get("contextual_section_sentence_limit", 2)
        ),
        contextual_sentence_token_limit=int(
            prediction_raw.get("contextual_sentence_token_limit", 48)
        ),
        sample_size=int(prediction_raw.get("sample_size", 5)),
        prediction_preview_chars=int(
            prediction_raw.get("prediction_preview_chars", 240)
        ),
        model_artifact_prefix=str(
            prediction_raw.get("model_artifact_prefix", "baseline")
        ),
    )
    perturbation_raw = dict(raw.get("perturbation", {}))
    perturbation = PerturbationConfig(
        enabled_recipes=tuple(
            str(value) for value in perturbation_raw.get("enabled_recipes", ())
        ) or PerturbationConfig().enabled_recipes,
        evaluation_splits=tuple(
            str(value) for value in perturbation_raw.get("evaluation_splits", ())
        ) or PerturbationConfig().evaluation_splits,
        include_section_markers=bool(
            perturbation_raw.get("include_section_markers", True)
        ),
        mask_placeholder_template=str(
            perturbation_raw.get(
                "mask_placeholder_template",
                "[MASKED_{SECTION}_SECTION]",
            )
        ),
        sample_size=int(perturbation_raw.get("sample_size", 5)),
        preview_chars=int(perturbation_raw.get("preview_chars", 240)),
        run_perturbed_evaluation=bool(
            perturbation_raw.get("run_perturbed_evaluation", False)
        ),
    )
    robustness_raw = dict(raw.get("robustness", {}))
    robustness = RobustnessConfig(
        selected_model_variants=tuple(
            str(value) for value in robustness_raw.get("selected_model_variants", ())
        ) or RobustnessConfig().selected_model_variants,
        selected_perturbation_recipes=tuple(
            str(value) for value in robustness_raw.get("selected_perturbation_recipes", ())
        ) or RobustnessConfig().selected_perturbation_recipes,
        evaluation_splits=tuple(
            str(value) for value in robustness_raw.get("evaluation_splits", ())
        ) or RobustnessConfig().evaluation_splits,
        export_per_example_predictions=bool(
            robustness_raw.get("export_per_example_predictions", True)
        ),
        export_comparative_predictions=bool(
            robustness_raw.get("export_comparative_predictions", True)
        ),
        export_prediction_probabilities=bool(
            robustness_raw.get("export_prediction_probabilities", True)
        ),
        compute_flip_rate=bool(robustness_raw.get("compute_flip_rate", True)),
        compute_relative_retention=bool(
            robustness_raw.get("compute_relative_retention", True)
        ),
        high_coverage_min_fraction=float(
            robustness_raw.get("high_coverage_min_fraction", 0.7)
        ),
        medium_coverage_min_fraction=float(
            robustness_raw.get("medium_coverage_min_fraction", 0.3)
        ),
        isolate_low_coverage_recipes=bool(
            robustness_raw.get("isolate_low_coverage_recipes", True)
        ),
        include_full_text_in_unperturbed_comparison=bool(
            robustness_raw.get("include_full_text_in_unperturbed_comparison", True)
        ),
        apa_centered_reporting=bool(
            robustness_raw.get("apa_centered_reporting", True)
        ),
        primary_model_variant=str(
            robustness_raw.get(
                "primary_model_variant",
                "averaged_passive_aggressive::pseudo_all_sections",
            )
        ),
        reference_context_variants=tuple(
            str(value) for value in robustness_raw.get("reference_context_variants", ())
        ) or RobustnessConfig().reference_context_variants,
        sample_size=int(robustness_raw.get("sample_size", 8)),
        failure_analysis_recipes=tuple(
            str(value) for value in robustness_raw.get("failure_analysis_recipes", ())
        ) or RobustnessConfig().failure_analysis_recipes,
        failure_analysis_case_limit=int(
            robustness_raw.get("failure_analysis_case_limit", 20)
        ),
        failure_analysis_preview_chars=int(
            robustness_raw.get("failure_analysis_preview_chars", 240)
        ),
        enable_failure_analysis_disagreement_analysis=bool(
            robustness_raw.get("enable_failure_analysis_disagreement_analysis", True)
        ),
        qualitative_example_count_per_recipe=int(
            robustness_raw.get("qualitative_example_count_per_recipe", 6)
        ),
        qualitative_examples_per_category=int(
            robustness_raw.get("qualitative_examples_per_category", 1)
        ),
        qualitative_preview_chars=int(
            robustness_raw.get("qualitative_preview_chars", 220)
        ),
        qualitative_include_model_variants=tuple(
            str(value) for value in robustness_raw.get("qualitative_include_model_variants", ())
        ),
        case_bundle_size=int(
            robustness_raw.get("case_bundle_size", 12)
        ),
        results_package_dirname=str(
            robustness_raw.get("results_package_dirname", "results_package")
        ),
        primary_qualitative_example_count=int(
            robustness_raw.get("primary_qualitative_example_count", 4)
        ),
        secondary_qualitative_example_count=int(
            robustness_raw.get("secondary_qualitative_example_count", 4)
        ),
        narrative_preview_chars=int(
            robustness_raw.get("narrative_preview_chars", 200)
        ),
        export_chart_ready_data=bool(
            robustness_raw.get("export_chart_ready_data", True)
        ),
        export_appendix_bundles=bool(
            robustness_raw.get("export_appendix_bundles", True)
        ),
        canonical_section_transfer_run_dir=resolve_optional_path(
            robustness_raw.get("canonical_section_transfer_run_dir"),
            project_root,
        ),
        canonical_baseline_run_dir=resolve_optional_path(
            robustness_raw.get("canonical_baseline_run_dir"),
            project_root,
        ),
        canonical_robustness_run_dir=resolve_optional_path(
            robustness_raw.get("canonical_robustness_run_dir"),
            project_root,
        ),
        paper_freeze_output_dirname=str(
            robustness_raw.get("paper_freeze_output_dirname", "paper_drafting_package")
        ),
        export_writing_support_bundles=bool(
            robustness_raw.get("export_writing_support_bundles", True)
        ),
        run_targeted_strengthening_check=bool(
            robustness_raw.get("run_targeted_strengthening_check", True)
        ),
        export_paper_selection_guidance=bool(
            robustness_raw.get("export_paper_selection_guidance", True)
        ),
        stability_comparison_model_variants=tuple(
            str(value)
            for value in robustness_raw.get("stability_comparison_model_variants", ())
        ) or RobustnessConfig().stability_comparison_model_variants,
        canonical_paper_drafting_package_dir=resolve_optional_path(
            robustness_raw.get("canonical_paper_drafting_package_dir"),
            project_root,
        ),
        section_importance_output_dirname=str(
            robustness_raw.get("section_importance_output_dirname", "section_importance")
        ),
        canonical_section_importance_run_dir=resolve_optional_path(
            robustness_raw.get("canonical_section_importance_run_dir"),
            project_root,
        ),
        section_importance_primary_model_variant=str(
            robustness_raw.get(
                "section_importance_primary_model_variant",
                "averaged_passive_aggressive::pseudo_all_sections",
            )
        ),
        section_importance_supporting_model_variants=tuple(
            str(value)
            for value in robustness_raw.get("section_importance_supporting_model_variants", ())
        ) or RobustnessConfig().section_importance_supporting_model_variants,
        section_importance_sections=tuple(
            str(value) for value in robustness_raw.get("section_importance_sections", ())
        ) or RobustnessConfig().section_importance_sections,
        section_importance_pairwise_keep_variants=tuple(
            str(value)
            for value in robustness_raw.get("section_importance_pairwise_keep_variants", ())
        ) or RobustnessConfig().section_importance_pairwise_keep_variants,
        run_section_importance_cross_model_check=bool(
            robustness_raw.get("run_section_importance_cross_model_check", True)
        ),
        section_importance_cross_model_recipes=tuple(
            str(value)
            for value in robustness_raw.get("section_importance_cross_model_recipes", ())
        ) or RobustnessConfig().section_importance_cross_model_recipes,
        section_importance_composite_weights={
            str(key): float(value)
            for key, value in dict(
                robustness_raw.get("section_importance_composite_weights", {})
            ).items()
        }
        or RobustnessConfig().section_importance_composite_weights,
        export_section_importance_chart_data=bool(
            robustness_raw.get("export_section_importance_chart_data", True)
        ),
        submission_package_output_dirname=str(
            robustness_raw.get("submission_package_output_dirname", "submission_package")
        ),
        submission_figure_formats=tuple(
            str(value)
            for value in robustness_raw.get("submission_figure_formats", ("png", "svg"))
        ) or RobustnessConfig().submission_figure_formats,
        export_claim_traceability=bool(
            robustness_raw.get("export_claim_traceability", True)
        ),
        export_caption_files=bool(
            robustness_raw.get("export_caption_files", True)
        ),
        export_placement_guides=bool(
            robustness_raw.get("export_placement_guides", True)
        ),
    )
    return AppConfig(
        project_name=str(raw.get("project_name", "legal_robustness")),
        project_root=project_root,
        data=data,
        output=output,
        logging=logging_config,
        runtime=runtime,
        sections=sections,
        section_transfer=section_transfer,
        prediction=prediction,
        perturbation=perturbation,
        robustness=robustness,
    )
