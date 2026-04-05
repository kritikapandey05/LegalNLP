from __future__ import annotations

from dataclasses import dataclass, field, is_dataclass
from pathlib import Path
from typing import Any


def _serialize(value: Any) -> Any:
    if is_dataclass(value):
        return {key: _serialize(getattr(value, key)) for key in value.__dataclass_fields__}
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, tuple):
        return [_serialize(item) for item in value]
    if isinstance(value, list):
        return [_serialize(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _serialize(item) for key, item in value.items()}
    return value


@dataclass(frozen=True)
class DataConfig:
    dataset_root: Path | None = None
    task_allowlist: tuple[str, ...] = ("cjpe", "rr")
    inspection_sample_rows: int = 2
    inspection_max_text_chars: int = 400
    inspection_max_list_items: int = 10
    inspection_max_tree_depth: int = 5
    loader_batch_size: int = 256
    max_malformed_row_fraction: float = 0.5
    export_normalized_samples: bool = True
    normalized_sample_size: int = 5
    fail_on_duplicate_ids: bool = False
    collapse_text_whitespace: bool = False
    rr_label_normalization: str = "preserve"


@dataclass(frozen=True)
class OutputConfig:
    root_dir: Path = Path("outputs")
    reports_dir: Path = Path("outputs/reports")
    dataset_inspection_dir: Path = Path("outputs/reports/dataset_inspection")
    caches_dir: Path = Path("outputs/caches")
    models_dir: Path = Path("outputs/models")
    perturbations_dir: Path = Path("outputs/perturbations")
    evaluations_dir: Path = Path("outputs/evaluations")
    analysis_dir: Path = Path("outputs/analysis")


@dataclass(frozen=True)
class LoggingConfig:
    level: str = "INFO"
    log_to_file: bool = True


@dataclass(frozen=True)
class RuntimeConfig:
    seed: int = 42
    deterministic: bool = True
    run_name: str | None = None


@dataclass(frozen=True)
class SectionConfig:
    rr_label_names: dict[str, str] = field(default_factory=dict)
    rr_section_mapping: dict[str, tuple[Any, ...]] = field(default_factory=dict)
    unmapped_label_behavior: str = "warn_and_route_to_other"
    allow_partial_mapping: bool = True
    fail_on_unmapped_labels: bool = False
    export_reconstruction_samples: bool = True
    reconstruction_sample_size: int = 5


@dataclass(frozen=True)
class SectionTransferConfig:
    label_mode: str = "broad"
    sentence_segmentation_method: str = "heuristic_legal"
    sentence_segmentation_abbreviations: tuple[str, ...] = (
        "mr",
        "mrs",
        "ms",
        "dr",
        "prof",
        "shri",
        "smt",
        "j",
        "jj",
        "cji",
        "no",
        "nos",
        "sec",
        "secs",
        "art",
        "arts",
        "uoi",
        "ltd",
        "co",
        "corp",
        "inc",
        "ors",
        "anr",
        "etc",
        "vs",
        "v",
    )
    segmentation_short_sentence_chars: int = 15
    segmentation_long_sentence_chars: int = 400
    sample_size: int = 10
    context_window_size: int = 1
    use_position_features: bool = True
    use_context_features: bool = True
    use_token_bigrams: bool = True
    feature_min_count: int = 2
    max_vocabulary_size: int = 50000
    classifier_type: str = "multinomial_naive_bayes"
    export_prediction_probabilities: bool = True
    dominant_section_ratio_threshold: float = 0.8
    sample_sentence_preview_count: int = 5


@dataclass(frozen=True)
class PredictionConfig:
    baseline_models: tuple[str, ...] = (
        "tfidf_logistic_regression",
        "multinomial_naive_bayes",
        "averaged_passive_aggressive",
        "section_contextual_logistic_regression",
    )
    input_variants: tuple[str, ...] = (
        "full_text",
        "pseudo_all_sections",
    )
    evaluation_splits: tuple[str, ...] = ("dev", "test")
    include_section_markers: bool = True
    max_tokens_per_document: int = 512
    min_token_chars: int = 2
    use_token_bigrams: bool = False
    hashing_dimension: int = 16384
    logistic_learning_rate: float = 0.05
    logistic_epochs: int = 3
    logistic_l2_weight: float = 0.000001
    passive_aggressive_epochs: int = 5
    passive_aggressive_aggressiveness: float = 1.0
    contextual_input_variants: tuple[str, ...] = ("pseudo_all_sections",)
    contextual_hashing_dimension: int = 32768
    contextual_learning_rate: float = 0.08
    contextual_epochs: int = 4
    contextual_l2_weight: float = 0.0000005
    contextual_section_sentence_limit: int = 2
    contextual_sentence_token_limit: int = 48
    sample_size: int = 5
    prediction_preview_chars: int = 240
    model_artifact_prefix: str = "baseline"


@dataclass(frozen=True)
class PerturbationConfig:
    enabled_recipes: tuple[str, ...] = (
        "drop_conclusion",
        "drop_precedents",
        "keep_facts_reasoning",
        "keep_reasoning_only",
        "mask_conclusion",
        "reorder_conclusion_first",
    )
    evaluation_splits: tuple[str, ...] = ("test",)
    include_section_markers: bool = True
    mask_placeholder_template: str = "[MASKED_{SECTION}_SECTION]"
    sample_size: int = 5
    preview_chars: int = 240
    run_perturbed_evaluation: bool = False


@dataclass(frozen=True)
class RobustnessConfig:
    selected_model_variants: tuple[str, ...] = (
        "tfidf_logistic_regression::pseudo_all_sections",
        "multinomial_naive_bayes::pseudo_all_sections",
        "averaged_passive_aggressive::pseudo_all_sections",
        "section_contextual_logistic_regression::pseudo_all_sections",
    )
    selected_perturbation_recipes: tuple[str, ...] = (
        "drop_precedents",
        "keep_reasoning_only",
    )
    evaluation_splits: tuple[str, ...] = ("test",)
    export_per_example_predictions: bool = True
    export_comparative_predictions: bool = True
    export_prediction_probabilities: bool = True
    compute_flip_rate: bool = True
    compute_relative_retention: bool = True
    high_coverage_min_fraction: float = 0.7
    medium_coverage_min_fraction: float = 0.3
    isolate_low_coverage_recipes: bool = True
    include_full_text_in_unperturbed_comparison: bool = True
    apa_centered_reporting: bool = True
    primary_model_variant: str = "averaged_passive_aggressive::pseudo_all_sections"
    reference_context_variants: tuple[str, ...] = (
        "full_text",
        "pseudo_all_sections",
        "pseudo_facts_reasoning",
    )
    sample_size: int = 8
    failure_analysis_recipes: tuple[str, ...] = (
        "drop_precedents",
        "keep_reasoning_only",
    )
    failure_analysis_case_limit: int = 20
    failure_analysis_preview_chars: int = 240
    enable_failure_analysis_disagreement_analysis: bool = True
    qualitative_example_count_per_recipe: int = 6
    qualitative_examples_per_category: int = 1
    qualitative_preview_chars: int = 220
    qualitative_include_model_variants: tuple[str, ...] = ()
    case_bundle_size: int = 12
    results_package_dirname: str = "results_package"
    primary_qualitative_example_count: int = 4
    secondary_qualitative_example_count: int = 4
    narrative_preview_chars: int = 200
    export_chart_ready_data: bool = True
    export_appendix_bundles: bool = True
    canonical_section_transfer_run_dir: Path | None = None
    canonical_baseline_run_dir: Path | None = None
    canonical_robustness_run_dir: Path | None = None
    paper_freeze_output_dirname: str = "paper_drafting_package"
    export_writing_support_bundles: bool = True
    run_targeted_strengthening_check: bool = True
    export_paper_selection_guidance: bool = True
    stability_comparison_model_variants: tuple[str, ...] = (
        "multinomial_naive_bayes::pseudo_all_sections",
        "tfidf_logistic_regression::pseudo_all_sections",
    )
    canonical_paper_drafting_package_dir: Path | None = None
    section_importance_output_dirname: str = "section_importance"
    canonical_section_importance_run_dir: Path | None = None
    section_importance_primary_model_variant: str = "averaged_passive_aggressive::pseudo_all_sections"
    section_importance_supporting_model_variants: tuple[str, ...] = (
        "multinomial_naive_bayes::pseudo_all_sections",
        "tfidf_logistic_regression::pseudo_all_sections",
    )
    section_importance_sections: tuple[str, ...] = (
        "facts",
        "precedents",
        "reasoning",
        "conclusion",
        "other",
    )
    section_importance_pairwise_keep_variants: tuple[str, ...] = (
        "keep_facts_reasoning",
        "keep_reasoning_precedents",
        "keep_facts_precedents",
    )
    run_section_importance_cross_model_check: bool = True
    section_importance_cross_model_recipes: tuple[str, ...] = (
        "keep_only_reasoning",
        "keep_only_facts",
        "keep_only_precedents",
        "drop_reasoning",
        "drop_precedents",
    )
    section_importance_composite_weights: dict[str, float] = field(
        default_factory=lambda: {
            "removal_impact": 0.45,
            "solo_sufficiency": 0.35,
            "flip_sensitivity": 0.2,
        }
    )
    export_section_importance_chart_data: bool = True
    submission_package_output_dirname: str = "submission_package"
    submission_figure_formats: tuple[str, ...] = ("png", "svg")
    export_claim_traceability: bool = True
    export_caption_files: bool = True
    export_placement_guides: bool = True


@dataclass(frozen=True)
class AppConfig:
    project_name: str
    project_root: Path
    data: DataConfig = field(default_factory=DataConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    sections: SectionConfig = field(default_factory=SectionConfig)
    section_transfer: SectionTransferConfig = field(default_factory=SectionTransferConfig)
    prediction: PredictionConfig = field(default_factory=PredictionConfig)
    perturbation: PerturbationConfig = field(default_factory=PerturbationConfig)
    robustness: RobustnessConfig = field(default_factory=RobustnessConfig)

    def to_dict(self) -> dict[str, Any]:
        return _serialize(self)
