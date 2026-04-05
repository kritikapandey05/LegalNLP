from __future__ import annotations

from pathlib import Path

from legal_robustness.config.schema import (
    AppConfig,
    DataConfig,
    LoggingConfig,
    OutputConfig,
    PerturbationConfig,
    PredictionConfig,
    RuntimeConfig,
    SectionConfig,
    SectionTransferConfig,
)
from legal_robustness.prediction import (
    build_unperturbed_model_comparison,
    build_prediction_input_text,
    render_baseline_prediction_metrics,
    render_unperturbed_model_comparison,
    train_prediction_baselines,
)
from legal_robustness.perturbations import (
    apply_perturbation,
    build_perturbation_specs,
    generate_perturbation_sets,
    render_perturbation_manifest,
)
from legal_robustness.section_transfer.types import CJPEPseudoSectionedCase


def build_test_config(project_root: Path) -> AppConfig:
    return AppConfig(
        project_name="prediction_test_project",
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
        runtime=RuntimeConfig(seed=13),
        sections=SectionConfig(),
        section_transfer=SectionTransferConfig(),
        prediction=PredictionConfig(
            baseline_models=(
                "tfidf_logistic_regression",
                "multinomial_naive_bayes",
                "averaged_passive_aggressive",
                "section_contextual_logistic_regression",
            ),
            input_variants=("full_text", "pseudo_all_sections", "pseudo_reasoning_only", "pseudo_without_conclusion"),
            evaluation_splits=("dev", "test"),
            max_tokens_per_document=128,
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
            prediction_preview_chars=120,
        ),
        perturbation=PerturbationConfig(
            enabled_recipes=(
                "drop_conclusion",
                "keep_facts_reasoning",
                "mask_conclusion",
                "reorder_conclusion_first",
            ),
            evaluation_splits=("test",),
            sample_size=3,
            preview_chars=100,
        ),
    )


def make_case(case_id: str, split: str, label: int, *, merits_token: str, conclusion: str) -> CJPEPseudoSectionedCase:
    facts = f"The assessee states the material tax facts for {case_id}."
    precedents = f"The court refers to precedent on {merits_token}."
    reasoning = f"The reasoning on {merits_token} strongly supports label {label}."
    other = "Miscellaneous procedural text."
    grouped_sections = {
        "facts": facts,
        "precedents": precedents,
        "reasoning": reasoning,
        "conclusion": conclusion,
        "other": other,
    }
    sentences = [facts, precedents, reasoning, conclusion, other]
    predicted_labels = ["facts", "precedents", "reasoning", "conclusion", "other"]
    raw_text = " ".join(sentences)
    return CJPEPseudoSectionedCase(
        case_id=case_id,
        cjpe_label=label,
        split=split,
        subset="single",
        raw_text=raw_text,
        sentences=sentences,
        sentence_indices=[0, 1, 2, 3, 4],
        sentence_start_chars=[0, 10, 20, 30, 40],
        sentence_end_chars=[9, 19, 29, 39, 49],
        predicted_broad_labels=predicted_labels,
        predicted_label_scores=[0.9, 0.91, 0.92, 0.93, 0.8],
        grouped_sections=grouped_sections,
        section_sentence_map={
            "facts": [0],
            "precedents": [1],
            "reasoning": [2],
            "conclusion": [3],
            "other": [4],
        },
        section_lengths_sentences={
            "facts": 1,
            "precedents": 1,
            "reasoning": 1,
            "conclusion": 1,
            "other": 1,
        },
        section_lengths_chars={key: len(value) for key, value in grouped_sections.items()},
        prediction_metadata={"classifier_type": "naive_bayes"},
        source_file="cjpe/sample.parquet",
        source_metadata={"source_row_index": 0},
    )


def make_cases() -> list[CJPEPseudoSectionedCase]:
    return [
        make_case("train-allow-1", "train", 1, merits_token="allow_token", conclusion="The appeal is allowed."),
        make_case("train-allow-2", "train", 1, merits_token="allow_token", conclusion="The petition is allowed."),
        make_case("train-dismiss-1", "train", 0, merits_token="dismiss_token", conclusion="The appeal is dismissed."),
        make_case("train-dismiss-2", "train", 0, merits_token="dismiss_token", conclusion="The petition is dismissed."),
        make_case("dev-allow-1", "dev", 1, merits_token="allow_token", conclusion="The appeal is allowed."),
        make_case("dev-dismiss-1", "dev", 0, merits_token="dismiss_token", conclusion="The appeal is dismissed."),
        make_case("test-allow-1", "test", 1, merits_token="allow_token", conclusion="The appeal is allowed."),
        make_case("test-dismiss-1", "test", 0, merits_token="dismiss_token", conclusion="The appeal is dismissed."),
    ]


def test_input_variant_generation_uses_expected_sections(tmp_path: Path) -> None:
    case = make_cases()[0]

    full_text, full_meta = build_prediction_input_text(
        case,
        variant_name="full_text",
        include_section_markers=True,
    )
    reasoning_only_text, reasoning_meta = build_prediction_input_text(
        case,
        variant_name="pseudo_reasoning_only",
        include_section_markers=True,
    )

    assert full_text == case.raw_text
    assert full_meta["used_predicted_sections"] is False
    assert "[REASONING_SECTION]" in reasoning_only_text
    assert "allow_token" in reasoning_only_text
    assert "appeal is allowed" not in reasoning_only_text.casefold()
    assert reasoning_meta["sections_used"] == ["reasoning"]


def test_train_prediction_baselines_on_synthetic_cases(tmp_path: Path) -> None:
    config = build_test_config(tmp_path)
    cases = make_cases()

    report, predictions_by_split, confusion_rows, sample_rows = train_prediction_baselines(
        cases,
        config=config,
        output_dir=tmp_path,
        source_section_transfer_run_dir=tmp_path / "section_transfer_run",
    )

    assert report["models"]["tfidf_logistic_regression"]["full_text"]["metrics_by_split"]["test"]["accuracy"] >= 0.5
    assert report["models"]["tfidf_logistic_regression"]["pseudo_all_sections"]["metrics_by_split"]["test"]["accuracy"] >= 0.5
    assert report["models"]["tfidf_logistic_regression"]["pseudo_reasoning_only"]["metrics_by_split"]["test"]["accuracy"] >= 0.5
    assert report["models"]["multinomial_naive_bayes"]["full_text"]["metrics_by_split"]["test"]["accuracy"] >= 0.5
    assert report["models"]["averaged_passive_aggressive"]["full_text"]["metrics_by_split"]["test"]["accuracy"] >= 0.5
    assert report["models"]["section_contextual_logistic_regression"]["pseudo_all_sections"]["metrics_by_split"]["test"]["accuracy"] >= 0.5
    assert predictions_by_split["test"]
    assert confusion_rows
    assert sample_rows
    assert any(
        "Skipping section_contextual_logistic_regression/full_text" in warning
        for warning in report["warnings"]
    )
    assert "Baseline Prediction Metrics" in render_baseline_prediction_metrics(report)
    comparison = build_unperturbed_model_comparison(
        report,
        primary_split="test",
        selected_model_variants=(
            "tfidf_logistic_regression::pseudo_all_sections",
            "multinomial_naive_bayes::pseudo_all_sections",
            "averaged_passive_aggressive::pseudo_all_sections",
            "section_contextual_logistic_regression::pseudo_all_sections",
        ),
    )
    assert len(comparison["pairwise_comparisons"]) == 6
    assert "Unperturbed Model Comparison" in render_unperturbed_model_comparison(comparison)


def test_perturbation_primitives_and_manifest(tmp_path: Path) -> None:
    config = build_test_config(tmp_path)
    test_case = [case for case in make_cases() if case.split == "test"][0]
    specs = build_perturbation_specs(config)
    drop_spec = next(spec for spec in specs if spec.name == "drop_conclusion")
    mask_spec = next(spec for spec in specs if spec.name == "mask_conclusion")
    reorder_spec = next(spec for spec in specs if spec.name == "reorder_conclusion_first")

    dropped = apply_perturbation(test_case, spec=drop_spec, config=config)
    masked = apply_perturbation(test_case, spec=mask_spec, config=config)
    reordered = apply_perturbation(test_case, spec=reorder_spec, config=config)
    rows_by_recipe, manifest, samples = generate_perturbation_sets(make_cases(), config=config)

    assert "allowed" not in dropped.perturbed_text.casefold()
    assert "[MASKED_CONCLUSION_SECTION]" in masked.perturbed_text
    assert reordered.perturbed_text.startswith("[CONCLUSION_SECTION]")
    assert rows_by_recipe["drop_conclusion"]
    assert manifest["recipes"]["drop_conclusion"]["case_count"] == 2
    assert samples
    assert "Perturbation Manifest" in render_perturbation_manifest(manifest)
