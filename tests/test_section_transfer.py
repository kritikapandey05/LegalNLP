from __future__ import annotations

from pathlib import Path

from legal_robustness.config.schema import (
    AppConfig,
    DataConfig,
    LoggingConfig,
    OutputConfig,
    RuntimeConfig,
    SectionConfig,
    SectionTransferConfig,
)
from legal_robustness.data.normalized_types import (
    RRSectionMappingEntry,
    RRSectionMappingReport,
    ReconstructedRRCase,
)
from legal_robustness.data.normalized_types import NormalizedCJPECase
from legal_robustness.section_transfer import (
    BroadSectionNaiveBayesModel,
    build_cjpe_prediction_samples,
    build_cjpe_reconstruction_samples,
    build_rr_sentence_supervision,
    infer_cjpe_sections,
    reconstruct_cjpe_predicted_sections,
    render_cjpe_sentence_segmentation_report,
    render_cjpe_reconstruction_summary,
    render_cjpe_section_prediction_summary,
    render_rr_section_tagger_metrics,
    render_rr_sentence_supervision_summary,
    render_section_transfer_readiness_summary,
    segment_cjpe_cases,
    split_legal_text_into_sentences,
    summarize_section_transfer_readiness,
    train_and_evaluate_rr_section_tagger,
)


def build_test_config(project_root: Path) -> AppConfig:
    return AppConfig(
        project_name="test_project",
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
        runtime=RuntimeConfig(),
        sections=SectionConfig(),
        section_transfer=SectionTransferConfig(
            sample_size=4,
            context_window_size=1,
            use_position_features=True,
            use_context_features=True,
            use_token_bigrams=True,
            feature_min_count=1,
            max_vocabulary_size=1000,
            sample_sentence_preview_count=3,
        ),
    )


def make_mapping_report() -> RRSectionMappingReport:
    entries = [
        RRSectionMappingEntry(
            label_key="0",
            raw_value=0,
            raw_type="int",
            label_name="Fact",
            mapped_section="facts",
            resolution_strategy="label_name",
            count=2,
            counts_by_split={"train": 2},
            cases_with_label=1,
        ),
        RRSectionMappingEntry(
            label_key="10",
            raw_value=10,
            raw_type="int",
            label_name="RatioOfTheDecision",
            mapped_section="reasoning",
            resolution_strategy="label_name",
            count=2,
            counts_by_split={"train": 2},
            cases_with_label=1,
        ),
        RRSectionMappingEntry(
            label_key="11",
            raw_value=11,
            raw_type="int",
            label_name="RulingByPresentCourt",
            mapped_section="conclusion",
            resolution_strategy="label_name",
            count=2,
            counts_by_split={"train": 2},
            cases_with_label=1,
        ),
    ]
    return RRSectionMappingReport(
        entries=entries,
        summary={"output_sections": ["facts", "precedents", "reasoning", "conclusion", "other"]},
        applied_config={},
        warnings=[],
    )


def make_rr_case(case_id: str, split: str, sentences: list[str], labels: list[int], label_names: list[str]) -> ReconstructedRRCase:
    section_sentence_map = {"facts": [], "precedents": [], "reasoning": [], "conclusion": [], "other": []}
    grouped_sections = {"facts": "", "precedents": "", "reasoning": "", "conclusion": "", "other": ""}
    section_lengths_chars = {key: 0 for key in grouped_sections}
    section_lengths_sentences = {key: 0 for key in grouped_sections}
    return ReconstructedRRCase(
        case_id=case_id,
        split=split,
        subset="IT",
        sentences=sentences,
        rr_labels=labels,
        rr_label_names=label_names,
        grouped_sections=grouped_sections,
        section_sentence_map=section_sentence_map,
        section_lengths_chars=section_lengths_chars,
        section_lengths_sentences=section_lengths_sentences,
        unmapped_labels_present=False,
        source_file="rr\\sample.parquet",
        source_metadata={"source_row_index": 0},
    )


def make_cjpe_case(case_id: str, text: str) -> NormalizedCJPECase:
    return NormalizedCJPECase(
        case_id=case_id,
        split="train",
        subset="single",
        label=1,
        raw_text=text,
        text_length_chars=len(text),
        text_length_tokens_approx=len(text.split()),
        expert_annotations={},
        source_file="cjpe\\sample.parquet",
        source_metadata={"source_row_index": 0},
    )


def test_build_rr_sentence_supervision_preserves_broad_labels(tmp_path: Path) -> None:
    config = build_test_config(tmp_path)
    rr_cases = [
        make_rr_case(
            "rr-1",
            "train",
            ["Facts sentence.", "Reasoning sentence.", "Conclusion sentence."],
            [0, 10, 11],
            ["Fact", "RatioOfTheDecision", "RulingByPresentCourt"],
        )
    ]
    mapping_report = make_mapping_report()

    result = build_rr_sentence_supervision(rr_cases, mapping_report=mapping_report, config=config)

    assert len(result.records) == 3
    assert result.records[0].broad_section_label == "facts"
    assert result.records[1].broad_section_label == "reasoning"
    assert result.records[2].broad_section_label == "conclusion"
    assert result.records[1].previous_context_text == "Facts sentence."
    assert result.records[1].next_context_text == "Conclusion sentence."
    assert "RR Sentence Supervision Summary" in render_rr_sentence_supervision_summary(result.report)


def test_split_legal_text_into_sentences_respects_abbreviations(tmp_path: Path) -> None:
    config = build_test_config(tmp_path)
    text = "Dr. Chandrachud, J. This appeal arises from the judgment. The Revenue is in appeal."

    spans = split_legal_text_into_sentences(text, config=config)

    assert len(spans) == 2
    assert spans[0].text == "Dr. Chandrachud, J. This appeal arises from the judgment."
    assert spans[1].text == "The Revenue is in appeal."


def test_segment_cjpe_cases_reports_sentence_statistics(tmp_path: Path) -> None:
    config = build_test_config(tmp_path)
    cases = [make_cjpe_case("cjpe-1", "Fact one. Fact two! Reasoning follows?")]

    result = segment_cjpe_cases(cases, config=config)

    assert len(result.records) == 1
    assert result.records[0].sentence_count == 3
    assert result.report["total_sentences"] == 3
    assert "CJPE Sentence Segmentation Report" in render_cjpe_sentence_segmentation_report(result.report)


def test_train_rr_section_tagger_on_synthetic_data(tmp_path: Path) -> None:
    config = build_test_config(tmp_path)
    mapping_report = make_mapping_report()
    rr_cases = [
        make_rr_case(
            "train-1",
            "train",
            [
                "The assessee filed the return of income.",
                "For the above reasons the appeal must fail.",
                "The appeal is dismissed.",
            ],
            [0, 10, 11],
            ["Fact", "RatioOfTheDecision", "RulingByPresentCourt"],
        ),
        make_rr_case(
            "train-2",
            "train",
            [
                "The dispute concerns assessment year 2011.",
                "We therefore hold that section 234B applies.",
                "The civil appeal stands allowed.",
            ],
            [0, 10, 11],
            ["Fact", "RatioOfTheDecision", "RulingByPresentCourt"],
        ),
        make_rr_case(
            "dev-1",
            "dev",
            [
                "The appeal arises from the assessment order.",
                "We hold that the tribunal was correct.",
                "The petition is dismissed.",
            ],
            [0, 10, 11],
            ["Fact", "RatioOfTheDecision", "RulingByPresentCourt"],
        ),
        make_rr_case(
            "test-1",
            "test",
            [
                "The assessee challenges the demand notice.",
                "For these reasons the levy is unsustainable.",
                "The appeal is allowed.",
            ],
            [0, 10, 11],
            ["Fact", "RatioOfTheDecision", "RulingByPresentCourt"],
        ),
    ]
    supervision = build_rr_sentence_supervision(rr_cases, mapping_report=mapping_report, config=config)

    training_result = train_and_evaluate_rr_section_tagger(
        supervision.records,
        config=config,
        output_dir=tmp_path,
    )

    assert Path(training_result.model_path).exists()
    assert training_result.metrics["split_counts"]["train"] == 6
    assert training_result.metrics["metrics_by_split"]["test"]["accuracy"] >= 0.66
    assert "RR Section Tagger Metrics" in render_rr_section_tagger_metrics(training_result.metrics)


def test_broad_section_tagger_predict_proba_shape(tmp_path: Path) -> None:
    config = build_test_config(tmp_path)
    mapping_report = make_mapping_report()
    rr_cases = [
        make_rr_case(
            "train-1",
            "train",
            [
                "The assessee filed the return.",
                "For these reasons section 234B applies.",
                "The appeal is dismissed.",
            ],
            [0, 10, 11],
            ["Fact", "RatioOfTheDecision", "RulingByPresentCourt"],
        )
    ]
    supervision = build_rr_sentence_supervision(rr_cases, mapping_report=mapping_report, config=config)
    model, _ = BroadSectionNaiveBayesModel.train(supervision.records, config=config)

    probabilities = model.predict_proba(supervision.records[0], config=config)

    assert set(probabilities) == {"facts", "reasoning", "conclusion"}
    assert abs(sum(probabilities.values()) - 1.0) <= 0.00001


def test_cjpe_inference_output_shape_and_summary(tmp_path: Path) -> None:
    config = build_test_config(tmp_path)
    mapping_report = make_mapping_report()
    rr_cases = [
        make_rr_case(
            "train-1",
            "train",
            [
                "The assessee filed the return.",
                "For these reasons section 234B applies.",
                "The appeal is dismissed.",
            ],
            [0, 10, 11],
            ["Fact", "RatioOfTheDecision", "RulingByPresentCourt"],
        )
    ]
    supervision = build_rr_sentence_supervision(rr_cases, mapping_report=mapping_report, config=config)
    model, _ = BroadSectionNaiveBayesModel.train(supervision.records, config=config)
    cjpe_cases = [make_cjpe_case("cjpe-1", "The assessee filed the return. For these reasons the levy fails. The appeal is dismissed.")]
    segmented = segment_cjpe_cases(cjpe_cases, config=config)

    result = infer_cjpe_sections(
        segmented_cases=segmented.records,
        cjpe_cases=cjpe_cases,
        model=model,
        model_path=str(tmp_path / "model.pkl"),
        config=config,
    )
    samples = build_cjpe_prediction_samples(
        result.records,
        sample_size=config.section_transfer.sample_size,
        preview_sentence_count=config.section_transfer.sample_sentence_preview_count,
    )

    assert len(result.records) == 1
    assert result.records[0].predicted_broad_labels
    assert len(result.records[0].predicted_broad_labels) == len(result.records[0].sentences)
    assert len(result.records[0].predicted_label_scores) == len(result.records[0].sentences)
    assert samples[0]["sample_predictions"]
    assert "CJPE Section Prediction Summary" in render_cjpe_section_prediction_summary(result.report)


def test_cjpe_reconstruction_and_readiness_summary(tmp_path: Path) -> None:
    config = build_test_config(tmp_path)
    mapping_report = make_mapping_report()
    rr_cases = [
        make_rr_case(
            "train-1",
            "train",
            [
                "The assessee filed the return.",
                "For these reasons section 234B applies.",
                "The appeal is dismissed.",
            ],
            [0, 10, 11],
            ["Fact", "RatioOfTheDecision", "RulingByPresentCourt"],
        ),
        make_rr_case(
            "test-1",
            "test",
            [
                "The dispute concerns assessment year 2011.",
                "For these reasons the order cannot stand.",
                "The appeal is allowed.",
            ],
            [0, 10, 11],
            ["Fact", "RatioOfTheDecision", "RulingByPresentCourt"],
        ),
    ]
    supervision = build_rr_sentence_supervision(rr_cases, mapping_report=mapping_report, config=config)
    training_result = train_and_evaluate_rr_section_tagger(
        supervision.records,
        config=config,
        output_dir=tmp_path,
    )
    model = BroadSectionNaiveBayesModel.train(
        [record for record in supervision.records if record.split == "train"],
        config=config,
    )[0]
    cjpe_cases = [make_cjpe_case("cjpe-1", "The assessee filed the return. For these reasons the levy fails. The appeal is dismissed.")]
    segmented = segment_cjpe_cases(cjpe_cases, config=config)
    predictions = infer_cjpe_sections(
        segmented_cases=segmented.records,
        cjpe_cases=cjpe_cases,
        model=model,
        model_path=str(tmp_path / "model.pkl"),
        config=config,
    )
    reconstructed = reconstruct_cjpe_predicted_sections(predictions.records, config=config)
    readiness = summarize_section_transfer_readiness(
        rr_supervision=supervision,
        training_result=training_result,
        cjpe_predictions=predictions,
        cjpe_reconstructed=reconstructed,
    )
    samples = build_cjpe_reconstruction_samples(
        reconstructed.records,
        sample_size=config.section_transfer.sample_size,
        preview_chars=120,
    )

    assert len(reconstructed.records) == 1
    assert reconstructed.records[0].grouped_sections["facts"] or reconstructed.records[0].grouped_sections["reasoning"]
    assert samples[0]["grouped_section_previews"]
    assert readiness["recommendations"]
    assert "CJPE Reconstruction Summary" in render_cjpe_reconstruction_summary(reconstructed.report)
    assert "Section Transfer Readiness Summary" in render_section_transfer_readiness_summary(readiness)
