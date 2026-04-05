from __future__ import annotations

from pathlib import Path

from legal_robustness.config.schema import (
    AppConfig,
    DataConfig,
    LoggingConfig,
    OutputConfig,
    RuntimeConfig,
    SectionConfig,
)
from legal_robustness.data.label_inventory import generate_rr_label_inventory
from legal_robustness.data.normalized_types import NormalizedRRCase
from legal_robustness.data.reconstruct import reconstruct_rr_sections, render_rr_reconstruction_report
from legal_robustness.data.rr_mapping import (
    discover_rr_label_names,
    render_rr_section_mapping_report,
    validate_rr_section_mapping,
)


def build_test_config(
    project_root: Path,
    *,
    rr_label_names: dict[str, str] | None = None,
    rr_section_mapping: dict[str, tuple[object, ...]] | None = None,
    unmapped_label_behavior: str = "warn_and_route_to_other",
) -> AppConfig:
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
        sections=SectionConfig(
            rr_label_names=rr_label_names or {"0": "Fact", "1": "Issue", "2": "RatioOfTheDecision", "3": "None"},
            rr_section_mapping=rr_section_mapping
            or {
                "facts": ("Fact", "Issue"),
                "reasoning": ("RatioOfTheDecision",),
                "other": ("None",),
            },
            unmapped_label_behavior=unmapped_label_behavior,
            allow_partial_mapping=True,
            fail_on_unmapped_labels=False,
            export_reconstruction_samples=True,
            reconstruction_sample_size=2,
        ),
    )


def build_rr_cases() -> list[NormalizedRRCase]:
    return [
        NormalizedRRCase(
            case_id="rr-1",
            split="train",
            subset="IT",
            sentences=["Fact sentence", "Issue sentence", "Reasoning sentence", "Fallback sentence"],
            rr_labels=[0, 1, 2, 99],
            num_sentences=4,
            num_labels=4,
            empty_sentence_count=0,
            empty_label_count=0,
            alignment_ok=True,
            source_file="rr\\IT_train.parquet",
            source_metadata={"raw_case_id": "rr-1"},
        ),
        NormalizedRRCase(
            case_id="rr-2",
            split="dev",
            subset="IT",
            sentences=["Only none sentence"],
            rr_labels=[3],
            num_sentences=1,
            num_labels=1,
            empty_sentence_count=0,
            empty_label_count=0,
            alignment_ok=True,
            source_file="rr\\IT_dev.parquet",
            source_metadata={"raw_case_id": "rr-2"},
        ),
    ]


def test_validate_rr_section_mapping_reports_unmapped_labels(tmp_path: Path) -> None:
    config = build_test_config(project_root=tmp_path)
    cases = build_rr_cases()
    inventory = generate_rr_label_inventory(cases, config=config)

    report = validate_rr_section_mapping(cases, inventory, config=config)

    assert report.summary["unmapped_label_count"] == 1
    assert report.summary["unmapped_labels"] == ["99"]
    assert report.summary["coverage_percent"] == 80.0
    entry_by_key = {entry.label_key: entry for entry in report.entries}
    assert entry_by_key["0"].mapped_section == "facts"
    assert entry_by_key["99"].mapped_section is None


def test_discover_rr_label_names_reads_dataset_readme(tmp_path: Path) -> None:
    dataset_root = tmp_path / "dataset"
    dataset_root.mkdir()
    (dataset_root / "README.md").write_text(
        '\n'.join(
            [
                "<details>",
                "  <summary>List of RR labels</summary>",
                '  "Fact", "Issue", "RatioOfTheDecision"',
                "</details>",
            ]
        ),
        encoding="utf-8",
    )

    label_names, source, warnings = discover_rr_label_names(dataset_root)

    assert source == "dataset_readme"
    assert warnings == []
    assert label_names == {"0": "Fact", "1": "Issue", "2": "RatioOfTheDecision"}


def test_reconstruct_rr_sections_routes_unmapped_labels_to_other(tmp_path: Path) -> None:
    config = build_test_config(project_root=tmp_path, unmapped_label_behavior="warn_and_route_to_other")
    cases = build_rr_cases()
    inventory = generate_rr_label_inventory(cases, config=config)
    mapping_report = validate_rr_section_mapping(cases, inventory, config=config)

    result = reconstruct_rr_sections(cases, mapping_report, config=config)

    record = result.records[0]
    assert record.grouped_sections["facts"] == "Fact sentence\nIssue sentence"
    assert record.grouped_sections["reasoning"] == "Reasoning sentence"
    assert record.grouped_sections["other"] == "Fallback sentence"
    assert record.section_sentence_map["facts"] == [0, 1]
    assert record.section_sentence_map["reasoning"] == [2]
    assert record.section_sentence_map["other"] == [3]
    assert record.unmapped_labels_present is True
    assert record.unmapped_labels == [99]


def test_reconstruct_rr_sections_can_keep_unmapped_bucket(tmp_path: Path) -> None:
    config = build_test_config(project_root=tmp_path, unmapped_label_behavior="warn_and_keep_unmapped_bucket")
    cases = build_rr_cases()
    inventory = generate_rr_label_inventory(cases, config=config)
    mapping_report = validate_rr_section_mapping(cases, inventory, config=config)

    result = reconstruct_rr_sections(cases, mapping_report, config=config)

    record = result.records[0]
    assert "unmapped" in record.grouped_sections
    assert record.grouped_sections["unmapped"] == "Fallback sentence"
    assert record.section_sentence_map["unmapped"] == [3]


def test_reconstruct_rr_sections_is_deterministic_and_preserves_empty_sections(tmp_path: Path) -> None:
    config = build_test_config(project_root=tmp_path)
    cases = build_rr_cases()
    inventory = generate_rr_label_inventory(cases, config=config)
    mapping_report = validate_rr_section_mapping(cases, inventory, config=config)

    first = reconstruct_rr_sections(cases, mapping_report, config=config)
    second = reconstruct_rr_sections(cases, mapping_report, config=config)

    assert [record.to_dict() for record in first.records] == [record.to_dict() for record in second.records]
    second_case = first.records[1]
    assert second_case.grouped_sections["facts"] == ""
    assert second_case.grouped_sections["reasoning"] == ""
    assert second_case.grouped_sections["other"] == "Only none sentence"


def test_rr_mapping_and_reconstruction_reports_render(tmp_path: Path) -> None:
    config = build_test_config(project_root=tmp_path)
    cases = build_rr_cases()
    inventory = generate_rr_label_inventory(cases, config=config)
    mapping_report = validate_rr_section_mapping(cases, inventory, config=config)
    reconstruction = reconstruct_rr_sections(cases, mapping_report, config=config)

    mapping_markdown = render_rr_section_mapping_report(mapping_report)
    reconstruction_markdown = render_rr_reconstruction_report(reconstruction.report)

    assert "RR Section Mapping Report" in mapping_markdown
    assert "Coverage percent" in mapping_markdown
    assert "RR Reconstruction Report" in reconstruction_markdown
    assert "Empty section counts" in reconstruction_markdown
