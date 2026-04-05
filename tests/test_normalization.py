from __future__ import annotations

from pathlib import Path

import pytest

from legal_robustness.config.schema import AppConfig, DataConfig, LoggingConfig, OutputConfig, RuntimeConfig
from legal_robustness.data.label_inventory import generate_rr_label_inventory
from legal_robustness.data.normalize import normalize_cjpe_cases, normalize_rr_cases
from legal_robustness.data.raw_types import RawCJPECase, RawRRCase
from legal_robustness.utils.exceptions import DatasetNormalizationError


def build_test_config(project_root: Path, fail_on_duplicate_ids: bool = False, collapse_text_whitespace: bool = False) -> AppConfig:
    return AppConfig(
        project_name="test_project",
        project_root=project_root,
        data=DataConfig(
            dataset_root=project_root / "dataset",
            fail_on_duplicate_ids=fail_on_duplicate_ids,
            collapse_text_whitespace=collapse_text_whitespace,
            rr_label_normalization="preserve",
        ),
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
    )


def test_normalize_cjpe_cases_generates_text_stats_and_preserves_provenance(tmp_path: Path) -> None:
    config = build_test_config(project_root=tmp_path, collapse_text_whitespace=True)
    raw_cases = [
        RawCJPECase(
            task="cjpe",
            split="train",
            dataset_name="single_train",
            subset="single",
            source_file="cjpe\\single_train-00000-of-00001.parquet",
            case_id=" case-1 ",
            raw_text="This   is a CJPE sample.\n",
            label=1,
            expert_annotations_raw={"expert_1": {"label": 1}, "expert_2": None},
            source_metadata={"source_row_index": 0},
        ),
        RawCJPECase(
            task="cjpe",
            split="dev",
            dataset_name="single_dev",
            subset="single",
            source_file="cjpe\\single_dev-00000-of-00001.parquet",
            case_id="case-2",
            raw_text="Second sample text.",
            label=0,
            expert_annotations_raw={"expert_1": None, "expert_2": None},
            source_metadata={"source_row_index": 1},
        ),
    ]

    result = normalize_cjpe_cases(raw_cases, config=config)

    assert len(result.records) == 2
    assert result.records[0].case_id == "case-1"
    assert result.records[0].raw_text == "This is a CJPE sample."
    assert result.records[0].text_length_tokens_approx == 5
    assert result.report["counts_by_split"] == {"dev": 1, "train": 1}
    assert result.report["label_counts_by_split"]["train"] == {"1": 1}
    assert result.report["missing_optional_fields"]["expert_2"] == 2
    assert result.report["duplicate_case_id_count"] == 0


def test_normalize_rr_cases_generates_structural_stats(tmp_path: Path) -> None:
    config = build_test_config(project_root=tmp_path)
    raw_cases = [
        RawRRCase(
            task="rr",
            split="train",
            dataset_name="IT_train",
            subset="IT",
            source_file="rr\\IT_train-00000-of-00001.parquet",
            case_id=" rr-1 ",
            sentences=["Sentence one.", "", "Sentence three."],
            rr_labels=[0, 1, None],
            alignment_ok=False,
            expert_annotations_raw={"expert_1": {"primary": [0, 1, 2]}},
            source_metadata={"source_row_index": 0},
        )
    ]

    result = normalize_rr_cases(raw_cases, config=config)

    assert len(result.records) == 1
    record = result.records[0]
    assert record.case_id == "rr-1"
    assert record.num_sentences == 3
    assert record.num_labels == 3
    assert record.empty_sentence_count == 1
    assert record.empty_label_count == 1
    assert record.alignment_ok is False
    assert result.report["non_aligned_cases"] == 1
    assert result.report["cases_with_empty_sentences"] == 1
    assert result.report["cases_with_empty_labels"] == 1


def test_normalization_duplicate_id_handling_can_fail(tmp_path: Path) -> None:
    config = build_test_config(project_root=tmp_path, fail_on_duplicate_ids=True)
    raw_cases = [
        RawCJPECase(
            task="cjpe",
            split="train",
            dataset_name="single_train",
            subset="single",
            source_file="cjpe\\single_train-00000-of-00001.parquet",
            case_id="dup-case",
            raw_text="First",
            label=1,
            source_metadata={},
        ),
        RawCJPECase(
            task="cjpe",
            split="dev",
            dataset_name="single_dev",
            subset="single",
            source_file="cjpe\\single_dev-00000-of-00001.parquet",
            case_id="dup-case",
            raw_text="Second",
            label=0,
            source_metadata={},
        ),
    ]

    with pytest.raises(DatasetNormalizationError, match="duplicate case ids"):
        normalize_cjpe_cases(raw_cases, config=config)


def test_generate_rr_label_inventory_tracks_counts_and_types(tmp_path: Path) -> None:
    config = build_test_config(project_root=tmp_path)
    rr_result = normalize_rr_cases(
        [
            RawRRCase(
                task="rr",
                split="train",
                dataset_name="IT_train",
                subset="IT",
                source_file="rr\\IT_train-00000-of-00001.parquet",
                case_id="rr-1",
                sentences=["A", "B"],
                rr_labels=[0, 1],
                alignment_ok=True,
                source_metadata={},
            ),
            RawRRCase(
                task="rr",
                split="dev",
                dataset_name="IT_dev",
                subset="IT",
                source_file="rr\\IT_dev-00000-of-00001.parquet",
                case_id="rr-2",
                sentences=["C"],
                rr_labels=[1],
                alignment_ok=True,
                source_metadata={},
            ),
        ],
        config=config,
    )

    inventory = generate_rr_label_inventory(rr_result.records, config=config)

    assert inventory.summary["total_unique_labels"] == 2
    assert inventory.summary["label_types"] == {"int": 2}
    entry_by_key = {entry.label_key: entry for entry in inventory.entries}
    assert entry_by_key["0"].count == 1
    assert entry_by_key["1"].count == 2
    assert entry_by_key["1"].counts_by_split == {"dev": 1, "train": 1}
