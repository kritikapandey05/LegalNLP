from __future__ import annotations

from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from legal_robustness.config.schema import AppConfig, DataConfig, LoggingConfig, OutputConfig, RuntimeConfig
from legal_robustness.data.discovery import discover_task_files
from legal_robustness.data.loaders import load_cjpe_raw_cases, load_rr_raw_cases


def write_parquet(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pylist(rows)
    pq.write_table(table, path)


def build_test_config(project_root: Path, dataset_root: Path, max_malformed_row_fraction: float = 0.5) -> AppConfig:
    return AppConfig(
        project_name="test_project",
        project_root=project_root,
        data=DataConfig(
            dataset_root=dataset_root,
            loader_batch_size=2,
            max_malformed_row_fraction=max_malformed_row_fraction,
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


def test_load_cjpe_raw_cases_preserves_split_and_expert_annotations(tmp_path: Path) -> None:
    dataset_root = tmp_path / "dataset"
    write_parquet(
        dataset_root / "cjpe" / "single_train-00000-of-00001.parquet",
        [
            {
                "id": "case-1",
                "text": "This is a CJPE sample.",
                "label": 1,
                "expert_1": {"label": 1, "rank1": ["reason-a"]},
            },
            {
                "id": "case-2",
                "text": "This is another CJPE sample.",
                "label": 0,
                "expert_1": None,
            },
        ],
    )
    manifest = discover_task_files(dataset_root=dataset_root, tasks=["cjpe"])
    config = build_test_config(project_root=tmp_path, dataset_root=dataset_root)

    result = load_cjpe_raw_cases(manifest.discovered_files, config=config)

    assert result.task == "cjpe"
    assert result.records_emitted == 2
    assert result.valid_records == 2
    assert result.counts_by_split == {"train": 2}
    assert result.records[0].subset == "single"
    assert result.records[0].expert_annotations_raw["expert_1"]["label"] == 1


def test_load_rr_raw_cases_handles_aligned_rows(tmp_path: Path) -> None:
    dataset_root = tmp_path / "dataset"
    write_parquet(
        dataset_root / "rr" / "IT_train-00000-of-00001.parquet",
        [
            {
                "id": "rr-1",
                "text": ["Sentence one.", "Sentence two."],
                "labels": [1, 2],
                "expert_1": {"primary": [1, 2]},
            }
        ],
    )
    manifest = discover_task_files(dataset_root=dataset_root, tasks=["rr"])
    config = build_test_config(project_root=tmp_path, dataset_root=dataset_root)

    result = load_rr_raw_cases(manifest.discovered_files, config=config)

    assert result.task == "rr"
    assert result.records_emitted == 1
    assert result.valid_records == 1
    assert result.malformed_rows == []
    assert result.counts_by_split == {"train": 1}
    assert result.records[0].subset == "IT"
    assert result.records[0].alignment_ok is True


def test_load_rr_raw_cases_records_alignment_mismatches_without_crashing(tmp_path: Path) -> None:
    dataset_root = tmp_path / "dataset"
    write_parquet(
        dataset_root / "rr" / "CL_train-00000-of-00001.parquet",
        [
            {
                "id": "rr-good",
                "text": ["Sentence one.", "Sentence two."],
                "labels": [1, 2],
                "expert_1": {"primary": [1, 2]},
            },
            {
                "id": "rr-bad",
                "text": ["Sentence one.", "Sentence two."],
                "labels": [1],
                "expert_1": {"primary": [1]},
            },
        ],
    )
    manifest = discover_task_files(dataset_root=dataset_root, tasks=["rr"])
    config = build_test_config(project_root=tmp_path, dataset_root=dataset_root, max_malformed_row_fraction=0.75)

    result = load_rr_raw_cases(manifest.discovered_files, config=config)

    assert result.records_emitted == 2
    assert result.valid_records == 1
    assert len(result.malformed_rows) == 1
    assert result.records[1].alignment_ok is False
    assert "mismatch" in result.malformed_rows[0].issue.lower()


def test_raw_loaders_handle_empty_discovery_results(tmp_path: Path) -> None:
    dataset_root = tmp_path / "dataset"
    dataset_root.mkdir()
    config = build_test_config(project_root=tmp_path, dataset_root=dataset_root)

    cjpe_result = load_cjpe_raw_cases([], config=config)
    rr_result = load_rr_raw_cases([], config=config)

    assert cjpe_result.records == []
    assert rr_result.records == []
    assert cjpe_result.warnings
    assert rr_result.warnings
