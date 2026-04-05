from __future__ import annotations

import logging
from pathlib import Path

import pytest

from legal_robustness.config.schema import AppConfig, DataConfig, LoggingConfig, OutputConfig, RuntimeConfig
from legal_robustness.data.inspection import DatasetInspector
from legal_robustness.utils.exceptions import DatasetPathError


def build_test_config(project_root: Path, dataset_root: Path | None) -> AppConfig:
    return AppConfig(
        project_name="test_project",
        project_root=project_root,
        data=DataConfig(dataset_root=dataset_root),
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


def test_dataset_inspector_handles_empty_directory(tmp_path: Path) -> None:
    dataset_root = tmp_path / "dataset"
    dataset_root.mkdir()
    config = build_test_config(project_root=tmp_path, dataset_root=dataset_root)

    report = DatasetInspector(config=config, logger=logging.getLogger("test")).inspect()

    assert report.total_files == 0
    assert report.supported_file_count == 0
    assert report.file_inspections == []
    assert report.directory_summaries[0].relative_path == "."


def test_dataset_inspector_rejects_invalid_dataset_path(tmp_path: Path) -> None:
    config = build_test_config(project_root=tmp_path, dataset_root=tmp_path / "missing")

    with pytest.raises(DatasetPathError, match="does not exist"):
        DatasetInspector(config=config, logger=logging.getLogger("test")).inspect()


def test_dataset_inspector_detects_task_directories_not_root_files(tmp_path: Path) -> None:
    dataset_root = tmp_path / "dataset"
    task_dir = dataset_root / "cjpe"
    task_dir.mkdir(parents=True)
    (dataset_root / "README.md").write_text("root note", encoding="utf-8")
    (task_dir / "sample.json").write_text('{"id": "case-1", "text": "example"}', encoding="utf-8")
    config = build_test_config(project_root=tmp_path, dataset_root=dataset_root)

    report = DatasetInspector(config=config, logger=logging.getLogger("test")).inspect()

    assert report.tasks_detected == ["cjpe"]
