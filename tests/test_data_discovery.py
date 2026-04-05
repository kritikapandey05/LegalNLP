from __future__ import annotations

from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from legal_robustness.data.discovery import detect_split_and_subset, discover_task_files, parse_dataset_filename


def write_parquet(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pylist(rows)
    pq.write_table(table, path)


def test_parse_dataset_filename_and_split_detection() -> None:
    dataset_name, shard_index, shard_count = parse_dataset_filename(Path("single_train-00000-of-00001.parquet"))
    assert dataset_name == "single_train"
    assert shard_index == 0
    assert shard_count == 1
    assert detect_split_and_subset("single_train") == ("train", "single")
    assert detect_split_and_subset("IT_train") == ("train", "IT")
    assert detect_split_and_subset("expert") == ("expert", None)
    assert detect_split_and_subset("test") == ("test", None)


def test_discover_task_files_handles_missing_or_empty_task_dirs(tmp_path: Path) -> None:
    dataset_root = tmp_path / "dataset"
    dataset_root.mkdir()
    manifest = discover_task_files(dataset_root=dataset_root, tasks=["cjpe", "rr"])

    assert manifest.discovered_files == []
    assert len(manifest.warnings) == 2


def test_discover_task_files_reads_manifest_metadata(tmp_path: Path) -> None:
    dataset_root = tmp_path / "dataset"
    write_parquet(
        dataset_root / "cjpe" / "single_train-00000-of-00001.parquet",
        [{"id": "case-1", "text": "Example text", "label": 1}],
    )

    manifest = discover_task_files(dataset_root=dataset_root, tasks=["cjpe"])

    assert len(manifest.discovered_files) == 1
    file_info = manifest.discovered_files[0]
    assert file_info.task == "cjpe"
    assert file_info.dataset_name == "single_train"
    assert file_info.split == "train"
    assert file_info.subset == "single"
    assert file_info.row_count == 1
    assert set(file_info.schema) == {"id", "text", "label"}
