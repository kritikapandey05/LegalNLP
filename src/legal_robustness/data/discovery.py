from __future__ import annotations

import logging
import re
from pathlib import Path

from legal_robustness.data.raw_types import DatasetFileShard, DiscoveredDatasetManifest
from legal_robustness.utils.exceptions import DatasetDiscoveryError
from legal_robustness.utils.paths import validate_dataset_root

PARQUET_SUFFIX = ".parquet"
SHARD_PATTERN = re.compile(r"^(?P<dataset_name>.+)-(?P<shard_index>\d{5})-of-(?P<shard_count>\d{5})$")
SPLIT_SUFFIX_PATTERN = re.compile(r"^(?P<subset>.+)_(?P<split>train|dev|test|validation|valid|val)$", re.IGNORECASE)
DIRECT_SPLIT_NAMES = {"train", "dev", "test", "validation", "valid", "val", "expert"}


def detect_split_and_subset(dataset_name: str) -> tuple[str, str | None]:
    direct_name = dataset_name.strip()
    if direct_name.lower() in DIRECT_SPLIT_NAMES:
        return direct_name.lower(), None

    match = SPLIT_SUFFIX_PATTERN.match(direct_name)
    if match is None:
        return direct_name.lower(), None

    return match.group("split").lower(), match.group("subset")


def parse_dataset_filename(file_path: Path) -> tuple[str, int | None, int | None]:
    match = SHARD_PATTERN.match(file_path.stem)
    if match is None:
        return file_path.stem, None, None
    return (
        match.group("dataset_name"),
        int(match.group("shard_index")),
        int(match.group("shard_count")),
    )


def inspect_parquet_metadata(file_path: Path) -> tuple[int | None, dict[str, str]]:
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise DatasetDiscoveryError(
            "pyarrow is required for dataset discovery on Parquet files. Install project dependencies first."
        ) from exc

    try:
        parquet_file = pq.ParquetFile(file_path)
    except Exception as exc:
        raise DatasetDiscoveryError(f"Failed to inspect Parquet metadata for {file_path}: {exc}") from exc

    row_count = parquet_file.metadata.num_rows if parquet_file.metadata is not None else None
    schema = {field.name: str(field.type) for field in parquet_file.schema_arrow}
    return row_count, schema


def discover_task_files(
    dataset_root: Path,
    tasks: tuple[str, ...] | list[str],
    logger: logging.Logger | None = None,
) -> DiscoveredDatasetManifest:
    dataset_root = validate_dataset_root(dataset_root)
    logger = logger or logging.getLogger(__name__)

    discovered_files: list[DatasetFileShard] = []
    warnings: list[str] = []
    tasks_requested = [task.lower() for task in tasks]

    for task in tasks_requested:
        task_dir = dataset_root / task
        if not task_dir.exists():
            warnings.append(f"Requested task directory was not found: {task_dir}")
            logger.warning("Requested task directory was not found: %s", task_dir)
            continue
        if not task_dir.is_dir():
            warnings.append(f"Requested task path is not a directory: {task_dir}")
            logger.warning("Requested task path is not a directory: %s", task_dir)
            continue

        parquet_files = sorted(path for path in task_dir.rglob(f"*{PARQUET_SUFFIX}") if path.is_file())
        if not parquet_files:
            warnings.append(f"No Parquet files were found for task '{task}' in {task_dir}")
            logger.warning("No Parquet files were found for task '%s' in %s", task, task_dir)
            continue

        for file_path in parquet_files:
            dataset_name, shard_index, shard_count = parse_dataset_filename(file_path)
            split, subset = detect_split_and_subset(dataset_name)
            row_count, schema = inspect_parquet_metadata(file_path)
            discovered_files.append(
                DatasetFileShard(
                    task=task,
                    dataset_name=dataset_name,
                    split=split,
                    subset=subset,
                    relative_path=str(file_path.relative_to(dataset_root)),
                    absolute_path=file_path.resolve(),
                    shard_index=shard_index,
                    shard_count=shard_count,
                    row_count=row_count,
                    schema=schema,
                )
            )

    discovered_files.sort(key=lambda item: (item.task, item.split, item.dataset_name, item.relative_path))
    return DiscoveredDatasetManifest(
        dataset_root=str(dataset_root),
        tasks_requested=tasks_requested,
        discovered_files=discovered_files,
        warnings=warnings,
    )


def group_files_by_task(files: list[DatasetFileShard]) -> dict[str, list[DatasetFileShard]]:
    grouped: dict[str, list[DatasetFileShard]] = {}
    for file_info in files:
        grouped.setdefault(file_info.task, []).append(file_info)
    return grouped


def group_files_by_task_and_split(files: list[DatasetFileShard]) -> dict[str, dict[str, list[DatasetFileShard]]]:
    grouped: dict[str, dict[str, list[DatasetFileShard]]] = {}
    for file_info in files:
        grouped.setdefault(file_info.task, {}).setdefault(file_info.split, []).append(file_info)
    return grouped
