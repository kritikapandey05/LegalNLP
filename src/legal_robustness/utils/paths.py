from __future__ import annotations

from pathlib import Path

from legal_robustness.utils.exceptions import DatasetPathError


def resolve_path(path_value: str | Path, base_dir: Path) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path.resolve()
    return (base_dir / path).resolve()


def resolve_optional_path(path_value: str | Path | None, base_dir: Path) -> Path | None:
    if path_value is None:
        return None
    text = str(path_value).strip()
    if not text:
        return None
    return resolve_path(text, base_dir)


def ensure_directory(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def validate_existing_directory(path: Path | None, description: str) -> Path:
    if path is None:
        raise DatasetPathError(
            f"{description} is not configured. Provide a value via --dataset-root, config, or environment variable."
        )
    resolved = path.resolve()
    if not resolved.exists():
        raise DatasetPathError(f"{description} does not exist: {resolved}")
    if not resolved.is_dir():
        raise DatasetPathError(f"{description} is not a directory: {resolved}")
    return resolved


def validate_dataset_root(dataset_root: Path | None) -> Path:
    return validate_existing_directory(dataset_root, "Dataset root")
