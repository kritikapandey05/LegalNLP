from __future__ import annotations

from pathlib import Path

import pytest

from legal_robustness.utils.exceptions import DatasetPathError
from legal_robustness.utils.paths import validate_dataset_root


def test_validate_dataset_root_rejects_unset_value() -> None:
    with pytest.raises(DatasetPathError, match="not configured"):
        validate_dataset_root(None)


def test_validate_dataset_root_rejects_missing_directory(tmp_path: Path) -> None:
    missing = tmp_path / "missing"
    with pytest.raises(DatasetPathError, match="does not exist"):
        validate_dataset_root(missing)


def test_validate_dataset_root_accepts_existing_directory(tmp_path: Path) -> None:
    dataset_root = tmp_path / "dataset"
    dataset_root.mkdir()
    assert validate_dataset_root(dataset_root) == dataset_root.resolve()
