from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from legal_robustness.utils.paths import ensure_directory


def create_stage_output_dir(base_dir: Path, stage_name: str, run_name: str | None = None) -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    suffix = run_name.strip().replace(" ", "_") if run_name else "default"
    output_dir = ensure_directory(base_dir / f"{timestamp}_{stage_name}_{suffix}")
    return output_dir


def write_json(path: Path, payload: Any) -> None:
    ensure_directory(path.parent)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    ensure_directory(path.parent)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False))
            handle.write("\n")


def write_yaml(path: Path, payload: Any) -> None:
    ensure_directory(path.parent)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False, allow_unicode=True)


def write_text(path: Path, payload: str) -> None:
    ensure_directory(path.parent)
    path.write_text(payload, encoding="utf-8")


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            rows.append(json.loads(stripped))
    return rows


def write_parquet(path: Path, rows: list[dict[str, Any]]) -> None:
    ensure_directory(path.parent)
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise RuntimeError("pyarrow is required to export Parquet artifacts.") from exc

    table = pa.Table.from_pylist(rows)
    pq.write_table(table, path)


def read_parquet(
    path: Path,
    *,
    columns: list[str] | None = None,
    filters: Any = None,
) -> list[dict[str, Any]]:
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise RuntimeError("pyarrow is required to read Parquet artifacts.") from exc

    table = pq.read_table(path, columns=columns, filters=filters)
    return table.to_pylist()
