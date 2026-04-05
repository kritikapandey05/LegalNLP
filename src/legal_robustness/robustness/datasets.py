from __future__ import annotations

import json
from pathlib import Path

from legal_robustness.prediction.types import BaselinePredictionRecord
from legal_robustness.perturbations.types import PerturbedCJPECase
from legal_robustness.utils.artifacts import read_parquet
from legal_robustness.utils.exceptions import PredictionError, PerturbationError


def resolve_baseline_run_dir(
    *,
    reports_dir: Path,
    explicit_run_dir: Path | None = None,
) -> Path:
    if explicit_run_dir is not None:
        run_dir = explicit_run_dir.resolve()
        if not run_dir.exists():
            raise PredictionError(f"Prediction-baseline artifact directory does not exist: {run_dir}")
        return run_dir

    root = (reports_dir / "prediction_baselines").resolve()
    if not root.exists():
        raise PredictionError(
            f"No prediction-baseline report directory exists at {root}. "
            "Run scripts/train_baseline.py first or pass --baseline-run-dir."
        )
    candidate_dirs = [path for path in root.iterdir() if path.is_dir()]
    if not candidate_dirs:
        raise PredictionError(
            f"No prediction-baseline runs were found under {root}. "
            "Run scripts/train_baseline.py first or pass --baseline-run-dir."
        )
    return max(candidate_dirs, key=lambda path: path.stat().st_mtime)


def load_baseline_report(run_dir: Path) -> dict:
    path = run_dir / "baseline_prediction_metrics.json"
    if not path.exists():
        raise PredictionError(f"Baseline report not found at {path}.")
    return json.loads(path.read_text(encoding="utf-8"))


def load_baseline_predictions(run_dir: Path, *, split: str) -> list[BaselinePredictionRecord]:
    path = run_dir / f"baseline_prediction_predictions_{split}.parquet"
    if not path.exists():
        raise PredictionError(f"Baseline prediction artifact not found at {path}.")
    return [_baseline_prediction_from_row(row) for row in read_parquet(path)]


def load_perturbation_manifest(run_dir: Path) -> dict:
    path = run_dir / "perturbation_manifest.json"
    if not path.exists():
        raise PerturbationError(f"Perturbation manifest not found at {path}.")
    return json.loads(path.read_text(encoding="utf-8"))


def load_perturbation_rows(
    run_dir: Path,
    *,
    recipe_name: str,
) -> list[PerturbedCJPECase]:
    path = run_dir / "cjpe_perturbation_sets" / f"{recipe_name}.parquet"
    if not path.exists():
        raise PerturbationError(f"Perturbation set for recipe {recipe_name!r} not found at {path}.")
    return [_perturbed_case_from_row(row) for row in read_parquet(path)]


def _baseline_prediction_from_row(row: dict[str, object]) -> BaselinePredictionRecord:
    return BaselinePredictionRecord(
        case_id=str(row.get("case_id", "")),
        split=str(row.get("split", "")),
        subset=row.get("subset"),
        gold_label=str(row.get("gold_label", "")),
        predicted_label=str(row.get("predicted_label", "")),
        input_variant=str(row.get("input_variant", "")),
        model_name=str(row.get("model_name", "")),
        correct=bool(row.get("correct", False)),
        predicted_score=float(row.get("predicted_score", 0.0)),
        predicted_probabilities={str(key): float(value) for key, value in dict(row.get("predicted_probabilities") or {}).items()},
        input_text_length_chars=int(row.get("input_text_length_chars", 0)),
        sections_used=[str(value) for value in list(row.get("sections_used") or [])],
        sections_omitted=[str(value) for value in list(row.get("sections_omitted") or [])],
        source_file=str(row.get("source_file", "")),
    )


def _perturbed_case_from_row(row: dict[str, object]) -> PerturbedCJPECase:
    return PerturbedCJPECase(
        case_id=str(row.get("case_id", "")),
        split=str(row.get("split", "")),
        subset=row.get("subset"),
        cjpe_label=row.get("cjpe_label"),
        perturbation_name=str(row.get("perturbation_name", "")),
        perturbation_family=str(row.get("perturbation_family", "")),
        base_input_variant=str(row.get("base_input_variant", "")),
        perturbed_text=str(row.get("perturbed_text", "")),
        target_section=row.get("target_section"),
        sections_kept=[str(value) for value in list(row.get("sections_kept") or [])],
        sections_dropped=[str(value) for value in list(row.get("sections_dropped") or [])],
        sections_masked=[str(value) for value in list(row.get("sections_masked") or [])],
        section_order=[str(value) for value in list(row.get("section_order") or [])],
        target_section_was_empty=bool(row.get("target_section_was_empty", False)),
        original_text_length_chars=int(row.get("original_text_length_chars", 0)),
        perturbed_text_length_chars=int(row.get("perturbed_text_length_chars", 0)),
        grouped_sections={str(key): str(value) for key, value in dict(row.get("grouped_sections") or {}).items()},
        source_file=str(row.get("source_file", "")),
        source_metadata=dict(row.get("source_metadata") or {}),
    )
