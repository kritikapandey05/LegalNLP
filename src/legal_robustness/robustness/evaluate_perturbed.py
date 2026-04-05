from __future__ import annotations

import logging
from collections import defaultdict
from typing import Any

from legal_robustness.config.schema import AppConfig
from legal_robustness.prediction.models import load_prediction_model
from legal_robustness.prediction.types import BaselinePredictionRecord
from legal_robustness.robustness.datasets import (
    load_baseline_predictions,
    load_baseline_report,
    load_perturbation_manifest,
    load_perturbation_rows,
)
from legal_robustness.robustness.metrics import (
    compute_coverage_summary,
    compute_recipe_metrics,
    confusion_matrix_rows_for_perturbation,
)
from legal_robustness.robustness.types import PerturbedPredictionRecord
from legal_robustness.utils.exceptions import PredictionError


def evaluate_selected_perturbations(
    *,
    baseline_run_dir,
    config: AppConfig,
    logger: logging.Logger | None = None,
) -> tuple[dict[str, Any], list[PerturbedPredictionRecord], list[dict[str, object]]]:
    logger = logger or logging.getLogger(__name__)
    baseline_report = load_baseline_report(baseline_run_dir)
    perturbation_manifest = load_perturbation_manifest(baseline_run_dir)
    report: dict[str, Any] = {
        "task": "perturbed_evaluation",
        "baseline_run_dir": str(baseline_run_dir),
        "selected_model_variants": list(config.robustness.selected_model_variants),
        "selected_perturbation_recipes": list(config.robustness.selected_perturbation_recipes),
        "evaluation_splits": list(config.robustness.evaluation_splits),
        "model_variant_results": {},
        "reference_context_variants": {},
        "warnings": [],
    }
    all_prediction_rows: list[PerturbedPredictionRecord] = []
    all_confusion_rows: list[dict[str, object]] = []
    baseline_predictions_by_split: dict[str, list[BaselinePredictionRecord]] = {
        split: load_baseline_predictions(baseline_run_dir, split=split)
        for split in config.robustness.evaluation_splits
    }

    for model_variant in config.robustness.selected_model_variants:
        model_name, input_variant = _parse_model_variant(model_variant)
        model_payload = baseline_report["models"].get(model_name, {}).get(input_variant)
        if model_payload is None:
            raise PredictionError(
                f"Baseline run {baseline_run_dir} does not contain model/input variant {model_variant}."
            )
        model = load_prediction_model(model_payload["model_path"])
        model_variant_key = f"{model_name}::{input_variant}"
        report["model_variant_results"][model_variant_key] = {
            "model_name": model_name,
            "input_variant": input_variant,
            "reference_metrics_by_split": model_payload.get("metrics_by_split", {}),
            "recipes": {},
        }
        report["reference_context_variants"][model_variant_key] = {
            variant_name: baseline_report["models"].get(model_name, {}).get(variant_name, {}).get("metrics_by_split", {})
            for variant_name in config.robustness.reference_context_variants
            if variant_name in baseline_report["models"].get(model_name, {})
        }

        for recipe_name in config.robustness.selected_perturbation_recipes:
            recipe_rows = load_perturbation_rows(baseline_run_dir, recipe_name=recipe_name)
            recipe_rows = [row for row in recipe_rows if row.split in config.robustness.evaluation_splits]
            if not recipe_rows:
                report["warnings"].append(
                    f"No perturbation rows were available for recipe {recipe_name} in splits {config.robustness.evaluation_splits}."
                )
                continue
            coverage_summary = compute_coverage_summary(
                recipe_rows,
                high_threshold=config.robustness.high_coverage_min_fraction,
                medium_threshold=config.robustness.medium_coverage_min_fraction,
            )
            prediction_rows, metrics_by_split, confusion_rows = _predict_recipe_rows(
                recipe_rows=recipe_rows,
                model=model,
                model_name=model_name,
                input_variant=input_variant,
                baseline_predictions_by_split=baseline_predictions_by_split,
                coverage_summary=coverage_summary,
                config=config,
            )
            all_prediction_rows.extend(prediction_rows)
            all_confusion_rows.extend(confusion_rows)
            report["model_variant_results"][model_variant_key]["recipes"][recipe_name] = {
                "perturbation_manifest_entry": perturbation_manifest["recipes"].get(recipe_name, {}),
                "coverage": coverage_summary,
                "metrics_by_split": metrics_by_split,
            }
            logger.info(
                "Evaluated perturbation recipe %s for %s with coverage %s.",
                recipe_name,
                model_variant_key,
                coverage_summary["coverage_band"],
            )

    return report, all_prediction_rows, all_confusion_rows


def _predict_recipe_rows(
    *,
    recipe_rows,
    model,
    model_name: str,
    input_variant: str,
    baseline_predictions_by_split: dict[str, list[BaselinePredictionRecord]],
    coverage_summary: dict[str, Any],
    config: AppConfig,
) -> tuple[list[PerturbedPredictionRecord], dict[str, Any], list[dict[str, object]]]:
    rows_by_split: dict[str, list[PerturbedPredictionRecord]] = defaultdict(list)
    confusion_rows: list[dict[str, object]] = []
    for split_name, baseline_rows in baseline_predictions_by_split.items():
        reference_lookup = {
            row.case_id: row
            for row in baseline_rows
            if row.model_name == model_name and row.input_variant == input_variant
        }
        for recipe_row in [row for row in recipe_rows if row.split == split_name]:
            reference_row = reference_lookup.get(recipe_row.case_id)
            if reference_row is None:
                raise PredictionError(
                    f"Missing unperturbed reference prediction for case {recipe_row.case_id} "
                    f"under {model_name}/{input_variant} on split {split_name}."
                )
            probabilities = model.predict_proba(recipe_row.perturbed_text)
            predicted_label, predicted_score = max(probabilities.items(), key=lambda item: item[1])
            record = PerturbedPredictionRecord(
                case_id=recipe_row.case_id,
                split=recipe_row.split,
                subset=recipe_row.subset,
                gold_label=str(recipe_row.cjpe_label),
                model_name=model_name,
                input_variant=input_variant,
                perturbation_recipe=recipe_row.perturbation_name,
                perturbation_family=recipe_row.perturbation_family,
                prediction=predicted_label,
                prediction_score=round(float(predicted_score), 6),
                predicted_probabilities=probabilities if config.robustness.export_prediction_probabilities else {},
                target_section=recipe_row.target_section,
                target_section_was_empty=recipe_row.target_section_was_empty,
                reference_prediction=reference_row.predicted_label,
                reference_prediction_score=reference_row.predicted_score,
                prediction_flipped=(predicted_label != reference_row.predicted_label),
                effective_coverage_group=coverage_summary["coverage_band"],
                source_file=recipe_row.source_file,
                source_metadata=dict(recipe_row.source_metadata),
            )
            rows_by_split[split_name].append(record)

    metrics_by_split: dict[str, Any] = {}
    label_order = list(model.label_order)
    for split_name, prediction_rows in rows_by_split.items():
        reference_rows = [
            row
            for row in baseline_predictions_by_split[split_name]
            if row.model_name == model_name and row.input_variant == input_variant and row.case_id in {record.case_id for record in prediction_rows}
        ]
        recipe_metrics = compute_recipe_metrics(
            prediction_rows,
            reference_rows,
            label_order=label_order,
            coverage_summary=coverage_summary,
        )
        metrics_by_split[split_name] = recipe_metrics
        confusion_rows.extend(
            confusion_matrix_rows_for_perturbation(
                confusion_matrix=recipe_metrics["overall_metrics"]["confusion_matrix"],
                label_order=label_order,
                model_name=model_name,
                input_variant=input_variant,
                perturbation_recipe=prediction_rows[0].perturbation_recipe,
                split=split_name,
            )
        )
    flattened_rows = [row for rows in rows_by_split.values() for row in rows]
    return flattened_rows, metrics_by_split, confusion_rows


def _parse_model_variant(value: str) -> tuple[str, str]:
    if "::" not in value:
        raise PredictionError(
            f"Robustness model variant must use the form 'model_name::input_variant', got {value!r}."
        )
    model_name, input_variant = value.split("::", maxsplit=1)
    return model_name, input_variant
