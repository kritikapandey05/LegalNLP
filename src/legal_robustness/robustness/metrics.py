from __future__ import annotations

from collections import Counter
from statistics import mean
from typing import Any

from legal_robustness.prediction.types import BaselinePredictionRecord
from legal_robustness.robustness.types import PerturbedPredictionRecord
from legal_robustness.section_transfer.diagnostics import build_classification_metrics


def compute_coverage_summary(
    rows,
    *,
    high_threshold: float,
    medium_threshold: float,
) -> dict[str, Any]:
    total_examples = len(rows)
    target_section = rows[0].target_section if rows else None
    empty_target_count = sum(1 for row in rows if row.target_section_was_empty)
    if total_examples == 0:
        coverage = 0.0
    elif target_section is None:
        coverage = 1.0
    else:
        coverage = (total_examples - empty_target_count) / total_examples

    if coverage >= high_threshold:
        coverage_band = "high_coverage"
    elif coverage >= medium_threshold:
        coverage_band = "medium_coverage"
    else:
        coverage_band = "low_coverage"

    if target_section is None:
        note = "Untargeted perturbation; coverage is treated as full because every example is directly transformed."
    elif coverage_band == "low_coverage":
        note = "Low effective coverage because most cases do not contain a non-empty target pseudo-section."
    elif coverage_band == "medium_coverage":
        note = "Medium effective coverage; interpret targeted effects with some caution."
    else:
        note = "High effective coverage; this perturbation is suitable for the first mainline robustness comparison."

    return {
        "total_examples": total_examples,
        "target_section": target_section,
        "empty_target_count": empty_target_count,
        "non_empty_target_count": total_examples - empty_target_count,
        "effective_non_empty_coverage": round(coverage, 6),
        "coverage_band": coverage_band,
        "recommended_for_future_experiments": coverage_band != "low_coverage",
        "note": note,
    }


def compute_recipe_metrics(
    perturbed_rows: list[PerturbedPredictionRecord],
    reference_rows: list[BaselinePredictionRecord],
    *,
    label_order: list[str],
    coverage_summary: dict[str, Any],
) -> dict[str, Any]:
    perturbed_metrics = _metrics_from_perturbed_rows(perturbed_rows, label_order=label_order)
    reference_metrics = _metrics_from_reference_rows(reference_rows, label_order=label_order)
    output = {
        "coverage": coverage_summary,
        "overall_metrics": {
            **perturbed_metrics,
            "accuracy_delta_vs_reference": round(
                perturbed_metrics["accuracy"] - reference_metrics["accuracy"], 6
            ),
            "macro_f1_delta_vs_reference": round(
                perturbed_metrics["macro_f1"] - reference_metrics["macro_f1"], 6
            ),
            "reference_accuracy": reference_metrics["accuracy"],
            "reference_macro_f1": reference_metrics["macro_f1"],
        },
    }

    if coverage_summary["target_section"] is not None and coverage_summary["non_empty_target_count"] > 0:
        non_empty_rows = [row for row in perturbed_rows if not row.target_section_was_empty]
        non_empty_case_ids = {row.case_id for row in non_empty_rows}
        non_empty_reference_rows = [row for row in reference_rows if row.case_id in non_empty_case_ids]
        non_empty_metrics = _metrics_from_perturbed_rows(non_empty_rows, label_order=label_order)
        non_empty_reference_metrics = _metrics_from_reference_rows(
            non_empty_reference_rows,
            label_order=label_order,
        )
        output["non_empty_target_metrics"] = {
            **non_empty_metrics,
            "accuracy_delta_vs_reference": round(
                non_empty_metrics["accuracy"] - non_empty_reference_metrics["accuracy"], 6
            ),
            "macro_f1_delta_vs_reference": round(
                non_empty_metrics["macro_f1"] - non_empty_reference_metrics["macro_f1"], 6
            ),
            "reference_accuracy": non_empty_reference_metrics["accuracy"],
            "reference_macro_f1": non_empty_reference_metrics["macro_f1"],
        }

    return output


def confusion_matrix_rows_for_perturbation(
    *,
    confusion_matrix: dict[str, dict[str, int]],
    label_order: list[str],
    model_name: str,
    input_variant: str,
    perturbation_recipe: str,
    split: str,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for actual_label in label_order:
        row: dict[str, object] = {
            "model_name": model_name,
            "input_variant": input_variant,
            "perturbation_recipe": perturbation_recipe,
            "split": split,
            "actual_label": actual_label,
        }
        for predicted_label in label_order:
            row[predicted_label] = confusion_matrix.get(actual_label, {}).get(predicted_label, 0)
        rows.append(row)
    return rows


def _metrics_from_perturbed_rows(
    rows: list[PerturbedPredictionRecord],
    *,
    label_order: list[str],
) -> dict[str, Any]:
    gold_labels = [row.gold_label for row in rows]
    predicted_labels = [row.prediction for row in rows]
    metrics = build_classification_metrics(gold_labels, predicted_labels, label_order=label_order)
    metrics["case_count"] = len(rows)
    metrics["gold_label_distribution"] = dict(sorted(Counter(gold_labels).items()))
    metrics["predicted_label_distribution"] = dict(sorted(Counter(predicted_labels).items()))
    metrics["average_prediction_score"] = round(
        mean(row.prediction_score for row in rows),
        6,
    ) if rows else 0.0
    metrics["flip_rate"] = round(
        sum(1 for row in rows if row.prediction_flipped) / len(rows),
        6,
    ) if rows else 0.0
    metrics["non_empty_target_flip_rate"] = round(
        sum(1 for row in rows if row.prediction_flipped and not row.target_section_was_empty)
        / max(sum(1 for row in rows if not row.target_section_was_empty), 1),
        6,
    ) if rows else 0.0
    return metrics


def _metrics_from_reference_rows(
    rows: list[BaselinePredictionRecord],
    *,
    label_order: list[str],
) -> dict[str, Any]:
    gold_labels = [row.gold_label for row in rows]
    predicted_labels = [row.predicted_label for row in rows]
    metrics = build_classification_metrics(gold_labels, predicted_labels, label_order=label_order)
    metrics["case_count"] = len(rows)
    return metrics
