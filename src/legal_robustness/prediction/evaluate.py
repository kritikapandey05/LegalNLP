from __future__ import annotations

from collections import Counter
from statistics import mean

from legal_robustness.prediction.types import BaselinePredictionRecord, PredictionInputExample
from legal_robustness.section_transfer.diagnostics import build_classification_metrics
from legal_robustness.utils.exceptions import PredictionError


def evaluate_prediction_examples(
    *,
    examples: list[PredictionInputExample],
    model,
    model_name: str,
    input_variant: str,
) -> tuple[dict[str, object], list[BaselinePredictionRecord]]:
    if not examples:
        raise PredictionError(f"Cannot evaluate model {model_name} on empty example list for {input_variant}.")

    gold_labels: list[str] = []
    predicted_labels: list[str] = []
    prediction_rows: list[BaselinePredictionRecord] = []
    input_lengths: list[int] = []

    for example in examples:
        gold_label = str(example.label)
        probabilities = model.predict_proba(example.input_text)
        predicted_label, predicted_score = max(probabilities.items(), key=lambda item: item[1])
        gold_labels.append(gold_label)
        predicted_labels.append(predicted_label)
        input_lengths.append(example.input_text_length_chars)
        prediction_rows.append(
            BaselinePredictionRecord(
                case_id=example.case_id,
                split=example.split,
                subset=example.subset,
                gold_label=gold_label,
                predicted_label=predicted_label,
                input_variant=input_variant,
                model_name=model_name,
                correct=(gold_label == predicted_label),
                predicted_score=round(float(predicted_score), 6),
                predicted_probabilities=probabilities,
                input_text_length_chars=example.input_text_length_chars,
                sections_used=list(example.sections_used),
                sections_omitted=list(example.sections_omitted),
                source_file=example.source_file,
            )
        )

    metrics = build_classification_metrics(
        gold_labels,
        predicted_labels,
        label_order=list(model.label_order),
    )
    metrics["case_count"] = len(examples)
    metrics["gold_label_distribution"] = dict(sorted(Counter(gold_labels).items()))
    metrics["predicted_label_distribution"] = dict(sorted(Counter(predicted_labels).items()))
    metrics["average_input_length_chars"] = round(mean(input_lengths), 3) if input_lengths else 0.0
    return metrics, prediction_rows


def confusion_matrix_rows_for_variant(
    *,
    confusion_matrix: dict[str, dict[str, int]],
    label_order: list[str],
    model_name: str,
    input_variant: str,
    split: str,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for actual_label in label_order:
        row: dict[str, object] = {
            "model_name": model_name,
            "input_variant": input_variant,
            "split": split,
            "actual_label": actual_label,
        }
        for predicted_label in label_order:
            row[predicted_label] = confusion_matrix.get(actual_label, {}).get(predicted_label, 0)
        rows.append(row)
    return rows
