from __future__ import annotations

from collections import Counter
from statistics import mean, median
from typing import Any


BROAD_SECTION_ORDER = ("facts", "precedents", "reasoning", "conclusion", "other")


def describe_numeric_series(values: list[int]) -> dict[str, int | float | None]:
    if not values:
        return {"count": 0, "min": None, "max": None, "mean": None, "median": None}
    return {
        "count": len(values),
        "min": min(values),
        "max": max(values),
        "mean": round(mean(values), 3),
        "median": median(values),
    }


def build_classification_metrics(
    true_labels: list[str],
    predicted_labels: list[str],
    *,
    label_order: list[str],
) -> dict[str, Any]:
    confusion_matrix = {
        actual: {predicted: 0 for predicted in label_order}
        for actual in label_order
    }
    for actual, predicted in zip(true_labels, predicted_labels, strict=False):
        confusion_matrix.setdefault(actual, {label: 0 for label in label_order})
        confusion_matrix[actual].setdefault(predicted, 0)
        confusion_matrix[actual][predicted] += 1

    correct = sum(
        confusion_matrix.get(label, {}).get(label, 0)
        for label in label_order
    )
    total = len(true_labels)
    accuracy = round(correct / total, 6) if total else 0.0
    per_class: dict[str, dict[str, float | int]] = {}
    f1_scores: list[float] = []
    for label in label_order:
        tp = confusion_matrix[label][label]
        fp = sum(confusion_matrix[other][label] for other in label_order if other != label)
        fn = sum(confusion_matrix[label][other] for other in label_order if other != label)
        support = sum(confusion_matrix[label].values())
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
        f1_scores.append(f1)
        per_class[label] = {
            "precision": round(precision, 6),
            "recall": round(recall, 6),
            "f1": round(f1, 6),
            "support": support,
        }

    return {
        "label_order": label_order,
        "accuracy": accuracy,
        "macro_f1": round(mean(f1_scores), 6) if f1_scores else 0.0,
        "per_class": per_class,
        "confusion_matrix": confusion_matrix,
    }


def confusion_matrix_csv_rows(confusion_matrix: dict[str, dict[str, int]], *, label_order: list[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for actual in label_order:
        row: dict[str, Any] = {"actual_label": actual}
        for predicted in label_order:
            row[predicted] = confusion_matrix.get(actual, {}).get(predicted, 0)
        rows.append(row)
    return rows


def render_rr_sentence_supervision_summary(report: dict[str, Any]) -> str:
    lines = [
        "# RR Sentence Supervision Summary",
        "",
        f"- Total supervision rows: `{report['total_sentences']}`",
        f"- Cases represented: `{report['cases_represented']}`",
        f"- Counts by split: `{report['counts_by_split']}`",
        f"- Broad section distribution: `{report['broad_section_distribution']}`",
        f"- Fine label distribution: `{report['fine_label_distribution']}`",
        f"- Sentence length chars: `{report['sentence_length_chars']}`",
        f"- Sentence length tokens: `{report['sentence_length_tokens_approx']}`",
        f"- Broad-section imbalance ratio: `{report['broad_section_imbalance_ratio']}`",
        "",
    ]
    if report.get("warnings"):
        lines.extend(["## Warnings", ""])
        for warning in report["warnings"]:
            lines.append(f"- {warning}")
        lines.append("")
    return "\n".join(lines).strip() + "\n"


def render_cjpe_sentence_segmentation_report(report: dict[str, Any]) -> str:
    lines = [
        "# CJPE Sentence Segmentation Report",
        "",
        f"- Segmentation method: `{report['segmentation_method']}`",
        f"- Total cases: `{report['total_cases']}`",
        f"- Total sentences: `{report['total_sentences']}`",
        f"- Counts by split: `{report['counts_by_split']}`",
        f"- Sentence count per case: `{report['sentence_count_per_case']}`",
        f"- Sentence length chars: `{report['sentence_length_chars']}`",
        f"- Short sentence ratio: `{report['short_sentence_ratio']}`",
        f"- Long sentence ratio: `{report['long_sentence_ratio']}`",
        "",
        "## Segmentation Anomalies",
        "",
        f"- Cases with zero sentences: `{report['cases_with_zero_sentences']}`",
        f"- Cases with high short-sentence ratios: `{report['high_short_sentence_ratio_cases']}`",
        f"- Cases with high long-sentence ratios: `{report['high_long_sentence_ratio_cases']}`",
        "",
    ]
    if report.get("warnings"):
        lines.extend(["## Warnings", ""])
        for warning in report["warnings"]:
            lines.append(f"- {warning}")
        lines.append("")
    return "\n".join(lines).strip() + "\n"


def render_rr_section_tagger_metrics(report: dict[str, Any]) -> str:
    lines = [
        "# RR Section Tagger Metrics",
        "",
        f"- Classifier type: `{report['classifier_type']}`",
        f"- Label mode: `{report['label_mode']}`",
        f"- Label order: `{report['label_order']}`",
        f"- Vocabulary size: `{report['vocabulary_size']}`",
        f"- Train rows: `{report['split_counts'].get('train', 0)}`",
        f"- Dev rows: `{report['split_counts'].get('dev', 0)}`",
        f"- Test rows: `{report['split_counts'].get('test', 0)}`",
        f"- Evaluation split for confusion matrix: `{report['confusion_matrix_split']}`",
        "",
        "## Metrics by Split",
        "",
    ]
    for split_name, metrics in report["metrics_by_split"].items():
        lines.extend(
            [
                f"### {split_name.upper()}",
                "",
                f"- Accuracy: `{metrics['accuracy']}`",
                f"- Macro F1: `{metrics['macro_f1']}`",
                f"- Per-class F1: `{ {label: values['f1'] for label, values in metrics['per_class'].items()} }`",
                "",
            ]
        )
    evidence = report.get("feature_settings", {})
    lines.extend(
        [
            "## Feature Settings",
            "",
            f"- Use position features: `{evidence.get('use_position_features')}`",
            f"- Use context features: `{evidence.get('use_context_features')}`",
            f"- Use token bigrams: `{evidence.get('use_token_bigrams')}`",
            f"- Feature min count: `{evidence.get('feature_min_count')}`",
            "",
        ]
    )
    if report.get("warnings"):
        lines.extend(["## Warnings", ""])
        for warning in report["warnings"]:
            lines.append(f"- {warning}")
        lines.append("")
    return "\n".join(lines).strip() + "\n"


def dominant_prediction_label(predictions: list[str]) -> str | None:
    if not predictions:
        return None
    counter = Counter(predictions)
    return counter.most_common(1)[0][0]
