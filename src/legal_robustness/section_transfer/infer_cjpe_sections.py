from __future__ import annotations

import logging
from collections import Counter

from legal_robustness.config.schema import AppConfig
from legal_robustness.data.normalized_types import NormalizedCJPECase
from legal_robustness.section_transfer.diagnostics import BROAD_SECTION_ORDER, describe_numeric_series
from legal_robustness.section_transfer.models import BroadSectionNaiveBayesModel
from legal_robustness.section_transfer.types import (
    CJPESegmentedCase,
    CJPESentencePredictionCase,
    CJPESentencePredictionResult,
)
from legal_robustness.utils.exceptions import SectionTransferError


def infer_cjpe_sections(
    *,
    segmented_cases: list[CJPESegmentedCase],
    cjpe_cases: list[NormalizedCJPECase],
    model: BroadSectionNaiveBayesModel,
    model_path: str,
    config: AppConfig,
    logger: logging.Logger | None = None,
) -> CJPESentencePredictionResult:
    logger = logger or logging.getLogger(__name__)
    cjpe_lookup = {case.case_id: case for case in cjpe_cases}
    records: list[CJPESentencePredictionCase] = []
    warnings: list[str] = []
    counts_by_split: Counter[str] = Counter()
    predicted_label_counts: Counter[str] = Counter()
    confidence_values: list[float] = []
    dominant_section_counts: Counter[str] = Counter()
    dominant_cases: list[str] = []
    cases_with_single_predicted_section = 0
    total_sentences = 0
    threshold = config.section_transfer.dominant_section_ratio_threshold

    for segmented_case in segmented_cases:
        normalized_case = cjpe_lookup.get(segmented_case.case_id)
        if normalized_case is None:
            raise SectionTransferError(
                f"Segmented CJPE case {segmented_case.case_id!r} could not be found in normalized CJPE records."
            )
        predicted_labels: list[str] = []
        predicted_scores: list[float] = []
        for sentence_index, sentence_text in enumerate(segmented_case.sentences):
            previous_context_text = _joined_context(
                segmented_case.sentences,
                start=max(0, sentence_index - config.section_transfer.context_window_size),
                end=sentence_index,
            )
            next_context_text = _joined_context(
                segmented_case.sentences,
                start=sentence_index + 1,
                end=min(
                    segmented_case.sentence_count,
                    sentence_index + 1 + config.section_transfer.context_window_size,
                ),
            )
            probabilities = model.predict_proba_from_parts(
                sentence_text=sentence_text,
                previous_context_text=previous_context_text,
                next_context_text=next_context_text,
                sentence_index=sentence_index,
                sentence_count=segmented_case.sentence_count,
                normalized_sentence_position=_normalized_sentence_position(
                    sentence_index,
                    segmented_case.sentence_count,
                ),
                document_position_bucket=_position_bucket(
                    sentence_index,
                    segmented_case.sentence_count,
                ),
                sentence_length_tokens_approx=len(sentence_text.split()),
                config=config,
            )
            predicted_label, predicted_score = max(probabilities.items(), key=lambda item: item[1])
            predicted_labels.append(predicted_label)
            predicted_scores.append(round(predicted_score, 6))
            predicted_label_counts[predicted_label] += 1
            confidence_values.append(predicted_score)
            total_sentences += 1

        case_distribution = Counter(predicted_labels)
        if case_distribution:
            dominant_label, dominant_count = case_distribution.most_common(1)[0]
            dominant_ratio = dominant_count / len(predicted_labels)
            dominant_section_counts[dominant_label] += 1
            if dominant_ratio >= threshold:
                dominant_cases.append(segmented_case.case_id)
            if len(case_distribution) == 1:
                cases_with_single_predicted_section += 1

        counts_by_split[segmented_case.split] += 1
        records.append(
            CJPESentencePredictionCase(
                case_id=segmented_case.case_id,
                split=segmented_case.split,
                subset=segmented_case.subset,
                label=segmented_case.label,
                raw_text=normalized_case.raw_text,
                sentences=list(segmented_case.sentences),
                sentence_indices=list(range(segmented_case.sentence_count)),
                sentence_start_chars=list(segmented_case.sentence_start_chars),
                sentence_end_chars=list(segmented_case.sentence_end_chars),
                predicted_broad_labels=predicted_labels,
                predicted_label_scores=predicted_scores,
                prediction_metadata={
                    "classifier_type": config.section_transfer.classifier_type,
                    "label_order": list(model.label_order),
                    "model_path": model_path,
                    "sentence_segmentation_method": config.section_transfer.sentence_segmentation_method,
                },
                source_file=segmented_case.source_file,
                source_metadata=dict(segmented_case.source_metadata),
            )
        )

    predicted_label_proportions = {
        label: round(predicted_label_counts.get(label, 0) / total_sentences, 6) if total_sentences else 0.0
        for label in BROAD_SECTION_ORDER
    }
    report = {
        "task": "cjpe_section_prediction",
        "total_cases": len(records),
        "total_sentences": total_sentences,
        "counts_by_split": dict(sorted(counts_by_split.items())),
        "predicted_label_counts": {label: predicted_label_counts.get(label, 0) for label in BROAD_SECTION_ORDER},
        "predicted_label_proportions": predicted_label_proportions,
        "confidence_summary": describe_numeric_series([int(value * 1_000_000) for value in confidence_values]),
        "confidence_summary_float": _describe_float_series(confidence_values),
        "dominant_section_counts": dict(sorted(dominant_section_counts.items())),
        "cases_above_dominant_threshold": len(dominant_cases),
        "dominant_section_ratio_threshold": threshold,
        "cases_with_single_predicted_section": cases_with_single_predicted_section,
        "dominant_cases_sample": dominant_cases[:25],
        "warnings": warnings,
    }
    logger.info(
        "Predicted CJPE broad sections for %s cases and %s sentences.",
        len(records),
        total_sentences,
    )
    return CJPESentencePredictionResult(records=records, warnings=warnings, report=report)


def build_cjpe_prediction_samples(
    records: list[CJPESentencePredictionCase],
    *,
    sample_size: int,
    preview_sentence_count: int,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for record in records[:sample_size]:
        rows.append(
            {
                "case_id": record.case_id,
                "split": record.split,
                "subset": record.subset,
                "label": record.label,
                "sentence_count": len(record.sentences),
                "sample_predictions": [
                    {
                        "sentence_index": sentence_index,
                        "sentence_text": record.sentences[sentence_index],
                        "predicted_broad_label": record.predicted_broad_labels[sentence_index],
                        "predicted_score": record.predicted_label_scores[sentence_index],
                    }
                    for sentence_index in range(min(len(record.sentences), preview_sentence_count))
                ],
                "prediction_metadata": record.prediction_metadata,
                "source_file": record.source_file,
            }
        )
    return rows


def render_cjpe_section_prediction_summary(report: dict[str, object]) -> str:
    lines = [
        "# CJPE Section Prediction Summary",
        "",
        f"- Total cases: `{report['total_cases']}`",
        f"- Total sentences: `{report['total_sentences']}`",
        f"- Counts by split: `{report['counts_by_split']}`",
        f"- Predicted label counts: `{report['predicted_label_counts']}`",
        f"- Predicted label proportions: `{report['predicted_label_proportions']}`",
        f"- Confidence summary: `{report['confidence_summary_float']}`",
        f"- Dominant section counts: `{report['dominant_section_counts']}`",
        f"- Cases above dominant-section threshold: `{report['cases_above_dominant_threshold']}`",
        f"- Cases with single predicted section: `{report['cases_with_single_predicted_section']}`",
        "",
    ]
    if report.get("warnings"):
        lines.extend(["## Warnings", ""])
        for warning in report["warnings"]:
            lines.append(f"- {warning}")
        lines.append("")
    return "\n".join(lines).strip() + "\n"


def _joined_context(sentences: list[str], *, start: int, end: int) -> str:
    if start >= end:
        return ""
    return " ".join(sentence.strip() for sentence in sentences[start:end] if sentence.strip())


def _normalized_sentence_position(sentence_index: int, sentence_count: int) -> float:
    if sentence_count <= 1:
        return 0.0
    return round(sentence_index / (sentence_count - 1), 6)


def _position_bucket(sentence_index: int, sentence_count: int) -> str:
    if sentence_count <= 1:
        return "start"
    normalized = sentence_index / (sentence_count - 1)
    if normalized <= 0.2:
        return "start"
    if normalized >= 0.8:
        return "end"
    return "middle"


def _describe_float_series(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "min": None, "max": None, "mean": None, "median": None}
    sorted_values = sorted(values)
    middle = len(sorted_values) // 2
    median = (
        sorted_values[middle]
        if len(sorted_values) % 2 == 1
        else (sorted_values[middle - 1] + sorted_values[middle]) / 2
    )
    return {
        "count": len(values),
        "min": round(min(values), 6),
        "max": round(max(values), 6),
        "mean": round(sum(values) / len(values), 6),
        "median": round(median, 6),
    }
