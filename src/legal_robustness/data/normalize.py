from __future__ import annotations

import logging
import re
from collections import Counter, defaultdict
from statistics import mean, median
from typing import Any

from legal_robustness.config.schema import AppConfig
from legal_robustness.data.normalized_types import NormalizedCJPECase, NormalizedRRCase, NormalizedTaskResult
from legal_robustness.data.raw_types import RawCJPECase, RawRRCase
from legal_robustness.utils.exceptions import DatasetNormalizationError

WHITESPACE_PATTERN = re.compile(r"\s+")


def normalize_cjpe_cases(
    raw_records: list[RawCJPECase],
    config: AppConfig,
    logger: logging.Logger | None = None,
) -> NormalizedTaskResult:
    logger = logger or logging.getLogger(__name__)
    warnings: list[str] = []
    records: list[NormalizedCJPECase] = []
    counts_by_split: Counter[str] = Counter()
    label_counts_by_split: dict[str, Counter[str]] = defaultdict(Counter)
    text_lengths: list[int] = []
    token_lengths: list[int] = []
    missing_optional_fields: Counter[str] = Counter()
    case_id_counts: Counter[str] = Counter()

    for raw_case in raw_records:
        normalized_text = _normalize_text(raw_case.raw_text, collapse_whitespace=config.data.collapse_text_whitespace)
        token_count = _approx_token_count(normalized_text)
        case_id = _normalize_case_id(raw_case.case_id)
        case_id_counts[case_id] += 1
        counts_by_split[raw_case.split] += 1
        label_counts_by_split[raw_case.split][str(raw_case.label)] += 1
        text_lengths.append(len(normalized_text))
        token_lengths.append(token_count)
        for key, value in raw_case.expert_annotations_raw.items():
            if value in (None, "", [], {}):
                missing_optional_fields[key] += 1

        records.append(
            NormalizedCJPECase(
                case_id=case_id,
                split=raw_case.split,
                subset=raw_case.subset,
                label=raw_case.label,
                raw_text=normalized_text,
                text_length_chars=len(normalized_text),
                text_length_tokens_approx=token_count,
                expert_annotations=dict(raw_case.expert_annotations_raw),
                source_file=raw_case.source_file,
                source_task=raw_case.task,
                source_metadata={
                    **raw_case.source_metadata,
                    "raw_case_id": raw_case.case_id,
                },
            )
        )

    duplicate_case_ids = {case_id: count for case_id, count in sorted(case_id_counts.items()) if count > 1}
    if duplicate_case_ids:
        warning = f"CJPE normalization found {len(duplicate_case_ids)} duplicate case ids."
        warnings.append(warning)
        logger.warning("%s", warning)
        if config.data.fail_on_duplicate_ids:
            raise DatasetNormalizationError("CJPE normalization failed due to duplicate case ids.")

    report = {
        "task": "cjpe",
        "total_cases": len(records),
        "counts_by_split": dict(sorted(counts_by_split.items())),
        "label_counts_by_split": {
            split: dict(sorted(counter.items()))
            for split, counter in sorted(label_counts_by_split.items())
        },
        "text_length_chars": _describe_numeric_series(text_lengths),
        "text_length_tokens_approx": _describe_numeric_series(token_lengths),
        "missing_optional_fields": dict(sorted(missing_optional_fields.items())),
        "duplicate_case_ids": duplicate_case_ids,
        "duplicate_case_id_count": len(duplicate_case_ids),
    }
    return NormalizedTaskResult(task="cjpe", records=records, warnings=warnings, duplicate_case_ids=duplicate_case_ids, report=report)


def normalize_rr_cases(
    raw_records: list[RawRRCase],
    config: AppConfig,
    logger: logging.Logger | None = None,
) -> NormalizedTaskResult:
    logger = logger or logging.getLogger(__name__)
    warnings: list[str] = []
    records: list[NormalizedRRCase] = []
    counts_by_split: Counter[str] = Counter()
    sentence_counts: list[int] = []
    label_counts: list[int] = []
    empty_sentence_case_count = 0
    empty_label_case_count = 0
    non_aligned_case_count = 0
    case_id_counts: Counter[str] = Counter()
    expert_missing_counts: Counter[str] = Counter()

    for raw_case in raw_records:
        case_id = _normalize_case_id(raw_case.case_id)
        sentences = [_normalize_text(sentence, collapse_whitespace=config.data.collapse_text_whitespace) for sentence in raw_case.sentences]
        labels = [_normalize_rr_label(label, strategy=config.data.rr_label_normalization) for label in raw_case.rr_labels]
        empty_sentence_count = sum(1 for sentence in sentences if not sentence.strip())
        empty_label_count = sum(1 for label in labels if _label_is_empty(label))

        case_id_counts[case_id] += 1
        counts_by_split[raw_case.split] += 1
        sentence_counts.append(len(sentences))
        label_counts.append(len(labels))
        if empty_sentence_count:
            empty_sentence_case_count += 1
        if empty_label_count:
            empty_label_case_count += 1
        if not raw_case.alignment_ok:
            non_aligned_case_count += 1
        for key, value in raw_case.expert_annotations_raw.items():
            if value in (None, "", [], {}):
                expert_missing_counts[key] += 1

        records.append(
            NormalizedRRCase(
                case_id=case_id,
                split=raw_case.split,
                subset=raw_case.subset,
                sentences=sentences,
                rr_labels=labels,
                num_sentences=len(sentences),
                num_labels=len(labels),
                empty_sentence_count=empty_sentence_count,
                empty_label_count=empty_label_count,
                alignment_ok=raw_case.alignment_ok,
                expert_annotations=dict(raw_case.expert_annotations_raw),
                source_file=raw_case.source_file,
                source_task=raw_case.task,
                source_metadata={
                    **raw_case.source_metadata,
                    "raw_case_id": raw_case.case_id,
                },
            )
        )

    duplicate_case_ids = {case_id: count for case_id, count in sorted(case_id_counts.items()) if count > 1}
    if duplicate_case_ids:
        warning = f"RR normalization found {len(duplicate_case_ids)} duplicate case ids."
        warnings.append(warning)
        logger.warning("%s", warning)
        if config.data.fail_on_duplicate_ids:
            raise DatasetNormalizationError("RR normalization failed due to duplicate case ids.")

    report = {
        "task": "rr",
        "total_cases": len(records),
        "counts_by_split": dict(sorted(counts_by_split.items())),
        "sentence_count": _describe_numeric_series(sentence_counts),
        "label_count": _describe_numeric_series(label_counts),
        "cases_with_empty_sentences": empty_sentence_case_count,
        "cases_with_empty_labels": empty_label_case_count,
        "non_aligned_cases": non_aligned_case_count,
        "duplicate_case_ids": duplicate_case_ids,
        "duplicate_case_id_count": len(duplicate_case_ids),
        "missing_optional_fields": dict(sorted(expert_missing_counts.items())),
    }
    return NormalizedTaskResult(task="rr", records=records, warnings=warnings, duplicate_case_ids=duplicate_case_ids, report=report)


def render_normalization_report(report: dict[str, Any]) -> str:
    lines = [
        f"# {report['task'].upper()} Normalization Report",
        "",
        f"- Total cases: `{report['total_cases']}`",
        f"- Counts by split: `{report['counts_by_split']}`",
    ]
    if report["task"] == "cjpe":
        lines.extend(
            [
                f"- Label counts by split: `{report['label_counts_by_split']}`",
                f"- Text length chars: `{report['text_length_chars']}`",
                f"- Text length tokens approx: `{report['text_length_tokens_approx']}`",
                f"- Missing optional fields: `{report['missing_optional_fields']}`",
                f"- Duplicate case ids: `{report['duplicate_case_id_count']}`",
            ]
        )
    else:
        lines.extend(
            [
                f"- Sentence count stats: `{report['sentence_count']}`",
                f"- Label count stats: `{report['label_count']}`",
                f"- Cases with empty sentences: `{report['cases_with_empty_sentences']}`",
                f"- Cases with empty labels: `{report['cases_with_empty_labels']}`",
                f"- Non-aligned cases: `{report['non_aligned_cases']}`",
                f"- Missing optional fields: `{report['missing_optional_fields']}`",
                f"- Duplicate case ids: `{report['duplicate_case_id_count']}`",
            ]
        )
    return "\n".join(lines).strip() + "\n"


def _normalize_case_id(value: str) -> str:
    return value.strip()


def _normalize_text(text: str, collapse_whitespace: bool) -> str:
    value = text.strip()
    if collapse_whitespace:
        value = WHITESPACE_PATTERN.sub(" ", value)
    return value


def _normalize_rr_label(label: Any, strategy: str) -> Any:
    if strategy == "preserve":
        return label
    if strategy == "stringify":
        return "" if label is None else str(label)
    if strategy == "lowercase_strings" and isinstance(label, str):
        return label.strip().lower()
    return label


def _label_is_empty(label: Any) -> bool:
    if label is None:
        return True
    if isinstance(label, str) and not label.strip():
        return True
    return False


def _approx_token_count(text: str) -> int:
    if not text:
        return 0
    return len(text.split())


def _describe_numeric_series(values: list[int]) -> dict[str, int | float | None]:
    if not values:
        return {"count": 0, "min": None, "max": None, "mean": None, "median": None}
    return {
        "count": len(values),
        "min": min(values),
        "max": max(values),
        "mean": round(mean(values), 3),
        "median": median(values),
    }
