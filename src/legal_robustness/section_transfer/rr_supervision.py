from __future__ import annotations

import logging
from collections import Counter

from legal_robustness.config.schema import AppConfig
from legal_robustness.data.normalized_types import RRSectionMappingReport, ReconstructedRRCase
from legal_robustness.data.rr_mapping import resolve_section_for_label
from legal_robustness.section_transfer.diagnostics import describe_numeric_series
from legal_robustness.section_transfer.types import (
    RRSentenceSupervisionRecord,
    RRSentenceSupervisionResult,
)
from legal_robustness.utils.exceptions import SectionTransferError


def build_rr_sentence_supervision(
    rr_cases: list[ReconstructedRRCase],
    mapping_report: RRSectionMappingReport,
    config: AppConfig,
    logger: logging.Logger | None = None,
) -> RRSentenceSupervisionResult:
    logger = logger or logging.getLogger(__name__)
    if config.section_transfer.label_mode != "broad":
        raise SectionTransferError(
            f"Unsupported section_transfer.label_mode {config.section_transfer.label_mode!r}. "
            "The current section-transfer pipeline supports 'broad' mode only."
        )

    records: list[RRSentenceSupervisionRecord] = []
    warnings: list[str] = []
    counts_by_split: Counter[str] = Counter()
    fine_label_distribution: Counter[str] = Counter()
    broad_section_distribution: Counter[str] = Counter()
    sentence_lengths_chars: list[int] = []
    sentence_lengths_tokens: list[int] = []
    cases_represented: set[str] = set()
    context_window = max(config.section_transfer.context_window_size, 0)

    for case in rr_cases:
        cases_represented.add(case.case_id)
        sentence_count = len(case.sentences)
        if sentence_count != len(case.rr_labels):
            raise SectionTransferError(
                f"RR reconstructed case {case.case_id!r} has {sentence_count} sentences but {len(case.rr_labels)} labels."
            )
        for sentence_index, sentence_text in enumerate(case.sentences):
            label_value = case.rr_labels[sentence_index]
            resolved_section, resolved_label_name = resolve_section_for_label(label_value, mapping_report)
            broad_section = resolved_section or "other"
            fine_label_name = case.rr_label_names[sentence_index] or resolved_label_name
            previous_context_text = _joined_context(
                case.sentences,
                start=max(0, sentence_index - context_window),
                end=sentence_index,
            )
            next_context_text = _joined_context(
                case.sentences,
                start=sentence_index + 1,
                end=min(sentence_count, sentence_index + 1 + context_window),
            )
            sentence_length_chars = len(sentence_text)
            sentence_length_tokens = len(sentence_text.split())
            record = RRSentenceSupervisionRecord(
                case_id=case.case_id,
                split=case.split,
                subset=case.subset,
                sentence_index=sentence_index,
                sentence_count=sentence_count,
                sentence_text=sentence_text,
                fine_rr_label=label_value,
                fine_rr_label_name=fine_label_name,
                broad_section_label=broad_section,
                previous_context_text=previous_context_text,
                next_context_text=next_context_text,
                sentence_length_chars=sentence_length_chars,
                sentence_length_tokens_approx=sentence_length_tokens,
                normalized_sentence_position=_normalized_sentence_position(sentence_index, sentence_count),
                document_position_bucket=_position_bucket(sentence_index, sentence_count),
                source_file=case.source_file,
                source_metadata=dict(case.source_metadata),
            )
            records.append(record)
            counts_by_split[case.split] += 1
            fine_label_distribution[str(fine_label_name or label_value)] += 1
            broad_section_distribution[broad_section] += 1
            sentence_lengths_chars.append(sentence_length_chars)
            sentence_lengths_tokens.append(sentence_length_tokens)

    imbalance_ratio = _imbalance_ratio(broad_section_distribution)
    report = {
        "task": "rr_sentence_supervision",
        "total_sentences": len(records),
        "cases_represented": len(cases_represented),
        "counts_by_split": dict(sorted(counts_by_split.items())),
        "fine_label_distribution": dict(sorted(fine_label_distribution.items())),
        "broad_section_distribution": dict(sorted(broad_section_distribution.items())),
        "sentence_length_chars": describe_numeric_series(sentence_lengths_chars),
        "sentence_length_tokens_approx": describe_numeric_series(sentence_lengths_tokens),
        "broad_section_imbalance_ratio": imbalance_ratio,
        "context_window_size": context_window,
        "warnings": warnings,
    }
    logger.info(
        "Built RR sentence supervision table with %s rows across %s cases.",
        len(records),
        len(cases_represented),
    )
    return RRSentenceSupervisionResult(records=records, warnings=warnings, report=report)


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


def _imbalance_ratio(counter: Counter[str]) -> float | None:
    non_zero = [count for count in counter.values() if count > 0]
    if not non_zero:
        return None
    return round(max(non_zero) / min(non_zero), 6)
