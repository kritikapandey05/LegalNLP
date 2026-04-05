from __future__ import annotations

import logging
import re
from collections import Counter

from legal_robustness.config.schema import AppConfig
from legal_robustness.data.normalized_types import NormalizedCJPECase
from legal_robustness.section_transfer.diagnostics import describe_numeric_series
from legal_robustness.section_transfer.types import (
    CJPESegmentedCase,
    CJPESentenceSegmentationResult,
    SentenceSpan,
)
from legal_robustness.utils.exceptions import SectionTransferError

WHITESPACE_PATTERN = re.compile(r"\s+")


def split_legal_text_into_sentences(text: str, config: AppConfig) -> list[SentenceSpan]:
    if config.section_transfer.sentence_segmentation_method != "heuristic_legal":
        raise SectionTransferError(
            "Unsupported section_transfer.sentence_segmentation_method. "
            "The current section-transfer pipeline supports 'heuristic_legal' only."
        )
    normalized_text = text.replace("\r\n", "\n").replace("\r", "\n")
    abbreviations = {value.strip().casefold() for value in config.section_transfer.sentence_segmentation_abbreviations}
    spans: list[SentenceSpan] = []
    start = _skip_whitespace(normalized_text, 0)
    index = 0
    i = 0
    while i < len(normalized_text):
        char = normalized_text[i]
        if char in ".?!":
            if _should_split_after_punctuation(normalized_text, i, abbreviations):
                span = _build_span(normalized_text, index=index, start=start, end=i + 1)
                if span is not None:
                    spans.append(span)
                    index += 1
                start = _skip_whitespace(normalized_text, i + 1)
        elif char == "\n":
            newline_end = i
            while newline_end < len(normalized_text) and normalized_text[newline_end] == "\n":
                newline_end += 1
            if newline_end - i >= 2:
                span = _build_span(normalized_text, index=index, start=start, end=i)
                if span is not None:
                    spans.append(span)
                    index += 1
                start = _skip_whitespace(normalized_text, newline_end)
                i = newline_end - 1
        i += 1

    tail = _build_span(normalized_text, index=index, start=start, end=len(normalized_text))
    if tail is not None:
        spans.append(tail)
    if not spans and normalized_text.strip():
        stripped = normalized_text.strip()
        leading = normalized_text.find(stripped)
        spans.append(
            SentenceSpan(
                sentence_index=0,
                text=_normalize_sentence_text(stripped),
                start_char=leading,
                end_char=leading + len(stripped),
            )
        )
    return spans


def segment_cjpe_cases(
    cjpe_cases: list[NormalizedCJPECase],
    config: AppConfig,
    logger: logging.Logger | None = None,
) -> CJPESentenceSegmentationResult:
    logger = logger or logging.getLogger(__name__)
    short_threshold = config.section_transfer.segmentation_short_sentence_chars
    long_threshold = config.section_transfer.segmentation_long_sentence_chars
    records: list[CJPESegmentedCase] = []
    warnings: list[str] = []
    counts_by_split: Counter[str] = Counter()
    sentence_count_per_case: list[int] = []
    sentence_length_chars: list[int] = []
    cases_with_zero_sentences = 0
    high_short_sentence_ratio_cases: list[str] = []
    high_long_sentence_ratio_cases: list[str] = []
    total_short_sentences = 0
    total_long_sentences = 0
    total_sentences = 0

    for case in cjpe_cases:
        spans = split_legal_text_into_sentences(case.raw_text, config=config)
        sentence_texts = [span.text for span in spans]
        starts = [span.start_char for span in spans]
        ends = [span.end_char for span in spans]
        short_sentence_count = sum(1 for text in sentence_texts if len(text) < short_threshold)
        long_sentence_count = sum(1 for text in sentence_texts if len(text) > long_threshold)
        sentence_count = len(sentence_texts)
        if sentence_count == 0:
            cases_with_zero_sentences += 1
        else:
            short_ratio = short_sentence_count / sentence_count
            long_ratio = long_sentence_count / sentence_count
            if short_ratio >= 0.3:
                high_short_sentence_ratio_cases.append(case.case_id)
            if long_ratio >= 0.2:
                high_long_sentence_ratio_cases.append(case.case_id)
        total_short_sentences += short_sentence_count
        total_long_sentences += long_sentence_count
        total_sentences += sentence_count
        sentence_count_per_case.append(sentence_count)
        sentence_length_chars.extend(len(text) for text in sentence_texts)
        counts_by_split[case.split] += 1
        records.append(
            CJPESegmentedCase(
                case_id=case.case_id,
                split=case.split,
                subset=case.subset,
                label=case.label,
                sentence_count=sentence_count,
                sentences=sentence_texts,
                sentence_start_chars=starts,
                sentence_end_chars=ends,
                short_sentence_count=short_sentence_count,
                long_sentence_count=long_sentence_count,
                raw_text_length_chars=case.text_length_chars,
                source_file=case.source_file,
                source_metadata=dict(case.source_metadata),
            )
        )

    report = {
        "task": "cjpe_sentence_segmentation",
        "segmentation_method": config.section_transfer.sentence_segmentation_method,
        "total_cases": len(records),
        "total_sentences": total_sentences,
        "counts_by_split": dict(sorted(counts_by_split.items())),
        "sentence_count_per_case": describe_numeric_series(sentence_count_per_case),
        "sentence_length_chars": describe_numeric_series(sentence_length_chars),
        "short_sentence_ratio": round(total_short_sentences / total_sentences, 6) if total_sentences else 0.0,
        "long_sentence_ratio": round(total_long_sentences / total_sentences, 6) if total_sentences else 0.0,
        "cases_with_zero_sentences": cases_with_zero_sentences,
        "high_short_sentence_ratio_cases": high_short_sentence_ratio_cases[:25],
        "high_long_sentence_ratio_cases": high_long_sentence_ratio_cases[:25],
        "warnings": warnings,
    }
    logger.info(
        "Segmented %s CJPE cases into %s total sentences.",
        len(records),
        total_sentences,
    )
    return CJPESentenceSegmentationResult(records=records, warnings=warnings, report=report)


def build_cjpe_sentence_samples(
    records: list[CJPESegmentedCase],
    *,
    sample_size: int,
    preview_sentence_count: int = 5,
) -> list[dict[str, object]]:
    return [
        {
            "case_id": record.case_id,
            "split": record.split,
            "subset": record.subset,
            "label": record.label,
            "sentence_count": record.sentence_count,
            "sample_sentences": record.sentences[:preview_sentence_count],
            "sentence_start_chars": record.sentence_start_chars[:preview_sentence_count],
            "sentence_end_chars": record.sentence_end_chars[:preview_sentence_count],
            "short_sentence_count": record.short_sentence_count,
            "long_sentence_count": record.long_sentence_count,
            "source_file": record.source_file,
        }
        for record in records[:sample_size]
    ]


def _build_span(text: str, *, index: int, start: int, end: int) -> SentenceSpan | None:
    if start >= len(text) or start >= end:
        return None
    raw_fragment = text[start:end]
    stripped = raw_fragment.strip()
    if not stripped:
        return None
    leading_offset = len(raw_fragment) - len(raw_fragment.lstrip())
    trailing_offset = len(raw_fragment.rstrip())
    return SentenceSpan(
        sentence_index=index,
        text=_normalize_sentence_text(stripped),
        start_char=start + leading_offset,
        end_char=start + trailing_offset,
    )


def _normalize_sentence_text(text: str) -> str:
    return WHITESPACE_PATTERN.sub(" ", text).strip()


def _skip_whitespace(text: str, index: int) -> int:
    while index < len(text) and text[index].isspace():
        index += 1
    return index


def _should_split_after_punctuation(text: str, index: int, abbreviations: set[str]) -> bool:
    char = text[index]
    if char in "?!":
        return True
    if index > 0 and index + 1 < len(text) and text[index - 1].isdigit() and text[index + 1].isdigit():
        return False
    token = _previous_token(text, index)
    if token and token.casefold().rstrip(".") in abbreviations:
        return False
    if token and len(token.rstrip(".")) == 1 and token.rstrip(".").isalpha():
        return False

    next_index = index + 1
    while next_index < len(text) and text[next_index].isspace():
        if text[next_index] == "\n":
            return True
        next_index += 1
    if next_index >= len(text):
        return True

    while next_index < len(text) and text[next_index] in "\"')]}":
        next_index += 1
    if next_index >= len(text):
        return True
    next_char = text[next_index]
    return next_char.isupper() or next_char.isdigit() or next_char in "([{\"'"


def _previous_token(text: str, index: int) -> str:
    start = index - 1
    while start >= 0 and text[start].isspace():
        start -= 1
    end = start + 1
    while start >= 0 and (text[start].isalnum() or text[start] in "._-"):
        start -= 1
    return text[start + 1 : end]
