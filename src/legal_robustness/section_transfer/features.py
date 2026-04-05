from __future__ import annotations

import re
from collections import Counter

from legal_robustness.config.schema import AppConfig
from legal_robustness.section_transfer.types import RRSentenceSupervisionRecord

TOKEN_PATTERN = re.compile(r"[A-Za-z0-9]+(?:'[A-Za-z0-9]+)?")
WHITESPACE_PATTERN = re.compile(r"\s+")


def normalize_feature_text(text: str) -> str:
    return WHITESPACE_PATTERN.sub(" ", text).strip().casefold()


def tokenize_feature_text(text: str) -> list[str]:
    normalized = normalize_feature_text(text)
    if not normalized:
        return []
    return TOKEN_PATTERN.findall(normalized)


def extract_record_features(
    record: RRSentenceSupervisionRecord,
    config: AppConfig,
) -> Counter[str]:
    return extract_features_from_parts(
        sentence_text=record.sentence_text,
        previous_context_text=record.previous_context_text,
        next_context_text=record.next_context_text,
        sentence_index=record.sentence_index,
        sentence_count=record.sentence_count,
        normalized_sentence_position=record.normalized_sentence_position,
        document_position_bucket=record.document_position_bucket,
        sentence_length_tokens_approx=record.sentence_length_tokens_approx,
        config=config,
    )


def extract_features_from_parts(
    *,
    sentence_text: str,
    previous_context_text: str,
    next_context_text: str,
    sentence_index: int,
    sentence_count: int,
    normalized_sentence_position: float,
    document_position_bucket: str,
    sentence_length_tokens_approx: int,
    config: AppConfig,
) -> Counter[str]:
    features: Counter[str] = Counter()
    sentence_tokens = tokenize_feature_text(sentence_text)
    for token in sentence_tokens:
        features[f"tok={token}"] += 1
    if config.section_transfer.use_token_bigrams:
        for left, right in zip(sentence_tokens, sentence_tokens[1:], strict=False):
            features[f"bigram={left}__{right}"] += 1

    if config.section_transfer.use_context_features:
        previous_tokens = tokenize_feature_text(previous_context_text)
        next_tokens = tokenize_feature_text(next_context_text)
        for token in previous_tokens:
            features[f"prev_tok={token}"] += 1
        for token in next_tokens:
            features[f"next_tok={token}"] += 1

    if config.section_transfer.use_position_features:
        features[f"position_bucket={document_position_bucket}"] += 1
        features[f"position_decile={_position_decile(normalized_sentence_position)}"] += 1
        features[f"sentence_length_bucket={_sentence_length_bucket(sentence_length_tokens_approx)}"] += 1
        if sentence_index == 0:
            features["is_first_sentence"] += 1
        if sentence_index == max(sentence_count - 1, 0):
            features["is_last_sentence"] += 1
    return features


def _position_decile(normalized_sentence_position: float) -> str:
    decile = min(int(normalized_sentence_position * 10), 9)
    return str(decile)


def _sentence_length_bucket(token_count: int) -> str:
    if token_count <= 8:
        return "short"
    if token_count <= 25:
        return "medium"
    return "long"
