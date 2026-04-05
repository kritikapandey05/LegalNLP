from __future__ import annotations

import re
import zlib
from collections import Counter


TOKEN_PATTERN = re.compile(r"[A-Za-z0-9_]+(?:'[A-Za-z0-9_]+)?")
WHITESPACE_PATTERN = re.compile(r"\s+")


def normalize_prediction_text(text: str) -> str:
    return WHITESPACE_PATTERN.sub(" ", text).strip().casefold()


def tokenize_prediction_text(
    text: str,
    *,
    max_tokens: int,
    min_token_chars: int,
) -> list[str]:
    normalized = normalize_prediction_text(text)
    if not normalized:
        return []
    tokens: list[str] = []
    for match in TOKEN_PATTERN.finditer(normalized):
        token = match.group(0)
        if len(token) < min_token_chars:
            continue
        tokens.append(token)
        if len(tokens) >= max_tokens:
            break
    return tokens


def hashed_term_counts(
    text: str,
    *,
    hashing_dimension: int,
    max_tokens: int,
    min_token_chars: int,
    use_token_bigrams: bool,
) -> Counter[int]:
    tokens = tokenize_prediction_text(
        text,
        max_tokens=max_tokens,
        min_token_chars=min_token_chars,
    )
    counts: Counter[int] = Counter()
    for token in tokens:
        counts[_stable_hash_index(token, hashing_dimension=hashing_dimension)] += 1
    if use_token_bigrams:
        for left, right in zip(tokens, tokens[1:], strict=False):
            counts[_stable_hash_index(f"bg:{left}__{right}", hashing_dimension=hashing_dimension)] += 1
    return counts


def _stable_hash_index(feature_name: str, *, hashing_dimension: int) -> int:
    return zlib.crc32(feature_name.encode("utf-8")) % hashing_dimension
