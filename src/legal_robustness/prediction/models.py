from __future__ import annotations

import math
import pickle
import random
import re
import zlib
from array import array
from collections import Counter
from pathlib import Path
from typing import Any

from legal_robustness.config.schema import AppConfig
from legal_robustness.prediction.features import hashed_term_counts, tokenize_prediction_text
from legal_robustness.section_transfer.diagnostics import BROAD_SECTION_ORDER
from legal_robustness.section_transfer.sentence_split import split_legal_text_into_sentences
from legal_robustness.utils.exceptions import PredictionError

SECTION_MARKER_PATTERN = re.compile(r"\[(FACTS|PRECEDENTS|REASONING|CONCLUSION|OTHER)_SECTION\]")


class HashedTfidfVectorizer:
    def __init__(
        self,
        *,
        hashing_dimension: int,
        max_tokens_per_document: int,
        min_token_chars: int,
        use_token_bigrams: bool,
        document_count: int,
        idf_by_index: dict[int, float],
        default_idf: float,
    ) -> None:
        self.hashing_dimension = hashing_dimension
        self.max_tokens_per_document = max_tokens_per_document
        self.min_token_chars = min_token_chars
        self.use_token_bigrams = use_token_bigrams
        self.document_count = document_count
        self.idf_by_index = idf_by_index
        self.default_idf = default_idf

    @classmethod
    def fit(cls, texts: list[str], *, config: AppConfig) -> "HashedTfidfVectorizer":
        if not texts:
            raise PredictionError("Cannot fit a TF-IDF vectorizer without training texts.")

        document_frequency: Counter[int] = Counter()
        for text in texts:
            counts = hashed_term_counts(
                text,
                hashing_dimension=config.prediction.hashing_dimension,
                max_tokens=config.prediction.max_tokens_per_document,
                min_token_chars=config.prediction.min_token_chars,
                use_token_bigrams=config.prediction.use_token_bigrams,
            )
            document_frequency.update(counts.keys())

        document_count = len(texts)
        default_idf = math.log((1 + document_count) / 1) + 1.0
        idf_by_index = {
            index: math.log((1 + document_count) / (1 + frequency)) + 1.0
            for index, frequency in document_frequency.items()
        }
        return cls(
            hashing_dimension=config.prediction.hashing_dimension,
            max_tokens_per_document=config.prediction.max_tokens_per_document,
            min_token_chars=config.prediction.min_token_chars,
            use_token_bigrams=config.prediction.use_token_bigrams,
            document_count=document_count,
            idf_by_index=idf_by_index,
            default_idf=default_idf,
        )

    def transform_text(self, text: str) -> dict[int, float]:
        counts = hashed_term_counts(
            text,
            hashing_dimension=self.hashing_dimension,
            max_tokens=self.max_tokens_per_document,
            min_token_chars=self.min_token_chars,
            use_token_bigrams=self.use_token_bigrams,
        )
        return self.transform_counts(counts)

    def transform_counts(self, counts: Counter[int]) -> dict[int, float]:
        if not counts:
            return {}
        weights: dict[int, float] = {}
        norm_sq = 0.0
        for index, count in counts.items():
            value = (1.0 + math.log(count)) * self.idf_by_index.get(index, self.default_idf)
            weights[index] = value
            norm_sq += value * value
        if norm_sq <= 0.0:
            return weights
        norm = math.sqrt(norm_sq)
        return {index: (value / norm) for index, value in weights.items()}

    def to_metadata(self) -> dict[str, Any]:
        return {
            "hashing_dimension": self.hashing_dimension,
            "max_tokens_per_document": self.max_tokens_per_document,
            "min_token_chars": self.min_token_chars,
            "use_token_bigrams": self.use_token_bigrams,
            "document_count": self.document_count,
            "observed_hashed_features": len(self.idf_by_index),
        }


class TfidfLogisticRegressionModel:
    def __init__(
        self,
        *,
        label_order: list[str],
        vectorizer: HashedTfidfVectorizer,
        binary_weights: array | None = None,
        binary_bias: float | None = None,
        ovr_weights: list[array] | None = None,
        ovr_biases: list[float] | None = None,
        training_summary: dict[str, Any] | None = None,
    ) -> None:
        self.label_order = label_order
        self.vectorizer = vectorizer
        self.binary_weights = binary_weights
        self.binary_bias = binary_bias
        self.ovr_weights = ovr_weights
        self.ovr_biases = ovr_biases
        self.training_summary = training_summary or {}

    @property
    def is_binary(self) -> bool:
        return self.binary_weights is not None

    @classmethod
    def train(
        cls,
        texts: list[str],
        labels: list[str],
        *,
        config: AppConfig,
    ) -> "TfidfLogisticRegressionModel":
        if not texts or not labels or len(texts) != len(labels):
            raise PredictionError("TF-IDF logistic regression training requires aligned texts and labels.")

        vectorizer = HashedTfidfVectorizer.fit(texts, config=config)
        features = [vectorizer.transform_text(text) for text in texts]
        label_order = sorted({str(label) for label in labels})
        if len(label_order) < 2:
            raise PredictionError("Judgment-prediction training requires at least two distinct labels.")

        summary = {
            "model_name": "tfidf_logistic_regression",
            "label_order": label_order,
            "vectorizer": vectorizer.to_metadata(),
            "epochs": config.prediction.logistic_epochs,
            "learning_rate": config.prediction.logistic_learning_rate,
            "l2_weight": config.prediction.logistic_l2_weight,
            "training_case_count": len(texts),
            "training_label_distribution": dict(sorted(Counter(labels).items())),
        }
        if len(label_order) == 2:
            weights, bias = _train_binary_logistic(
                features,
                labels,
                positive_label=label_order[1],
                hashing_dimension=vectorizer.hashing_dimension,
                learning_rate=config.prediction.logistic_learning_rate,
                epochs=config.prediction.logistic_epochs,
                l2_weight=config.prediction.logistic_l2_weight,
                seed=config.runtime.seed,
            )
            return cls(
                label_order=label_order,
                vectorizer=vectorizer,
                binary_weights=weights,
                binary_bias=bias,
                training_summary=summary,
            )

        weights, biases = _train_ovr_logistic(
            features,
            labels,
            label_order=label_order,
            hashing_dimension=vectorizer.hashing_dimension,
            learning_rate=config.prediction.logistic_learning_rate,
            epochs=config.prediction.logistic_epochs,
            l2_weight=config.prediction.logistic_l2_weight,
            seed=config.runtime.seed,
        )
        return cls(
            label_order=label_order,
            vectorizer=vectorizer,
            ovr_weights=weights,
            ovr_biases=biases,
            training_summary=summary,
        )

    def predict_label(self, text: str) -> str:
        probabilities = self.predict_proba(text)
        return max(probabilities.items(), key=lambda item: item[1])[0]

    def predict_proba(self, text: str) -> dict[str, float]:
        return self.predict_proba_from_features(self.vectorizer.transform_text(text))

    def predict_proba_from_features(self, features: dict[int, float]) -> dict[str, float]:
        if self.is_binary:
            assert self.binary_weights is not None
            assert self.binary_bias is not None
            positive_score = _sigmoid(_dot(self.binary_weights, features) + self.binary_bias)
            return {
                self.label_order[0]: round(1.0 - positive_score, 6),
                self.label_order[1]: round(positive_score, 6),
            }

        assert self.ovr_weights is not None
        assert self.ovr_biases is not None
        raw_scores = []
        for index, label in enumerate(self.label_order):
            score = _sigmoid(_dot(self.ovr_weights[index], features) + self.ovr_biases[index])
            raw_scores.append((label, score))
        total = sum(score for _, score in raw_scores) or 1.0
        return {
            label: round(score / total, 6)
            for label, score in raw_scores
        }

    def save(self, path: Path) -> None:
        with path.open("wb") as handle:
            pickle.dump(self, handle)

    def to_metadata(self) -> dict[str, Any]:
        return dict(self.training_summary)


class MultinomialNaiveBayesTextModel:
    def __init__(
        self,
        *,
        label_order: list[str],
        hashing_dimension: int,
        max_tokens_per_document: int,
        min_token_chars: int,
        use_token_bigrams: bool,
        class_document_counts: dict[str, int],
        feature_counts_by_label: dict[str, dict[int, int]],
        total_feature_counts_by_label: dict[str, int],
        alpha: float = 1.0,
        training_summary: dict[str, Any] | None = None,
    ) -> None:
        self.label_order = label_order
        self.hashing_dimension = hashing_dimension
        self.max_tokens_per_document = max_tokens_per_document
        self.min_token_chars = min_token_chars
        self.use_token_bigrams = use_token_bigrams
        self.class_document_counts = class_document_counts
        self.feature_counts_by_label = feature_counts_by_label
        self.total_feature_counts_by_label = total_feature_counts_by_label
        self.alpha = alpha
        self.total_documents = sum(class_document_counts.values())
        self.training_summary = training_summary or {}
        self._log_probability_cache: dict[str, dict[int, float]] = {
            label: {} for label in label_order
        }

    @classmethod
    def train(
        cls,
        texts: list[str],
        labels: list[str],
        *,
        config: AppConfig,
        alpha: float = 1.0,
    ) -> "MultinomialNaiveBayesTextModel":
        if not texts or not labels or len(texts) != len(labels):
            raise PredictionError("Multinomial Naive Bayes training requires aligned texts and labels.")

        label_order = sorted({str(label) for label in labels})
        if len(label_order) < 2:
            raise PredictionError("Judgment-prediction training requires at least two distinct labels.")

        class_document_counts: Counter[str] = Counter()
        feature_counts_by_label: dict[str, Counter[int]] = {
            label: Counter() for label in label_order
        }
        total_feature_counts_by_label: Counter[str] = Counter()
        observed_features: set[int] = set()
        for text, label in zip(texts, labels, strict=False):
            counts = hashed_term_counts(
                text,
                hashing_dimension=config.prediction.hashing_dimension,
                max_tokens=config.prediction.max_tokens_per_document,
                min_token_chars=config.prediction.min_token_chars,
                use_token_bigrams=config.prediction.use_token_bigrams,
            )
            string_label = str(label)
            class_document_counts[string_label] += 1
            feature_counts_by_label[string_label].update(counts)
            total_feature_counts_by_label[string_label] += sum(counts.values())
            observed_features.update(counts.keys())

        training_summary = {
            "model_name": "multinomial_naive_bayes",
            "label_order": label_order,
            "hashing_dimension": config.prediction.hashing_dimension,
            "max_tokens_per_document": config.prediction.max_tokens_per_document,
            "min_token_chars": config.prediction.min_token_chars,
            "use_token_bigrams": config.prediction.use_token_bigrams,
            "alpha": alpha,
            "training_case_count": len(texts),
            "training_label_distribution": dict(sorted(Counter(labels).items())),
            "observed_hashed_features": len(observed_features),
        }
        return cls(
            label_order=label_order,
            hashing_dimension=config.prediction.hashing_dimension,
            max_tokens_per_document=config.prediction.max_tokens_per_document,
            min_token_chars=config.prediction.min_token_chars,
            use_token_bigrams=config.prediction.use_token_bigrams,
            class_document_counts=dict(class_document_counts),
            feature_counts_by_label={
                label: dict(counter)
                for label, counter in feature_counts_by_label.items()
            },
            total_feature_counts_by_label=dict(total_feature_counts_by_label),
            alpha=alpha,
            training_summary=training_summary,
        )

    def predict_label(self, text: str) -> str:
        probabilities = self.predict_proba(text)
        return max(probabilities.items(), key=lambda item: item[1])[0]

    def predict_proba(self, text: str) -> dict[str, float]:
        counts = hashed_term_counts(
            text,
            hashing_dimension=self.hashing_dimension,
            max_tokens=self.max_tokens_per_document,
            min_token_chars=self.min_token_chars,
            use_token_bigrams=self.use_token_bigrams,
        )
        return self.predict_proba_from_counts(counts)

    def predict_proba_from_counts(self, counts: Counter[int]) -> dict[str, float]:
        scores = self.predict_log_scores_from_counts(counts)
        max_score = max(scores.values())
        stabilized = {
            label: math.exp(score - max_score)
            for label, score in scores.items()
        }
        total = sum(stabilized.values()) or 1.0
        return {
            label: round(value / total, 6)
            for label, value in sorted(stabilized.items())
        }

    def predict_log_scores_from_counts(self, counts: Counter[int]) -> dict[str, float]:
        scores: dict[str, float] = {}
        label_count = len(self.label_order)
        for label in self.label_order:
            prior_numerator = self.class_document_counts.get(label, 0) + self.alpha
            prior_denominator = self.total_documents + (self.alpha * label_count)
            score = math.log(prior_numerator / prior_denominator)
            for feature_index, value in counts.items():
                score += value * self._feature_log_probability(label, feature_index)
            scores[label] = score
        return scores

    def save(self, path: Path) -> None:
        with path.open("wb") as handle:
            pickle.dump(self, handle)

    def to_metadata(self) -> dict[str, Any]:
        return dict(self.training_summary)

    def _feature_log_probability(self, label: str, feature_index: int) -> float:
        cached = self._log_probability_cache[label]
        if feature_index in cached:
            return cached[feature_index]
        feature_count = self.feature_counts_by_label.get(label, {}).get(feature_index, 0)
        denominator = self.total_feature_counts_by_label.get(label, 0) + (
            self.alpha * self.hashing_dimension
        )
        log_probability = math.log((feature_count + self.alpha) / denominator)
        cached[feature_index] = log_probability
        return log_probability


class AveragedPassiveAggressiveModel:
    def __init__(
        self,
        *,
        label_order: list[str],
        vectorizer: HashedTfidfVectorizer,
        binary_weights: array | None = None,
        binary_bias: float | None = None,
        ovr_weights: list[array] | None = None,
        ovr_biases: list[float] | None = None,
        aggressiveness: float = 1.0,
        training_summary: dict[str, Any] | None = None,
    ) -> None:
        self.label_order = label_order
        self.vectorizer = vectorizer
        self.binary_weights = binary_weights
        self.binary_bias = binary_bias
        self.ovr_weights = ovr_weights
        self.ovr_biases = ovr_biases
        self.aggressiveness = aggressiveness
        self.training_summary = training_summary or {}

    @property
    def is_binary(self) -> bool:
        return self.binary_weights is not None

    @classmethod
    def train(
        cls,
        texts: list[str],
        labels: list[str],
        *,
        config: AppConfig,
    ) -> "AveragedPassiveAggressiveModel":
        if not texts or not labels or len(texts) != len(labels):
            raise PredictionError("Averaged passive-aggressive training requires aligned texts and labels.")

        vectorizer = HashedTfidfVectorizer.fit(texts, config=config)
        features = [vectorizer.transform_text(text) for text in texts]
        label_order = sorted({str(label) for label in labels})
        if len(label_order) < 2:
            raise PredictionError("Judgment-prediction training requires at least two distinct labels.")

        summary = {
            "model_name": "averaged_passive_aggressive",
            "label_order": label_order,
            "vectorizer": vectorizer.to_metadata(),
            "epochs": config.prediction.passive_aggressive_epochs,
            "aggressiveness": config.prediction.passive_aggressive_aggressiveness,
            "training_case_count": len(texts),
            "training_label_distribution": dict(sorted(Counter(labels).items())),
        }
        if len(label_order) == 2:
            weights, bias = _train_binary_passive_aggressive(
                features,
                labels,
                positive_label=label_order[1],
                hashing_dimension=vectorizer.hashing_dimension,
                aggressiveness=config.prediction.passive_aggressive_aggressiveness,
                epochs=config.prediction.passive_aggressive_epochs,
                seed=config.runtime.seed,
            )
            return cls(
                label_order=label_order,
                vectorizer=vectorizer,
                binary_weights=weights,
                binary_bias=bias,
                aggressiveness=config.prediction.passive_aggressive_aggressiveness,
                training_summary=summary,
            )

        weights, biases = _train_ovr_passive_aggressive(
            features,
            labels,
            label_order=label_order,
            hashing_dimension=vectorizer.hashing_dimension,
            aggressiveness=config.prediction.passive_aggressive_aggressiveness,
            epochs=config.prediction.passive_aggressive_epochs,
            seed=config.runtime.seed,
        )
        return cls(
            label_order=label_order,
            vectorizer=vectorizer,
            ovr_weights=weights,
            ovr_biases=biases,
            aggressiveness=config.prediction.passive_aggressive_aggressiveness,
            training_summary=summary,
        )

    def predict_label(self, text: str) -> str:
        probabilities = self.predict_proba(text)
        return max(probabilities.items(), key=lambda item: item[1])[0]

    def predict_proba(self, text: str) -> dict[str, float]:
        return self.predict_proba_from_features(self.vectorizer.transform_text(text))

    def predict_proba_from_features(self, features: dict[int, float]) -> dict[str, float]:
        if self.is_binary:
            assert self.binary_weights is not None
            assert self.binary_bias is not None
            positive_score = _sigmoid(_dot(self.binary_weights, features) + self.binary_bias)
            return {
                self.label_order[0]: round(1.0 - positive_score, 6),
                self.label_order[1]: round(positive_score, 6),
            }

        assert self.ovr_weights is not None
        assert self.ovr_biases is not None
        raw_scores = []
        for index, label in enumerate(self.label_order):
            score = _sigmoid(_dot(self.ovr_weights[index], features) + self.ovr_biases[index])
            raw_scores.append((label, score))
        total = sum(score for _, score in raw_scores) or 1.0
        return {
            label: round(score / total, 6)
            for label, score in raw_scores
        }

    def save(self, path: Path) -> None:
        with path.open("wb") as handle:
            pickle.dump(self, handle)

    def to_metadata(self) -> dict[str, Any]:
        return dict(self.training_summary)


class SectionContextualVectorizer:
    def __init__(
        self,
        *,
        hashing_dimension: int,
        min_token_chars: int,
        use_token_bigrams: bool,
        document_count: int,
        section_sentence_limit: int,
        sentence_token_limit: int,
        idf_by_index: dict[int, float],
        default_idf: float,
    ) -> None:
        self.hashing_dimension = hashing_dimension
        self.min_token_chars = min_token_chars
        self.use_token_bigrams = use_token_bigrams
        self.document_count = document_count
        self.section_sentence_limit = section_sentence_limit
        self.sentence_token_limit = sentence_token_limit
        self.idf_by_index = idf_by_index
        self.default_idf = default_idf

    @classmethod
    def fit(cls, texts: list[str], *, config: AppConfig) -> "SectionContextualVectorizer":
        if not texts:
            raise PredictionError("Cannot fit a contextual vectorizer without training texts.")
        document_frequency: Counter[int] = Counter()
        for text in texts:
            counts = _contextual_feature_counts(text, config=config)
            document_frequency.update(counts.keys())
        document_count = len(texts)
        default_idf = math.log((1 + document_count) / 1) + 1.0
        idf_by_index = {
            index: math.log((1 + document_count) / (1 + frequency)) + 1.0
            for index, frequency in document_frequency.items()
        }
        return cls(
            hashing_dimension=config.prediction.contextual_hashing_dimension,
            min_token_chars=config.prediction.min_token_chars,
            use_token_bigrams=config.prediction.use_token_bigrams,
            document_count=document_count,
            section_sentence_limit=config.prediction.contextual_section_sentence_limit,
            sentence_token_limit=config.prediction.contextual_sentence_token_limit,
            idf_by_index=idf_by_index,
            default_idf=default_idf,
        )

    def transform_text(self, text: str, *, config: AppConfig) -> dict[int, float]:
        return self.transform_counts(_contextual_feature_counts(text, config=config))

    def transform_counts(self, counts: Counter[int]) -> dict[int, float]:
        if not counts:
            return {}
        weights: dict[int, float] = {}
        norm_sq = 0.0
        for index, count in counts.items():
            value = (1.0 + math.log(count)) * self.idf_by_index.get(index, self.default_idf)
            weights[index] = value
            norm_sq += value * value
        if norm_sq <= 0.0:
            return weights
        norm = math.sqrt(norm_sq)
        return {index: (value / norm) for index, value in weights.items()}

    def to_metadata(self) -> dict[str, Any]:
        return {
            "hashing_dimension": self.hashing_dimension,
            "min_token_chars": self.min_token_chars,
            "use_token_bigrams": self.use_token_bigrams,
            "document_count": self.document_count,
            "section_sentence_limit": self.section_sentence_limit,
            "sentence_token_limit": self.sentence_token_limit,
            "observed_hashed_features": len(self.idf_by_index),
            "feature_style": "section_and_sentence_contextual_tfidf",
        }


class SectionContextualLogisticRegressionModel:
    def __init__(
        self,
        *,
        label_order: list[str],
        vectorizer: SectionContextualVectorizer,
        config_snapshot: dict[str, Any],
        binary_weights: array | None = None,
        binary_bias: float | None = None,
        ovr_weights: list[array] | None = None,
        ovr_biases: list[float] | None = None,
        training_summary: dict[str, Any] | None = None,
    ) -> None:
        self.label_order = label_order
        self.vectorizer = vectorizer
        self.config_snapshot = config_snapshot
        self.binary_weights = binary_weights
        self.binary_bias = binary_bias
        self.ovr_weights = ovr_weights
        self.ovr_biases = ovr_biases
        self.training_summary = training_summary or {}

    @property
    def is_binary(self) -> bool:
        return self.binary_weights is not None

    @classmethod
    def train(
        cls,
        texts: list[str],
        labels: list[str],
        *,
        config: AppConfig,
    ) -> "SectionContextualLogisticRegressionModel":
        if not texts or not labels or len(texts) != len(labels):
            raise PredictionError("Contextual logistic regression training requires aligned texts and labels.")

        vectorizer = SectionContextualVectorizer.fit(texts, config=config)
        features = [vectorizer.transform_text(text, config=config) for text in texts]
        label_order = sorted({str(label) for label in labels})
        if len(label_order) < 2:
            raise PredictionError("Judgment-prediction training requires at least two distinct labels.")

        config_snapshot = {
            "sentence_segmentation_method": config.section_transfer.sentence_segmentation_method,
            "sentence_segmentation_abbreviations": list(config.section_transfer.sentence_segmentation_abbreviations),
            "contextual_hashing_dimension": config.prediction.contextual_hashing_dimension,
            "contextual_learning_rate": config.prediction.contextual_learning_rate,
            "contextual_epochs": config.prediction.contextual_epochs,
            "contextual_l2_weight": config.prediction.contextual_l2_weight,
            "contextual_section_sentence_limit": config.prediction.contextual_section_sentence_limit,
            "contextual_sentence_token_limit": config.prediction.contextual_sentence_token_limit,
            "min_token_chars": config.prediction.min_token_chars,
            "use_token_bigrams": config.prediction.use_token_bigrams,
        }
        summary = {
            "model_name": "section_contextual_logistic_regression",
            "label_order": label_order,
            "vectorizer": vectorizer.to_metadata(),
            "config_snapshot": config_snapshot,
            "training_case_count": len(texts),
            "training_label_distribution": dict(sorted(Counter(labels).items())),
        }
        if len(label_order) == 2:
            weights, bias = _train_binary_logistic(
                features,
                labels,
                positive_label=label_order[1],
                hashing_dimension=vectorizer.hashing_dimension,
                learning_rate=config.prediction.contextual_learning_rate,
                epochs=config.prediction.contextual_epochs,
                l2_weight=config.prediction.contextual_l2_weight,
                seed=config.runtime.seed,
            )
            return cls(
                label_order=label_order,
                vectorizer=vectorizer,
                config_snapshot=config_snapshot,
                binary_weights=weights,
                binary_bias=bias,
                training_summary=summary,
            )

        weights, biases = _train_ovr_logistic(
            features,
            labels,
            label_order=label_order,
            hashing_dimension=vectorizer.hashing_dimension,
            learning_rate=config.prediction.contextual_learning_rate,
            epochs=config.prediction.contextual_epochs,
            l2_weight=config.prediction.contextual_l2_weight,
            seed=config.runtime.seed,
        )
        return cls(
            label_order=label_order,
            vectorizer=vectorizer,
            config_snapshot=config_snapshot,
            ovr_weights=weights,
            ovr_biases=biases,
            training_summary=summary,
        )

    def predict_label(self, text: str) -> str:
        probabilities = self.predict_proba(text)
        return max(probabilities.items(), key=lambda item: item[1])[0]

    def predict_proba(self, text: str) -> dict[str, float]:
        features = self.vectorizer.transform_text(
            text,
            config=_build_contextual_prediction_config(self.config_snapshot),
        )
        return self.predict_proba_from_features(features)

    def predict_proba_from_features(self, features: dict[int, float]) -> dict[str, float]:
        if self.is_binary:
            assert self.binary_weights is not None
            assert self.binary_bias is not None
            positive_score = _sigmoid(_dot(self.binary_weights, features) + self.binary_bias)
            return {
                self.label_order[0]: round(1.0 - positive_score, 6),
                self.label_order[1]: round(positive_score, 6),
            }

        assert self.ovr_weights is not None
        assert self.ovr_biases is not None
        raw_scores = []
        for index, label in enumerate(self.label_order):
            score = _sigmoid(_dot(self.ovr_weights[index], features) + self.ovr_biases[index])
            raw_scores.append((label, score))
        total = sum(score for _, score in raw_scores) or 1.0
        return {
            label: round(score / total, 6)
            for label, score in raw_scores
        }

    def save(self, path: Path) -> None:
        with path.open("wb") as handle:
            pickle.dump(self, handle)

    def to_metadata(self) -> dict[str, Any]:
        return dict(self.training_summary)


def load_prediction_model(
    path: Path | str,
) -> TfidfLogisticRegressionModel | MultinomialNaiveBayesTextModel | AveragedPassiveAggressiveModel | SectionContextualLogisticRegressionModel:
    path = Path(path)
    with path.open("rb") as handle:
        model = pickle.load(handle)
    if not isinstance(
        model,
        (
            TfidfLogisticRegressionModel,
            MultinomialNaiveBayesTextModel,
            AveragedPassiveAggressiveModel,
            SectionContextualLogisticRegressionModel,
        ),
    ):
        raise PredictionError(
            f"Loaded object from {path} is not a supported prediction model."
        )
    return model


def _train_binary_logistic(
    features: list[dict[int, float]],
    labels: list[str],
    *,
    positive_label: str,
    hashing_dimension: int,
    learning_rate: float,
    epochs: int,
    l2_weight: float,
    seed: int,
) -> tuple[array, float]:
    weights = array("f", [0.0]) * hashing_dimension
    bias = 0.0
    order = list(range(len(features)))
    rng = random.Random(seed)
    for epoch in range(epochs):
        rng.shuffle(order)
        epoch_learning_rate = learning_rate / math.sqrt(epoch + 1)
        for row_index in order:
            row = features[row_index]
            gold = 1.0 if labels[row_index] == positive_label else 0.0
            probability = _sigmoid(_dot(weights, row) + bias)
            error = probability - gold
            bias -= epoch_learning_rate * error
            for feature_index, value in row.items():
                weights[feature_index] -= epoch_learning_rate * (
                    (error * value) + (l2_weight * weights[feature_index])
                )
    return weights, bias


def _train_ovr_logistic(
    features: list[dict[int, float]],
    labels: list[str],
    *,
    label_order: list[str],
    hashing_dimension: int,
    learning_rate: float,
    epochs: int,
    l2_weight: float,
    seed: int,
) -> tuple[list[array], list[float]]:
    weights: list[array] = [array("f", [0.0]) * hashing_dimension for _ in label_order]
    biases: list[float] = [0.0 for _ in label_order]
    order = list(range(len(features)))
    rng = random.Random(seed)
    for class_index, positive_label in enumerate(label_order):
        for epoch in range(epochs):
            rng.shuffle(order)
            epoch_learning_rate = learning_rate / math.sqrt(epoch + 1)
            for row_index in order:
                row = features[row_index]
                gold = 1.0 if labels[row_index] == positive_label else 0.0
                probability = _sigmoid(_dot(weights[class_index], row) + biases[class_index])
                error = probability - gold
                biases[class_index] -= epoch_learning_rate * error
                for feature_index, value in row.items():
                    weights[class_index][feature_index] -= epoch_learning_rate * (
                        (error * value) + (l2_weight * weights[class_index][feature_index])
                    )
    return weights, biases


def _train_binary_passive_aggressive(
    features: list[dict[int, float]],
    labels: list[str],
    *,
    positive_label: str,
    hashing_dimension: int,
    aggressiveness: float,
    epochs: int,
    seed: int,
) -> tuple[array, float]:
    weights = array("f", [0.0]) * hashing_dimension
    totals = array("d", [0.0]) * hashing_dimension
    timestamps = array("I", [0]) * hashing_dimension
    bias = 0.0
    bias_total = 0.0
    bias_timestamp = 0
    order = list(range(len(features)))
    rng = random.Random(seed)
    step = 1

    for _ in range(epochs):
        rng.shuffle(order)
        for row_index in order:
            row = features[row_index]
            gold = 1.0 if labels[row_index] == positive_label else -1.0
            score = _dot(weights, row) + bias
            loss = max(0.0, 1.0 - (gold * score))
            if loss > 0.0:
                norm_sq = sum(value * value for value in row.values()) or 1.0
                tau = min(aggressiveness, loss / (norm_sq + 1e-12))
                for feature_index, value in row.items():
                    totals[feature_index] += (step - timestamps[feature_index]) * weights[feature_index]
                    timestamps[feature_index] = step
                    weights[feature_index] += tau * gold * value
                bias_total += (step - bias_timestamp) * bias
                bias_timestamp = step
                bias += tau * gold
            step += 1

    divisor = max(step - 1, 1)
    for feature_index in range(hashing_dimension):
        totals[feature_index] += (step - timestamps[feature_index]) * weights[feature_index]
    bias_total += (step - bias_timestamp) * bias
    averaged_weights = array("f", [float(total / divisor) for total in totals])
    averaged_bias = bias_total / divisor
    return averaged_weights, averaged_bias


def _train_ovr_passive_aggressive(
    features: list[dict[int, float]],
    labels: list[str],
    *,
    label_order: list[str],
    hashing_dimension: int,
    aggressiveness: float,
    epochs: int,
    seed: int,
) -> tuple[list[array], list[float]]:
    weights: list[array] = []
    biases: list[float] = []
    for class_index, positive_label in enumerate(label_order):
        class_weights, class_bias = _train_binary_passive_aggressive(
            features,
            labels,
            positive_label=positive_label,
            hashing_dimension=hashing_dimension,
            aggressiveness=aggressiveness,
            epochs=epochs,
            seed=seed + class_index,
        )
        weights.append(class_weights)
        biases.append(class_bias)
    return weights, biases


def _dot(weights: array, row: dict[int, float]) -> float:
    total = 0.0
    for feature_index, value in row.items():
        total += weights[feature_index] * value
    return total


def _sigmoid(value: float) -> float:
    clipped = max(min(value, 30.0), -30.0)
    return 1.0 / (1.0 + math.exp(-clipped))


class _ConfigNamespace:
    def __init__(self, **kwargs: Any) -> None:
        for key, value in kwargs.items():
            setattr(self, key, value)


def _build_contextual_prediction_config(config_snapshot: dict[str, Any]) -> Any:
    return _ConfigNamespace(
        prediction=_ConfigNamespace(
            contextual_hashing_dimension=int(config_snapshot["contextual_hashing_dimension"]),
            contextual_section_sentence_limit=int(config_snapshot["contextual_section_sentence_limit"]),
            contextual_sentence_token_limit=int(config_snapshot["contextual_sentence_token_limit"]),
            min_token_chars=int(config_snapshot["min_token_chars"]),
            use_token_bigrams=bool(config_snapshot["use_token_bigrams"]),
        ),
        section_transfer=_ConfigNamespace(
            sentence_segmentation_method=str(config_snapshot["sentence_segmentation_method"]),
            sentence_segmentation_abbreviations=tuple(
                str(value) for value in config_snapshot["sentence_segmentation_abbreviations"]
            ),
        ),
    )


def _contextual_feature_counts(text: str, *, config: Any) -> Counter[int]:
    hashing_dimension = int(config.prediction.contextual_hashing_dimension)
    min_token_chars = int(config.prediction.min_token_chars)
    use_token_bigrams = bool(config.prediction.use_token_bigrams)
    section_sentence_limit = int(config.prediction.contextual_section_sentence_limit)
    sentence_token_limit = int(config.prediction.contextual_sentence_token_limit)

    counts: Counter[int] = Counter()
    sections = _extract_marked_sections(text)
    if sections:
        observed_sections = {section for section, _ in sections}
        for section in BROAD_SECTION_ORDER:
            counts[_contextual_hash(f"present:{section}:{int(section in observed_sections)}", hashing_dimension=hashing_dimension)] += 1
    else:
        sections = [("document", text)]

    counts[_contextual_hash(f"layout:{'>'.join(section for section, _ in sections)}", hashing_dimension=hashing_dimension)] += 1
    previous_section = None
    all_sentences: list[tuple[str, str]] = []
    for order_index, (section_name, section_text) in enumerate(sections):
        normalized_section = section_text.strip()
        if not normalized_section:
            continue
        section_sentences = [
            span.text for span in split_legal_text_into_sentences(normalized_section, config=config)
        ]
        if not section_sentences:
            section_sentences = [normalized_section]
        counts[_contextual_hash(f"section_order:{order_index}:{section_name}", hashing_dimension=hashing_dimension)] += 1
        counts[_contextual_hash(f"section_sentence_bucket:{section_name}:{_count_bucket(len(section_sentences))}", hashing_dimension=hashing_dimension)] += 1
        counts[_contextual_hash(f"section_char_bucket:{section_name}:{_char_bucket(len(normalized_section))}", hashing_dimension=hashing_dimension)] += 1
        if previous_section is not None:
            counts[_contextual_hash(f"transition:{previous_section}->{section_name}", hashing_dimension=hashing_dimension)] += 1
        previous_section = section_name

        _add_token_namespace_counts(
            counts,
            namespace=f"section:{section_name}",
            text=normalized_section,
            hashing_dimension=hashing_dimension,
            token_limit=max(sentence_token_limit * 3, sentence_token_limit),
            min_token_chars=min_token_chars,
            use_token_bigrams=use_token_bigrams,
        )

        leading_sentences = section_sentences[:section_sentence_limit]
        trailing_sentences = section_sentences[-section_sentence_limit:] if section_sentence_limit else []
        for sentence_index, sentence in enumerate(leading_sentences):
            _add_token_namespace_counts(
                counts,
                namespace=f"leading:{section_name}:{sentence_index}",
                text=sentence,
                hashing_dimension=hashing_dimension,
                token_limit=sentence_token_limit,
                min_token_chars=min_token_chars,
                use_token_bigrams=use_token_bigrams,
            )
        for sentence_index, sentence in enumerate(trailing_sentences):
            _add_token_namespace_counts(
                counts,
                namespace=f"trailing:{section_name}:{sentence_index}",
                text=sentence,
                hashing_dimension=hashing_dimension,
                token_limit=sentence_token_limit,
                min_token_chars=min_token_chars,
                use_token_bigrams=use_token_bigrams,
            )
        all_sentences.extend((section_name, sentence) for sentence in section_sentences)

    sentence_count = len(all_sentences) or 1
    counts[_contextual_hash(f"document_sentence_bucket:{_count_bucket(sentence_count)}", hashing_dimension=hashing_dimension)] += 1
    for sentence_index, (section_name, sentence) in enumerate(all_sentences):
        position_bucket = _sentence_position_bucket(sentence_index, sentence_count)
        _add_token_namespace_counts(
            counts,
            namespace=f"docpos:{position_bucket}:{section_name}",
            text=sentence,
            hashing_dimension=hashing_dimension,
            token_limit=sentence_token_limit,
            min_token_chars=min_token_chars,
            use_token_bigrams=False,
        )

    if all_sentences:
        _add_token_namespace_counts(
            counts,
            namespace="document_first",
            text=all_sentences[0][1],
            hashing_dimension=hashing_dimension,
            token_limit=sentence_token_limit,
            min_token_chars=min_token_chars,
            use_token_bigrams=use_token_bigrams,
        )
        _add_token_namespace_counts(
            counts,
            namespace="document_last",
            text=all_sentences[-1][1],
            hashing_dimension=hashing_dimension,
            token_limit=sentence_token_limit,
            min_token_chars=min_token_chars,
            use_token_bigrams=use_token_bigrams,
        )

    return counts


def _extract_marked_sections(text: str) -> list[tuple[str, str]]:
    matches = list(SECTION_MARKER_PATTERN.finditer(text))
    if not matches:
        return []
    sections: list[tuple[str, str]] = []
    for index, match in enumerate(matches):
        start = match.end()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        section_name = match.group(1).casefold()
        section_text = text[start:end].strip()
        if section_text:
            sections.append((section_name, section_text))
    return sections


def _add_token_namespace_counts(
    counts: Counter[int],
    *,
    namespace: str,
    text: str,
    hashing_dimension: int,
    token_limit: int,
    min_token_chars: int,
    use_token_bigrams: bool,
) -> None:
    tokens = tokenize_prediction_text(
        text,
        max_tokens=token_limit,
        min_token_chars=min_token_chars,
    )
    for token in tokens:
        counts[_contextual_hash(f"{namespace}:{token}", hashing_dimension=hashing_dimension)] += 1
    if use_token_bigrams:
        for left, right in zip(tokens, tokens[1:], strict=False):
            counts[_contextual_hash(f"{namespace}:bg:{left}__{right}", hashing_dimension=hashing_dimension)] += 1


def _contextual_hash(feature_name: str, *, hashing_dimension: int) -> int:
    return zlib.crc32(feature_name.encode("utf-8")) % hashing_dimension


def _count_bucket(value: int) -> str:
    if value <= 1:
        return "one"
    if value <= 3:
        return "few"
    if value <= 8:
        return "medium"
    return "many"


def _char_bucket(value: int) -> str:
    if value <= 200:
        return "short"
    if value <= 800:
        return "medium"
    if value <= 2000:
        return "long"
    return "very_long"


def _sentence_position_bucket(sentence_index: int, sentence_count: int) -> str:
    if sentence_count <= 1:
        return "singleton"
    normalized_position = sentence_index / max(sentence_count - 1, 1)
    if normalized_position <= 0.2:
        return "opening"
    if normalized_position <= 0.5:
        return "early_middle"
    if normalized_position <= 0.8:
        return "late_middle"
    return "closing"
