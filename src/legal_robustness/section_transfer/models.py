from __future__ import annotations

import logging
import math
import pickle
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from legal_robustness.config.schema import AppConfig
from legal_robustness.section_transfer.diagnostics import (
    BROAD_SECTION_ORDER,
    build_classification_metrics,
)
from legal_robustness.section_transfer.features import extract_features_from_parts, extract_record_features
from legal_robustness.section_transfer.types import RRSentenceSupervisionRecord, SectionTaggerTrainingResult
from legal_robustness.utils.exceptions import SectionTransferError


class BroadSectionNaiveBayesModel:
    def __init__(
        self,
        *,
        label_order: list[str],
        vocabulary: list[str],
        class_document_counts: dict[str, int],
        feature_counts_by_label: dict[str, dict[str, int]],
        total_feature_counts_by_label: dict[str, int],
        alpha: float = 1.0,
    ) -> None:
        self.label_order = label_order
        self.vocabulary = vocabulary
        self.vocabulary_set = set(vocabulary)
        self.class_document_counts = class_document_counts
        self.feature_counts_by_label = feature_counts_by_label
        self.total_feature_counts_by_label = total_feature_counts_by_label
        self.alpha = alpha
        self.total_documents = sum(class_document_counts.values())
        self._log_probability_cache: dict[str, dict[str, float]] = defaultdict(dict)

    @classmethod
    def train(
        cls,
        records: list[RRSentenceSupervisionRecord],
        *,
        config: AppConfig,
        alpha: float = 1.0,
    ) -> tuple["BroadSectionNaiveBayesModel", dict[str, Any]]:
        if not records:
            raise SectionTransferError("Cannot train the RR section tagger because no training records were supplied.")

        label_order = [
            label
            for label in BROAD_SECTION_ORDER
            if any(record.broad_section_label == label for record in records)
        ]
        unseen_labels = sorted({record.broad_section_label for record in records} - set(label_order))
        label_order.extend(unseen_labels)
        if not label_order:
            raise SectionTransferError("No broad labels were available for RR section tagger training.")

        feature_frequency: Counter[str] = Counter()
        cached_features: list[tuple[str, Counter[str]]] = []
        for record in records:
            features = extract_record_features(record, config=config)
            cached_features.append((record.broad_section_label, features))
            feature_frequency.update(features.keys())

        vocabulary = [
            feature_name
            for feature_name, _ in feature_frequency.most_common(config.section_transfer.max_vocabulary_size)
            if feature_frequency[feature_name] >= config.section_transfer.feature_min_count
        ]
        if not vocabulary:
            raise SectionTransferError("RR section tagger training produced an empty feature vocabulary.")
        vocabulary_set = set(vocabulary)

        class_document_counts: Counter[str] = Counter()
        feature_counts_by_label: dict[str, Counter[str]] = defaultdict(Counter)
        total_feature_counts_by_label: Counter[str] = Counter()
        for label, features in cached_features:
            class_document_counts[label] += 1
            for feature_name, value in features.items():
                if feature_name not in vocabulary_set:
                    continue
                feature_counts_by_label[label][feature_name] += value
                total_feature_counts_by_label[label] += value

        model = cls(
            label_order=label_order,
            vocabulary=vocabulary,
            class_document_counts=dict(class_document_counts),
            feature_counts_by_label={label: dict(counter) for label, counter in feature_counts_by_label.items()},
            total_feature_counts_by_label=dict(total_feature_counts_by_label),
            alpha=alpha,
        )
        training_summary = {
            "vocabulary_size": len(vocabulary),
            "label_order": label_order,
            "class_document_counts": dict(sorted(class_document_counts.items())),
        }
        return model, training_summary

    def predict(self, record: RRSentenceSupervisionRecord, *, config: AppConfig) -> str:
        scores = self.predict_log_scores(record, config=config)
        return max(scores.items(), key=lambda item: item[1])[0]

    def predict_proba(self, record: RRSentenceSupervisionRecord, *, config: AppConfig) -> dict[str, float]:
        scores = self.predict_log_scores(record, config=config)
        max_score = max(scores.values())
        stabilized = {label: math.exp(score - max_score) for label, score in scores.items()}
        total = sum(stabilized.values()) or 1.0
        return {
            label: round(value / total, 6)
            for label, value in sorted(stabilized.items())
        }

    def predict_log_scores(self, record: RRSentenceSupervisionRecord, *, config: AppConfig) -> dict[str, float]:
        features = extract_record_features(record, config=config)
        return self.predict_log_scores_from_features(features)

    def predict_from_parts(
        self,
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
    ) -> str:
        features = extract_features_from_parts(
            sentence_text=sentence_text,
            previous_context_text=previous_context_text,
            next_context_text=next_context_text,
            sentence_index=sentence_index,
            sentence_count=sentence_count,
            normalized_sentence_position=normalized_sentence_position,
            document_position_bucket=document_position_bucket,
            sentence_length_tokens_approx=sentence_length_tokens_approx,
            config=config,
        )
        scores = self.predict_log_scores_from_features(features)
        return max(scores.items(), key=lambda item: item[1])[0]

    def predict_proba_from_parts(
        self,
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
    ) -> dict[str, float]:
        features = extract_features_from_parts(
            sentence_text=sentence_text,
            previous_context_text=previous_context_text,
            next_context_text=next_context_text,
            sentence_index=sentence_index,
            sentence_count=sentence_count,
            normalized_sentence_position=normalized_sentence_position,
            document_position_bucket=document_position_bucket,
            sentence_length_tokens_approx=sentence_length_tokens_approx,
            config=config,
        )
        return self.predict_proba_from_features(features)

    def predict_proba_from_features(self, features: Counter[str]) -> dict[str, float]:
        scores = self.predict_log_scores_from_features(features)
        max_score = max(scores.values())
        stabilized = {label: math.exp(score - max_score) for label, score in scores.items()}
        total = sum(stabilized.values()) or 1.0
        return {
            label: round(value / total, 6)
            for label, value in sorted(stabilized.items())
        }

    def predict_log_scores_from_features(self, features: Counter[str]) -> dict[str, float]:
        filtered_features = {
            feature_name: value
            for feature_name, value in features.items()
            if feature_name in self.vocabulary_set
        }
        scores: dict[str, float] = {}
        label_count = len(self.label_order)
        for label in self.label_order:
            prior_numerator = self.class_document_counts.get(label, 0) + self.alpha
            prior_denominator = self.total_documents + (self.alpha * label_count)
            score = math.log(prior_numerator / prior_denominator)
            for feature_name, value in filtered_features.items():
                score += value * self._feature_log_probability(label, feature_name)
            scores[label] = score
        return scores

    def save(self, path: Path) -> None:
        with path.open("wb") as handle:
            pickle.dump(self, handle)

    def to_metadata(self) -> dict[str, Any]:
        return {
            "classifier_type": "multinomial_naive_bayes",
            "label_order": list(self.label_order),
            "vocabulary_size": len(self.vocabulary),
            "alpha": self.alpha,
            "class_document_counts": dict(self.class_document_counts),
        }

    def _feature_log_probability(self, label: str, feature_name: str) -> float:
        cached = self._log_probability_cache[label]
        if feature_name in cached:
            return cached[feature_name]
        feature_count = self.feature_counts_by_label.get(label, {}).get(feature_name, 0)
        denominator = self.total_feature_counts_by_label.get(label, 0) + (self.alpha * len(self.vocabulary))
        log_probability = math.log((feature_count + self.alpha) / denominator)
        cached[feature_name] = log_probability
        return log_probability


def train_and_evaluate_rr_section_tagger(
    supervision_records: list[RRSentenceSupervisionRecord],
    *,
    config: AppConfig,
    output_dir: Path,
    logger: logging.Logger | None = None,
) -> SectionTaggerTrainingResult:
    logger = logger or logging.getLogger(__name__)
    train_records = [record for record in supervision_records if record.split == "train"]
    if not train_records:
        raise SectionTransferError("RR section tagger training requires train-split supervision rows.")

    model, training_summary = BroadSectionNaiveBayesModel.train(train_records, config=config)
    split_records = {
        split_name: [record for record in supervision_records if record.split == split_name]
        for split_name in ("train", "dev", "test")
    }
    metrics_by_split: dict[str, dict[str, Any]] = {}
    prediction_samples: list[dict[str, Any]] = []
    warnings: list[str] = []
    evaluation_split = "test" if split_records["test"] else "dev" if split_records["dev"] else "train"
    confusion_matrix = None

    for split_name, records in split_records.items():
        if not records:
            warnings.append(f"No {split_name}-split RR supervision rows were available for evaluation.")
            continue
        gold_labels = [record.broad_section_label for record in records]
        predicted_labels = [model.predict(record, config=config) for record in records]
        split_metrics = build_classification_metrics(
            gold_labels,
            predicted_labels,
            label_order=model.label_order,
        )
        metrics_by_split[split_name] = split_metrics
        if split_name == evaluation_split:
            confusion_matrix = split_metrics["confusion_matrix"]
            prediction_samples = _build_prediction_samples(
                records,
                predicted_labels,
                model,
                config=config,
                sample_size=config.section_transfer.sample_size,
            )

    model_path = output_dir / "rr_section_tagger_model.pkl"
    metadata_path = output_dir / "rr_section_tagger_model_metadata.json"
    model.save(model_path)
    metrics = {
        "task": "rr_section_tagger",
        "classifier_type": config.section_transfer.classifier_type,
        "label_mode": config.section_transfer.label_mode,
        "label_order": model.label_order,
        "split_counts": {
            split_name: len(records)
            for split_name, records in sorted(split_records.items())
        },
        "vocabulary_size": training_summary["vocabulary_size"],
        "class_document_counts": training_summary["class_document_counts"],
        "feature_settings": {
            "use_position_features": config.section_transfer.use_position_features,
            "use_context_features": config.section_transfer.use_context_features,
            "use_token_bigrams": config.section_transfer.use_token_bigrams,
            "feature_min_count": config.section_transfer.feature_min_count,
            "max_vocabulary_size": config.section_transfer.max_vocabulary_size,
        },
        "metrics_by_split": metrics_by_split,
        "confusion_matrix_split": evaluation_split,
        "confusion_matrix": confusion_matrix or {},
        "warnings": warnings,
    }
    logger.info(
        "Trained RR section tagger with vocabulary size %s and evaluation split %s.",
        training_summary["vocabulary_size"],
        evaluation_split,
    )
    return SectionTaggerTrainingResult(
        model_path=str(model_path),
        metadata_path=str(metadata_path),
        metrics=metrics,
        prediction_samples=prediction_samples,
        warnings=warnings,
    )


def load_section_tagger_model(path: Path) -> BroadSectionNaiveBayesModel:
    with path.open("rb") as handle:
        model = pickle.load(handle)
    if not isinstance(model, BroadSectionNaiveBayesModel):
        raise SectionTransferError(
            f"Loaded object from {path} is not a BroadSectionNaiveBayesModel."
        )
    return model


def _build_prediction_samples(
    records: list[RRSentenceSupervisionRecord],
    predicted_labels: list[str],
    model: BroadSectionNaiveBayesModel,
    *,
    config: AppConfig,
    sample_size: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for record, predicted_label in zip(records, predicted_labels, strict=False):
        probabilities = model.predict_proba(record, config=config) if config.section_transfer.export_prediction_probabilities else {}
        rows.append(
            {
                "case_id": record.case_id,
                "split": record.split,
                "subset": record.subset,
                "sentence_index": record.sentence_index,
                "sentence_text": record.sentence_text,
                "gold_broad_section_label": record.broad_section_label,
                "predicted_broad_section_label": predicted_label,
                "correct": record.broad_section_label == predicted_label,
                "prediction_probabilities": probabilities,
            }
        )
        if len(rows) >= sample_size:
            break
    return rows
