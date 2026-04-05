from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from legal_robustness.config.schema import AppConfig
from legal_robustness.prediction.datasets import (
    build_prediction_example_samples,
    build_prediction_examples,
    group_prediction_examples_by_split,
)
from legal_robustness.prediction.evaluate import (
    confusion_matrix_rows_for_variant,
    evaluate_prediction_examples,
)
from legal_robustness.prediction.models import (
    AveragedPassiveAggressiveModel,
    MultinomialNaiveBayesTextModel,
    SectionContextualLogisticRegressionModel,
    TfidfLogisticRegressionModel,
)
from legal_robustness.prediction.types import BaselineModelRunResult, BaselinePredictionRecord
from legal_robustness.utils.exceptions import PredictionError


def train_prediction_baselines(
    pseudo_sectioned_cases,
    *,
    config: AppConfig,
    output_dir: Path,
    source_section_transfer_run_dir: Path,
    logger: logging.Logger | None = None,
) -> tuple[dict[str, Any], dict[str, list[BaselinePredictionRecord]], list[dict[str, Any]], list[dict[str, Any]]]:
    logger = logger or logging.getLogger(__name__)
    model_results: dict[str, dict[str, Any]] = {}
    predictions_by_split: dict[str, list[BaselinePredictionRecord]] = {
        split: [] for split in config.prediction.evaluation_splits
    }
    confusion_rows: list[dict[str, Any]] = []
    sample_rows: list[dict[str, Any]] = []
    warnings: list[str] = []

    for model_name in config.prediction.baseline_models:
        model_results[model_name] = {}
        for variant_name in config.prediction.input_variants:
            if (
                model_name == "section_contextual_logistic_regression"
                and variant_name not in config.prediction.contextual_input_variants
            ):
                warnings.append(
                    f"Skipping {model_name}/{variant_name} because contextual_input_variants is restricted to {config.prediction.contextual_input_variants}."
                )
                continue
            logger.info("Training %s baseline for input variant %s", model_name, variant_name)
            examples = build_prediction_examples(
                pseudo_sectioned_cases,
                variant_name=variant_name,
                config=config,
            )
            split_examples = group_prediction_examples_by_split(examples)
            train_examples = split_examples.get("train", [])
            if not train_examples:
                warnings.append(f"Skipping {model_name}/{variant_name} because no train split examples were available.")
                continue

            train_texts = [example.input_text for example in train_examples]
            train_labels = [str(example.label) for example in train_examples]
            model = _train_prediction_model(
                model_name=model_name,
                train_texts=train_texts,
                train_labels=train_labels,
                config=config,
            )
            model_dir = output_dir / "models"
            model_dir.mkdir(parents=True, exist_ok=True)
            model_path = model_dir / (
                f"{config.prediction.model_artifact_prefix}__{model_name}__{variant_name}.pkl"
            )
            model.save(model_path)

            metrics_by_split: dict[str, Any] = {}
            split_counts = {
                split_name: len(rows)
                for split_name, rows in sorted(split_examples.items())
            }
            for split_name in config.prediction.evaluation_splits:
                examples_for_split = split_examples.get(split_name, [])
                if not examples_for_split:
                    warnings.append(f"No {split_name} examples were available for {model_name}/{variant_name}.")
                    continue
                metrics, prediction_rows = evaluate_prediction_examples(
                    examples=examples_for_split,
                    model=model,
                    model_name=model_name,
                    input_variant=variant_name,
                )
                metrics_by_split[split_name] = metrics
                predictions_by_split.setdefault(split_name, []).extend(prediction_rows)
                confusion_rows.extend(
                    confusion_matrix_rows_for_variant(
                        confusion_matrix=metrics["confusion_matrix"],
                        label_order=model.label_order,
                        model_name=model_name,
                        input_variant=variant_name,
                        split=split_name,
                    )
                )

            sample_rows.extend(
                build_prediction_example_samples(
                    examples,
                    sample_size=min(config.prediction.sample_size, 2),
                    preview_chars=config.prediction.prediction_preview_chars,
                )
            )
            model_results[model_name][variant_name] = BaselineModelRunResult(
                model_name=model_name,
                input_variant=variant_name,
                model_path=str(model_path),
                metrics_by_split=metrics_by_split,
                training_summary={
                    **model.to_metadata(),
                    "split_counts": split_counts,
                    "average_train_input_length_chars": round(
                        sum(example.input_text_length_chars for example in train_examples) / len(train_examples),
                        3,
                    ),
                },
                warnings=[],
            ).to_dict()

    report = {
        "task": "baseline_prediction",
        "source_section_transfer_run_dir": str(source_section_transfer_run_dir),
        "baseline_models": list(config.prediction.baseline_models),
        "input_variants": list(config.prediction.input_variants),
        "evaluation_splits": list(config.prediction.evaluation_splits),
        "models": model_results,
        "test_variant_deltas_vs_full_text": _variant_deltas_vs_full_text(model_results),
        "interpretation": _build_interpretation(model_results),
        "warnings": warnings,
    }
    return report, predictions_by_split, confusion_rows, sample_rows


def _variant_deltas_vs_full_text(model_results: dict[str, dict[str, Any]]) -> dict[str, Any]:
    deltas: dict[str, Any] = {}
    for model_name, variants in model_results.items():
        full_text_metrics = variants.get("full_text", {}).get("metrics_by_split", {}).get("test")
        if not full_text_metrics:
            continue
        model_deltas: dict[str, Any] = {}
        for variant_name, variant_report in variants.items():
            if variant_name == "full_text":
                continue
            test_metrics = variant_report.get("metrics_by_split", {}).get("test")
            if not test_metrics:
                continue
            model_deltas[variant_name] = {
                "accuracy_delta": round(test_metrics["accuracy"] - full_text_metrics["accuracy"], 6),
                "macro_f1_delta": round(test_metrics["macro_f1"] - full_text_metrics["macro_f1"], 6),
            }
        if model_deltas:
            deltas[model_name] = model_deltas
    return deltas


def _build_interpretation(model_results: dict[str, dict[str, Any]]) -> list[str]:
    ranked: list[tuple[str, str, float]] = []
    for model_name, variants in model_results.items():
        for variant_name, variant_report in variants.items():
            test_metrics = variant_report.get("metrics_by_split", {}).get("test")
            if test_metrics:
                ranked.append((model_name, variant_name, test_metrics["macro_f1"]))
    ranked.sort(key=lambda item: item[2], reverse=True)
    interpretations: list[str] = []
    if ranked:
        interpretations.append(
            f"Best test macro F1 in this run came from `{ranked[0][0]} / {ranked[0][1]}` at `{ranked[0][2]}`."
        )
    for model_name, variants in model_results.items():
        if variants.get("pseudo_reasoning_only", {}).get("metrics_by_split", {}).get("test"):
            interpretations.append(
                f"For `{model_name}`, the reasoning-only variant remains a direct first probe of how much predictive signal survives section-aware compression."
            )
        if variants.get("pseudo_without_conclusion", {}).get("metrics_by_split", {}).get("test"):
            interpretations.append(
                f"For `{model_name}`, the without-conclusion variant must still be interpreted carefully because transferred conclusion labels are the noisiest pseudo-section."
            )
        if variants.get("pseudo_without_precedents", {}).get("metrics_by_split", {}).get("test"):
            interpretations.append(
                f"For `{model_name}`, the without-precedents variant remains the clean baseline for precedent-drop and distractor experiments."
            )
    return interpretations


def _train_prediction_model(
    *,
    model_name: str,
    train_texts: list[str],
    train_labels: list[str],
    config: AppConfig,
):
    if model_name == "tfidf_logistic_regression":
        return TfidfLogisticRegressionModel.train(
            train_texts,
            train_labels,
            config=config,
        )
    if model_name == "multinomial_naive_bayes":
        return MultinomialNaiveBayesTextModel.train(
            train_texts,
            train_labels,
            config=config,
        )
    if model_name == "averaged_passive_aggressive":
        return AveragedPassiveAggressiveModel.train(
            train_texts,
            train_labels,
            config=config,
        )
    if model_name == "section_contextual_logistic_regression":
        return SectionContextualLogisticRegressionModel.train(
            train_texts,
            train_labels,
            config=config,
        )
    raise PredictionError(f"Unsupported baseline model requested: {model_name}")
