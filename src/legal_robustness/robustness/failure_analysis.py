from __future__ import annotations

from collections import Counter, defaultdict
from itertools import combinations
from typing import Any

from legal_robustness.robustness.types import PerturbedPredictionRecord
from legal_robustness.section_transfer.types import CJPEPseudoSectionedCase


def build_failure_analysis(
    prediction_rows: list[PerturbedPredictionRecord],
    *,
    pseudo_sectioned_cases: list[CJPEPseudoSectionedCase],
    primary_split: str,
    focused_recipes: tuple[str, ...] | list[str],
    selected_model_variants: tuple[str, ...] | list[str],
    primary_model_variant: str | None,
    contextual_model_variant: str | None = None,
    case_limit: int,
    preview_chars: int,
    enable_disagreement_analysis: bool,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    case_lookup = {
        case.case_id: case
        for case in pseudo_sectioned_cases
        if case.split == primary_split
    }
    selected_variants = tuple(selected_model_variants)
    available_variants = {
        f"{row.model_name}::{row.input_variant}"
        for row in prediction_rows
        if row.split == primary_split
    }
    selected_variants = tuple(
        model_variant for model_variant in selected_variants if model_variant in available_variants
    )
    if not selected_variants:
        return (
            {
                "task": "failure_analysis",
                "primary_split": primary_split,
                "primary_model_variant": primary_model_variant,
                "focused_recipes": list(focused_recipes),
                "model_variants": [],
                "focused_recipe_summaries": [],
                "summary_takeaways": [
                    "Failure analysis could not run because no selected model variants were available in the perturbed predictions."
                ],
            },
            [],
        )

    if primary_model_variant not in selected_variants:
        primary_model_variant = selected_variants[0]
    contextual_model_variant = contextual_model_variant or _infer_model_variant(
        selected_variants,
        model_name_fragment="section_contextual_logistic_regression",
    )

    grouped_rows: dict[tuple[str, str], list[PerturbedPredictionRecord]] = defaultdict(list)
    for row in prediction_rows:
        if row.split != primary_split:
            continue
        if row.perturbation_recipe not in focused_recipes:
            continue
        grouped_rows[(row.perturbation_recipe, row.case_id)].append(row)

    focused_recipe_summaries: list[dict[str, Any]] = []
    export_rows: list[dict[str, Any]] = []
    summary_takeaways: list[str] = []

    for recipe_name in focused_recipes:
        case_rows_for_recipe = [
            (case_id, rows)
            for (group_recipe, case_id), rows in grouped_rows.items()
            if group_recipe == recipe_name
        ]
        case_rows_for_recipe.sort(key=lambda item: item[0])
        recipe_summary, recipe_export_rows = _build_recipe_failure_summary(
            recipe_name=recipe_name,
            grouped_rows=case_rows_for_recipe,
            case_lookup=case_lookup,
            selected_variants=selected_variants,
            primary_model_variant=primary_model_variant,
            contextual_model_variant=contextual_model_variant,
            case_limit=case_limit,
            preview_chars=preview_chars,
            enable_disagreement_analysis=enable_disagreement_analysis,
        )
        if recipe_summary is None:
            continue
        focused_recipe_summaries.append(recipe_summary)
        export_rows.extend(recipe_export_rows)
        if recipe_summary.get("takeaways"):
            summary_takeaways.extend(recipe_summary["takeaways"][:1])

    return (
        {
            "task": "failure_analysis",
            "primary_split": primary_split,
            "primary_model_variant": primary_model_variant,
            "contextual_model_variant": contextual_model_variant,
            "focused_recipes": list(focused_recipes),
            "model_variants": list(selected_variants),
            "focused_recipe_summaries": focused_recipe_summaries,
            "summary_takeaways": summary_takeaways,
        },
        export_rows,
    )


def _build_recipe_failure_summary(
    *,
    recipe_name: str,
    grouped_rows: list[tuple[str, list[PerturbedPredictionRecord]]],
    case_lookup: dict[str, CJPEPseudoSectionedCase],
    selected_variants: tuple[str, ...],
    primary_model_variant: str,
    contextual_model_variant: str | None,
    case_limit: int,
    preview_chars: int,
    enable_disagreement_analysis: bool,
) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
    if not grouped_rows:
        return None, []

    per_model_slices: dict[str, Counter[str]] = {
        model_variant: Counter() for model_variant in selected_variants
    }
    pairwise_prediction_disagreements: Counter[str] = Counter()
    pairwise_flip_disagreements: Counter[str] = Counter()
    case_entries: list[dict[str, Any]] = []
    prediction_disagreement_count = 0
    flip_disagreement_count = 0
    empty_target_count = 0
    non_empty_target_count = 0
    primary_model_correct_while_others_fail_count = 0
    primary_model_wrong_while_others_correct_count = 0
    primary_model_flipped_while_others_stable_count = 0
    contextual_unique_success_count = 0
    contextual_unique_failure_count = 0
    all_models_fail_count = 0
    all_models_flip_count = 0
    all_models_stable_count = 0
    linear_models_agree_contextual_diff_count = 0
    contextual_and_apa_agree_nb_logistic_diff_count = 0

    for case_id, rows in grouped_rows:
        row_by_variant = {
            f"{row.model_name}::{row.input_variant}": row
            for row in rows
        }
        if not row_by_variant:
            continue
        active_variants = [
            model_variant for model_variant in selected_variants if model_variant in row_by_variant
        ]
        if len(active_variants) < 2:
            continue

        primary_row = row_by_variant.get(primary_model_variant)
        if primary_row is None:
            continue

        case = case_lookup.get(case_id)
        representative_row = primary_row
        if representative_row.target_section_was_empty:
            empty_target_count += 1
        else:
            non_empty_target_count += 1

        per_model_payload: dict[str, Any] = {}
        predicted_labels = set()
        flip_values = set()
        correct_variants: list[str] = []
        wrong_variants: list[str] = []
        flipped_variants: list[str] = []
        stable_variants: list[str] = []
        correctness_by_variant: dict[str, bool] = {}
        prediction_by_variant: dict[str, str] = {}

        for model_variant in active_variants:
            row = row_by_variant[model_variant]
            correct = row.prediction == row.gold_label
            predicted_labels.add(row.prediction)
            flip_values.add(row.prediction_flipped)
            correctness_by_variant[model_variant] = correct
            prediction_by_variant[model_variant] = row.prediction
            if correct:
                correct_variants.append(model_variant)
            else:
                wrong_variants.append(model_variant)
            if row.prediction_flipped:
                flipped_variants.append(model_variant)
            else:
                stable_variants.append(model_variant)
            slice_name = _stability_correctness_slice(
                prediction_flipped=row.prediction_flipped,
                correct=correct,
            )
            per_model_slices[model_variant][slice_name] += 1
            per_model_payload[model_variant] = {
                "reference_prediction": row.reference_prediction,
                "perturbed_prediction": row.prediction,
                "reference_correct": row.reference_prediction == row.gold_label,
                "perturbed_correct": correct,
                "prediction_flipped": row.prediction_flipped,
                "prediction_score": row.prediction_score,
                "reference_prediction_score": row.reference_prediction_score,
            }

        prediction_disagreement = len(predicted_labels) > 1
        flip_disagreement = len(flip_values) > 1
        if prediction_disagreement:
            prediction_disagreement_count += 1
        if flip_disagreement:
            flip_disagreement_count += 1

        if enable_disagreement_analysis:
            for left_model, right_model in combinations(active_variants, 2):
                pair_key = f"{left_model}__vs__{right_model}"
                if row_by_variant[left_model].prediction != row_by_variant[right_model].prediction:
                    pairwise_prediction_disagreements[pair_key] += 1
                if row_by_variant[left_model].prediction_flipped != row_by_variant[right_model].prediction_flipped:
                    pairwise_flip_disagreements[pair_key] += 1

        primary_model_correct_while_others_fail = (
            primary_model_variant in correct_variants and bool(wrong_variants)
        )
        primary_model_wrong_while_others_correct = (
            primary_model_variant in wrong_variants and bool(correct_variants)
        )
        primary_model_flipped_while_others_stable = (
            primary_model_variant in flipped_variants and bool(stable_variants)
        )

        primary_model_correct_while_others_fail_count += int(primary_model_correct_while_others_fail)
        primary_model_wrong_while_others_correct_count += int(primary_model_wrong_while_others_correct)
        primary_model_flipped_while_others_stable_count += int(primary_model_flipped_while_others_stable)

        contextual_unique_success = False
        contextual_unique_failure = False
        linear_models_agree_contextual_diff = False
        contextual_and_apa_agree_nb_logistic_diff = False
        if contextual_model_variant and contextual_model_variant in active_variants:
            contextual_correct = correctness_by_variant.get(contextual_model_variant, False)
            linear_variants = [
                model_variant
                for model_variant in active_variants
                if model_variant != contextual_model_variant
            ]
            linear_correctness = [correctness_by_variant[model_variant] for model_variant in linear_variants]
            linear_predictions = [prediction_by_variant[model_variant] for model_variant in linear_variants]
            contextual_unique_success = contextual_correct and linear_correctness and all(not value for value in linear_correctness)
            contextual_unique_failure = (not contextual_correct) and any(linear_correctness)
            if linear_predictions:
                linear_models_agree_contextual_diff = (
                    len(set(linear_predictions)) == 1
                    and prediction_by_variant[contextual_model_variant] != linear_predictions[0]
                )
            apa_variant = _infer_model_variant(active_variants, model_name_fragment="averaged_passive_aggressive")
            nb_variant = _infer_model_variant(active_variants, model_name_fragment="multinomial_naive_bayes")
            logistic_variant = _infer_model_variant(active_variants, model_name_fragment="tfidf_logistic_regression")
            if all(value is not None for value in (apa_variant, nb_variant, logistic_variant)):
                contextual_and_apa_agree_nb_logistic_diff = (
                    prediction_by_variant.get(contextual_model_variant) == prediction_by_variant.get(apa_variant)
                    and prediction_by_variant.get(contextual_model_variant) != prediction_by_variant.get(nb_variant)
                    and prediction_by_variant.get(contextual_model_variant) != prediction_by_variant.get(logistic_variant)
                )

        contextual_unique_success_count += int(contextual_unique_success)
        contextual_unique_failure_count += int(contextual_unique_failure)
        all_models_fail_count += int(bool(active_variants) and all(not correctness_by_variant[variant] for variant in active_variants))
        all_models_flip_count += int(bool(active_variants) and all(row_by_variant[variant].prediction_flipped for variant in active_variants))
        all_models_stable_count += int(bool(active_variants) and all(not row_by_variant[variant].prediction_flipped for variant in active_variants))
        linear_models_agree_contextual_diff_count += int(linear_models_agree_contextual_diff)
        contextual_and_apa_agree_nb_logistic_diff_count += int(contextual_and_apa_agree_nb_logistic_diff)

        section_presence = _build_section_presence(case)
        section_previews = _build_section_previews(case, preview_chars=preview_chars)
        interestingness_score = (
            (3 if not representative_row.target_section_was_empty else 0)
            + (3 if prediction_disagreement else 0)
            + (2 if flip_disagreement else 0)
            + (2 if primary_model_correct_while_others_fail else 0)
            + (2 if primary_model_wrong_while_others_correct else 0)
            + (1 if primary_model_flipped_while_others_stable else 0)
            + (4 if contextual_unique_success else 0)
            + (3 if contextual_unique_failure else 0)
            + (2 if linear_models_agree_contextual_diff else 0)
            + (2 if contextual_and_apa_agree_nb_logistic_diff else 0)
            + (1 if bool(active_variants) and all(not correctness_by_variant[variant] for variant in active_variants) else 0)
            + (1 if bool(active_variants) and all(row_by_variant[variant].prediction_flipped for variant in active_variants) else 0)
        )
        case_entries.append(
            {
                "case_id": case_id,
                "split": representative_row.split,
                "gold_label": representative_row.gold_label,
                "perturbation_recipe": recipe_name,
                "target_section": representative_row.target_section,
                "target_section_was_empty": representative_row.target_section_was_empty,
                "section_presence_summary": section_presence,
                "section_previews": section_previews,
                "per_model_predictions": per_model_payload,
                "prediction_disagreement": prediction_disagreement,
                "flip_disagreement": flip_disagreement,
                "primary_model_correct_while_others_fail": primary_model_correct_while_others_fail,
                "primary_model_wrong_while_others_correct": primary_model_wrong_while_others_correct,
                "primary_model_flipped_while_others_stable": primary_model_flipped_while_others_stable,
                "contextual_unique_success": contextual_unique_success,
                "contextual_unique_failure": contextual_unique_failure,
                "all_models_fail": bool(active_variants) and all(not correctness_by_variant[variant] for variant in active_variants),
                "all_models_flip": bool(active_variants) and all(row_by_variant[variant].prediction_flipped for variant in active_variants),
                "all_models_stable": bool(active_variants) and all(not row_by_variant[variant].prediction_flipped for variant in active_variants),
                "linear_models_agree_contextual_diff": linear_models_agree_contextual_diff,
                "contextual_and_apa_agree_nb_logistic_diff": contextual_and_apa_agree_nb_logistic_diff,
                "interestingness_score": interestingness_score,
                "source_file": case.source_file if case is not None else representative_row.source_file,
                "source_metadata": dict(case.source_metadata) if case is not None else dict(representative_row.source_metadata),
            }
        )

    case_entries.sort(
        key=lambda row: (
            row["interestingness_score"],
            not row["target_section_was_empty"],
            row["prediction_disagreement"],
            row["flip_disagreement"],
        ),
        reverse=True,
    )
    recipe_export_rows = case_entries[:case_limit]
    total_case_count = len(case_entries)
    prediction_disagreement_rate = round(
        prediction_disagreement_count / total_case_count,
        6,
    ) if total_case_count else 0.0
    flip_disagreement_rate = round(
        flip_disagreement_count / total_case_count,
        6,
    ) if total_case_count else 0.0

    pairwise_prediction_summary = [
        {
            "pair_key": pair_key,
            "count": count,
            "rate": round(count / total_case_count, 6) if total_case_count else 0.0,
        }
        for pair_key, count in pairwise_prediction_disagreements.most_common()
    ]
    pairwise_flip_summary = [
        {
            "pair_key": pair_key,
            "count": count,
            "rate": round(count / total_case_count, 6) if total_case_count else 0.0,
        }
        for pair_key, count in pairwise_flip_disagreements.most_common()
    ]

    takeaways = [
        _build_recipe_takeaway(
            recipe_name=recipe_name,
            prediction_disagreement_rate=prediction_disagreement_rate,
            flip_disagreement_rate=flip_disagreement_rate,
            primary_model_variant=primary_model_variant,
            primary_model_correct_while_others_fail_count=primary_model_correct_while_others_fail_count,
            primary_model_wrong_while_others_correct_count=primary_model_wrong_while_others_correct_count,
            primary_model_flipped_while_others_stable_count=primary_model_flipped_while_others_stable_count,
        )
    ]

    return (
        {
            "recipe_name": recipe_name,
            "case_count": total_case_count,
            "non_empty_target_case_count": non_empty_target_count,
            "empty_target_case_count": empty_target_count,
            "prediction_disagreement_count": prediction_disagreement_count,
            "prediction_disagreement_rate": prediction_disagreement_rate,
            "flip_disagreement_count": flip_disagreement_count,
            "flip_disagreement_rate": flip_disagreement_rate,
            "primary_model_variant": primary_model_variant,
            "contextual_model_variant": contextual_model_variant,
            "primary_model_correct_while_others_fail_count": primary_model_correct_while_others_fail_count,
            "primary_model_wrong_while_others_correct_count": primary_model_wrong_while_others_correct_count,
            "primary_model_flipped_while_others_stable_count": primary_model_flipped_while_others_stable_count,
            "contextual_unique_success_count": contextual_unique_success_count,
            "contextual_unique_failure_count": contextual_unique_failure_count,
            "all_models_fail_count": all_models_fail_count,
            "all_models_flip_count": all_models_flip_count,
            "all_models_stable_count": all_models_stable_count,
            "linear_models_agree_contextual_diff_count": linear_models_agree_contextual_diff_count,
            "contextual_and_apa_agree_nb_logistic_diff_count": contextual_and_apa_agree_nb_logistic_diff_count,
            "per_model_stability_slices": {
                model_variant: dict(counter)
                for model_variant, counter in per_model_slices.items()
            },
            "pairwise_prediction_disagreement": pairwise_prediction_summary,
            "pairwise_flip_disagreement": pairwise_flip_summary,
            "takeaways": takeaways,
        },
        recipe_export_rows,
    )


def _build_recipe_takeaway(
    *,
    recipe_name: str,
    prediction_disagreement_rate: float,
    flip_disagreement_rate: float,
    primary_model_variant: str,
    primary_model_correct_while_others_fail_count: int,
    primary_model_wrong_while_others_correct_count: int,
    primary_model_flipped_while_others_stable_count: int,
) -> str:
    if primary_model_correct_while_others_fail_count > primary_model_wrong_while_others_correct_count:
        return (
            f"For `{recipe_name}`, `{primary_model_variant}` kept the correct label while at least one competing model failed in "
            f"`{primary_model_correct_while_others_fail_count}` focused cases, which suggests a real robustness advantage."
        )
    if primary_model_wrong_while_others_correct_count > 0:
        return (
            f"For `{recipe_name}`, `{primary_model_variant}` still lost head-to-head on "
            f"`{primary_model_wrong_while_others_correct_count}` focused cases, so the aggregate win is not uniform."
        )
    if primary_model_flipped_while_others_stable_count > 0:
        return (
            f"For `{recipe_name}`, `{primary_model_variant}` flipped while at least one competing model stayed stable in "
            f"`{primary_model_flipped_while_others_stable_count}` cases, which reinforces the stability-versus-correctness tradeoff."
        )
    return (
        f"For `{recipe_name}`, model disagreement rate was `{prediction_disagreement_rate}` and flip disagreement rate was "
        f"`{flip_disagreement_rate}`, which makes it a useful focused failure-analysis slice."
    )


def _stability_correctness_slice(*, prediction_flipped: bool, correct: bool) -> str:
    if not prediction_flipped and correct:
        return "stable_and_correct"
    if not prediction_flipped and not correct:
        return "stable_but_wrong"
    if prediction_flipped and correct:
        return "flipped_to_correct"
    return "flipped_to_wrong"


def _build_section_presence(case: CJPEPseudoSectionedCase | None) -> dict[str, bool]:
    if case is None:
        return {}
    return {
        section: bool(text.strip())
        for section, text in case.grouped_sections.items()
    }


def _build_section_previews(
    case: CJPEPseudoSectionedCase | None,
    *,
    preview_chars: int,
) -> dict[str, str]:
    if case is None:
        return {}
    return {
        section: _truncate_text(text, preview_chars=preview_chars)
        for section, text in case.grouped_sections.items()
        if text.strip()
    }


def _truncate_text(text: str, *, preview_chars: int) -> str:
    if len(text) <= preview_chars:
        return text
    return text[:preview_chars].rstrip() + "..."


def _infer_model_variant(
    model_variants: tuple[str, ...] | list[str],
    *,
    model_name_fragment: str,
) -> str | None:
    for model_variant in model_variants:
        if model_name_fragment in model_variant:
            return model_variant
    return None
