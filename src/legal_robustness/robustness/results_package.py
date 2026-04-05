from __future__ import annotations

from typing import Any


def build_table_main_results(
    apa_focused_table: dict[str, Any],
    *,
    primary_recipe: str,
    secondary_recipe: str,
) -> dict[str, Any]:
    row_lookup = {
        row["recipe_name"]: row
        for row in apa_focused_table.get("rows", [])
    }
    reference_row = (
        row_lookup.get(primary_recipe)
        or row_lookup.get(secondary_recipe)
        or (apa_focused_table.get("rows", [None])[0] if apa_focused_table.get("rows") else None)
    )
    rows: list[dict[str, Any]] = []
    if reference_row is not None:
        rows.append(
            {
                "condition_key": "unperturbed",
                "condition_label": "Unperturbed",
                "target_section": "all_sections",
                "macro_f1": reference_row["unperturbed_macro_f1"],
                "delta_macro_f1": 0.0,
                "accuracy": reference_row["unperturbed_accuracy"],
                "delta_accuracy": 0.0,
                "flip_rate": 0.0,
                "effective_coverage": 1.0,
                "coverage_band": "reference",
                "interpretation": "Reference APA performance on pseudo_all_sections before any section-aware perturbation is applied.",
            }
        )
    for recipe_name in (primary_recipe, secondary_recipe):
        row = row_lookup.get(recipe_name)
        if row is None:
            continue
        rows.append(
            {
                "condition_key": recipe_name,
                "condition_label": _recipe_display_name(recipe_name),
                "target_section": row["target_section"],
                "macro_f1": row["perturbed_macro_f1"],
                "delta_macro_f1": row["delta_macro_f1"],
                "accuracy": row["perturbed_accuracy"],
                "delta_accuracy": row["delta_accuracy"],
                "flip_rate": row["flip_rate"],
                "effective_coverage": row["effective_coverage"],
                "coverage_band": row["coverage_band"],
                "interpretation": row["short_interpretation"],
            }
        )
    return {
        "task": "table_main_results",
        "primary_model_variant": apa_focused_table.get("primary_model_variant"),
        "primary_split": apa_focused_table.get("primary_split"),
        "primary_recipe": primary_recipe,
        "secondary_recipe": secondary_recipe,
        "rows": rows,
    }


def render_table_main_results(report: dict[str, Any]) -> str:
    lines = [
        "# Table Main Results",
        "",
        f"- Primary split: `{report['primary_split']}`",
        f"- Primary model variant: `{report['primary_model_variant']}`",
        "",
        "| Condition | Macro F1 | Delta F1 | Accuracy | Delta Acc. | Flip Rate | Coverage | Interpretation |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in report.get("rows", []):
        lines.append(
            f"| {row['condition_label']} | {_fmt(row['macro_f1'])} | {_fmt(row['delta_macro_f1'])} | "
            f"{_fmt(row['accuracy'])} | {_fmt(row['delta_accuracy'])} | {_fmt(row['flip_rate'])} | "
            f"{_fmt(row['effective_coverage'])} | {row['interpretation']} |"
        )
    lines.extend(
        [
            "",
            "Pseudo-section caveat: these perturbations operate over transferred/predicted CJPE sections rather than gold section annotations.",
            "",
        ]
    )
    return "\n".join(lines)


def build_table_model_comparison(
    unperturbed_comparison: dict[str, Any],
    comparative_metrics: dict[str, Any],
    *,
    primary_model_variant: str,
    focused_recipes: tuple[str, ...] | list[str],
) -> dict[str, Any]:
    unperturbed_lookup = {
        row["model_variant"]: row
        for row in unperturbed_comparison.get("model_variants", [])
        if row.get("input_variant") == "pseudo_all_sections"
    }
    recipe_lookup = {
        row["recipe_name"]: row
        for row in comparative_metrics.get("recipes", [])
    }
    model_order = _ordered_model_variants(
        primary_model_variant=primary_model_variant,
        unperturbed_lookup=unperturbed_lookup,
    )
    rows: list[dict[str, Any]] = []
    for model_variant in model_order:
        unperturbed = unperturbed_lookup.get(model_variant)
        if unperturbed is None:
            continue
        row = {
            "model_variant": model_variant,
            "model_label": _model_display_name(model_variant),
            "unperturbed_accuracy": unperturbed["accuracy"],
            "unperturbed_macro_f1": unperturbed["macro_f1"],
        }
        for recipe_name in focused_recipes:
            recipe = recipe_lookup.get(recipe_name)
            model_metrics = _find_model_metrics(recipe, model_variant)
            prefix = recipe_name
            row[f"{prefix}_accuracy"] = model_metrics["accuracy"] if model_metrics else None
            row[f"{prefix}_macro_f1"] = model_metrics["macro_f1"] if model_metrics else None
            row[f"{prefix}_delta_macro_f1"] = (
                model_metrics["macro_f1_delta_vs_reference"] if model_metrics else None
            )
            row[f"{prefix}_flip_rate"] = model_metrics["flip_rate"] if model_metrics else None
        rows.append(row)
    return {
        "task": "table_model_comparison",
        "primary_split": unperturbed_comparison.get("primary_split"),
        "primary_input_variant": comparative_metrics.get("primary_input_variant"),
        "primary_model_variant": primary_model_variant,
        "focused_recipes": list(focused_recipes),
        "rows": rows,
    }


def render_table_model_comparison(report: dict[str, Any]) -> str:
    primary_recipe = report["focused_recipes"][0] if report.get("focused_recipes") else "keep_reasoning_only"
    secondary_recipe = report["focused_recipes"][1] if len(report.get("focused_recipes", [])) > 1 else "drop_precedents"
    lines = [
        "# Table Model Comparison",
        "",
        f"- Primary split: `{report['primary_split']}`",
        f"- Primary input variant: `{report['primary_input_variant']}`",
        f"- Primary model variant: `{report['primary_model_variant']}`",
        "",
        "| Model | Unpert. Acc. | Unpert. Macro F1 | "
        f"{_recipe_display_name(primary_recipe)} Macro F1 | {_recipe_display_name(primary_recipe)} Delta F1 | {_recipe_display_name(primary_recipe)} Flip | "
        f"{_recipe_display_name(secondary_recipe)} Macro F1 | {_recipe_display_name(secondary_recipe)} Delta F1 | {_recipe_display_name(secondary_recipe)} Flip |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in report.get("rows", []):
        lines.append(
            f"| {row['model_label']} | {_fmt(row['unperturbed_accuracy'])} | {_fmt(row['unperturbed_macro_f1'])} | "
            f"{_fmt(row.get(f'{primary_recipe}_macro_f1'))} | {_fmt(row.get(f'{primary_recipe}_delta_macro_f1'))} | {_fmt(row.get(f'{primary_recipe}_flip_rate'))} | "
            f"{_fmt(row.get(f'{secondary_recipe}_macro_f1'))} | {_fmt(row.get(f'{secondary_recipe}_delta_macro_f1'))} | {_fmt(row.get(f'{secondary_recipe}_flip_rate'))} |"
        )
    lines.append("")
    return "\n".join(lines)


def build_chart_data_main_performance(
    unperturbed_comparison: dict[str, Any],
) -> dict[str, Any]:
    rows = [
        {
            "model_variant": row["model_variant"],
            "model_label": _model_display_name(row["model_variant"]),
            "accuracy": row["accuracy"],
            "macro_f1": row["macro_f1"],
            "input_variant": row["input_variant"],
        }
        for row in unperturbed_comparison.get("model_variants", [])
        if row.get("input_variant") == "pseudo_all_sections"
    ]
    rows.sort(key=lambda row: row["macro_f1"], reverse=True)
    for rank, row in enumerate(rows, start=1):
        row["rank_by_macro_f1"] = rank
    return {
        "task": "chart_data_main_performance",
        "primary_split": unperturbed_comparison.get("primary_split"),
        "primary_input_variant": "pseudo_all_sections",
        "suggested_figure": "unperturbed_model_comparison_bar",
        "rows": rows,
    }


def build_chart_data_robustness_deltas(
    comparative_metrics: dict[str, Any],
    *,
    focused_recipes: tuple[str, ...] | list[str],
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for recipe in comparative_metrics.get("recipes", []):
        if recipe["recipe_name"] not in focused_recipes:
            continue
        for model_row in recipe.get("model_metrics", []):
            rows.append(
                {
                    "recipe_name": recipe["recipe_name"],
                    "recipe_label": _recipe_display_name(recipe["recipe_name"]),
                    "model_variant": model_row["model_variant"],
                    "model_label": _model_display_name(model_row["model_variant"]),
                    "accuracy_delta": model_row["accuracy_delta_vs_reference"],
                    "macro_f1_delta": model_row["macro_f1_delta_vs_reference"],
                    "accuracy_retention": model_row.get("accuracy_retention"),
                    "macro_f1_retention": model_row.get("macro_f1_retention"),
                }
            )
    return {
        "task": "chart_data_robustness_deltas",
        "primary_split": comparative_metrics.get("primary_split"),
        "suggested_figures": [
            "perturbation_delta_bar",
            "relative_retention_comparison",
        ],
        "rows": rows,
    }


def build_chart_data_flip_rates(
    comparative_metrics: dict[str, Any],
    stability_summary: dict[str, Any],
    *,
    focused_recipes: tuple[str, ...] | list[str],
) -> dict[str, Any]:
    flip_rows: list[dict[str, Any]] = []
    for recipe in comparative_metrics.get("recipes", []):
        if recipe["recipe_name"] not in focused_recipes:
            continue
        for model_row in recipe.get("model_metrics", []):
            flip_rows.append(
                {
                    "recipe_name": recipe["recipe_name"],
                    "recipe_label": _recipe_display_name(recipe["recipe_name"]),
                    "model_variant": model_row["model_variant"],
                    "model_label": _model_display_name(model_row["model_variant"]),
                    "flip_rate": model_row["flip_rate"],
                    "non_empty_target_flip_rate": model_row["non_empty_target_flip_rate"],
                }
            )
    stability_rows: list[dict[str, Any]] = []
    for recipe in stability_summary.get("recipe_summaries", []):
        if recipe["recipe_name"] not in focused_recipes:
            continue
        for model_variant, slices in recipe.get("per_model_stability_slices", {}).items():
            stability_rows.append(
                {
                    "recipe_name": recipe["recipe_name"],
                    "recipe_label": _recipe_display_name(recipe["recipe_name"]),
                    "model_variant": model_variant,
                    "model_label": _model_display_name(model_variant),
                    "stable_and_correct": slices.get("stable_and_correct", 0),
                    "stable_but_wrong": slices.get("stable_but_wrong", 0),
                    "flipped_to_correct": slices.get("flipped_to_correct", 0),
                    "flipped_to_wrong": slices.get("flipped_to_wrong", 0),
                }
            )
    return {
        "task": "chart_data_flip_rates",
        "primary_split": comparative_metrics.get("primary_split"),
        "suggested_figures": [
            "flip_rate_comparison",
            "stability_vs_correctness_stacked_bar",
        ],
        "flip_rate_rows": flip_rows,
        "stability_slice_rows": stability_rows,
    }


def build_chart_data_coverage(
    comparative_metrics: dict[str, Any],
    *,
    focused_recipes: tuple[str, ...] | list[str],
) -> dict[str, Any]:
    rows = [
        {
            "recipe_name": recipe["recipe_name"],
            "recipe_label": _recipe_display_name(recipe["recipe_name"]),
            "target_section": recipe["target_section"],
            "effective_non_empty_coverage": recipe["effective_non_empty_coverage"],
            "coverage_band": recipe["coverage_band"],
            "non_empty_target_count": recipe["non_empty_target_count"],
            "empty_target_count": recipe["empty_target_count"],
        }
        for recipe in comparative_metrics.get("recipes", [])
        if recipe["recipe_name"] in focused_recipes
    ]
    return {
        "task": "chart_data_coverage",
        "primary_split": comparative_metrics.get("primary_split"),
        "suggested_figures": [
            "coverage_aware_probe_comparison",
        ],
        "rows": rows,
    }


def build_table_stability_vs_correctness(
    stability_summary: dict[str, Any],
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for recipe in stability_summary.get("recipe_summaries", []):
        for pairwise in recipe.get("pairwise_primary_comparisons", []):
            rows.append(
                {
                    "recipe_name": recipe["recipe_name"],
                    "recipe_label": _recipe_display_name(recipe["recipe_name"]),
                    "primary_model_variant": stability_summary.get("primary_model_variant"),
                    "comparator_model_variant": pairwise["comparator_model_variant"],
                    "comparator_model_label": _model_display_name(pairwise["comparator_model_variant"]),
                    "primary_correct_comparator_wrong_count": pairwise["primary_correct_comparator_wrong_count"],
                    "comparator_correct_primary_wrong_count": pairwise["comparator_correct_primary_wrong_count"],
                    "primary_flipped_comparator_stable_count": pairwise["primary_flipped_comparator_stable_count"],
                    "comparator_flipped_primary_stable_count": pairwise["comparator_flipped_primary_stable_count"],
                    "joint_case_count": pairwise["joint_case_count"],
                }
            )
    return {
        "task": "table_stability_vs_correctness",
        "primary_split": stability_summary.get("primary_split"),
        "primary_model_variant": stability_summary.get("primary_model_variant"),
        "rows": rows,
    }


def render_table_stability_vs_correctness(report: dict[str, Any]) -> str:
    lines = [
        "# Table Stability vs Correctness",
        "",
        f"- Primary split: `{report['primary_split']}`",
        f"- Primary model variant: `{report['primary_model_variant']}`",
        "",
        "| Perturbation | Comparator | APA correct / comparator wrong | Comparator correct / APA wrong | APA flipped / comparator stable | Comparator flipped / APA stable |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for row in report.get("rows", []):
        lines.append(
            f"| {row['recipe_label']} | {row['comparator_model_label']} | "
            f"{row['primary_correct_comparator_wrong_count']} | {row['comparator_correct_primary_wrong_count']} | "
            f"{row['primary_flipped_comparator_stable_count']} | {row['comparator_flipped_primary_stable_count']} |"
        )
    lines.append("")
    return "\n".join(lines)


def build_stability_vs_correctness_narrative(
    stability_summary: dict[str, Any],
) -> dict[str, Any]:
    recipe_lookup = {
        row["recipe_name"]: row
        for row in stability_summary.get("recipe_summaries", [])
    }
    paragraphs: list[str] = []
    primary_model_variant = stability_summary.get("primary_model_variant")
    keep_reasoning = recipe_lookup.get("keep_reasoning_only")
    drop_precedents = recipe_lookup.get("drop_precedents")
    if keep_reasoning is not None:
        nb_pair = _find_pairwise_row(
            keep_reasoning,
            "multinomial_naive_bayes::pseudo_all_sections",
        )
        log_pair = _find_pairwise_row(
            keep_reasoning,
            "tfidf_logistic_regression::pseudo_all_sections",
        )
        reasoning_sentence = (
            f"Under `keep_reasoning_only`, `{primary_model_variant}` remains the strongest central model in absolute terms, "
            f"but its stability profile is more mixed than the headline score alone suggests."
        )
        if nb_pair is not None:
            reasoning_sentence += (
                f" Against naive Bayes, APA is correct while NB is wrong on `{nb_pair['primary_correct_comparator_wrong_count']}` cases, "
                f"versus `{nb_pair['comparator_correct_primary_wrong_count']}` in the opposite direction, but APA also flips while NB stays stable on "
                f"`{nb_pair['primary_flipped_comparator_stable_count']}` cases."
            )
        if log_pair is not None:
            reasoning_sentence += (
                f" The contrast with logistic regression is even sharper: APA flips while logistic stays stable on "
                f"`{log_pair['primary_flipped_comparator_stable_count']}` cases, whereas logistic flips while APA stays stable on only "
                f"`{log_pair['comparator_flipped_primary_stable_count']}`."
            )
        paragraphs.append(reasoning_sentence)
    if drop_precedents is not None:
        nb_pair = _find_pairwise_row(
            drop_precedents,
            "multinomial_naive_bayes::pseudo_all_sections",
        )
        log_pair = _find_pairwise_row(
            drop_precedents,
            "tfidf_logistic_regression::pseudo_all_sections",
        )
        precedent_sentence = (
            "Under `drop_precedents`, the same tension remains visible but is less extreme. "
            "Precedent removal produces a smaller aggregate degradation than reasoning-only ablation, yet it still exposes a meaningful tradeoff between correctness and stability."
        )
        if nb_pair is not None:
            precedent_sentence += (
                f" APA stays correct while NB is wrong on `{nb_pair['primary_correct_comparator_wrong_count']}` cases and loses that edge on "
                f"`{nb_pair['comparator_correct_primary_wrong_count']}` cases, which makes the APA-vs-NB comparison substantively balanced rather than one-sided."
            )
        if log_pair is not None:
            precedent_sentence += (
                f" Relative to logistic regression, APA gains a larger correctness advantage, but it also flips while logistic stays stable on "
                f"`{log_pair['primary_flipped_comparator_stable_count']}` cases."
            )
        paragraphs.append(precedent_sentence)
    paragraphs.append(
        "Taken together, these slices support a nuanced robustness claim: the strongest model is not automatically the most stable, and the lowest flip rate is not sufficient evidence of the best task performance. For the pilot paper-facing argument, APA remains central because it offers the strongest overall predictive baseline, while NB and logistic supply the key comparator evidence that retention and stability capture distinct behaviors under section-aware perturbation."
    )
    return {
        "task": "stability_vs_correctness_narrative",
        "primary_model_variant": primary_model_variant,
        "paragraphs": paragraphs,
    }


def render_stability_vs_correctness_narrative(report: dict[str, Any]) -> str:
    lines = [
        "# Stability vs Correctness Narrative",
        "",
        f"- Primary model variant: `{report['primary_model_variant']}`",
        "",
    ]
    lines.extend(report.get("paragraphs", []))
    lines.append("")
    return "\n".join(lines)


def build_recipe_qualitative_bundle(
    examples: list[dict[str, Any]],
    *,
    recipe_name: str,
    count: int,
    preview_chars: int,
) -> dict[str, Any]:
    filtered = [
        _truncate_example_previews(example, preview_chars=preview_chars)
        for example in examples
        if example.get("perturbation_recipe") == recipe_name
    ]
    filtered.sort(
        key=lambda row: (
            _qualitative_category_rank(recipe_name, row.get("selection_category", "")),
            row.get("priority_score", 0),
            row.get("interestingness_score", 0),
        ),
        reverse=True,
    )
    selected = filtered[:count]
    category_counts: dict[str, int] = {}
    for row in selected:
        category = row.get("selection_category", "unknown")
        category_counts[category] = category_counts.get(category, 0) + 1
    return {
        "task": "recipe_qualitative_bundle",
        "recipe_name": recipe_name,
        "recipe_label": _recipe_display_name(recipe_name),
        "selected_example_count": len(selected),
        "selection_category_counts": category_counts,
        "examples": selected,
    }


def render_recipe_qualitative_bundle(report: dict[str, Any]) -> str:
    lines = [
        f"# Qualitative Examples: {report['recipe_label']}",
        "",
        f"- Selected examples: `{report['selected_example_count']}`",
        f"- Category counts: `{report['selection_category_counts']}`",
        "",
    ]
    for example in report.get("examples", []):
        lines.extend(_render_example_block(example))
    return "\n".join(lines).strip() + "\n"


def build_results_narratives(
    table_main_results: dict[str, Any],
    table_model_comparison: dict[str, Any],
    packaging_next_step: dict[str, Any],
) -> dict[str, Any]:
    main_rows = {row["condition_key"]: row for row in table_main_results.get("rows", [])}
    model_rows = table_model_comparison.get("rows", [])
    ranked_models = sorted(
        model_rows,
        key=lambda row: row["unperturbed_macro_f1"],
        reverse=True,
    )
    main_paragraphs: list[str] = []
    if ranked_models:
        top = ranked_models[0]
        second = ranked_models[1] if len(ranked_models) > 1 else None
        paragraph = (
            f"On unperturbed `pseudo_all_sections`, {top['model_label']} is the strongest baseline in the current comparison "
            f"with macro F1 `{_fmt(top['unperturbed_macro_f1'])}` and accuracy `{_fmt(top['unperturbed_accuracy'])}`."
        )
        if second is not None:
            paragraph += (
                f" The nearest supporting comparator is {second['model_label']} at macro F1 `{_fmt(second['unperturbed_macro_f1'])}`, "
                f"which keeps the central APA-vs-NB comparison substantively close rather than trivial."
            )
        main_paragraphs.append(paragraph)
    reasoning = main_rows.get("keep_reasoning_only")
    precedents = main_rows.get("drop_precedents")
    if reasoning is not None:
        main_paragraphs.append(
            f"The primary probe is `keep_reasoning_only`. For APA, restricting the input to predicted reasoning text lowers macro F1 from "
            f"`{_fmt(main_rows['unperturbed']['macro_f1'])}` to `{_fmt(reasoning['macro_f1'])}` "
            f"(delta `{_fmt(reasoning['delta_macro_f1'])}`) and lowers accuracy from `{_fmt(main_rows['unperturbed']['accuracy'])}` "
            f"to `{_fmt(reasoning['accuracy'])}` (delta `{_fmt(reasoning['delta_accuracy'])}`), while flip rate rises to `{_fmt(reasoning['flip_rate'])}` at full coverage."
        )
    if precedents is not None:
        main_paragraphs.append(
            f"The secondary probe is `drop_precedents`. Removing predicted precedent content causes a smaller but still meaningful degradation for APA, "
            f"with macro F1 moving to `{_fmt(precedents['macro_f1'])}` (delta `{_fmt(precedents['delta_macro_f1'])}`) and accuracy moving to "
            f"`{_fmt(precedents['accuracy'])}` (delta `{_fmt(precedents['delta_accuracy'])}`) at coverage `{_fmt(precedents['effective_coverage'])}`."
        )
    main_paragraphs.append(
        "These pilot results support a focused claim rather than a broad one: reasoning-only ablation is the harsher stress test, precedent removal is a smaller but still interpretable disruption, and robustness should be discussed jointly in terms of absolute performance, metric retention, and prediction stability."
    )

    supporting_paragraphs = [
        "The supporting comparison models sharpen the interpretation. Naive Bayes often retains performance better by delta, while logistic regression is often the most stable by flip rate. Those patterns make it clear that absolute strength, retention, and stability should not be collapsed into a single notion of robustness.",
        "Conclusion-target perturbations remain deliberately de-emphasized because transferred conclusion coverage is weak. They can stay in caveat-heavy appendices, but they are not strong enough to anchor the main results argument.",
        "The contextual approximation is retained as context rather than promoted. In the current environment it does not beat the best simple baselines, so it helps bound expectations without changing the main APA-centered story.",
        f"Given the current evidence, the package now supports the manuscript-facing claim that {packaging_next_step.get('central_claim', 'section-aware perturbations reveal different forms of robustness even among simple document classifiers')}."
    ]
    return {
        "task": "results_narratives",
        "main_paragraphs": main_paragraphs,
        "supporting_paragraphs": supporting_paragraphs,
    }


def render_results_narrative_main(report: dict[str, Any]) -> str:
    lines = ["# Results Narrative Main", ""]
    lines.extend(report.get("main_paragraphs", []))
    lines.append("")
    return "\n".join(lines)


def render_results_narrative_supporting(report: dict[str, Any]) -> str:
    lines = ["# Results Narrative Supporting", ""]
    lines.extend(report.get("supporting_paragraphs", []))
    lines.append("")
    return "\n".join(lines)


def build_appendix_bundle(
    *,
    recipe_name: str,
    recipe_summary_row: dict[str, Any] | None,
    model_comparison: dict[str, Any],
    qualitative_bundle: dict[str, Any],
    case_bundle_filename: str,
) -> dict[str, Any]:
    comparison_rows: list[dict[str, Any]] = []
    for row in model_comparison.get("rows", []):
        comparison_rows.append(
            {
                "model_label": row["model_label"],
                "macro_f1": row.get(f"{recipe_name}_macro_f1"),
                "delta_macro_f1": row.get(f"{recipe_name}_delta_macro_f1"),
                "flip_rate": row.get(f"{recipe_name}_flip_rate"),
            }
        )
    return {
        "task": "appendix_bundle",
        "recipe_name": recipe_name,
        "recipe_label": _recipe_display_name(recipe_name),
        "recipe_summary_row": recipe_summary_row,
        "model_comparison_rows": comparison_rows,
        "qualitative_bundle": qualitative_bundle,
        "case_bundle_filename": case_bundle_filename,
        "methodological_caveat": "All section references in this bundle come from transferred/predicted CJPE pseudo-sections rather than gold section annotations.",
        "short_interpretation": (
            "This probe removes supporting precedent content while leaving the rest of the pseudo-sectioned document intact."
            if recipe_name == "drop_precedents"
            else "This probe keeps only the predicted reasoning section and tests how much decision signal survives without facts or precedents."
        ),
    }


def render_appendix_bundle(report: dict[str, Any]) -> str:
    summary_row = report.get("recipe_summary_row") or {}
    lines = [
        f"# Appendix Bundle: {report['recipe_label']}",
        "",
        f"- Methodological caveat: {report['methodological_caveat']}",
        f"- Interpretation: {report['short_interpretation']}",
        f"- Case bundle reference: `{report['case_bundle_filename']}`",
        "",
        "## Compact Summary Table",
        "",
        "| Model | Macro F1 | Delta F1 | Flip Rate |",
        "| --- | ---: | ---: | ---: |",
    ]
    for row in report.get("model_comparison_rows", []):
        lines.append(
            f"| {row['model_label']} | {_fmt(row['macro_f1'])} | {_fmt(row['delta_macro_f1'])} | {_fmt(row['flip_rate'])} |"
        )
    if summary_row:
        lines.extend(
            [
                "",
                f"Coverage for this probe is `{_fmt(summary_row.get('effective_coverage'))}` and the APA-centered interpretation is: {summary_row.get('interpretation')}",
                "",
            ]
        )
    lines.extend(["## Selected Qualitative Examples", ""])
    for example in report.get("qualitative_bundle", {}).get("examples", []):
        lines.extend(_render_example_block(example))
    return "\n".join(lines).strip() + "\n"


def build_results_package_manifest(
    *,
    baseline_run_dir: str | Any,
    robustness_run_dir: str | Any,
    package_dirname: str,
    primary_model_variant: str,
    models_included: list[str],
    perturbations_included: list[str],
    chart_data_files: list[str],
    main_summary_files: list[str],
    qualitative_files: list[str],
    appendix_files: list[str],
    caveats: list[str],
) -> dict[str, Any]:
    return {
        "task": "results_package_manifest",
        "baseline_run_dir": str(baseline_run_dir),
        "robustness_run_dir": str(robustness_run_dir),
        "results_package_dirname": package_dirname,
        "primary_model_variant": primary_model_variant,
        "models_included": models_included,
        "perturbations_included": perturbations_included,
        "main_summary_files": main_summary_files,
        "chart_data_files": chart_data_files,
        "qualitative_files": qualitative_files,
        "appendix_files": appendix_files,
        "known_caveats": caveats,
    }


def render_results_package_manifest(report: dict[str, Any]) -> str:
    lines = [
        "# Results Package Manifest",
        "",
        f"- Baseline run dir: `{report['baseline_run_dir']}`",
        f"- Robustness run dir: `{report['robustness_run_dir']}`",
        f"- Results package dir: `{report['results_package_dirname']}`",
        f"- Primary model variant: `{report['primary_model_variant']}`",
        f"- Models included: `{report['models_included']}`",
        f"- Perturbations included: `{report['perturbations_included']}`",
        "",
        "## Main Summary Files",
        "",
    ]
    for path in report.get("main_summary_files", []):
        lines.append(f"- `{path}`")
    lines.extend(["", "## Chart Data Files", ""])
    for path in report.get("chart_data_files", []):
        lines.append(f"- `{path}`")
    lines.extend(["", "## Qualitative Files", ""])
    for path in report.get("qualitative_files", []):
        lines.append(f"- `{path}`")
    lines.extend(["", "## Appendix Files", ""])
    for path in report.get("appendix_files", []):
        lines.append(f"- `{path}`")
    lines.extend(["", "## Known Caveats", ""])
    for caveat in report.get("known_caveats", []):
        lines.append(f"- {caveat}")
    lines.append("")
    return "\n".join(lines)


def build_paper_results_packaging_next_step_summary(
    comparative_next_step_summary: dict[str, Any],
) -> dict[str, Any]:
    primary_model_variant = comparative_next_step_summary.get("recommended_primary_model_variant")
    primary_probe = comparative_next_step_summary.get("primary_writeup_perturbation")
    secondary_probe = comparative_next_step_summary.get("secondary_writeup_perturbation")
    ready = comparative_next_step_summary.get("ready_for_pilot_results_section", False)
    return {
        "task": "paper_results_packaging_next_step_summary",
        "primary_model_variant": primary_model_variant,
        "primary_probe": primary_probe,
        "secondary_probe": secondary_probe,
        "ready_to_begin_pilot_results_drafting": ready,
        "one_more_targeted_experiment_necessary": not ready,
        "central_claim": "On pseudo-sectioned CJPE, reasoning-only ablation produces the strongest performance drop for the APA baseline, while precedent removal yields a smaller but still meaningful degradation; across models, absolute strength, retention, and stability capture distinct robustness behaviors.",
        "visible_caveats": [
            "CJPE sections are transferred/predicted from RR supervision rather than gold annotations.",
            "The main results package is intentionally focused on keep_reasoning_only and drop_precedents.",
            "The contextual approximation is included as comparison context rather than the main model.",
        ],
        "recommendations": [
            f"Begin drafting the pilot results section with `{primary_model_variant}` as the central model." if primary_model_variant else "Begin drafting the pilot results section with APA as the central model.",
            f"Use `{primary_probe}` as the primary probe and `{secondary_probe}` as the secondary probe." if primary_probe and secondary_probe else "Use the focused perturbation probes as the core paper evidence.",
            "Do not block drafting on one more experiment unless the manuscript later needs a targeted rebuttal-style follow-up.",
            "Keep the pseudo-section caveat visible in both the main text and appendix-style evidence bundles.",
        ],
    }


def render_paper_results_packaging_next_step_summary(report: dict[str, Any]) -> str:
    lines = [
        "# Paper Results Packaging Next-Step Summary",
        "",
        f"- Primary model variant: `{report['primary_model_variant']}`",
        f"- Primary probe: `{report['primary_probe']}`",
        f"- Secondary probe: `{report['secondary_probe']}`",
        f"- Ready to begin pilot results drafting: `{report['ready_to_begin_pilot_results_drafting']}`",
        f"- One more targeted experiment necessary: `{report['one_more_targeted_experiment_necessary']}`",
        "",
        "## Central Claim",
        "",
        report["central_claim"],
        "",
        "## Visible Caveats",
        "",
    ]
    for caveat in report.get("visible_caveats", []):
        lines.append(f"- {caveat}")
    lines.extend(["", "## Recommendations", ""])
    for recommendation in report.get("recommendations", []):
        lines.append(f"- {recommendation}")
    lines.append("")
    return "\n".join(lines)


def _ordered_model_variants(
    *,
    primary_model_variant: str,
    unperturbed_lookup: dict[str, dict[str, Any]],
) -> list[str]:
    variants = sorted(
        unperturbed_lookup.keys(),
        key=lambda key: unperturbed_lookup[key]["macro_f1"],
        reverse=True,
    )
    if primary_model_variant in variants:
        variants.remove(primary_model_variant)
        return [primary_model_variant] + variants
    return variants


def _find_model_metrics(recipe: dict[str, Any] | None, model_variant: str) -> dict[str, Any] | None:
    if recipe is None:
        return None
    for row in recipe.get("model_metrics", []):
        if row["model_variant"] == model_variant:
            return row
    return None


def _find_pairwise_row(recipe_summary: dict[str, Any], comparator_variant: str) -> dict[str, Any] | None:
    for row in recipe_summary.get("pairwise_primary_comparisons", []):
        if row["comparator_model_variant"] == comparator_variant:
            return row
    return None


def _truncate_example_previews(example: dict[str, Any], *, preview_chars: int) -> dict[str, Any]:
    updated = dict(example)
    updated["section_previews"] = {
        key: _truncate_text(str(value), preview_chars=preview_chars)
        for key, value in dict(example.get("section_previews", {})).items()
    }
    return updated


def _render_example_block(example: dict[str, Any]) -> list[str]:
    lines = [
        f"## Case {example['case_id']} | {example['selection_category']}",
        "",
        f"- Gold label: `{example['gold_label']}`",
        f"- Selection rationale: {example['selection_reason']}",
        f"- Short interpretation: {_qualitative_interpretation(example)}",
        f"- Section presence: `{example['section_presence_summary']}`",
        "- Predictions before/after:",
    ]
    for model_variant, payload in example.get("per_model_predictions", {}).items():
        lines.append(
            f"  `{_model_display_name(model_variant)}`: `{payload['reference_prediction']}` -> `{payload['perturbed_prediction']}`, "
            f"flipped `{payload['prediction_flipped']}`, perturbed_correct `{payload['perturbed_correct']}`"
        )
    lines.append("- Section previews:")
    for section_name, preview in example.get("section_previews", {}).items():
        lines.append(f"  `{section_name}`: {preview}")
    lines.append("")
    return lines


def _qualitative_interpretation(example: dict[str, Any]) -> str:
    category = example.get("selection_category")
    if category == "apa_unique_success":
        return "APA preserves the correct prediction where most competing models do not, making the case useful for the main qualitative argument."
    if category == "apa_unique_failure":
        return "APA misses the correct label despite at least one simpler model succeeding, which makes the case useful as counterevidence and keeps the write-up honest."
    if category == "precedent_sensitive_flip":
        return "Removing precedents changes the decision enough to trigger an APA flip, which makes the case a clean precedent-sensitivity example."
    if category == "reasoning_sufficient_stable_case":
        return "Reasoning alone appears sufficient for the correct APA decision here, even after the rest of the pseudo-sectioned document is removed."
    if category == "reasoning_insufficient_failure_case":
        return "Reasoning alone is not enough to preserve the correct APA decision on this case, which makes the ablation cost concrete."
    if category == "consensus_failure":
        return "All models fail after perturbation, so the example highlights a shared weakness rather than a model-specific error."
    if category == "consensus_robustness":
        return "All models remain correct after perturbation, making this a useful control case rather than a failure case."
    return "This case was selected because it remains qualitatively informative even after the higher-priority categories were filled."


def _qualitative_category_rank(recipe_name: str, category: str) -> int:
    if recipe_name == "keep_reasoning_only":
        order = [
            "apa_unique_success",
            "reasoning_sufficient_stable_case",
            "reasoning_insufficient_failure_case",
            "apa_unique_failure",
            "consensus_failure",
            "consensus_robustness",
            "high_interest_remainder",
        ]
    else:
        order = [
            "precedent_sensitive_flip",
            "apa_unique_success",
            "apa_unique_failure",
            "consensus_robustness",
            "consensus_failure",
            "high_interest_remainder",
        ]
    if category not in order:
        return 0
    return len(order) - order.index(category)


def _model_display_name(model_variant: str) -> str:
    if "averaged_passive_aggressive" in model_variant:
        return "APA"
    if "multinomial_naive_bayes" in model_variant:
        return "NB"
    if "tfidf_logistic_regression" in model_variant:
        return "Logistic"
    if "section_contextual_logistic_regression" in model_variant:
        return "Contextual approx."
    return model_variant


def _recipe_display_name(recipe_name: str) -> str:
    mapping = {
        "drop_precedents": "Drop Precedents",
        "keep_reasoning_only": "Reasoning Only",
        "keep_facts_reasoning": "Facts + Reasoning",
        "drop_conclusion": "Drop Conclusion",
        "mask_conclusion": "Mask Conclusion",
    }
    return mapping.get(recipe_name, recipe_name.replace("_", " ").title())


def _truncate_text(text: str, *, preview_chars: int) -> str:
    if len(text) <= preview_chars:
        return text
    return text[:preview_chars].rstrip() + "..."


def _fmt(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        if value == int(value):
            return str(int(value))
        return f"{value:.6f}"
    return str(value)
