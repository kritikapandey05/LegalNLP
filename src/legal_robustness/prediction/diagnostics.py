from __future__ import annotations

from itertools import combinations
from typing import Any


def render_baseline_prediction_metrics(report: dict[str, Any]) -> str:
    lines = [
        "# Baseline Prediction Metrics",
        "",
        f"- Source section-transfer run: `{report['source_section_transfer_run_dir']}`",
        f"- Baseline models: `{report['baseline_models']}`",
        f"- Input variants: `{report['input_variants']}`",
        f"- Evaluation splits: `{report['evaluation_splits']}`",
        "- Note: pseudo-section variants use transferred/predicted sections, not gold section annotation.",
        "",
    ]
    for model_name, variants in report["models"].items():
        lines.extend([f"## {model_name}", ""])
        for variant_name, variant_report in variants.items():
            lines.extend(
                [
                    f"### {variant_name}",
                    "",
                    f"- Train cases: `{variant_report['training_summary']['split_counts'].get('train', 0)}`",
                    f"- Dev cases: `{variant_report['training_summary']['split_counts'].get('dev', 0)}`",
                    f"- Test cases: `{variant_report['training_summary']['split_counts'].get('test', 0)}`",
                    f"- Label order: `{variant_report['training_summary']['label_order']}`",
                    "",
                ]
            )
            for split_name, metrics in variant_report["metrics_by_split"].items():
                lines.extend(
                    [
                        f"- {split_name} accuracy: `{metrics['accuracy']}`",
                        f"- {split_name} macro F1: `{metrics['macro_f1']}`",
                    ]
                )
            lines.append("")

    comparisons = report.get("test_variant_deltas_vs_full_text", {})
    if comparisons:
        lines.extend(["## Test-Set Deltas Vs Full Text", ""])
        for model_name, model_comparisons in comparisons.items():
            for variant_name, delta_report in model_comparisons.items():
                lines.append(
                    f"- `{model_name} / {variant_name}`: accuracy delta `{delta_report['accuracy_delta']}`, "
                    f"macro F1 delta `{delta_report['macro_f1_delta']}`"
                )
        lines.append("")

    interpretations = report.get("interpretation", [])
    if interpretations:
        lines.extend(["## Interpretation", ""])
        for line in interpretations:
            lines.append(f"- {line}")
        lines.append("")

    warnings = report.get("warnings", [])
    if warnings:
        lines.extend(["## Warnings", ""])
        for warning in warnings:
            lines.append(f"- {warning}")
        lines.append("")
    return "\n".join(lines).strip() + "\n"


def build_unperturbed_model_comparison(
    report: dict[str, Any],
    *,
    primary_split: str,
    selected_model_variants: tuple[str, ...] | list[str] | None = None,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    selected = list(selected_model_variants or ())
    if selected:
        for model_variant in selected:
            model_name, input_variant = _parse_model_variant(model_variant)
            metrics = report.get("models", {}).get(model_name, {}).get(input_variant, {}).get("metrics_by_split", {}).get(primary_split)
            if not metrics:
                continue
            rows.append(_build_model_variant_row(model_name, input_variant, metrics))
    else:
        for model_name, variants in report.get("models", {}).items():
            for input_variant, variant_payload in variants.items():
                metrics = variant_payload.get("metrics_by_split", {}).get(primary_split)
                if not metrics:
                    continue
                rows.append(_build_model_variant_row(model_name, input_variant, metrics))

    strongest_by_macro_f1 = max(rows, key=lambda row: row["macro_f1"], default=None)
    strongest_by_accuracy = max(rows, key=lambda row: row["accuracy"], default=None)
    strongest_by_macro_f1_by_input_variant: dict[str, dict[str, Any]] = {}
    strongest_by_accuracy_by_input_variant: dict[str, dict[str, Any]] = {}
    for row in rows:
        current_macro = strongest_by_macro_f1_by_input_variant.get(row["input_variant"])
        if current_macro is None or row["macro_f1"] > current_macro["macro_f1"]:
            strongest_by_macro_f1_by_input_variant[row["input_variant"]] = row
        current_accuracy = strongest_by_accuracy_by_input_variant.get(row["input_variant"])
        if current_accuracy is None or row["accuracy"] > current_accuracy["accuracy"]:
            strongest_by_accuracy_by_input_variant[row["input_variant"]] = row
    pairwise_comparisons: list[dict[str, Any]] = []
    rows_by_input_variant: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        rows_by_input_variant.setdefault(row["input_variant"], []).append(row)
    for input_variant, variant_rows in sorted(rows_by_input_variant.items()):
        for left, right in combinations(variant_rows, 2):
            pairwise_comparisons.append(
                {
                    "input_variant": input_variant,
                    "left_model_variant": left["model_variant"],
                    "right_model_variant": right["model_variant"],
                    "accuracy_gap_right_minus_left": round(right["accuracy"] - left["accuracy"], 6),
                    "macro_f1_gap_right_minus_left": round(right["macro_f1"] - left["macro_f1"], 6),
                    "stronger_by_accuracy": (
                        right["model_variant"]
                        if right["accuracy"] > left["accuracy"]
                        else left["model_variant"]
                    ),
                    "stronger_by_macro_f1": (
                        right["model_variant"]
                        if right["macro_f1"] > left["macro_f1"]
                        else left["model_variant"]
                    ),
                }
            )

    takeaways: list[str] = []
    if strongest_by_macro_f1 is not None:
        takeaways.append(
            f"Best unperturbed `{primary_split}` macro F1 came from `{strongest_by_macro_f1['model_variant']}` at `{strongest_by_macro_f1['macro_f1']}`."
        )
    if strongest_by_accuracy is not None and strongest_by_accuracy != strongest_by_macro_f1:
        takeaways.append(
            f"Best unperturbed `{primary_split}` accuracy came from `{strongest_by_accuracy['model_variant']}` at `{strongest_by_accuracy['accuracy']}`."
        )
    if strongest_by_macro_f1_by_input_variant:
        for input_variant, row in sorted(strongest_by_macro_f1_by_input_variant.items()):
            takeaways.append(
                f"Best unperturbed `{primary_split}` macro F1 for `{input_variant}` came from `{row['model_variant']}` at `{row['macro_f1']}`."
            )
    if rows:
        weakest_by_macro_f1 = min(rows, key=lambda row: row["macro_f1"])
        takeaways.append(
            f"Unperturbed macro-F1 gap between strongest `{strongest_by_macro_f1['model_variant']}` and weakest `{weakest_by_macro_f1['model_variant']}` was `{round(strongest_by_macro_f1['macro_f1'] - weakest_by_macro_f1['macro_f1'], 6)}`."
        )

    return {
        "task": "unperturbed_model_comparison",
        "primary_split": primary_split,
        "model_variants": rows,
        "strongest_by_accuracy": strongest_by_accuracy,
        "strongest_by_macro_f1": strongest_by_macro_f1,
        "strongest_by_accuracy_by_input_variant": strongest_by_accuracy_by_input_variant,
        "strongest_by_macro_f1_by_input_variant": strongest_by_macro_f1_by_input_variant,
        "pairwise_comparisons": pairwise_comparisons,
        "takeaways": takeaways,
    }


def expand_unperturbed_model_variants(
    selected_model_variants: tuple[str, ...] | list[str],
    *,
    include_full_text: bool,
) -> tuple[str, ...]:
    expanded: list[str] = []
    seen: set[str] = set()
    for model_variant in selected_model_variants:
        if model_variant not in seen:
            expanded.append(model_variant)
            seen.add(model_variant)
        if not include_full_text:
            continue
        model_name, _ = _parse_model_variant(model_variant)
        full_text_variant = f"{model_name}::full_text"
        if full_text_variant not in seen:
            expanded.append(full_text_variant)
            seen.add(full_text_variant)
    return tuple(expanded)


def render_unperturbed_model_comparison(report: dict[str, Any]) -> str:
    lines = [
        "# Unperturbed Model Comparison",
        "",
        f"- Primary split: `{report['primary_split']}`",
        "",
        "## Model Variants",
        "",
    ]
    for row in report["model_variants"]:
        lines.append(
            f"- `{row['model_variant']}`: accuracy `{row['accuracy']}`, macro F1 `{row['macro_f1']}`, "
            f"case count `{row['case_count']}`"
        )
    if report.get("strongest_by_macro_f1_by_input_variant"):
        lines.extend(["", "## Best By Input Variant", ""])
        for input_variant, row in sorted(report["strongest_by_macro_f1_by_input_variant"].items()):
            lines.append(
                f"- `{input_variant}`: strongest macro F1 came from `{row['model_variant']}` at `{row['macro_f1']}`"
            )
    if report.get("pairwise_comparisons"):
        lines.extend(["", "## Pairwise Comparison", ""])
        for row in report["pairwise_comparisons"]:
            lines.append(
                f"- `{row['input_variant']}` | `{row['left_model_variant']}` vs `{row['right_model_variant']}`: "
                f"macro-F1 gap (right-left) `{row['macro_f1_gap_right_minus_left']}`, "
                f"accuracy gap (right-left) `{row['accuracy_gap_right_minus_left']}`, "
                f"stronger by macro F1 `{row['stronger_by_macro_f1']}`"
            )
    if report.get("takeaways"):
        lines.extend(["", "## Takeaways", ""])
        for takeaway in report["takeaways"]:
            lines.append(f"- {takeaway}")
    return "\n".join(lines).strip() + "\n"


def _build_model_variant_row(model_name: str, input_variant: str, metrics: dict[str, Any]) -> dict[str, Any]:
    return {
        "model_variant": f"{model_name}::{input_variant}",
        "model_name": model_name,
        "input_variant": input_variant,
        "accuracy": metrics["accuracy"],
        "macro_f1": metrics["macro_f1"],
        "case_count": metrics["case_count"],
        "per_class": metrics.get("per_class", {}),
        "gold_label_distribution": metrics.get("gold_label_distribution", {}),
        "predicted_label_distribution": metrics.get("predicted_label_distribution", {}),
        "confusion_matrix": metrics.get("confusion_matrix", {}),
    }


def _parse_model_variant(value: str) -> tuple[str, str]:
    model_name, input_variant = value.split("::", maxsplit=1)
    return model_name, input_variant
