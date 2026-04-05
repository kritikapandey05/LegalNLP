from __future__ import annotations

from typing import Any


def build_focused_perturbation_interpretation(
    apa_focused_table: dict[str, Any],
) -> dict[str, Any]:
    rows = {
        row["recipe_name"]: row
        for row in apa_focused_table.get("rows", [])
    }
    entries: list[dict[str, Any]] = []
    drop_precedents = rows.get("drop_precedents")
    if drop_precedents is not None:
        entries.append(
            {
                "recipe_name": "drop_precedents",
                "conceptual_focus": [
                    "dependence on precedent content",
                    "sensitivity to removing legal-citation support",
                    "robustness to loss of supporting doctrinal structure",
                ],
                "current_readout": (
                    f"For APA, dropping precedents changes macro F1 by `{drop_precedents['delta_macro_f1']}` "
                    f"at effective coverage `{drop_precedents['effective_coverage']}`."
                ),
            }
        )
    keep_reasoning_only = rows.get("keep_reasoning_only")
    if keep_reasoning_only is not None:
        entries.append(
            {
                "recipe_name": "keep_reasoning_only",
                "conceptual_focus": [
                    "how much decision signal survives in reasoning alone",
                    "whether models can perform without facts and precedents",
                    "whether reasoning sections carry dominant label cues",
                ],
                "current_readout": (
                    f"For APA, reasoning-only changes macro F1 by `{keep_reasoning_only['delta_macro_f1']}` "
                    f"at effective coverage `{keep_reasoning_only['effective_coverage']}`."
                ),
            }
        )
    return {
        "task": "focused_perturbation_interpretation",
        "primary_model_variant": apa_focused_table.get("primary_model_variant"),
        "entries": entries,
    }


def render_focused_perturbation_interpretation(report: dict[str, Any]) -> str:
    lines = [
        "# Focused Perturbation Interpretation",
        "",
        f"- Primary model variant: `{report['primary_model_variant']}`",
        "",
    ]
    for entry in report.get("entries", []):
        lines.extend([f"## {entry['recipe_name']}", ""])
        lines.append("- What this probe tests:")
        for point in entry["conceptual_focus"]:
            lines.append(f"  {point}")
        lines.append(f"- Current empirical readout: {entry['current_readout']}")
        lines.append("")
    return "\n".join(lines).strip() + "\n"


def build_pilot_results_section_summary(
    unperturbed_comparison: dict[str, Any],
    apa_focused_table: dict[str, Any],
    stability_summary: dict[str, Any],
    qualitative_report: dict[str, Any],
    next_step_summary: dict[str, Any],
) -> dict[str, Any]:
    ranked_models = sorted(
        [
            row
            for row in unperturbed_comparison.get("model_variants", [])
            if row.get("input_variant") == "pseudo_all_sections"
        ],
        key=lambda row: row["macro_f1"],
        reverse=True,
    )
    focused_findings: list[str] = []
    for row in apa_focused_table.get("rows", []):
        focused_findings.append(
            f"`{row['recipe_name']}`: APA macro F1 moves from `{row['unperturbed_macro_f1']}` to `{row['perturbed_macro_f1']}` "
            f"(delta `{row['delta_macro_f1']}`) with flip rate `{row['flip_rate']}` and coverage `{row['effective_coverage']}`."
        )
    stability_takeaway = (
        stability_summary.get("summary_takeaways", [None])[0]
        if stability_summary.get("summary_takeaways")
        else None
    )
    recommendation = (
        next_step_summary.get("recommendations", [None])[0]
        if next_step_summary.get("recommendations")
        else None
    )
    return {
        "task": "pilot_results_section_summary",
        "primary_model_variant": apa_focused_table.get("primary_model_variant"),
        "ready_for_pilot_results_section": next_step_summary.get("ready_for_pilot_results_section", False),
        "unperturbed_model_ranking": ranked_models,
        "focused_findings": focused_findings,
        "stability_vs_correctness_insight": stability_takeaway,
        "qualitative_example_overview": qualitative_report.get("per_recipe_summary", []),
        "caveats": [
            "The CJPE section structure is transferred/predicted from RR supervision and should not be treated as gold annotation.",
            "The focused results are intentionally limited to `drop_precedents` and `keep_reasoning_only` because those are the highest-value current probes.",
            "The contextual approximation remains weaker than the best simple baselines in this environment.",
        ],
        "recommendation": recommendation,
        "recommended_next_major_step": next_step_summary.get("recommended_next_major_step"),
    }


def render_pilot_results_section_summary(report: dict[str, Any]) -> str:
    lines = [
        "# Pilot Results Section Summary",
        "",
        f"- Primary model variant: `{report['primary_model_variant']}`",
        f"- Ready for pilot results section: `{report['ready_for_pilot_results_section']}`",
        f"- Recommended next major step: `{report.get('recommended_next_major_step')}`",
        "",
        "## Unperturbed Model Ranking",
        "",
    ]
    for row in report.get("unperturbed_model_ranking", []):
        lines.append(
            f"- `{row['model_variant']}`: accuracy `{row['accuracy']}`, macro F1 `{row['macro_f1']}`"
        )
    lines.extend(["", "## Focused Perturbation Findings", ""])
    for finding in report.get("focused_findings", []):
        lines.append(f"- {finding}")
    lines.extend(["", "## Stability vs Correctness Insight", ""])
    lines.append(f"- {report.get('stability_vs_correctness_insight')}")
    lines.extend(["", "## Caveats", ""])
    for caveat in report.get("caveats", []):
        lines.append(f"- {caveat}")
    lines.extend(["", "## Recommendation", ""])
    lines.append(f"- {report.get('recommendation')}")
    return "\n".join(lines).strip() + "\n"
