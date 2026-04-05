from __future__ import annotations

from pathlib import Path
from typing import Any


def build_claim_to_evidence_traceability(
    *,
    results_package_dir: Path,
    section_importance_dir: Path,
    submission_package_dir: Path,
    table_main_results: dict[str, Any],
    table_model_comparison: dict[str, Any],
    stability_table: dict[str, Any],
    section_importance_scores: dict[str, Any],
    section_importance_cross_model_check: dict[str, Any],
    figure_manifest: dict[str, Any],
) -> dict[str, Any]:
    main_rows = {row["condition_key"]: row for row in table_main_results.get("rows", [])}
    model_rows = {row["model_label"]: row for row in table_model_comparison.get("rows", [])}
    section_rows = {row["section"]: row for row in section_importance_scores.get("section_rows", [])}
    figure_lookup = {row["figure_id"]: row for row in figure_manifest.get("figures", [])}

    claims = [
        {
            "claim_id": "C1",
            "claim_text": "APA is the strongest central baseline on unperturbed pseudo_all_sections in the frozen focused comparison.",
            "supporting_artifacts": [
                str(results_package_dir / "table_model_comparison.md"),
                *figure_lookup["fig_model_comparison"]["output_files"],
            ],
            "supporting_metrics": {
                "apa_accuracy": model_rows["APA"]["unperturbed_accuracy"],
                "apa_macro_f1": model_rows["APA"]["unperturbed_macro_f1"],
            },
            "confidence_note": "High confidence within the frozen four-model pilot comparison.",
        },
        {
            "claim_id": "C2",
            "claim_text": "Keeping only reasoning causes the largest APA degradation among the focused primary probes.",
            "supporting_artifacts": [
                str(results_package_dir / "table_main_results.md"),
                *figure_lookup["fig_main_robustness"]["output_files"],
            ],
            "supporting_metrics": {
                "macro_f1_delta": main_rows["keep_reasoning_only"]["delta_macro_f1"],
                "accuracy_delta": main_rows["keep_reasoning_only"]["delta_accuracy"],
                "flip_rate": main_rows["keep_reasoning_only"]["flip_rate"],
                "coverage": main_rows["keep_reasoning_only"]["effective_coverage"],
            },
            "confidence_note": "High confidence because the probe has full effective coverage in the frozen package.",
        },
        {
            "claim_id": "C3",
            "claim_text": "Dropping precedents causes a smaller but still meaningful APA degradation.",
            "supporting_artifacts": [
                str(results_package_dir / "table_main_results.md"),
                *figure_lookup["fig_main_robustness"]["output_files"],
            ],
            "supporting_metrics": {
                "macro_f1_delta": main_rows["drop_precedents"]["delta_macro_f1"],
                "accuracy_delta": main_rows["drop_precedents"]["delta_accuracy"],
                "flip_rate": main_rows["drop_precedents"]["flip_rate"],
                "coverage": main_rows["drop_precedents"]["effective_coverage"],
            },
            "confidence_note": "High confidence because coverage remains strong and the effect direction is stable in the frozen package.",
        },
        {
            "claim_id": "C4",
            "claim_text": "Absolute strength, perturbation retention, and prediction stability are not the same notion of robustness.",
            "supporting_artifacts": [
                str(results_package_dir / "table_model_comparison.md"),
                str(results_package_dir / "table_stability_vs_correctness.md"),
                *figure_lookup["fig_flip_rates"]["output_files"],
            ],
            "supporting_metrics": {
                "apa_keep_reasoning_only_flip_rate": _find_model_recipe_metric(
                    table_model_comparison,
                    "APA",
                    "keep_reasoning_only_flip_rate",
                ),
                "logistic_keep_reasoning_only_flip_rate": _find_model_recipe_metric(
                    table_model_comparison,
                    "Logistic",
                    "keep_reasoning_only_flip_rate",
                ),
                "nb_drop_precedents_delta": _find_model_recipe_metric(
                    table_model_comparison,
                    "NB",
                    "drop_precedents_delta_macro_f1",
                ),
            },
            "confidence_note": "High confidence as a comparative interpretation inside the frozen model set.",
        },
        {
            "claim_id": "C5",
            "claim_text": "Under APA, the current section-importance ranking is precedents > facts > reasoning > other > conclusion.",
            "supporting_artifacts": [
                str(section_importance_dir / "section_importance_scores.md"),
                str(section_importance_dir / "section_importance_ranking.md"),
                *figure_lookup["fig_section_importance"]["output_files"],
            ],
            "supporting_metrics": {
                "precedents_rank": section_rows["precedents"]["rank"],
                "facts_rank": section_rows["facts"]["rank"],
                "reasoning_rank": section_rows["reasoning"]["rank"],
                "conclusion_rank": section_rows["conclusion"]["rank"],
            },
            "confidence_note": "APA-centered claim; do not present it as fully model-invariant.",
        },
        {
            "claim_id": "C6",
            "claim_text": "Conclusion importance remains low confidence because conclusion coverage is sparse.",
            "supporting_artifacts": [
                str(section_importance_dir / "section_importance_scores.md"),
                *figure_lookup["fig_coverage"]["output_files"],
            ],
            "supporting_metrics": {
                "conclusion_coverage": section_rows["conclusion"]["combined_effective_coverage"],
                "conclusion_confidence_label": section_rows["conclusion"]["confidence_label"],
            },
            "confidence_note": "This caveat should remain explicit in the paper.",
        },
        {
            "claim_id": "C7",
            "claim_text": "All section-aware claims in this project are made over transferred/predicted pseudo-sections rather than gold annotations.",
            "supporting_artifacts": [
                str(results_package_dir / "table_main_results.md"),
                str(section_importance_dir / "section_importance_scores.md"),
                str(submission_package_dir / "paper_limitations_ethics_support.md"),
            ],
            "supporting_metrics": {
                "artifact_type": "methodological_caveat",
            },
            "confidence_note": "Mandatory caveat for all section-aware claims.",
        },
        {
            "claim_id": "C8",
            "claim_text": "The section-importance ordering is useful for the paper, but it is best framed as APA-centered because the reduced cross-model sanity check is mixed.",
            "supporting_artifacts": [
                str(section_importance_dir / "section_importance_cross_model_check.md"),
                str(section_importance_dir / "section_importance_next_step_summary.md"),
            ],
            "supporting_metrics": {
                "reasoning_vs_precedents_alignment": section_importance_cross_model_check["alignment_counts"][
                    "reasoning_gt_precedents_by_removal_impact"
                ],
                "precedents_vs_facts_alignment": section_importance_cross_model_check["alignment_counts"][
                    "precedents_gt_facts_by_solo_sufficiency"
                ],
            },
            "confidence_note": "Use this as a framing caveat rather than a negative result.",
        },
    ]
    return {
        "task": "claim_to_evidence_traceability",
        "claims": claims,
    }


def render_claim_to_evidence_traceability(report: dict[str, Any]) -> str:
    lines = ["# Claim to Evidence Traceability", ""]
    for claim in report.get("claims", []):
        lines.extend([f"## {claim['claim_id']}", ""])
        lines.append(f"- Claim: {claim['claim_text']}")
        lines.append("- Supporting artifacts:")
        for artifact in claim.get("supporting_artifacts", []):
            lines.append(f"  - `{artifact}`")
        lines.append(f"- Supporting metrics: `{claim['supporting_metrics']}`")
        lines.append(f"- Confidence / caveat note: {claim['confidence_note']}")
        lines.append("")
    return "\n".join(lines).strip() + "\n"


def _find_model_recipe_metric(
    table_model_comparison: dict[str, Any],
    model_label: str,
    field_name: str,
) -> Any:
    for row in table_model_comparison.get("rows", []):
        if row["model_label"] == model_label:
            return row.get(field_name)
    return None
