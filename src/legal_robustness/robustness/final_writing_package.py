from __future__ import annotations

from pathlib import Path
from typing import Any


def build_paper_abstract_support(
    *,
    table_main_results: dict[str, Any],
    table_model_comparison: dict[str, Any],
    section_importance_scores: dict[str, Any],
) -> dict[str, Any]:
    main_rows = {row["condition_key"]: row for row in table_main_results.get("rows", [])}
    model_rows = sorted(
        table_model_comparison.get("rows", []),
        key=lambda row: row["unperturbed_macro_f1"],
        reverse=True,
    )
    section_rows = sorted(
        section_importance_scores.get("section_rows", []),
        key=lambda row: row["rank"],
    )
    return {
        "task": "paper_abstract_support",
        "top_model": model_rows[0]["model_label"] if model_rows else "APA",
        "unperturbed_macro_f1": main_rows["unperturbed"]["macro_f1"],
        "unperturbed_accuracy": main_rows["unperturbed"]["accuracy"],
        "primary_probe": {
            "name": "keep_reasoning_only",
            "macro_f1_delta": main_rows["keep_reasoning_only"]["delta_macro_f1"],
            "accuracy_delta": main_rows["keep_reasoning_only"]["delta_accuracy"],
        },
        "secondary_probe": {
            "name": "drop_precedents",
            "macro_f1_delta": main_rows["drop_precedents"]["delta_macro_f1"],
            "accuracy_delta": main_rows["drop_precedents"]["delta_accuracy"],
        },
        "section_ranking": [row["section"] for row in section_rows],
        "conclusion_confidence": section_rows[-1]["confidence_label"] if section_rows else "low_confidence_importance_estimate",
    }


def render_paper_abstract_support(report: dict[str, Any]) -> str:
    lines = [
        "# Paper Abstract Support",
        "",
        "Suggested abstract ingredients:",
        f"- Setup: evaluate section-aware robustness for legal judgment prediction using RR-to-CJPE transferred pseudo-sections and a focused APA-centered baseline.",
        f"- Main baseline result: APA reaches accuracy `{_fmt(report['unperturbed_accuracy'])}` and macro F1 `{_fmt(report['unperturbed_macro_f1'])}` on unperturbed `pseudo_all_sections`.",
        f"- Primary robustness result: `keep_reasoning_only` changes macro F1 by `{_fmt(report['primary_probe']['macro_f1_delta'])}` and accuracy by `{_fmt(report['primary_probe']['accuracy_delta'])}`.",
        f"- Secondary robustness result: `drop_precedents` changes macro F1 by `{_fmt(report['secondary_probe']['macro_f1_delta'])}` and accuracy by `{_fmt(report['secondary_probe']['accuracy_delta'])}`.",
        f"- Section-importance result: the APA-centered ranking is `{report['section_ranking']}`.",
        f"- Caveat sentence: conclusion importance remains `{report['conclusion_confidence']}` because conclusion coverage is sparse and CJPE sections are predicted/transferred rather than gold.",
        "",
    ]
    return "\n".join(lines)


def build_paper_intro_support(
    *,
    primary_model_variant: str,
    primary_probe: str,
    secondary_probe: str,
) -> dict[str, Any]:
    return {
        "task": "paper_intro_support",
        "primary_model_variant": primary_model_variant,
        "primary_probe": primary_probe,
        "secondary_probe": secondary_probe,
        "contribution_points": [
            "Legal judgment prediction is usually reported with aggregate accuracy, even though judgments are internally structured and different sections may carry different legal signals.",
            "Section-aware robustness offers a more interpretable stress test than generic corruption because it asks which predicted parts of the judgment the model depends on.",
            "RR supervision enables rhetorical structure transfer to CJPE, making section-aware evaluation possible at CJPE scale even without gold CJPE section annotations.",
            "The paper centers APA as the main model, keep_reasoning_only as the primary probe, and drop_precedents as the supporting probe.",
        ],
    }


def render_paper_intro_support(report: dict[str, Any]) -> str:
    lines = [
        "# Paper Intro Support",
        "",
        "High-level introduction framing:",
        f"- Central model: `{report['primary_model_variant']}`",
        f"- Primary probe: `{report['primary_probe']}`",
        f"- Secondary probe: `{report['secondary_probe']}`",
        "",
        "Contribution framing bullets:",
    ]
    for point in report.get("contribution_points", []):
        lines.append(f"- {point}")
    lines.append("")
    return "\n".join(lines)


def build_paper_method_support(
    *,
    freeze_manifest: dict[str, Any],
    primary_model_variant: str,
    primary_probe: str,
    secondary_probe: str,
    section_importance_summary: dict[str, Any],
) -> dict[str, Any]:
    return {
        "task": "paper_method_support",
        "section_transfer_run": freeze_manifest["canonical_section_transfer_run_dir"],
        "baseline_run": freeze_manifest["canonical_baseline_run_dir"],
        "robustness_run": freeze_manifest["canonical_robustness_run_dir"],
        "primary_model_variant": primary_model_variant,
        "primary_probe": primary_probe,
        "secondary_probe": secondary_probe,
        "section_importance_formula": section_importance_summary.get("composite_formula"),
    }


def render_paper_method_support(report: dict[str, Any]) -> str:
    lines = [
        "# Paper Method Support",
        "",
        "Method outline:",
        f"- RR-to-CJPE section transfer source: `{report['section_transfer_run']}`",
        f"- Canonical baseline artifacts: `{report['baseline_run']}`",
        f"- Canonical robustness artifacts: `{report['robustness_run']}`",
        f"- Main prediction model: `{report['primary_model_variant']}`",
        f"- Focused perturbations: `{report['primary_probe']}` and `{report['secondary_probe']}`",
        "- Section labels are broad predicted pseudo-sections: facts, precedents, reasoning, conclusion, other.",
        f"- Section-importance scoring formula: `{report['section_importance_formula']}`",
        "- Evaluation should be described as focused, coverage-aware, and pilot-scale rather than exhaustive.",
        "",
    ]
    return "\n".join(lines)


def build_paper_results_support(
    *,
    table_main_results: dict[str, Any],
    table_model_comparison: dict[str, Any],
    section_importance_scores: dict[str, Any],
    section_importance_cross_model_check: dict[str, Any],
) -> dict[str, Any]:
    main_rows = {row["condition_key"]: row for row in table_main_results.get("rows", [])}
    model_rows = {row["model_label"]: row for row in table_model_comparison.get("rows", [])}
    section_rows = sorted(section_importance_scores.get("section_rows", []), key=lambda row: row["rank"])
    return {
        "task": "paper_results_support",
        "unperturbed": main_rows["unperturbed"],
        "keep_reasoning_only": main_rows["keep_reasoning_only"],
        "drop_precedents": main_rows["drop_precedents"],
        "apa_row": model_rows.get("APA"),
        "nb_row": model_rows.get("NB"),
        "logistic_row": model_rows.get("Logistic"),
        "section_ranking": [row["section"] for row in section_rows],
        "section_cross_model_summary": section_importance_cross_model_check.get("alignment_counts", {}),
    }


def render_paper_results_support(report: dict[str, Any]) -> str:
    reasoning = report["keep_reasoning_only"]
    precedents = report["drop_precedents"]
    lines = [
        "# Paper Results Support",
        "",
        f"APA main result: unperturbed accuracy `{_fmt(report['unperturbed']['accuracy'])}` and macro F1 `{_fmt(report['unperturbed']['macro_f1'])}` on `pseudo_all_sections`.",
        "",
        f"Primary probe result: under `keep_reasoning_only`, macro F1 changes by `{_fmt(reasoning['delta_macro_f1'])}` and accuracy changes by `{_fmt(reasoning['delta_accuracy'])}` with flip rate `{_fmt(reasoning['flip_rate'])}`.",
        "",
        f"Secondary probe result: under `drop_precedents`, macro F1 changes by `{_fmt(precedents['delta_macro_f1'])}` and accuracy changes by `{_fmt(precedents['delta_accuracy'])}` with flip rate `{_fmt(precedents['flip_rate'])}`.",
        "",
        f"Comparator framing: NB remains a supporting retention comparator, while logistic remains the main stability comparator. Use this to explain why robustness retention and flip-rate stability are not identical.",
        "",
        f"Section-importance result: the current APA-centered ranking is `{report['section_ranking']}`.",
        "",
        "Recommended artifact references in the results section:",
        "- `table_main_results.md` for the headline APA table.",
        "- `fig_main_robustness.png` for the centered robustness figure.",
        "- `fig_model_comparison.png` and `fig_flip_rates.png` for the comparator story.",
        "- `fig_section_importance.png` for the ranking figure.",
        "",
        f"Cross-model caveat: section-importance alignment is mixed across models (`{report['section_cross_model_summary']}`), so the ranking should be presented as APA-centered rather than universal.",
        "",
    ]
    return "\n".join(lines)


def build_paper_limitations_ethics_support() -> dict[str, Any]:
    return {
        "task": "paper_limitations_ethics_support",
        "points": [
            "CJPE sections are transferred/predicted pseudo-sections rather than gold annotations.",
            "The focused probes are intentionally narrow and should be described as a pilot robustness setup.",
            "Conclusion importance is low confidence because conclusion coverage is sparse.",
            "The contextual approximation remains supporting context rather than a competitive main model.",
            "Legal AI results should be framed as analytical evidence about model behavior, not as deployment justification.",
        ],
    }


def render_paper_limitations_ethics_support(report: dict[str, Any]) -> str:
    lines = ["# Paper Limitations and Ethics Support", ""]
    for point in report.get("points", []):
        lines.append(f"- {point}")
    lines.append("")
    return "\n".join(lines)


def build_paper_conclusion_support(
    *,
    section_importance_next_step_summary: dict[str, Any],
) -> dict[str, Any]:
    return {
        "task": "paper_conclusion_support",
        "ranking": [
            row["section"]
            for row in section_importance_next_step_summary.get("section_ranking", [])
        ],
        "exact_next_action": section_importance_next_step_summary.get("next_action"),
    }


def render_paper_conclusion_support(report: dict[str, Any]) -> str:
    lines = [
        "# Paper Conclusion Support",
        "",
        f"- Final substantive claim: section-aware perturbations expose a measurable difference between reasoning-only ablation and precedent removal for the APA baseline.",
        f"- Section-importance claim: the current APA-centered ranking is `{report['ranking']}`.",
        "- Future-work sentence: stronger contextual models and gold section annotations remain the most natural next validation targets, but they are not required for the current pilot paper.",
        f"- Immediate transition sentence: {report['exact_next_action']}",
        "",
    ]
    return "\n".join(lines)


def build_submission_package_manifest(
    *,
    freeze_manifest: dict[str, Any],
    results_package_dir: Path,
    paper_drafting_package_dir: Path,
    section_importance_dir: Path,
    submission_package_dir: Path,
    figure_manifest: dict[str, Any],
    writing_files: list[Path],
    qualitative_files: list[Path],
    appendix_files: list[Path],
    manifest_files: list[Path],
    layout_files: list[Path],
    reproducibility_file: Path,
    traceability_file: Path,
    consistency_file: Path,
) -> dict[str, Any]:
    return {
        "task": "submission_package_manifest",
        "canonical_section_transfer_run_dir": freeze_manifest["canonical_section_transfer_run_dir"],
        "canonical_baseline_run_dir": freeze_manifest["canonical_baseline_run_dir"],
        "canonical_robustness_run_dir": freeze_manifest["canonical_robustness_run_dir"],
        "canonical_results_package_dir": str(results_package_dir),
        "canonical_paper_drafting_package_dir": str(paper_drafting_package_dir),
        "canonical_section_importance_dir": str(section_importance_dir),
        "submission_package_dir": str(submission_package_dir),
        "primary_model_variant": freeze_manifest["primary_model_variant"],
        "primary_probe": freeze_manifest["primary_probe"],
        "secondary_probe": freeze_manifest["secondary_probe"],
        "final_tables": [
            str(results_package_dir / "table_main_results.md"),
            str(results_package_dir / "table_model_comparison.md"),
            str(results_package_dir / "table_stability_vs_correctness.md"),
            str(section_importance_dir / "section_importance_scores.md"),
        ],
        "final_figures": [
            output_file
            for figure in figure_manifest.get("figures", [])
            for output_file in figure.get("output_files", [])
        ],
        "qualitative_files": [str(path) for path in qualitative_files],
        "appendix_files": [str(path) for path in appendix_files],
        "writing_support_files": [str(path) for path in writing_files],
        "manifest_files": [str(path) for path in manifest_files],
        "layout_files": [str(path) for path in layout_files],
        "reproducibility_file": str(reproducibility_file),
        "traceability_file": str(traceability_file),
        "consistency_file": str(consistency_file),
        "known_caveats": [
            "CJPE sections are transferred/predicted pseudo-sections rather than gold section annotations.",
            "The main story remains centered on keep_reasoning_only and drop_precedents.",
            "The section-importance ranking is APA-centered and only partially cross-model consistent.",
            "Conclusion importance remains low confidence because conclusion coverage is sparse.",
        ],
    }


def render_submission_package_manifest(report: dict[str, Any]) -> str:
    lines = [
        "# Submission Package Manifest",
        "",
        f"- Canonical section-transfer run: `{report['canonical_section_transfer_run_dir']}`",
        f"- Canonical baseline run: `{report['canonical_baseline_run_dir']}`",
        f"- Canonical robustness run: `{report['canonical_robustness_run_dir']}`",
        f"- Canonical results package dir: `{report['canonical_results_package_dir']}`",
        f"- Canonical paper drafting package dir: `{report['canonical_paper_drafting_package_dir']}`",
        f"- Canonical section-importance dir: `{report['canonical_section_importance_dir']}`",
        f"- Submission package dir: `{report['submission_package_dir']}`",
        "",
        "## Final Tables",
        "",
    ]
    for path in report.get("final_tables", []):
        lines.append(f"- `{path}`")
    lines.extend(["", "## Final Figures", ""])
    for path in report.get("final_figures", []):
        lines.append(f"- `{path}`")
    lines.extend(["", "## Writing Support Files", ""])
    for path in report.get("writing_support_files", []):
        lines.append(f"- `{path}`")
    lines.extend(["", "## Qualitative / Appendix Files", ""])
    for path in report.get("qualitative_files", []):
        lines.append(f"- `{path}`")
    for path in report.get("appendix_files", []):
        lines.append(f"- `{path}`")
    lines.extend(["", "## Manifests / Guides", ""])
    for path in report.get("manifest_files", []):
        lines.append(f"- `{path}`")
    for path in report.get("layout_files", []):
        lines.append(f"- `{path}`")
    lines.extend(
        [
            "",
            "## Reproducibility / Traceability / Consistency",
            "",
            f"- `{report['reproducibility_file']}`",
            f"- `{report['traceability_file']}`",
            f"- `{report['consistency_file']}`",
            "",
            "## Known Caveats",
            "",
        ]
    )
    for caveat in report.get("known_caveats", []):
        lines.append(f"- {caveat}")
    lines.append("")
    return "\n".join(lines)


def build_main_text_layout_guide() -> dict[str, Any]:
    return {
        "task": "main_text_layout_guide",
        "placements": [
            {
                "slot": "Results opening paragraph",
                "artifact": "table_main_results.md",
                "claim": "APA headline robustness performance under the two focused probes.",
            },
            {
                "slot": "Results figure 1",
                "artifact": "fig_main_robustness.png",
                "claim": "Reasoning-only ablation is harsher than precedent removal for APA.",
            },
            {
                "slot": "Results figure 2",
                "artifact": "fig_model_comparison.png",
                "claim": "APA is strongest overall, while comparator models show different retention behavior.",
            },
            {
                "slot": "Section-importance subsection",
                "artifact": "fig_section_importance.png",
                "claim": "The APA-centered section-importance ranking is precedents > facts > reasoning > other > conclusion.",
            },
        ],
    }


def render_main_text_layout_guide(report: dict[str, Any]) -> str:
    lines = ["# Main Text Layout Guide", ""]
    for row in report.get("placements", []):
        lines.append(f"- {row['slot']}: `{row['artifact']}` supports {row['claim']}")
    lines.append("")
    return "\n".join(lines)


def build_appendix_layout_guide() -> dict[str, Any]:
    return {
        "task": "appendix_layout_guide",
        "placements": [
            {
                "slot": "Appendix robustness detail",
                "artifact": "table_stability_vs_correctness.md",
                "claim": "Absolute strength and stability diverge under focused perturbations.",
            },
            {
                "slot": "Appendix figure",
                "artifact": "fig_flip_rates.png",
                "claim": "Flip-rate stability differs from retention.",
            },
            {
                "slot": "Appendix figure",
                "artifact": "fig_coverage.png",
                "claim": "Conclusion importance remains low confidence because coverage is sparse.",
            },
            {
                "slot": "Appendix examples",
                "artifact": "qualitative_examples_primary.md and qualitative_examples_secondary.md",
                "claim": "Qualitative anchors for reasoning-only and precedent-removal behavior.",
            },
        ],
    }


def render_appendix_layout_guide(report: dict[str, Any]) -> str:
    lines = ["# Appendix Layout Guide", ""]
    for row in report.get("placements", []):
        lines.append(f"- {row['slot']}: `{row['artifact']}` supports {row['claim']}")
    lines.append("")
    return "\n".join(lines)


def build_table_captions() -> dict[str, Any]:
    return {
        "task": "table_captions",
        "captions": [
            {
                "table_name": "table_main_results.md",
                "caption": "Table 1. APA headline results on the focused probes. Reasoning-only ablation produces the largest degradation, while precedent removal produces a smaller but still meaningful drop.",
            },
            {
                "table_name": "table_model_comparison.md",
                "caption": "Table 2. Four-model comparison on pseudo_all_sections, showing that absolute strength, perturbation retention, and prediction stability are distinct properties.",
            },
            {
                "table_name": "section_importance_scores.md",
                "caption": "Table 3. APA-centered, coverage-aware section-importance scores and ranking.",
            },
        ],
    }


def render_table_captions(report: dict[str, Any]) -> str:
    lines = ["# Table Captions", ""]
    for row in report.get("captions", []):
        lines.append(f"## {row['table_name']}")
        lines.append("")
        lines.append(row["caption"])
        lines.append("")
    return "\n".join(lines).strip() + "\n"


def build_paper_handoff_summary(
    *,
    submission_package_manifest: dict[str, Any],
    final_consistency_check: dict[str, Any],
    section_importance_next_step_summary: dict[str, Any],
) -> dict[str, Any]:
    open_first = [
        str(Path(submission_package_manifest["submission_package_dir"]) / "paper_results_support.md"),
        str(Path(submission_package_manifest["submission_package_dir"]) / "claim_to_evidence_traceability.md"),
        str(Path(submission_package_manifest["submission_package_dir"]) / "fig_main_robustness.png"),
        str(Path(submission_package_manifest["submission_package_dir"]) / "fig_section_importance.png"),
        str(Path(submission_package_manifest["submission_package_dir"]) / "submission_package_manifest.md"),
    ]
    return {
        "task": "paper_handoff_summary",
        "technical_work_done": final_consistency_check.get("overall_status") == "pass",
        "evidence_package_frozen_and_consistent": final_consistency_check.get("overall_status") == "pass",
        "open_first": open_first,
        "immediate_next_step": section_importance_next_step_summary.get(
            "next_action",
            "Begin manuscript drafting from the frozen package.",
        ),
        "required_from_here": [
            "Write the manuscript sections using the final writing-support files.",
            "Use the frozen tables and generated figures as the source of cited numbers.",
        ],
        "optional_from_here": [
            "Polish figure styling for venue aesthetics.",
            "Shorten or merge appendix bundles during manuscript assembly if space requires it.",
        ],
    }


def render_paper_handoff_summary(report: dict[str, Any]) -> str:
    lines = [
        "# Paper Handoff Summary",
        "",
        f"- Technical work done: `{report['technical_work_done']}`",
        f"- Evidence package frozen and consistent: `{report['evidence_package_frozen_and_consistent']}`",
        "",
        "## Open First",
        "",
    ]
    for path in report.get("open_first", []):
        lines.append(f"- `{path}`")
    lines.extend(["", "## Immediate Next Step", "", report["immediate_next_step"], "", "## Required From Here", ""])
    for item in report.get("required_from_here", []):
        lines.append(f"- {item}")
    lines.extend(["", "## Optional From Here", ""])
    for item in report.get("optional_from_here", []):
        lines.append(f"- {item}")
    lines.append("")
    return "\n".join(lines)


def _fmt(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)
