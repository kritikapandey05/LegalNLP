from __future__ import annotations

from typing import Any

from legal_robustness.section_transfer.diagnostics import BROAD_SECTION_ORDER
from legal_robustness.section_transfer.types import (
    CJPEPseudoSectionedResult,
    CJPESentencePredictionResult,
    RRSentenceSupervisionResult,
    SectionTaggerTrainingResult,
)

MAJOR_SECTIONS = ("facts", "reasoning", "conclusion")


def summarize_section_transfer_readiness(
    *,
    rr_supervision: RRSentenceSupervisionResult,
    training_result: SectionTaggerTrainingResult,
    cjpe_predictions: CJPESentencePredictionResult,
    cjpe_reconstructed: CJPEPseudoSectionedResult,
) -> dict[str, Any]:
    rr_distribution = rr_supervision.report["broad_section_distribution"]
    cjpe_distribution = cjpe_predictions.report["predicted_label_counts"]
    total_cases = cjpe_reconstructed.report["total_cases"] or 1
    presence_counts = cjpe_reconstructed.report["section_presence_counts"]
    all_major_sections_count = sum(
        1
        for record in cjpe_reconstructed.records
        if all(record.section_lengths_sentences.get(section, 0) > 0 for section in MAJOR_SECTIONS)
    )
    major_section_ratios = {
        section: round(presence_counts.get(section, 0) / total_cases, 6)
        for section in BROAD_SECTION_ORDER
    }
    rr_total = rr_supervision.report["total_sentences"] or 1
    cjpe_total = cjpe_predictions.report["total_sentences"] or 1
    distribution_shift = {
        section: round(
            (cjpe_distribution.get(section, 0) / cjpe_total)
            - (rr_distribution.get(section, 0) / rr_total),
            6,
        )
        for section in BROAD_SECTION_ORDER
    }
    readiness_recommendations = _build_recommendations(
        training_metrics=training_result.metrics,
        cjpe_reconstruction_report=cjpe_reconstructed.report,
        cjpe_prediction_report=cjpe_predictions.report,
        all_major_sections_ratio=round(all_major_sections_count / total_cases, 6),
    )
    return {
        "task": "section_transfer_readiness",
        "rr_supervision_distribution": rr_distribution,
        "rr_test_metrics": training_result.metrics["metrics_by_split"].get("test", {}),
        "cjpe_prediction_distribution": cjpe_distribution,
        "cjpe_prediction_proportions": cjpe_predictions.report["predicted_label_proportions"],
        "distribution_shift_rr_to_cjpe": distribution_shift,
        "total_pseudo_sectioned_cjpe_cases": cjpe_reconstructed.report["total_cases"],
        "section_presence_ratios": major_section_ratios,
        "all_major_sections_count": all_major_sections_count,
        "all_major_sections_ratio": round(all_major_sections_count / total_cases, 6),
        "dominant_section_case_count": cjpe_reconstructed.report["dominant_section_case_count"],
        "dominant_section_case_ratio": round(
            cjpe_reconstructed.report["dominant_section_case_count"] / total_cases,
            6,
        ),
        "confidence_summary": cjpe_predictions.report["confidence_summary_float"],
        "recommendations": readiness_recommendations,
    }


def render_section_transfer_readiness_summary(report: dict[str, Any]) -> str:
    lines = [
        "# Section Transfer Readiness Summary",
        "",
        f"- Total pseudo-sectioned CJPE cases: `{report['total_pseudo_sectioned_cjpe_cases']}`",
        f"- RR test metrics: accuracy=`{report['rr_test_metrics'].get('accuracy')}` macro_f1=`{report['rr_test_metrics'].get('macro_f1')}`",
        f"- CJPE prediction distribution: `{report['cjpe_prediction_distribution']}`",
        f"- CJPE section presence ratios: `{report['section_presence_ratios']}`",
        f"- Cases with all major sections present: `{report['all_major_sections_count']}` ({report['all_major_sections_ratio']})",
        f"- Dominant-section case ratio: `{report['dominant_section_case_ratio']}`",
        f"- Prediction confidence summary: `{report['confidence_summary']}`",
        f"- RR-to-CJPE distribution shift: `{report['distribution_shift_rr_to_cjpe']}`",
        "",
        "## Recommendations",
        "",
    ]
    for recommendation in report["recommendations"]:
        lines.append(f"- {recommendation}")
    return "\n".join(lines).strip() + "\n"


def _build_recommendations(
    *,
    training_metrics: dict[str, Any],
    cjpe_reconstruction_report: dict[str, Any],
    cjpe_prediction_report: dict[str, Any],
    all_major_sections_ratio: float,
) -> list[str]:
    recommendations: list[str] = []
    test_metrics = training_metrics["metrics_by_split"].get("test", {})
    macro_f1 = float(test_metrics.get("macro_f1", 0.0))
    conclusion_f1 = float(test_metrics.get("per_class", {}).get("conclusion", {}).get("f1", 0.0))
    other_f1 = float(test_metrics.get("per_class", {}).get("other", {}).get("f1", 0.0))
    presence_ratios = {
        section: round(
            cjpe_reconstruction_report["section_presence_counts"].get(section, 0)
            / max(cjpe_reconstruction_report["total_cases"], 1),
            6,
        )
        for section in BROAD_SECTION_ORDER
    }
    dominant_ratio = round(
        cjpe_reconstruction_report["dominant_section_case_count"]
        / max(cjpe_reconstruction_report["total_cases"], 1),
        6,
    )
    if macro_f1 < 0.6:
        recommendations.append("Not suitable as gold structural annotation; use predicted sections as weak structure only.")
    else:
        recommendations.append("Suitable as weak structural annotation for large-scale CJPE preprocessing, with manual spot checks.")
    recommendations.append("Suitable for section-aware perturbation prototyping and section masking experiments on CJPE.")
    recommendations.append("Suitable for section ablation studies where section boundaries are treated as pseudo-labels rather than gold labels.")
    if presence_ratios["precedents"] < 0.25:
        recommendations.append("Precedent-based perturbations should be used cautiously because predicted precedent coverage is sparse.")
    else:
        recommendations.append("Precedent distractor injection experiments are plausible because precedents are present in a meaningful share of CJPE cases.")
    if all_major_sections_ratio < 0.5:
        recommendations.append("Large-scale robustness experiments should exclude cases missing one of facts, reasoning, or conclusion, or at least stratify by section coverage.")
    else:
        recommendations.append("A large subset of CJPE cases contains all major sections, which supports section-aware robustness experiments.")
    if conclusion_f1 < 0.5 or other_f1 < 0.4:
        recommendations.append("Conclusion and other predictions are comparatively noisy; treat them as lower-confidence structure during experiment design.")
    if dominant_ratio > 0.5:
        recommendations.append("Prediction collapse is substantial; inspect dominant-section cases before trusting section-specific perturbations.")
    else:
        recommendations.append("Prediction collapse is limited enough for pilot section-aware perturbation experiments.")
    recommendations.append("Proceed to CJPE judgment-prediction baselines using full text plus pseudo-sectioned variants as parallel inputs.")
    return recommendations
