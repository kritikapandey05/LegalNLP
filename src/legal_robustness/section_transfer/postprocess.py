from __future__ import annotations

import logging
from collections import Counter, defaultdict
from statistics import mean

from legal_robustness.config.schema import AppConfig
from legal_robustness.section_transfer.diagnostics import BROAD_SECTION_ORDER, describe_numeric_series
from legal_robustness.section_transfer.types import (
    CJPEPseudoSectionedCase,
    CJPEPseudoSectionedResult,
    CJPESentencePredictionCase,
)


def reconstruct_cjpe_predicted_sections(
    predicted_cases: list[CJPESentencePredictionCase],
    *,
    config: AppConfig,
    logger: logging.Logger | None = None,
) -> CJPEPseudoSectionedResult:
    logger = logger or logging.getLogger(__name__)
    warnings: list[str] = []
    records: list[CJPEPseudoSectionedCase] = []
    counts_by_split: Counter[str] = Counter()
    section_presence_counts: Counter[str] = Counter()
    empty_section_counts: Counter[str] = Counter()
    section_lengths_sentences: dict[str, list[int]] = defaultdict(list)
    section_lengths_chars: dict[str, list[int]] = defaultdict(list)
    dominant_section_cases = 0
    dominant_section_ratio_threshold = config.section_transfer.dominant_section_ratio_threshold
    section_pattern_counts: Counter[str] = Counter()
    non_empty_sections_per_case: list[int] = []

    for case in predicted_cases:
        grouped_sentences = {section: [] for section in BROAD_SECTION_ORDER}
        section_sentence_map = {section: [] for section in BROAD_SECTION_ORDER}
        for sentence_index, (sentence_text, predicted_label) in enumerate(
            zip(case.sentences, case.predicted_broad_labels, strict=False)
        ):
            grouped_sentences.setdefault(predicted_label, [])
            section_sentence_map.setdefault(predicted_label, [])
            grouped_sentences[predicted_label].append(sentence_text)
            section_sentence_map[predicted_label].append(sentence_index)

        grouped_sections = {
            section: "\n".join(grouped_sentences.get(section, []))
            for section in BROAD_SECTION_ORDER
        }
        lengths_sentences = {
            section: len(section_sentence_map.get(section, []))
            for section in BROAD_SECTION_ORDER
        }
        lengths_chars = {
            section: len(grouped_sections[section])
            for section in BROAD_SECTION_ORDER
        }
        non_empty_sections = [section for section in BROAD_SECTION_ORDER if lengths_sentences[section] > 0]
        non_empty_sections_per_case.append(len(non_empty_sections))
        section_pattern_counts["|".join(non_empty_sections) if non_empty_sections else "<empty>"] += 1
        case_distribution = Counter(case.predicted_broad_labels)
        if case_distribution:
            dominant_ratio = case_distribution.most_common(1)[0][1] / len(case.predicted_broad_labels)
            if dominant_ratio >= dominant_section_ratio_threshold:
                dominant_section_cases += 1

        for section in BROAD_SECTION_ORDER:
            section_lengths_sentences[section].append(lengths_sentences[section])
            section_lengths_chars[section].append(lengths_chars[section])
            if lengths_sentences[section] > 0:
                section_presence_counts[section] += 1
            else:
                empty_section_counts[section] += 1

        counts_by_split[case.split] += 1
        records.append(
            CJPEPseudoSectionedCase(
                case_id=case.case_id,
                cjpe_label=case.label,
                split=case.split,
                subset=case.subset,
                raw_text=case.raw_text,
                sentences=list(case.sentences),
                sentence_indices=list(case.sentence_indices),
                sentence_start_chars=list(case.sentence_start_chars),
                sentence_end_chars=list(case.sentence_end_chars),
                predicted_broad_labels=list(case.predicted_broad_labels),
                predicted_label_scores=list(case.predicted_label_scores),
                grouped_sections=grouped_sections,
                section_sentence_map=section_sentence_map,
                section_lengths_sentences=lengths_sentences,
                section_lengths_chars=lengths_chars,
                prediction_metadata=dict(case.prediction_metadata),
                source_file=case.source_file,
                source_metadata=dict(case.source_metadata),
            )
        )

    report = {
        "task": "cjpe_reconstructed_sections",
        "total_cases": len(records),
        "counts_by_split": dict(sorted(counts_by_split.items())),
        "section_presence_counts": {section: section_presence_counts.get(section, 0) for section in BROAD_SECTION_ORDER},
        "empty_section_counts": {section: empty_section_counts.get(section, 0) for section in BROAD_SECTION_ORDER},
        "section_length_sentences": {
            section: describe_numeric_series(values)
            for section, values in sorted(section_lengths_sentences.items())
        },
        "section_length_chars": {
            section: describe_numeric_series(values)
            for section, values in sorted(section_lengths_chars.items())
        },
        "dominant_section_case_count": dominant_section_cases,
        "dominant_section_ratio_threshold": dominant_section_ratio_threshold,
        "average_sections_present": round(mean(non_empty_sections_per_case), 6) if non_empty_sections_per_case else 0.0,
        "section_pattern_counts": dict(sorted(section_pattern_counts.items())),
        "warnings": warnings,
    }
    logger.info(
        "Reconstructed pseudo-sections for %s CJPE cases.",
        len(records),
    )
    return CJPEPseudoSectionedResult(records=records, warnings=warnings, report=report)


def build_cjpe_reconstruction_samples(
    records: list[CJPEPseudoSectionedCase],
    *,
    sample_size: int,
    preview_chars: int,
) -> list[dict[str, object]]:
    return [
        {
            "case_id": record.case_id,
            "split": record.split,
            "subset": record.subset,
            "cjpe_label": record.cjpe_label,
            "predicted_broad_labels": record.predicted_broad_labels[:10],
            "section_sentence_map": record.section_sentence_map,
            "grouped_section_previews": {
                section: _truncate_text(text, max_chars=preview_chars)
                for section, text in record.grouped_sections.items()
            },
        }
        for record in records[:sample_size]
    ]


def render_cjpe_reconstruction_summary(report: dict[str, object]) -> str:
    lines = [
        "# CJPE Reconstruction Summary",
        "",
        f"- Total cases: `{report['total_cases']}`",
        f"- Counts by split: `{report['counts_by_split']}`",
        f"- Section presence counts: `{report['section_presence_counts']}`",
        f"- Empty section counts: `{report['empty_section_counts']}`",
        f"- Dominant-section case count: `{report['dominant_section_case_count']}`",
        f"- Average sections present: `{report['average_sections_present']}`",
        "",
        "## Sentence-Length Stats",
        "",
    ]
    for section, stats in report["section_length_sentences"].items():
        lines.append(f"- `{section}`: `{stats}`")
    lines.extend(["", "## Character-Length Stats", ""])
    for section, stats in report["section_length_chars"].items():
        lines.append(f"- `{section}`: `{stats}`")
    if report.get("warnings"):
        lines.extend(["", "## Warnings", ""])
        for warning in report["warnings"]:
            lines.append(f"- {warning}")
    return "\n".join(lines).strip() + "\n"


def _truncate_text(text: str, *, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip() + "..."
