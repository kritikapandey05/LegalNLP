from __future__ import annotations

from collections import defaultdict
from pathlib import Path

from legal_robustness.config.schema import AppConfig
from legal_robustness.prediction.input_variants import build_prediction_input_text
from legal_robustness.prediction.types import PredictionInputExample
from legal_robustness.section_transfer.types import CJPEPseudoSectionedCase
from legal_robustness.utils.artifacts import read_parquet
from legal_robustness.utils.exceptions import PredictionError


def resolve_section_transfer_run_dir(
    *,
    reports_dir: Path,
    explicit_run_dir: Path | None = None,
) -> Path:
    if explicit_run_dir is not None:
        run_dir = explicit_run_dir.resolve()
        if not run_dir.exists():
            raise PredictionError(f"Section-transfer artifact directory does not exist: {run_dir}")
        return run_dir

    section_transfer_root = (reports_dir / "section_transfer").resolve()
    if not section_transfer_root.exists():
        raise PredictionError(
            f"No section-transfer report directory exists at {section_transfer_root}. "
            "Run scripts/run_section_transfer.py first or pass --section-transfer-dir."
        )
    candidate_dirs = [path for path in section_transfer_root.iterdir() if path.is_dir()]
    if not candidate_dirs:
        raise PredictionError(
            f"No section-transfer runs were found under {section_transfer_root}. "
            "Run scripts/run_section_transfer.py first or pass --section-transfer-dir."
        )
    return max(candidate_dirs, key=lambda path: path.stat().st_mtime)


def load_pseudo_sectioned_cases_from_run_dir(
    run_dir: Path,
    *,
    splits: tuple[str, ...] | list[str] | None = None,
) -> list[CJPEPseudoSectionedCase]:
    parquet_path = run_dir / "cjpe_reconstructed_sections.parquet"
    if not parquet_path.exists():
        raise PredictionError(
            f"The section-transfer run directory {run_dir} does not contain cjpe_reconstructed_sections.parquet."
        )
    return load_pseudo_sectioned_cases_from_parquet(parquet_path, splits=splits)


def load_pseudo_sectioned_cases_from_parquet(
    path: Path,
    *,
    splits: tuple[str, ...] | list[str] | None = None,
) -> list[CJPEPseudoSectionedCase]:
    filters = None
    if splits:
        normalized_splits = [str(value) for value in splits]
        filters = [("split", "in", normalized_splits)]
    rows = read_parquet(path, filters=filters)
    return [_case_from_row(row) for row in rows]


def build_prediction_examples(
    cases: list[CJPEPseudoSectionedCase],
    *,
    variant_name: str,
    config: AppConfig,
) -> list[PredictionInputExample]:
    rows: list[PredictionInputExample] = []
    for case in cases:
        input_text, variant_metadata = build_prediction_input_text(
            case,
            variant_name=variant_name,
            include_section_markers=config.prediction.include_section_markers,
        )
        rows.append(
            PredictionInputExample(
                case_id=case.case_id,
                split=case.split,
                subset=case.subset,
                label=case.cjpe_label,
                input_variant=variant_name,
                input_text=input_text,
                input_text_length_chars=len(input_text),
                sections_used=list(variant_metadata["sections_used"]),
                sections_omitted=list(variant_metadata["sections_omitted"]),
                empty_selected_sections=list(variant_metadata["empty_selected_sections"]),
                variant_metadata=variant_metadata,
                source_file=case.source_file,
                source_metadata=dict(case.source_metadata),
            )
        )
    return rows


def group_prediction_examples_by_split(
    examples: list[PredictionInputExample],
) -> dict[str, list[PredictionInputExample]]:
    grouped: dict[str, list[PredictionInputExample]] = defaultdict(list)
    for example in examples:
        grouped[example.split].append(example)
    return dict(grouped)


def build_prediction_example_samples(
    examples: list[PredictionInputExample],
    *,
    sample_size: int,
    preview_chars: int,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for example in examples[:sample_size]:
        rows.append(
            {
                "case_id": example.case_id,
                "split": example.split,
                "subset": example.subset,
                "label": example.label,
                "input_variant": example.input_variant,
                "sections_used": example.sections_used,
                "sections_omitted": example.sections_omitted,
                "empty_selected_sections": example.empty_selected_sections,
                "input_text_preview": _truncate_text(example.input_text, max_chars=preview_chars),
            }
        )
    return rows


def _case_from_row(row: dict[str, object]) -> CJPEPseudoSectionedCase:
    return CJPEPseudoSectionedCase(
        case_id=str(row.get("case_id", "")),
        cjpe_label=row.get("cjpe_label"),
        split=str(row.get("split", "")),
        subset=row.get("subset"),
        raw_text=str(row.get("raw_text", "")),
        sentences=list(row.get("sentences") or []),
        sentence_indices=[int(value) for value in list(row.get("sentence_indices") or [])],
        sentence_start_chars=[int(value) for value in list(row.get("sentence_start_chars") or [])],
        sentence_end_chars=[int(value) for value in list(row.get("sentence_end_chars") or [])],
        predicted_broad_labels=list(row.get("predicted_broad_labels") or []),
        predicted_label_scores=[float(value) for value in list(row.get("predicted_label_scores") or [])],
        grouped_sections={str(key): str(value) for key, value in dict(row.get("grouped_sections") or {}).items()},
        section_sentence_map={
            str(key): [int(value) for value in list(values)]
            for key, values in dict(row.get("section_sentence_map") or {}).items()
        },
        section_lengths_sentences={
            str(key): int(value)
            for key, value in dict(row.get("section_lengths_sentences") or {}).items()
        },
        section_lengths_chars={
            str(key): int(value)
            for key, value in dict(row.get("section_lengths_chars") or {}).items()
        },
        prediction_metadata=dict(row.get("prediction_metadata") or {}),
        source_file=str(row.get("source_file", "")),
        source_metadata=dict(row.get("source_metadata") or {}),
    )


def _truncate_text(text: str, *, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip() + "..."
