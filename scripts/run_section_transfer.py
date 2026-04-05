from __future__ import annotations

import csv
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from legal_robustness.config.loader import load_app_config
from legal_robustness.data.discovery import discover_task_files, group_files_by_task
from legal_robustness.data.label_inventory import generate_rr_label_inventory
from legal_robustness.data.loaders import load_cjpe_raw_cases, load_rr_raw_cases
from legal_robustness.data.normalize import normalize_cjpe_cases, normalize_rr_cases
from legal_robustness.data.reconstruct import reconstruct_rr_sections
from legal_robustness.data.rr_mapping import validate_rr_section_mapping
from legal_robustness.section_transfer import (
    build_cjpe_sentence_samples,
    build_cjpe_prediction_samples,
    build_cjpe_reconstruction_samples,
    build_rr_sentence_supervision,
    confusion_matrix_csv_rows,
    infer_cjpe_sections,
    load_section_tagger_model,
    reconstruct_cjpe_predicted_sections,
    render_cjpe_reconstruction_summary,
    render_cjpe_section_prediction_summary,
    render_cjpe_sentence_segmentation_report,
    render_rr_section_tagger_metrics,
    render_rr_sentence_supervision_summary,
    render_section_transfer_readiness_summary,
    segment_cjpe_cases,
    summarize_section_transfer_readiness,
    train_and_evaluate_rr_section_tagger,
)
from legal_robustness.utils.artifacts import (
    create_stage_output_dir,
    write_json,
    write_jsonl,
    write_parquet,
    write_text,
    write_yaml,
)
from legal_robustness.utils.cli import build_common_parser
from legal_robustness.utils.logging import configure_logging, get_logger
from legal_robustness.utils.paths import ensure_directory
from legal_robustness.utils.seeds import seed_everything


def parse_args() -> object:
    parser = build_common_parser(
        "Run the RR-to-CJPE section-transfer pipeline: supervision export, CJPE segmentation, RR broad-section training, CJPE inference, pseudo-section reconstruction, and readiness reporting."
    )
    return parser.parse_args()


def _sample_records(records: list[object], sample_size: int) -> list[dict]:
    return [record.to_dict() for record in records[:sample_size]]


def _write_confusion_matrix_csv(path: Path, rows: list[dict[str, object]]) -> None:
    ensure_directory(path.parent)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    config = load_app_config(
        config_path=Path(args.config),
        dataset_root_override=Path(args.dataset_root) if args.dataset_root else None,
        run_name_override=args.run_name,
        log_level_override=args.log_level,
        project_root=PROJECT_ROOT,
    )
    output_dir = create_stage_output_dir(
        base_dir=config.output.reports_dir / "section_transfer",
        stage_name="section_transfer",
        run_name=config.runtime.run_name,
    )
    configure_logging(
        level=config.logging.level,
        log_file=output_dir / "section_transfer.log" if config.logging.log_to_file else None,
    )
    logger = get_logger("legal_robustness.scripts.run_section_transfer")
    seed_everything(config.runtime.seed, deterministic=config.runtime.deterministic)
    write_yaml(output_dir / "config_snapshot.yaml", config.to_dict())

    manifest = discover_task_files(
        dataset_root=config.data.dataset_root,
        tasks=("cjpe", "rr"),
        logger=logger,
    )
    grouped_files = group_files_by_task(manifest.discovered_files)
    cjpe_raw = load_cjpe_raw_cases(grouped_files.get("cjpe", []), config=config, logger=logger)
    rr_raw = load_rr_raw_cases(grouped_files.get("rr", []), config=config, logger=logger)
    cjpe_normalized = normalize_cjpe_cases(cjpe_raw.records, config=config, logger=logger)
    rr_normalized = normalize_rr_cases(rr_raw.records, config=config, logger=logger)
    rr_label_inventory = generate_rr_label_inventory(rr_normalized.records, config=config)
    rr_section_mapping = validate_rr_section_mapping(
        rr_normalized.records,
        label_inventory=rr_label_inventory,
        config=config,
        logger=logger,
    )
    rr_reconstructed = reconstruct_rr_sections(
        rr_normalized.records,
        mapping_report=rr_section_mapping,
        config=config,
        logger=logger,
    )

    rr_supervision = build_rr_sentence_supervision(
        rr_reconstructed.records,
        mapping_report=rr_section_mapping,
        config=config,
        logger=logger,
    )
    cjpe_segmented = segment_cjpe_cases(
        cjpe_normalized.records,
        config=config,
        logger=logger,
    )
    training_result = train_and_evaluate_rr_section_tagger(
        rr_supervision.records,
        config=config,
        output_dir=output_dir,
        logger=logger,
    )
    trained_model = load_section_tagger_model(Path(training_result.model_path))
    cjpe_predictions = infer_cjpe_sections(
        segmented_cases=cjpe_segmented.records,
        cjpe_cases=cjpe_normalized.records,
        model=trained_model,
        model_path=training_result.model_path,
        config=config,
        logger=logger,
    )
    cjpe_reconstructed = reconstruct_cjpe_predicted_sections(
        cjpe_predictions.records,
        config=config,
        logger=logger,
    )
    readiness_summary = summarize_section_transfer_readiness(
        rr_supervision=rr_supervision,
        training_result=training_result,
        cjpe_predictions=cjpe_predictions,
        cjpe_reconstructed=cjpe_reconstructed,
    )

    write_json(output_dir / "rr_sentence_supervision_summary.json", rr_supervision.report)
    write_text(
        output_dir / "rr_sentence_supervision_summary.md",
        render_rr_sentence_supervision_summary(rr_supervision.report),
    )
    write_parquet(
        output_dir / "rr_sentence_supervision.parquet",
        [record.to_dict() for record in rr_supervision.records],
    )
    write_jsonl(
        output_dir / "rr_sentence_supervision_sample.jsonl",
        _sample_records(rr_supervision.records, config.section_transfer.sample_size),
    )

    write_json(output_dir / "cjpe_sentence_segmentation_report.json", cjpe_segmented.report)
    write_text(
        output_dir / "cjpe_sentence_segmentation_report.md",
        render_cjpe_sentence_segmentation_report(cjpe_segmented.report),
    )
    write_parquet(
        output_dir / "cjpe_sentences.parquet",
        [record.to_dict() for record in cjpe_segmented.records],
    )
    write_jsonl(
        output_dir / "cjpe_sentences_sample.jsonl",
        build_cjpe_sentence_samples(
            cjpe_segmented.records,
            sample_size=config.section_transfer.sample_size,
        ),
    )

    write_json(output_dir / "rr_section_tagger_metrics.json", training_result.metrics)
    write_text(
        output_dir / "rr_section_tagger_metrics.md",
        render_rr_section_tagger_metrics(training_result.metrics),
    )
    write_json(
        Path(training_result.metadata_path),
        {
            "classifier_type": training_result.metrics["classifier_type"],
            "label_mode": training_result.metrics["label_mode"],
            "label_order": training_result.metrics["label_order"],
            "vocabulary_size": training_result.metrics["vocabulary_size"],
            "feature_settings": training_result.metrics["feature_settings"],
            "split_counts": training_result.metrics["split_counts"],
        },
    )
    _write_confusion_matrix_csv(
        output_dir / "rr_section_tagger_confusion_matrix.csv",
        confusion_matrix_csv_rows(
            training_result.metrics["confusion_matrix"],
            label_order=training_result.metrics["label_order"],
        ),
    )
    write_jsonl(
        output_dir / "rr_section_tagger_predictions_sample.jsonl",
        training_result.prediction_samples,
    )
    write_json(output_dir / "cjpe_section_prediction_summary.json", cjpe_predictions.report)
    write_text(
        output_dir / "cjpe_section_prediction_summary.md",
        render_cjpe_section_prediction_summary(cjpe_predictions.report),
    )
    write_parquet(
        output_dir / "cjpe_predicted_sections.parquet",
        [record.to_dict() for record in cjpe_predictions.records],
    )
    write_jsonl(
        output_dir / "cjpe_predicted_sections_sample.jsonl",
        build_cjpe_prediction_samples(
            cjpe_predictions.records,
            sample_size=config.section_transfer.sample_size,
            preview_sentence_count=config.section_transfer.sample_sentence_preview_count,
        ),
    )
    write_json(output_dir / "cjpe_reconstruction_summary.json", cjpe_reconstructed.report)
    write_text(
        output_dir / "cjpe_reconstruction_summary.md",
        render_cjpe_reconstruction_summary(cjpe_reconstructed.report),
    )
    write_parquet(
        output_dir / "cjpe_reconstructed_sections.parquet",
        [record.to_dict() for record in cjpe_reconstructed.records],
    )
    write_jsonl(
        output_dir / "cjpe_reconstructed_sections_sample.jsonl",
        build_cjpe_reconstruction_samples(
            cjpe_reconstructed.records,
            sample_size=config.section_transfer.sample_size,
            preview_chars=config.data.inspection_max_text_chars,
        ),
    )
    write_json(output_dir / "section_transfer_readiness_summary.json", readiness_summary)
    write_text(
        output_dir / "section_transfer_readiness_summary.md",
        render_section_transfer_readiness_summary(readiness_summary),
    )

    logger.info(
        "RR supervision rows=%s broad_distribution=%s",
        rr_supervision.report["total_sentences"],
        rr_supervision.report["broad_section_distribution"],
    )
    logger.info(
        "CJPE segmentation cases=%s total_sentences=%s sentence_count_stats=%s",
        cjpe_segmented.report["total_cases"],
        cjpe_segmented.report["total_sentences"],
        cjpe_segmented.report["sentence_count_per_case"],
    )
    logger.info(
        "RR section tagger metrics: dev=%s test=%s",
        training_result.metrics["metrics_by_split"].get("dev", {}),
        training_result.metrics["metrics_by_split"].get("test", {}),
    )
    logger.info(
        "CJPE section inference: cases=%s sentences=%s dominant_cases=%s",
        cjpe_predictions.report["total_cases"],
        cjpe_predictions.report["total_sentences"],
        cjpe_predictions.report["cases_above_dominant_threshold"],
    )
    logger.info(
        "CJPE pseudo-section reconstruction: cases=%s section_presence=%s",
        cjpe_reconstructed.report["total_cases"],
        cjpe_reconstructed.report["section_presence_counts"],
    )
    logger.info(
        "Section-transfer readiness: all_major_sections_ratio=%s recommendations=%s",
        readiness_summary["all_major_sections_ratio"],
        len(readiness_summary["recommendations"]),
    )
    logger.info("Section-transfer artifacts written to %s", output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
