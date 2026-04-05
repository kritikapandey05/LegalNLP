from __future__ import annotations

import csv
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from legal_robustness.config.loader import load_app_config
from legal_robustness.prediction import (
    build_unperturbed_model_comparison,
    expand_unperturbed_model_variants,
    load_pseudo_sectioned_cases_from_run_dir,
    render_baseline_prediction_metrics,
    render_unperturbed_model_comparison,
    resolve_section_transfer_run_dir,
    train_prediction_baselines,
)
from legal_robustness.perturbations import (
    generate_perturbation_sets,
    render_perturbation_manifest,
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
        "Train first CJPE judgment-prediction baselines on pseudo-sectioned inputs and export deterministic perturbation primitives."
    )
    parser.add_argument(
        "--section-transfer-dir",
        default=None,
        help="Optional path to an existing section-transfer run directory. Defaults to the most recent run under outputs/reports/section_transfer.",
    )
    return parser.parse_args()


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
        base_dir=config.output.reports_dir / "prediction_baselines",
        stage_name="prediction_baselines",
        run_name=config.runtime.run_name,
    )
    configure_logging(
        level=config.logging.level,
        log_file=output_dir / "train_baseline.log" if config.logging.log_to_file else None,
    )
    logger = get_logger("legal_robustness.scripts.train_baseline")
    seed_everything(config.runtime.seed, deterministic=config.runtime.deterministic)
    write_yaml(output_dir / "config_snapshot.yaml", config.to_dict())

    section_transfer_run_dir = resolve_section_transfer_run_dir(
        reports_dir=config.output.reports_dir,
        explicit_run_dir=Path(args.section_transfer_dir) if args.section_transfer_dir else None,
    )
    logger.info("Using section-transfer artifacts from %s", section_transfer_run_dir)
    pseudo_sectioned_cases = load_pseudo_sectioned_cases_from_run_dir(section_transfer_run_dir)

    baseline_report, predictions_by_split, confusion_rows, baseline_samples = train_prediction_baselines(
        pseudo_sectioned_cases,
        config=config,
        output_dir=output_dir,
        source_section_transfer_run_dir=section_transfer_run_dir,
        logger=logger,
    )
    perturbation_rows_by_recipe, perturbation_manifest, perturbation_samples = generate_perturbation_sets(
        pseudo_sectioned_cases,
        config=config,
    )

    write_json(output_dir / "baseline_prediction_metrics.json", baseline_report)
    write_text(
        output_dir / "baseline_prediction_metrics.md",
        render_baseline_prediction_metrics(baseline_report),
    )
    selected_model_variants = expand_unperturbed_model_variants(
        config.robustness.selected_model_variants,
        include_full_text=config.robustness.include_full_text_in_unperturbed_comparison,
    )
    unperturbed_comparison = build_unperturbed_model_comparison(
        baseline_report,
        primary_split=config.prediction.evaluation_splits[-1],
        selected_model_variants=selected_model_variants,
    )
    write_json(output_dir / "unperturbed_model_comparison.json", unperturbed_comparison)
    write_text(
        output_dir / "unperturbed_model_comparison.md",
        render_unperturbed_model_comparison(unperturbed_comparison),
    )
    for split_name in config.prediction.evaluation_splits:
        write_parquet(
            output_dir / f"baseline_prediction_predictions_{split_name}.parquet",
            [record.to_dict() for record in predictions_by_split.get(split_name, [])],
        )
    _write_confusion_matrix_csv(
        output_dir / "baseline_prediction_confusion_matrix.csv",
        confusion_rows,
    )
    write_jsonl(
        output_dir / "baseline_prediction_input_samples.jsonl",
        baseline_samples,
    )

    perturbation_dir = output_dir / "cjpe_perturbation_sets"
    ensure_directory(perturbation_dir)
    for recipe_name, rows in perturbation_rows_by_recipe.items():
        write_parquet(
            perturbation_dir / f"{recipe_name}.parquet",
            [row.to_dict() for row in rows],
        )
    write_json(output_dir / "perturbation_manifest.json", perturbation_manifest)
    write_text(
        output_dir / "perturbation_manifest.md",
        render_perturbation_manifest(perturbation_manifest),
    )
    write_jsonl(
        output_dir / "perturbation_samples.jsonl",
        perturbation_samples,
    )

    logger.info(
        "Baseline metrics written for models=%s variants=%s",
        baseline_report["baseline_models"],
        baseline_report["input_variants"],
    )
    logger.info(
        "Unperturbed comparison written for model_variants=%s",
        unperturbed_comparison["model_variants"],
    )
    logger.info(
        "Perturbation artifacts written for recipes=%s total_examples=%s",
        list(perturbation_rows_by_recipe),
        perturbation_manifest["total_examples"],
    )
    logger.info("Baseline and perturbation-prep artifacts written to %s", output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
