from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from legal_robustness.config.loader import load_app_config
from legal_robustness.prediction import load_pseudo_sectioned_cases_from_run_dir
from legal_robustness.robustness import (
    build_chart_data_section_importance_coverage,
    build_chart_data_section_importance_ranking,
    build_chart_data_section_importance_scores,
    build_section_importance_cross_model_check,
    build_section_importance_next_step_summary,
    build_section_importance_scores,
    build_updated_results_package_manifest,
    evaluate_section_importance_model,
    render_results_package_manifest,
    render_section_importance_cross_model_check,
    render_section_importance_narrative_main,
    render_section_importance_narrative_supporting,
    render_section_importance_next_step_summary,
    render_section_importance_ranking,
    render_section_importance_scores,
)
from legal_robustness.robustness.datasets import load_baseline_report, resolve_baseline_run_dir
from legal_robustness.utils.artifacts import (
    create_stage_output_dir,
    read_json,
    write_json,
    write_text,
    write_yaml,
)
from legal_robustness.utils.cli import build_common_parser
from legal_robustness.utils.logging import configure_logging, get_logger
from legal_robustness.utils.seeds import seed_everything


def parse_args() -> object:
    parser = build_common_parser(
        "Run the APA-centered section-importance matrix and export ranking-oriented paper-facing outputs."
    )
    parser.add_argument(
        "--baseline-run-dir",
        default=None,
        help="Optional path to the canonical prediction-baseline run directory.",
    )
    parser.add_argument(
        "--robustness-run-dir",
        default=None,
        help="Optional path to the canonical robustness run directory that contains the frozen results package.",
    )
    parser.add_argument(
        "--results-package-dir",
        default=None,
        help="Optional path to the canonical results_package directory. Defaults to <robustness-run-dir>/results_package.",
    )
    return parser.parse_args()


def _resolve_required_path(
    *,
    explicit_value: str | None,
    configured_value: Path | None,
    label: str,
) -> Path:
    if explicit_value:
        return Path(explicit_value).resolve()
    if configured_value is not None:
        return configured_value.resolve()
    raise RuntimeError(f"{label} is required. Pass it explicitly or set it in the config.")


def _resolve_latest_robustness_run(reports_dir: Path) -> Path:
    root = (reports_dir / "robustness").resolve()
    if not root.exists():
        raise RuntimeError(f"No robustness report directory exists at {root}.")
    candidate_dirs = [path for path in root.iterdir() if path.is_dir()]
    if not candidate_dirs:
        raise RuntimeError(f"No robustness runs were found under {root}.")
    return max(candidate_dirs, key=lambda path: path.stat().st_mtime)


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
        base_dir=config.output.reports_dir / config.robustness.section_importance_output_dirname,
        stage_name="section_importance",
        run_name=config.runtime.run_name,
    )
    configure_logging(
        level=config.logging.level,
        log_file=output_dir / "run_section_importance.log" if config.logging.log_to_file else None,
    )
    logger = get_logger("legal_robustness.scripts.run_section_importance")
    seed_everything(config.runtime.seed, deterministic=config.runtime.deterministic)
    write_yaml(output_dir / "config_snapshot.yaml", config.to_dict())

    baseline_run_dir = resolve_baseline_run_dir(
        reports_dir=config.output.reports_dir,
        explicit_run_dir=Path(args.baseline_run_dir) if args.baseline_run_dir else config.robustness.canonical_baseline_run_dir,
    )
    robustness_run_dir = (
        Path(args.robustness_run_dir).resolve()
        if args.robustness_run_dir
        else (
            config.robustness.canonical_robustness_run_dir.resolve()
            if config.robustness.canonical_robustness_run_dir is not None
            else _resolve_latest_robustness_run(config.output.reports_dir)
        )
    )
    results_package_dir = (
        Path(args.results_package_dir).resolve()
        if args.results_package_dir
        else (robustness_run_dir / config.robustness.results_package_dirname).resolve()
    )

    baseline_report = load_baseline_report(baseline_run_dir)
    section_transfer_run_dir = Path(baseline_report["source_section_transfer_run_dir"]).resolve()
    pseudo_sectioned_cases = load_pseudo_sectioned_cases_from_run_dir(
        section_transfer_run_dir,
        splits=config.robustness.evaluation_splits,
    )
    logger.info(
        "Loaded %s pseudo-sectioned cases for splits %s from %s",
        len(pseudo_sectioned_cases),
        config.robustness.evaluation_splits,
        section_transfer_run_dir,
    )

    primary_model_variant = config.robustness.section_importance_primary_model_variant
    primary_report, _ = evaluate_section_importance_model(
        baseline_run_dir=baseline_run_dir,
        pseudo_sectioned_cases=pseudo_sectioned_cases,
        model_variant=primary_model_variant,
        config=config,
    )
    primary_split = config.robustness.evaluation_splits[0]
    scores_report = build_section_importance_scores(
        primary_report,
        config=config,
        primary_split=primary_split,
    )

    supporting_reports: list[dict[str, object]] = []
    if config.robustness.run_section_importance_cross_model_check:
        for model_variant in config.robustness.section_importance_supporting_model_variants:
            report, _ = evaluate_section_importance_model(
                baseline_run_dir=baseline_run_dir,
                pseudo_sectioned_cases=pseudo_sectioned_cases,
                model_variant=model_variant,
                config=config,
                spec_names=config.robustness.section_importance_cross_model_recipes,
            )
            supporting_reports.append(report)
    cross_model_check = build_section_importance_cross_model_check(
        primary_model_report=primary_report,
        supporting_model_reports=supporting_reports,
        primary_split=primary_split,
    )

    chart_data_scores = build_chart_data_section_importance_scores(scores_report)
    chart_data_ranking = build_chart_data_section_importance_ranking(scores_report)
    chart_data_coverage = build_chart_data_section_importance_coverage(scores_report)
    narrative_main = render_section_importance_narrative_main(scores_report)
    narrative_supporting = render_section_importance_narrative_supporting(
        scores_report,
        cross_model_check,
    )
    next_step_summary = build_section_importance_next_step_summary(
        scores_report,
        cross_model_check,
    )

    write_json(output_dir / "section_importance_scores.json", scores_report)
    write_text(output_dir / "section_importance_scores.md", render_section_importance_scores(scores_report))
    write_text(output_dir / "section_importance_ranking.md", render_section_importance_ranking(scores_report))
    write_json(output_dir / "section_importance_cross_model_check.json", cross_model_check)
    write_text(
        output_dir / "section_importance_cross_model_check.md",
        render_section_importance_cross_model_check(cross_model_check),
    )
    if config.robustness.export_section_importance_chart_data:
        write_json(output_dir / "chart_data_section_importance_scores.json", chart_data_scores)
        write_json(output_dir / "chart_data_section_importance_ranking.json", chart_data_ranking)
        write_json(output_dir / "chart_data_section_importance_coverage.json", chart_data_coverage)
    write_text(output_dir / "section_importance_narrative_main.md", narrative_main)
    write_text(output_dir / "section_importance_narrative_supporting.md", narrative_supporting)
    write_json(output_dir / "section_importance_next_step_summary.json", next_step_summary)
    write_text(
        output_dir / "section_importance_next_step_summary.md",
        render_section_importance_next_step_summary(next_step_summary),
    )

    existing_manifest = read_json(results_package_dir / "results_package_manifest.json")
    updated_manifest = build_updated_results_package_manifest(
        existing_manifest=existing_manifest,
        source_results_package_dir=results_package_dir,
        section_importance_output_dir=output_dir,
        section_importance_files={
            "main_summary_files": [
                "section_importance_scores.md",
                "section_importance_ranking.md",
                "section_importance_narrative_main.md",
                "section_importance_narrative_supporting.md",
                "section_importance_next_step_summary.md",
            ],
            "chart_data_files": [
                "chart_data_section_importance_scores.json",
                "chart_data_section_importance_ranking.json",
                "chart_data_section_importance_coverage.json",
            ],
            "appendix_files": [
                "section_importance_cross_model_check.md",
            ],
            "qualitative_files": [],
        },
        additional_caveats=next_step_summary["visible_caveats"],
    )
    write_json(output_dir / "results_package_manifest.json", updated_manifest)
    write_text(
        output_dir / "results_package_manifest.md",
        render_results_package_manifest(updated_manifest),
    )

    ranking_preview = [
        f"{row['rank']}. {row['section']} ({row['confidence_label']})"
        for row in scores_report.get("section_rows", [])[:5]
    ]
    logger.info(
        "Section-importance package created from baseline=%s robustness=%s results_package=%s",
        baseline_run_dir,
        robustness_run_dir,
        results_package_dir,
    )
    logger.info("Top section-importance ranking: %s", ranking_preview)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
