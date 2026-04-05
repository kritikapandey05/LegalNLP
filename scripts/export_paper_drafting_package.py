from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from legal_robustness.config.loader import load_app_config
from legal_robustness.robustness import (
    build_paper_consistency_check,
    build_paper_freeze_manifest,
    build_paper_readiness_summary,
    build_recipe_qualitative_bundle,
    build_stability_vs_correctness_narrative,
    build_targeted_strengthening_check,
    render_draft_support_appendix,
    render_draft_support_introduction,
    render_draft_support_limitations,
    render_draft_support_method,
    render_draft_support_results,
    render_paper_consistency_check,
    render_paper_figure_selection,
    render_paper_freeze_manifest,
    render_paper_readiness_summary,
    render_paper_reproducibility_commands,
    render_paper_table_selection,
    render_targeted_strengthening_check,
)
from legal_robustness.utils.artifacts import (
    create_stage_output_dir,
    read_json,
    read_jsonl,
    write_json,
    write_text,
    write_yaml,
)
from legal_robustness.utils.cli import build_common_parser
from legal_robustness.utils.logging import configure_logging, get_logger
from legal_robustness.utils.paths import ensure_directory
from legal_robustness.utils.seeds import seed_everything


def parse_args() -> object:
    parser = build_common_parser(
        "Freeze the canonical robustness evidence and export the final paper-drafting support package."
    )
    parser.add_argument(
        "--baseline-run-dir",
        default=None,
        help="Optional path to the canonical baseline run directory.",
    )
    parser.add_argument(
        "--robustness-run-dir",
        default=None,
        help="Optional path to the canonical robustness run directory.",
    )
    parser.add_argument(
        "--results-package-dir",
        default=None,
        help="Optional path to the canonical results_package directory. Defaults to <robustness-run-dir>/results_package.",
    )
    parser.add_argument(
        "--section-transfer-run-dir",
        default=None,
        help="Optional path to the canonical section-transfer run directory. Defaults to the source_section_transfer_run_dir in the baseline report.",
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
        base_dir=config.output.reports_dir / config.robustness.paper_freeze_output_dirname,
        stage_name="paper_drafting_package",
        run_name=config.runtime.run_name,
    )
    configure_logging(
        level=config.logging.level,
        log_file=output_dir / "export_paper_drafting_package.log" if config.logging.log_to_file else None,
    )
    logger = get_logger("legal_robustness.scripts.export_paper_drafting_package")
    seed_everything(config.runtime.seed, deterministic=config.runtime.deterministic)
    write_yaml(output_dir / "config_snapshot.yaml", config.to_dict())

    baseline_run_dir = _resolve_required_path(
        explicit_value=args.baseline_run_dir,
        configured_value=config.robustness.canonical_baseline_run_dir,
        label="canonical baseline run directory",
    )
    robustness_run_dir = _resolve_required_path(
        explicit_value=args.robustness_run_dir,
        configured_value=config.robustness.canonical_robustness_run_dir,
        label="canonical robustness run directory",
    )
    results_package_dir = (
        Path(args.results_package_dir).resolve()
        if args.results_package_dir
        else (robustness_run_dir / config.robustness.results_package_dirname).resolve()
    )

    baseline_report = read_json(baseline_run_dir / "baseline_prediction_metrics.json")
    section_transfer_run_dir = (
        Path(args.section_transfer_run_dir).resolve()
        if args.section_transfer_run_dir
        else (
            config.robustness.canonical_section_transfer_run_dir.resolve()
            if config.robustness.canonical_section_transfer_run_dir is not None
            else Path(baseline_report["source_section_transfer_run_dir"]).resolve()
        )
    )

    results_package_manifest = read_json(results_package_dir / "results_package_manifest.json")
    table_main_results = read_json(results_package_dir / "table_main_results.json")
    table_model_comparison = read_json(results_package_dir / "table_model_comparison.json")
    packaging_next_step_summary = read_json(
        results_package_dir / "paper_results_packaging_next_step_summary.json"
    )
    qualitative_examples = read_jsonl(results_package_dir / "paper_qualitative_examples.jsonl")

    comparative_metrics = read_json(robustness_run_dir / "comparative_robustness_metrics.json")
    unperturbed_comparison = read_json(robustness_run_dir / "unperturbed_model_comparison.json")
    failure_analysis_cases = read_jsonl(robustness_run_dir / "failure_analysis_cases.jsonl")
    stability_summary = read_json(robustness_run_dir / "stability_vs_correctness_summary.json")

    primary_model_variant = packaging_next_step_summary["primary_model_variant"]
    primary_probe = packaging_next_step_summary["primary_probe"]
    secondary_probe = packaging_next_step_summary["secondary_probe"]
    supporting_model_variants = [
        model_variant
        for model_variant in results_package_manifest.get("models_included", [])
        if model_variant != primary_model_variant
    ]

    stability_narrative = build_stability_vs_correctness_narrative(stability_summary)
    primary_bundle = build_recipe_qualitative_bundle(
        qualitative_examples,
        recipe_name=primary_probe,
        count=config.robustness.primary_qualitative_example_count,
        preview_chars=config.robustness.narrative_preview_chars,
    )
    secondary_bundle = build_recipe_qualitative_bundle(
        qualitative_examples,
        recipe_name=secondary_probe,
        count=config.robustness.secondary_qualitative_example_count,
        preview_chars=config.robustness.narrative_preview_chars,
    )

    introduction_path = output_dir / "draft_support_introduction.md"
    method_path = output_dir / "draft_support_method.md"
    results_path = output_dir / "draft_support_results.md"
    limitations_path = output_dir / "draft_support_limitations.md"
    appendix_path = output_dir / "draft_support_appendix.md"
    table_selection_path = output_dir / "paper_table_selection.md"
    figure_selection_path = output_dir / "paper_figure_selection.md"
    reproducibility_path = output_dir / "paper_reproducibility_commands.md"
    strengthening_json_path = output_dir / "targeted_strengthening_check.json"
    strengthening_md_path = output_dir / "targeted_strengthening_check.md"
    readiness_json_path = output_dir / "paper_readiness_summary.json"
    readiness_md_path = output_dir / "paper_readiness_summary.md"
    freeze_json_path = output_dir / "paper_freeze_manifest.json"
    freeze_md_path = output_dir / "paper_freeze_manifest.md"
    consistency_json_path = output_dir / "paper_consistency_check.json"
    consistency_md_path = output_dir / "paper_consistency_check.md"

    if config.robustness.export_writing_support_bundles:
        write_text(
            introduction_path,
            render_draft_support_introduction(
                primary_model_variant=primary_model_variant,
                primary_probe=primary_probe,
                secondary_probe=secondary_probe,
            ),
        )
        write_text(
            method_path,
            render_draft_support_method(
                section_transfer_run_dir=section_transfer_run_dir,
                baseline_run_dir=baseline_run_dir,
                robustness_run_dir=robustness_run_dir,
                primary_model_variant=primary_model_variant,
                primary_probe=primary_probe,
                secondary_probe=secondary_probe,
                supporting_model_variants=supporting_model_variants,
            ),
        )
        write_text(
            results_path,
            render_draft_support_results(
                table_main_results=table_main_results,
                table_model_comparison=table_model_comparison,
                stability_narrative=stability_narrative,
                primary_bundle=primary_bundle,
                secondary_bundle=secondary_bundle,
            ),
        )
        write_text(limitations_path, render_draft_support_limitations())
        write_text(
            reproducibility_path,
            render_paper_reproducibility_commands(
                section_transfer_run_dir=section_transfer_run_dir,
                baseline_run_dir=baseline_run_dir,
                robustness_run_dir=robustness_run_dir,
                results_package_dir=results_package_dir,
                drafting_package_dir=output_dir,
            ),
        )
        write_text(
            appendix_path,
            render_draft_support_appendix(
                freeze_manifest={
                    "canonical_section_transfer_run_dir": str(section_transfer_run_dir),
                    "canonical_baseline_run_dir": str(baseline_run_dir),
                    "canonical_robustness_run_dir": str(robustness_run_dir),
                },
                reproducibility_commands_path=reproducibility_path,
                primary_bundle_path=results_package_dir / "qualitative_examples_primary.md",
                secondary_bundle_path=results_package_dir / "qualitative_examples_secondary.md",
            ),
        )

    if config.robustness.export_paper_selection_guidance:
        write_text(
            table_selection_path,
            render_paper_table_selection(
                primary_bundle_path=results_package_dir / "qualitative_examples_primary.md",
                secondary_bundle_path=results_package_dir / "qualitative_examples_secondary.md",
            ),
        )
        write_text(
            figure_selection_path,
            render_paper_figure_selection(),
        )

    targeted_strengthening_check = None
    if config.robustness.run_targeted_strengthening_check:
        targeted_strengthening_check = build_targeted_strengthening_check(
            comparative_metrics=comparative_metrics,
            primary_model_variant=primary_model_variant,
            target_recipe=secondary_probe,
        )
        write_json(strengthening_json_path, targeted_strengthening_check)
        write_text(
            strengthening_md_path,
            render_targeted_strengthening_check(targeted_strengthening_check),
        )

    manuscript_artifacts: dict[str, list[Path]] = {
        "results": [
            results_package_dir / "table_main_results.md",
            results_package_dir / "table_model_comparison.md",
            results_package_dir / "qualitative_examples_primary.md",
        ],
        "appendix": [
            results_package_dir / "appendix_keep_reasoning_only_bundle.md",
            results_package_dir / "appendix_drop_precedents_bundle.md",
            results_package_dir / "table_stability_vs_correctness.md",
        ],
    }
    if config.robustness.export_writing_support_bundles:
        manuscript_artifacts["introduction"] = [introduction_path]
        manuscript_artifacts["method"] = [method_path, reproducibility_path]
        manuscript_artifacts["results"].append(results_path)
        manuscript_artifacts["limitations"] = [limitations_path]
        manuscript_artifacts["appendix"].append(appendix_path)
    if config.robustness.export_paper_selection_guidance:
        manuscript_artifacts["selection_guidance"] = [table_selection_path, figure_selection_path]

    freeze_manifest = build_paper_freeze_manifest(
        section_transfer_run_dir=section_transfer_run_dir,
        baseline_run_dir=baseline_run_dir,
        robustness_run_dir=robustness_run_dir,
        results_package_dir=results_package_dir,
        drafting_package_dir=output_dir,
        primary_model_variant=primary_model_variant,
        primary_probe=primary_probe,
        secondary_probe=secondary_probe,
        supporting_model_variants=supporting_model_variants,
        manuscript_artifacts=manuscript_artifacts,
    )
    write_json(freeze_json_path, freeze_manifest)
    write_text(freeze_md_path, render_paper_freeze_manifest(freeze_manifest))

    appendix_bundle_paths = {
        "keep_reasoning_only": results_package_dir / "appendix_keep_reasoning_only_bundle.md",
        "drop_precedents": results_package_dir / "appendix_drop_precedents_bundle.md",
    }
    consistency_check = build_paper_consistency_check(
        freeze_manifest=freeze_manifest,
        results_package_manifest=results_package_manifest,
        unperturbed_comparison=unperturbed_comparison,
        comparative_metrics=comparative_metrics,
        table_main_results=table_main_results,
        table_model_comparison=table_model_comparison,
        qualitative_examples=qualitative_examples,
        failure_analysis_cases=failure_analysis_cases,
        appendix_bundle_paths=appendix_bundle_paths,
    )
    write_json(consistency_json_path, consistency_check)
    write_text(
        consistency_md_path,
        render_paper_consistency_check(consistency_check),
    )

    paper_readiness_summary = build_paper_readiness_summary(
        freeze_manifest=freeze_manifest,
        consistency_check=consistency_check,
        packaging_next_step_summary=packaging_next_step_summary,
        targeted_strengthening_check=targeted_strengthening_check,
    )
    write_json(readiness_json_path, paper_readiness_summary)
    write_text(
        readiness_md_path,
        render_paper_readiness_summary(paper_readiness_summary),
    )

    logger.info(
        "Paper drafting package created from baseline=%s robustness=%s results_package=%s",
        baseline_run_dir,
        robustness_run_dir,
        results_package_dir,
    )
    logger.info(
        "Consistency status=%s paper_ready=%s",
        consistency_check["overall_status"],
        paper_readiness_summary["ready_for_pilot_paper_drafting"],
    )

    if consistency_check["overall_status"] == "fail":
        logger.error("Paper consistency check failed. Review %s before drafting.", consistency_md_path)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
