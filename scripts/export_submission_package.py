from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from legal_robustness.config.loader import load_app_config
from legal_robustness.robustness import (
    build_claim_to_evidence_traceability,
    build_figure_captions,
    build_final_package_consistency_check,
    build_main_text_layout_guide,
    build_paper_abstract_support,
    build_paper_conclusion_support,
    build_paper_intro_support,
    build_paper_limitations_ethics_support,
    build_paper_method_support,
    build_paper_results_support,
    build_paper_handoff_summary,
    build_submission_package_manifest,
    build_table_captions,
    render_appendix_layout_guide,
    render_claim_to_evidence_traceability,
    render_figure_captions,
    render_figure_manifest,
    render_final_package_consistency_check,
    render_main_text_layout_guide,
    render_paper_abstract_support,
    render_paper_conclusion_support,
    render_paper_handoff_summary,
    render_paper_intro_support,
    render_paper_limitations_ethics_support,
    render_paper_method_support,
    render_paper_results_support,
    render_submission_figures,
    render_submission_package_manifest,
    render_table_captions,
)
from legal_robustness.robustness.final_writing_package import build_appendix_layout_guide
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
        "Export the final submission-style paper support package from the frozen evidence."
    )
    parser.add_argument("--baseline-run-dir", default=None)
    parser.add_argument("--robustness-run-dir", default=None)
    parser.add_argument("--results-package-dir", default=None)
    parser.add_argument("--paper-drafting-package-dir", default=None)
    parser.add_argument("--section-importance-dir", default=None)
    return parser.parse_args()


def _resolve_optional_path(explicit_value: str | None, configured_value: Path | None) -> Path | None:
    if explicit_value:
        return Path(explicit_value).resolve()
    if configured_value is not None:
        return configured_value.resolve()
    return None


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
        base_dir=config.output.reports_dir / config.robustness.submission_package_output_dirname,
        stage_name="submission_package",
        run_name=config.runtime.run_name,
    )
    configure_logging(
        level=config.logging.level,
        log_file=output_dir / "export_submission_package.log" if config.logging.log_to_file else None,
    )
    logger = get_logger("legal_robustness.scripts.export_submission_package")
    seed_everything(config.runtime.seed, deterministic=config.runtime.deterministic)
    write_yaml(output_dir / "config_snapshot.yaml", config.to_dict())

    paper_drafting_package_dir = _resolve_optional_path(
        args.paper_drafting_package_dir,
        config.robustness.canonical_paper_drafting_package_dir,
    )
    if paper_drafting_package_dir is None:
        raise RuntimeError(
            "A paper drafting package directory is required. Pass --paper-drafting-package-dir or set canonical_paper_drafting_package_dir."
        )
    freeze_manifest = read_json(paper_drafting_package_dir / "paper_freeze_manifest.json")
    prior_consistency_check = read_json(paper_drafting_package_dir / "paper_consistency_check.json")
    reproducibility_text = (paper_drafting_package_dir / "paper_reproducibility_commands.md").read_text(
        encoding="utf-8"
    )

    baseline_run_dir = (
        _resolve_optional_path(args.baseline_run_dir, config.robustness.canonical_baseline_run_dir)
        or Path(freeze_manifest["canonical_baseline_run_dir"]).resolve()
    )
    robustness_run_dir = (
        _resolve_optional_path(args.robustness_run_dir, config.robustness.canonical_robustness_run_dir)
        or Path(freeze_manifest["canonical_robustness_run_dir"]).resolve()
    )
    results_package_dir = (
        Path(args.results_package_dir).resolve()
        if args.results_package_dir
        else (
            (robustness_run_dir / config.robustness.results_package_dirname).resolve()
            if robustness_run_dir is not None
            else Path(freeze_manifest["canonical_results_package_dir"]).resolve()
        )
    )
    section_importance_dir = _resolve_optional_path(
        args.section_importance_dir,
        config.robustness.canonical_section_importance_run_dir,
    )
    if section_importance_dir is None:
        raise RuntimeError(
            "A section-importance package directory is required. Pass --section-importance-dir or set canonical_section_importance_run_dir."
        )

    table_main_results = read_json(results_package_dir / "table_main_results.json")
    table_model_comparison = read_json(results_package_dir / "table_model_comparison.json")
    table_stability_vs_correctness = read_json(results_package_dir / "table_stability_vs_correctness.json")
    results_package_manifest = read_json(results_package_dir / "results_package_manifest.json")
    chart_data_main_performance = read_json(results_package_dir / "chart_data_main_performance.json")
    chart_data_robustness_deltas = read_json(results_package_dir / "chart_data_robustness_deltas.json")
    chart_data_flip_rates = read_json(results_package_dir / "chart_data_flip_rates.json")

    section_importance_scores = read_json(section_importance_dir / "section_importance_scores.json")
    section_importance_cross_model_check = read_json(
        section_importance_dir / "section_importance_cross_model_check.json"
    )
    section_importance_next_step_summary = read_json(
        section_importance_dir / "section_importance_next_step_summary.json"
    )
    chart_data_section_importance_scores = read_json(
        section_importance_dir / "chart_data_section_importance_scores.json"
    )
    chart_data_section_importance_coverage = read_json(
        section_importance_dir / "chart_data_section_importance_coverage.json"
    )

    figure_manifest = render_submission_figures(
        output_dir=output_dir,
        figure_formats=config.robustness.submission_figure_formats,
        chart_data_main_performance=chart_data_main_performance,
        chart_data_robustness_deltas=chart_data_robustness_deltas,
        chart_data_flip_rates=chart_data_flip_rates,
        chart_data_section_importance_scores=chart_data_section_importance_scores,
        chart_data_section_importance_coverage=chart_data_section_importance_coverage,
        source_file_map={
            "chart_data_main_performance": results_package_dir / "chart_data_main_performance.json",
            "chart_data_robustness_deltas": results_package_dir / "chart_data_robustness_deltas.json",
            "chart_data_flip_rates": results_package_dir / "chart_data_flip_rates.json",
            "chart_data_section_importance_scores": section_importance_dir / "chart_data_section_importance_scores.json",
            "chart_data_section_importance_coverage": section_importance_dir / "chart_data_section_importance_coverage.json",
        },
    )
    figure_manifest_json_path = output_dir / "figure_manifest.json"
    figure_manifest_md_path = output_dir / "figure_manifest.md"
    write_json(figure_manifest_json_path, figure_manifest)
    write_text(figure_manifest_md_path, render_figure_manifest(figure_manifest))

    abstract_support = build_paper_abstract_support(
        table_main_results=table_main_results,
        table_model_comparison=table_model_comparison,
        section_importance_scores=section_importance_scores,
    )
    intro_support = build_paper_intro_support(
        primary_model_variant=freeze_manifest["primary_model_variant"],
        primary_probe=freeze_manifest["primary_probe"],
        secondary_probe=freeze_manifest["secondary_probe"],
    )
    method_support = build_paper_method_support(
        freeze_manifest=freeze_manifest,
        primary_model_variant=freeze_manifest["primary_model_variant"],
        primary_probe=freeze_manifest["primary_probe"],
        secondary_probe=freeze_manifest["secondary_probe"],
        section_importance_summary={
            "composite_formula": section_importance_scores["composite_scoring_formula"]["description"],
        },
    )
    results_support = build_paper_results_support(
        table_main_results=table_main_results,
        table_model_comparison=table_model_comparison,
        section_importance_scores=section_importance_scores,
        section_importance_cross_model_check=section_importance_cross_model_check,
    )
    limitations_support = build_paper_limitations_ethics_support()
    conclusion_support = build_paper_conclusion_support(
        section_importance_next_step_summary=section_importance_next_step_summary,
    )

    abstract_path = output_dir / "paper_abstract_support.md"
    intro_path = output_dir / "paper_intro_support.md"
    method_path = output_dir / "paper_method_support.md"
    results_path = output_dir / "paper_results_support.md"
    limitations_path = output_dir / "paper_limitations_ethics_support.md"
    conclusion_path = output_dir / "paper_conclusion_support.md"
    reproducibility_path = output_dir / "paper_reproducibility_commands.md"
    write_text(abstract_path, render_paper_abstract_support(abstract_support))
    write_text(intro_path, render_paper_intro_support(intro_support))
    write_text(method_path, render_paper_method_support(method_support))
    write_text(results_path, render_paper_results_support(results_support))
    write_text(limitations_path, render_paper_limitations_ethics_support(limitations_support))
    write_text(conclusion_path, render_paper_conclusion_support(conclusion_support))
    write_text(reproducibility_path, reproducibility_text)

    claim_traceability = build_claim_to_evidence_traceability(
        results_package_dir=results_package_dir,
        section_importance_dir=section_importance_dir,
        submission_package_dir=output_dir,
        table_main_results=table_main_results,
        table_model_comparison=table_model_comparison,
        stability_table=table_stability_vs_correctness,
        section_importance_scores=section_importance_scores,
        section_importance_cross_model_check=section_importance_cross_model_check,
        figure_manifest=figure_manifest,
    )
    claim_traceability_json_path = output_dir / "claim_to_evidence_traceability.json"
    claim_traceability_md_path = output_dir / "claim_to_evidence_traceability.md"
    write_json(claim_traceability_json_path, claim_traceability)
    write_text(
        claim_traceability_md_path,
        render_claim_to_evidence_traceability(claim_traceability),
    )

    main_layout = build_main_text_layout_guide()
    appendix_layout = build_appendix_layout_guide()
    main_layout_path = output_dir / "main_text_layout_guide.md"
    appendix_layout_path = output_dir / "appendix_layout_guide.md"
    if config.robustness.export_placement_guides:
        write_text(main_layout_path, render_main_text_layout_guide(main_layout))
        write_text(appendix_layout_path, render_appendix_layout_guide(appendix_layout))

    figure_captions_path = output_dir / "figure_captions.md"
    table_captions_path = output_dir / "table_captions.md"
    manifest_files = [figure_manifest_json_path, figure_manifest_md_path]
    if config.robustness.export_caption_files:
        write_text(
            figure_captions_path,
            render_figure_captions(build_figure_captions(figure_manifest)),
        )
        write_text(
            table_captions_path,
            render_table_captions(build_table_captions()),
        )
        manifest_files.extend([figure_captions_path, table_captions_path])

    submission_manifest_json_path = output_dir / "submission_package_manifest.json"
    submission_manifest_md_path = output_dir / "submission_package_manifest.md"
    final_consistency_json_path = output_dir / "final_package_consistency_check.json"
    final_consistency_md_path = output_dir / "final_package_consistency_check.md"

    writing_files = [
        abstract_path,
        intro_path,
        method_path,
        results_path,
        limitations_path,
        conclusion_path,
    ]
    layout_files = []
    if config.robustness.export_placement_guides:
        layout_files = [main_layout_path, appendix_layout_path]
        manifest_files.extend(layout_files)

    submission_manifest = build_submission_package_manifest(
        freeze_manifest=freeze_manifest,
        results_package_dir=results_package_dir,
        paper_drafting_package_dir=paper_drafting_package_dir,
        section_importance_dir=section_importance_dir,
        submission_package_dir=output_dir,
        figure_manifest=figure_manifest,
        writing_files=writing_files,
        qualitative_files=[
            results_package_dir / "qualitative_examples_primary.md",
            results_package_dir / "qualitative_examples_secondary.md",
            results_package_dir / "paper_qualitative_examples.md",
        ],
        appendix_files=[
            results_package_dir / "appendix_keep_reasoning_only_bundle.md",
            results_package_dir / "appendix_drop_precedents_bundle.md",
            section_importance_dir / "section_importance_cross_model_check.md",
        ],
        manifest_files=[
            submission_manifest_json_path,
            submission_manifest_md_path,
            *manifest_files,
        ],
        layout_files=layout_files,
        reproducibility_file=reproducibility_path,
        traceability_file=claim_traceability_md_path,
        consistency_file=final_consistency_md_path,
    )
    write_json(submission_manifest_json_path, submission_manifest)
    write_text(
        submission_manifest_md_path,
        render_submission_package_manifest(submission_manifest),
    )

    final_consistency_check = build_final_package_consistency_check(
        prior_consistency_check=prior_consistency_check,
        submission_package_manifest=submission_manifest,
        figure_manifest=figure_manifest,
        claim_traceability=claim_traceability,
        writing_file_paths=writing_files,
        traceability_path=claim_traceability_md_path,
        allowed_root_paths=[
            results_package_dir,
            paper_drafting_package_dir,
            section_importance_dir,
            output_dir,
            baseline_run_dir,
            robustness_run_dir,
            Path(freeze_manifest["canonical_section_transfer_run_dir"]),
        ],
    )
    write_json(final_consistency_json_path, final_consistency_check)
    write_text(
        final_consistency_md_path,
        render_final_package_consistency_check(final_consistency_check),
    )

    handoff_summary = build_paper_handoff_summary(
        submission_package_manifest=submission_manifest,
        final_consistency_check=final_consistency_check,
        section_importance_next_step_summary=section_importance_next_step_summary,
    )
    handoff_summary_json_path = output_dir / "paper_handoff_summary.json"
    handoff_summary_md_path = output_dir / "paper_handoff_summary.md"
    write_json(handoff_summary_json_path, handoff_summary)
    write_text(handoff_summary_md_path, render_paper_handoff_summary(handoff_summary))

    logger.info(
        "Submission package created from baseline=%s robustness=%s results_package=%s section_importance=%s",
        baseline_run_dir,
        robustness_run_dir,
        results_package_dir,
        section_importance_dir,
    )
    logger.info(
        "Final consistency status=%s handoff technical_work_done=%s",
        final_consistency_check["overall_status"],
        handoff_summary["technical_work_done"],
    )
    if final_consistency_check["overall_status"] == "fail":
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
