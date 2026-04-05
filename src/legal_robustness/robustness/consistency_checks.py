from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Any


def build_paper_consistency_check(
    *,
    freeze_manifest: dict[str, Any],
    results_package_manifest: dict[str, Any],
    unperturbed_comparison: dict[str, Any],
    comparative_metrics: dict[str, Any],
    table_main_results: dict[str, Any],
    table_model_comparison: dict[str, Any],
    qualitative_examples: list[dict[str, Any]],
    failure_analysis_cases: list[dict[str, Any]],
    appendix_bundle_paths: dict[str, Path],
    tolerance: float = 1e-6,
) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []
    checks.append(
        _check_table_main_results(
            table_main_results=table_main_results,
            unperturbed_comparison=unperturbed_comparison,
            comparative_metrics=comparative_metrics,
            primary_model_variant=freeze_manifest["primary_model_variant"],
            tolerance=tolerance,
        )
    )
    checks.append(
        _check_table_model_comparison(
            table_model_comparison=table_model_comparison,
            unperturbed_comparison=unperturbed_comparison,
            comparative_metrics=comparative_metrics,
            tolerance=tolerance,
        )
    )
    checks.append(
        _check_qualitative_examples(
            qualitative_examples=qualitative_examples,
            failure_analysis_cases=failure_analysis_cases,
        )
    )
    checks.append(
        _check_appendix_bundles(
            primary_probe=freeze_manifest["primary_probe"],
            secondary_probe=freeze_manifest["secondary_probe"],
            appendix_bundle_paths=appendix_bundle_paths,
        )
    )
    checks.append(
        _check_manifest_paths(
            freeze_manifest=freeze_manifest,
        )
    )
    checks.append(
        _check_run_alignment(
            freeze_manifest=freeze_manifest,
            results_package_manifest=results_package_manifest,
        )
    )
    checks.append(
        _check_results_package_manifest_files(
            freeze_manifest=freeze_manifest,
            results_package_manifest=results_package_manifest,
        )
    )
    checks.append(
        _check_artifact_scope(
            freeze_manifest=freeze_manifest,
        )
    )
    fail_count = sum(check["status"] == "fail" for check in checks)
    warning_count = sum(check["status"] == "warning" for check in checks)
    overall_status = "fail" if fail_count else ("warning" if warning_count else "pass")
    return {
        "task": "paper_consistency_check",
        "overall_status": overall_status,
        "fail_count": fail_count,
        "warning_count": warning_count,
        "checks": checks,
    }


def render_paper_consistency_check(report: dict[str, Any]) -> str:
    lines = [
        "# Paper Consistency Check",
        "",
        f"- Overall status: `{report['overall_status']}`",
        f"- Fail count: `{report['fail_count']}`",
        f"- Warning count: `{report['warning_count']}`",
        "",
    ]
    for check in report.get("checks", []):
        lines.append(f"## {check['name']}")
        lines.append("")
        lines.append(f"- Status: `{check['status']}`")
        lines.append(f"- Message: {check['message']}")
        if check.get("details"):
            lines.append("- Details:")
            for detail in check["details"]:
                lines.append(f"  {detail}")
        lines.append("")
    return "\n".join(lines).strip() + "\n"


def _check_table_main_results(
    *,
    table_main_results: dict[str, Any],
    unperturbed_comparison: dict[str, Any],
    comparative_metrics: dict[str, Any],
    primary_model_variant: str,
    tolerance: float,
) -> dict[str, Any]:
    details: list[str] = []
    unperturbed_lookup = {
        row["model_variant"]: row
        for row in unperturbed_comparison.get("model_variants", [])
    }
    recipe_lookup = {
        row["recipe_name"]: row
        for row in comparative_metrics.get("recipes", [])
    }
    for row in table_main_results.get("rows", []):
        if row["condition_key"] == "unperturbed":
            source = unperturbed_lookup.get(primary_model_variant)
            if source is None:
                return _fail_check(
                    "table_main_results",
                    "Primary unperturbed APA row is missing from unperturbed comparison.",
                )
            if not _close(row["macro_f1"], source["macro_f1"], tolerance):
                details.append("Unperturbed macro F1 mismatch.")
            if not _close(row["accuracy"], source["accuracy"], tolerance):
                details.append("Unperturbed accuracy mismatch.")
            continue
        recipe = recipe_lookup.get(row["condition_key"])
        model_metrics = _find_model_metrics(recipe, primary_model_variant)
        if recipe is None or model_metrics is None:
            return _fail_check(
                "table_main_results",
                f"Missing comparative metrics for `{row['condition_key']}`.",
            )
        if not _close(row["macro_f1"], model_metrics["macro_f1"], tolerance):
            details.append(f"{row['condition_key']} macro F1 mismatch.")
        if not _close(row["delta_macro_f1"], model_metrics["macro_f1_delta_vs_reference"], tolerance):
            details.append(f"{row['condition_key']} delta macro F1 mismatch.")
        if not _close(row["accuracy"], model_metrics["accuracy"], tolerance):
            details.append(f"{row['condition_key']} accuracy mismatch.")
        if not _close(row["delta_accuracy"], model_metrics["accuracy_delta_vs_reference"], tolerance):
            details.append(f"{row['condition_key']} delta accuracy mismatch.")
        if not _close(row["flip_rate"], model_metrics["flip_rate"], tolerance):
            details.append(f"{row['condition_key']} flip-rate mismatch.")
    if details:
        return _fail_check(
            "table_main_results",
            "Manuscript main-results table diverges from the canonical metrics.",
            details,
        )
    return _pass_check(
        "table_main_results",
        "Main-results table matches the canonical unperturbed and perturbed APA metrics.",
    )


def _check_table_model_comparison(
    *,
    table_model_comparison: dict[str, Any],
    unperturbed_comparison: dict[str, Any],
    comparative_metrics: dict[str, Any],
    tolerance: float,
) -> dict[str, Any]:
    details: list[str] = []
    unperturbed_lookup = {
        row["model_variant"]: row
        for row in unperturbed_comparison.get("model_variants", [])
    }
    recipe_lookup = {
        row["recipe_name"]: row
        for row in comparative_metrics.get("recipes", [])
    }
    for row in table_model_comparison.get("rows", []):
        model_variant = row["model_variant"]
        unperturbed = unperturbed_lookup.get(model_variant)
        if unperturbed is None:
            details.append(f"Missing unperturbed row for `{model_variant}`.")
            continue
        if not _close(row["unperturbed_accuracy"], unperturbed["accuracy"], tolerance):
            details.append(f"Unperturbed accuracy mismatch for `{model_variant}`.")
        if not _close(row["unperturbed_macro_f1"], unperturbed["macro_f1"], tolerance):
            details.append(f"Unperturbed macro F1 mismatch for `{model_variant}`.")
        for recipe_name in table_model_comparison.get("focused_recipes", []):
            model_metrics = _find_model_metrics(recipe_lookup.get(recipe_name), model_variant)
            if model_metrics is None:
                details.append(f"Missing `{recipe_name}` metrics for `{model_variant}`.")
                continue
            if not _close(row.get(f"{recipe_name}_macro_f1"), model_metrics["macro_f1"], tolerance):
                details.append(f"{recipe_name} macro F1 mismatch for `{model_variant}`.")
            if not _close(
                row.get(f"{recipe_name}_delta_macro_f1"),
                model_metrics["macro_f1_delta_vs_reference"],
                tolerance,
            ):
                details.append(f"{recipe_name} delta mismatch for `{model_variant}`.")
            if not _close(row.get(f"{recipe_name}_flip_rate"), model_metrics["flip_rate"], tolerance):
                details.append(f"{recipe_name} flip-rate mismatch for `{model_variant}`.")
    if details:
        return _fail_check(
            "table_model_comparison",
            "Model-comparison table is not fully aligned with the canonical metrics.",
            details,
        )
    return _pass_check(
        "table_model_comparison",
        "Model-comparison table matches the canonical unperturbed and perturbation metrics.",
    )


def _check_qualitative_examples(
    *,
    qualitative_examples: list[dict[str, Any]],
    failure_analysis_cases: list[dict[str, Any]],
) -> dict[str, Any]:
    valid_keys = {
        (row["perturbation_recipe"], row["case_id"])
        for row in failure_analysis_cases
    }
    missing = [
        f"{row['perturbation_recipe']}::{row['case_id']}"
        for row in qualitative_examples
        if (row["perturbation_recipe"], row["case_id"]) not in valid_keys
    ]
    if missing:
        return _fail_check(
            "qualitative_examples",
            "Some qualitative examples do not resolve back to the underlying failure-analysis cases.",
            missing,
        )
    return _pass_check(
        "qualitative_examples",
        "All qualitative examples resolve back to canonical failure-analysis cases.",
    )


def _check_appendix_bundles(
    *,
    primary_probe: str,
    secondary_probe: str,
    appendix_bundle_paths: dict[str, Path],
) -> dict[str, Any]:
    expected = {primary_probe, secondary_probe}
    actual = set(appendix_bundle_paths.keys())
    details: list[str] = []
    if actual != expected:
        details.append(f"Expected appendix probes `{sorted(expected)}`, found `{sorted(actual)}`.")
    for recipe_name, path in appendix_bundle_paths.items():
        if not path.exists():
            details.append(f"Missing appendix bundle file `{path}`.")
    if details:
        return _fail_check(
            "appendix_bundles",
            "Appendix bundle set is incomplete or misaligned with the frozen primary probes.",
            details,
        )
    return _pass_check(
        "appendix_bundles",
        "Appendix bundles match the frozen perturbation probes and resolve to real files.",
    )


def _check_manifest_paths(
    *,
    freeze_manifest: dict[str, Any],
) -> dict[str, Any]:
    missing: list[str] = []
    for key in (
        "canonical_section_transfer_run_dir",
        "canonical_baseline_run_dir",
        "canonical_robustness_run_dir",
        "canonical_results_package_dir",
        "paper_drafting_package_dir",
    ):
        path = Path(freeze_manifest[key])
        if not path.exists():
            missing.append(f"{key}: {path}")
    for section_name, paths in freeze_manifest.get("artifact_map_by_manuscript_section", {}).items():
        for raw_path in paths:
            path = Path(raw_path)
            if not path.exists():
                missing.append(f"{section_name}: {path}")
    if missing:
        return _fail_check(
            "manifest_paths",
            "The paper-freeze manifest references missing paths.",
            missing,
        )
    return _pass_check(
        "manifest_paths",
        "All canonical run paths and cited artifact paths resolve successfully.",
    )


def _check_run_alignment(
    *,
    freeze_manifest: dict[str, Any],
    results_package_manifest: dict[str, Any],
) -> dict[str, Any]:
    details: list[str] = []
    if results_package_manifest.get("baseline_run_dir") != freeze_manifest.get("canonical_baseline_run_dir"):
        details.append("Results-package manifest baseline run does not match the paper freeze baseline run.")
    if results_package_manifest.get("robustness_run_dir") != freeze_manifest.get("canonical_robustness_run_dir"):
        details.append("Results-package manifest robustness run does not match the paper freeze robustness run.")
    if details:
        return _fail_check(
            "run_alignment",
            "The freeze manifest and the canonical results-package manifest are not aligned.",
            details,
        )
    return _pass_check(
        "run_alignment",
        "Canonical baseline and robustness run references are aligned across the freeze and results-package manifests.",
    )


def _check_artifact_scope(
    *,
    freeze_manifest: dict[str, Any],
) -> dict[str, Any]:
    allowed_prefixes = [
        Path(freeze_manifest["canonical_results_package_dir"]).resolve(),
        Path(freeze_manifest["paper_drafting_package_dir"]).resolve(),
    ]
    out_of_scope: list[str] = []
    for paths in freeze_manifest.get("artifact_map_by_manuscript_section", {}).values():
        for raw_path in paths:
            path = Path(raw_path).resolve()
            if not any(_is_relative_to(path, prefix) for prefix in allowed_prefixes):
                out_of_scope.append(str(path))
    if out_of_scope:
        return _warning_check(
            "artifact_scope",
            "Some cited artifacts fall outside the frozen results package or drafting package directories. Review them before citing.",
            out_of_scope,
        )
    return _pass_check(
        "artifact_scope",
        "All cited manuscript artifacts stay inside the frozen results package or drafting package directories.",
    )


def _check_results_package_manifest_files(
    *,
    freeze_manifest: dict[str, Any],
    results_package_manifest: dict[str, Any],
) -> dict[str, Any]:
    results_package_dir = Path(freeze_manifest["canonical_results_package_dir"])
    missing: list[str] = []
    for key in ("main_summary_files", "chart_data_files", "qualitative_files", "appendix_files"):
        for relative_path in results_package_manifest.get(key, []):
            path = results_package_dir / relative_path
            if not path.exists():
                missing.append(str(path))
    if missing:
        return _fail_check(
            "results_package_manifest_files",
            "The canonical results-package manifest references missing files.",
            missing,
        )
    return _pass_check(
        "results_package_manifest_files",
        "All files listed inside the canonical results-package manifest resolve correctly.",
    )


def _find_model_metrics(recipe: dict[str, Any] | None, model_variant: str) -> dict[str, Any] | None:
    if recipe is None:
        return None
    for row in recipe.get("model_metrics", []):
        if row["model_variant"] == model_variant:
            return row
    return None


def _is_relative_to(path: Path, other: Path) -> bool:
    try:
        path.relative_to(other)
        return True
    except ValueError:
        return False


def _close(left: float | None, right: float | None, tolerance: float) -> bool:
    if left is None or right is None:
        return left == right
    return abs(left - right) <= tolerance


def _pass_check(name: str, message: str) -> dict[str, Any]:
    return {"name": name, "status": "pass", "message": message, "details": []}


def _warning_check(name: str, message: str, details: list[str]) -> dict[str, Any]:
    return {"name": name, "status": "warning", "message": message, "details": details}


def _fail_check(name: str, message: str, details: list[str] | None = None) -> dict[str, Any]:
    return {"name": name, "status": "fail", "message": message, "details": details or []}


def build_final_package_consistency_check(
    *,
    prior_consistency_check: dict[str, Any],
    submission_package_manifest: dict[str, Any],
    figure_manifest: dict[str, Any],
    claim_traceability: dict[str, Any],
    writing_file_paths: list[Path],
    traceability_path: Path,
    allowed_root_paths: list[Path],
) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []
    checks.append(_check_prior_consistency_status(prior_consistency_check))
    checks.append(_check_figure_manifest_outputs(figure_manifest))
    checks.append(_check_figure_source_hashes(figure_manifest))
    checks.append(_check_traceability_paths(claim_traceability))
    checks.append(_check_submission_manifest_paths(submission_package_manifest))
    checks.append(
        _check_writing_file_references(
            writing_file_paths=writing_file_paths,
            traceability_path=traceability_path,
            submission_package_manifest=submission_package_manifest,
        )
    )
    checks.append(
        _check_submission_path_scope(
            submission_package_manifest=submission_package_manifest,
            claim_traceability=claim_traceability,
            allowed_root_paths=allowed_root_paths,
        )
    )
    fail_count = sum(check["status"] == "fail" for check in checks)
    warning_count = sum(check["status"] == "warning" for check in checks)
    overall_status = "fail" if fail_count else ("warning" if warning_count else "pass")
    return {
        "task": "final_package_consistency_check",
        "overall_status": overall_status,
        "fail_count": fail_count,
        "warning_count": warning_count,
        "checks": checks,
    }


def render_final_package_consistency_check(report: dict[str, Any]) -> str:
    lines = [
        "# Final Package Consistency Check",
        "",
        f"- Overall status: `{report['overall_status']}`",
        f"- Fail count: `{report['fail_count']}`",
        f"- Warning count: `{report['warning_count']}`",
        "",
    ]
    for check in report.get("checks", []):
        lines.extend([f"## {check['name']}", ""])
        lines.append(f"- Status: `{check['status']}`")
        lines.append(f"- Message: {check['message']}")
        if check.get("details"):
            lines.append("- Details:")
            for detail in check["details"]:
                lines.append(f"  {detail}")
        lines.append("")
    return "\n".join(lines).strip() + "\n"


def _check_prior_consistency_status(prior_consistency_check: dict[str, Any]) -> dict[str, Any]:
    status = prior_consistency_check.get("overall_status")
    if status != "pass":
        return _fail_check(
            "prior_paper_consistency",
            "The frozen paper-drafting package consistency check is not pass.",
            [f"Frozen status: {status}"],
        )
    return _pass_check(
        "prior_paper_consistency",
        "The frozen paper-drafting package already passed its consistency check.",
    )


def _check_figure_manifest_outputs(figure_manifest: dict[str, Any]) -> dict[str, Any]:
    missing: list[str] = []
    for figure in figure_manifest.get("figures", []):
        spec_path = Path(figure["spec_file"])
        if not spec_path.exists():
            missing.append(str(spec_path))
        for output_file in figure.get("output_files", []):
            path = Path(output_file)
            if not path.exists():
                missing.append(str(path))
    if missing:
        return _fail_check(
            "figure_outputs",
            "Some figure spec or output files are missing.",
            missing,
        )
    return _pass_check(
        "figure_outputs",
        "All figure specs and rendered outputs exist.",
    )


def _check_figure_source_hashes(figure_manifest: dict[str, Any]) -> dict[str, Any]:
    mismatches: list[str] = []
    for figure in figure_manifest.get("figures", []):
        for source in figure.get("source_files", []):
            path = Path(source["path"])
            if not path.exists():
                mismatches.append(f"Missing source file `{path}`.")
                continue
            current_hash = hashlib.sha256(path.read_bytes()).hexdigest()
            if current_hash != source["sha256"]:
                mismatches.append(f"Hash mismatch for `{path}`.")
    if mismatches:
        return _fail_check(
            "figure_source_hashes",
            "At least one figure source file changed after figure generation.",
            mismatches,
        )
    return _pass_check(
        "figure_source_hashes",
        "All figure source hashes still match the frozen chart data.",
    )


def _check_traceability_paths(claim_traceability: dict[str, Any]) -> dict[str, Any]:
    missing: list[str] = []
    for claim in claim_traceability.get("claims", []):
        for artifact in claim.get("supporting_artifacts", []):
            path = Path(artifact)
            if not path.exists():
                missing.append(f"{claim['claim_id']}: {path}")
    if missing:
        return _fail_check(
            "claim_traceability_paths",
            "Some claim-to-evidence paths do not resolve.",
            missing,
        )
    return _pass_check(
        "claim_traceability_paths",
        "All claim-to-evidence artifact paths resolve successfully.",
    )


def _check_submission_manifest_paths(submission_package_manifest: dict[str, Any]) -> dict[str, Any]:
    missing: list[str] = []
    path_keys = (
        "final_tables",
        "final_figures",
        "qualitative_files",
        "appendix_files",
        "writing_support_files",
        "manifest_files",
        "layout_files",
    )
    for key in path_keys:
        for raw_path in submission_package_manifest.get(key, []):
            path = Path(raw_path)
            if not path.exists():
                missing.append(str(path))
    for key in ("reproducibility_file", "traceability_file"):
        path = Path(submission_package_manifest[key])
        if not path.exists():
            missing.append(str(path))
    if missing:
        return _fail_check(
            "submission_manifest_paths",
            "The submission-package manifest references missing files.",
            missing,
        )
    return _pass_check(
        "submission_manifest_paths",
        "All files listed in the submission-package manifest exist.",
    )


def _check_writing_file_references(
    *,
    writing_file_paths: list[Path],
    traceability_path: Path,
    submission_package_manifest: dict[str, Any],
) -> dict[str, Any]:
    valid_names = {
        Path(raw_path).name
        for key in (
            "final_tables",
            "final_figures",
            "qualitative_files",
            "appendix_files",
            "writing_support_files",
            "manifest_files",
            "layout_files",
        )
        for raw_path in submission_package_manifest.get(key, [])
    }
    valid_names.update(
        {
            Path(submission_package_manifest["reproducibility_file"]).name,
            Path(submission_package_manifest["traceability_file"]).name,
            Path(submission_package_manifest["consistency_file"]).name,
            traceability_path.name,
        }
    )
    pattern = re.compile(r"`([^`]+\.(?:md|json|png|svg))`")
    unresolved: list[str] = []
    for writing_path in writing_file_paths:
        content = writing_path.read_text(encoding="utf-8")
        for match in pattern.findall(content):
            name = Path(match).name
            if name not in valid_names:
                unresolved.append(f"{writing_path.name}: {match}")
    if unresolved:
        return _warning_check(
            "writing_file_references",
            "Some writing files mention artifact names that are not present in the final manifest.",
            unresolved,
        )
    return _pass_check(
        "writing_file_references",
        "Writing-support files reference artifact names that are present in the final manifest.",
    )


def _check_submission_path_scope(
    *,
    submission_package_manifest: dict[str, Any],
    claim_traceability: dict[str, Any],
    allowed_root_paths: list[Path],
) -> dict[str, Any]:
    offenders: list[str] = []
    allowed = [path.resolve() for path in allowed_root_paths]

    def _check_path(raw_path: str) -> None:
        path = Path(raw_path).resolve()
        if not any(_is_relative_to(path, root) or path == root for root in allowed):
            offenders.append(str(path))

    for key in (
        "final_tables",
        "final_figures",
        "qualitative_files",
        "appendix_files",
        "writing_support_files",
        "manifest_files",
        "layout_files",
    ):
        for raw_path in submission_package_manifest.get(key, []):
            _check_path(raw_path)
    for key in ("reproducibility_file", "traceability_file", "consistency_file"):
        _check_path(submission_package_manifest[key])
    for claim in claim_traceability.get("claims", []):
        for raw_path in claim.get("supporting_artifacts", []):
            _check_path(raw_path)
    if offenders:
        return _warning_check(
            "submission_path_scope",
            "Some referenced files fall outside the canonical frozen roots or the new submission package.",
            offenders,
        )
    return _pass_check(
        "submission_path_scope",
        "All referenced files remain inside the expected frozen package roots or the submission package.",
    )
