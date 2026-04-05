from __future__ import annotations

from pathlib import Path
from typing import Any


def build_paper_freeze_manifest(
    *,
    section_transfer_run_dir: Path,
    baseline_run_dir: Path,
    robustness_run_dir: Path,
    results_package_dir: Path,
    drafting_package_dir: Path,
    primary_model_variant: str,
    primary_probe: str,
    secondary_probe: str,
    supporting_model_variants: tuple[str, ...] | list[str],
    manuscript_artifacts: dict[str, list[Path]],
) -> dict[str, Any]:
    return {
        "task": "paper_freeze_manifest",
        "canonical_section_transfer_run_dir": str(section_transfer_run_dir),
        "canonical_baseline_run_dir": str(baseline_run_dir),
        "canonical_robustness_run_dir": str(robustness_run_dir),
        "canonical_results_package_dir": str(results_package_dir),
        "paper_drafting_package_dir": str(drafting_package_dir),
        "primary_model_variant": primary_model_variant,
        "primary_probe": primary_probe,
        "secondary_probe": secondary_probe,
        "supporting_model_variants": list(supporting_model_variants),
        "artifact_map_by_manuscript_section": {
            section_name: [str(path) for path in paths]
            for section_name, paths in manuscript_artifacts.items()
        },
    }


def render_paper_freeze_manifest(report: dict[str, Any]) -> str:
    lines = [
        "# Paper Freeze Manifest",
        "",
        f"- Canonical section-transfer run: `{report['canonical_section_transfer_run_dir']}`",
        f"- Canonical baseline run: `{report['canonical_baseline_run_dir']}`",
        f"- Canonical robustness run: `{report['canonical_robustness_run_dir']}`",
        f"- Canonical results package dir: `{report['canonical_results_package_dir']}`",
        f"- Paper drafting package dir: `{report['paper_drafting_package_dir']}`",
        f"- Primary model variant: `{report['primary_model_variant']}`",
        f"- Primary probe: `{report['primary_probe']}`",
        f"- Secondary probe: `{report['secondary_probe']}`",
        f"- Supporting model variants: `{report['supporting_model_variants']}`",
        "",
    ]
    for section_name, paths in report.get("artifact_map_by_manuscript_section", {}).items():
        lines.extend([f"## {section_name.replace('_', ' ').title()}", ""])
        for path in paths:
            lines.append(f"- `{path}`")
        lines.append("")
    return "\n".join(lines).strip() + "\n"
