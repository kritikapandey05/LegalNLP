from __future__ import annotations

from typing import Any


def render_perturbation_manifest(report: dict[str, Any]) -> str:
    lines = [
        "# Perturbation Manifest",
        "",
        f"- Evaluation splits: `{report['evaluation_splits']}`",
        f"- Recipe count: `{report['recipe_count']}`",
        f"- Total examples: `{report['total_examples']}`",
        "- Note: all perturbations operate on transferred/predicted CJPE sections, not gold section labels.",
        "",
        "## Recipes",
        "",
    ]
    for recipe_name, recipe_report in report["recipes"].items():
        lines.extend(
            [
                f"### {recipe_name}",
                "",
                f"- Family: `{recipe_report['family']}`",
                f"- Cases: `{recipe_report['case_count']}`",
                f"- Empty-target cases: `{recipe_report['empty_target_case_count']}`",
                f"- Average perturbed text length chars: `{recipe_report['average_perturbed_text_length_chars']}`",
                f"- Target section: `{recipe_report['target_section']}`",
                f"- Sections to keep: `{recipe_report['sections_to_keep']}`",
                f"- Section order: `{recipe_report['section_order']}`",
                "",
            ]
        )
    return "\n".join(lines).strip() + "\n"
