from __future__ import annotations

import hashlib
import subprocess
import uuid
from pathlib import Path
from typing import Any

from legal_robustness.utils.artifacts import write_json
from legal_robustness.utils.paths import ensure_directory

PALETTE = {
    "background": "#FFFFFF",
    "panel_background": "#F8FAFC",
    "grid": "#CBD5E1",
    "axis": "#475569",
    "text": "#0F172A",
    "muted_text": "#334155",
    "apa": "#0F766E",
    "nb": "#2563EB",
    "logistic": "#DC2626",
    "contextual": "#6B7280",
    "reasoning_probe": "#D97706",
    "precedent_probe": "#0891B2",
    "importance_high": "#1D4ED8",
    "importance_medium": "#D97706",
    "importance_low": "#DC2626",
}


def render_submission_figures(
    *,
    output_dir: Path,
    figure_formats: tuple[str, ...] | list[str],
    chart_data_main_performance: dict[str, Any],
    chart_data_robustness_deltas: dict[str, Any],
    chart_data_flip_rates: dict[str, Any],
    chart_data_section_importance_scores: dict[str, Any],
    chart_data_section_importance_coverage: dict[str, Any],
    source_file_map: dict[str, Path],
) -> dict[str, Any]:
    specs = [
        _build_main_robustness_spec(
            chart_data_main_performance=chart_data_main_performance,
            chart_data_robustness_deltas=chart_data_robustness_deltas,
            source_files=[
                source_file_map["chart_data_main_performance"],
                source_file_map["chart_data_robustness_deltas"],
            ],
        ),
        _build_model_comparison_spec(
            chart_data_main_performance=chart_data_main_performance,
            chart_data_robustness_deltas=chart_data_robustness_deltas,
            source_files=[
                source_file_map["chart_data_main_performance"],
                source_file_map["chart_data_robustness_deltas"],
            ],
        ),
        _build_flip_rates_spec(
            chart_data_flip_rates=chart_data_flip_rates,
            source_files=[source_file_map["chart_data_flip_rates"]],
        ),
        _build_section_importance_spec(
            chart_data_section_importance_scores=chart_data_section_importance_scores,
            source_files=[source_file_map["chart_data_section_importance_scores"]],
        ),
        _build_coverage_spec(
            chart_data_section_importance_coverage=chart_data_section_importance_coverage,
            source_files=[source_file_map["chart_data_section_importance_coverage"]],
        ),
    ]

    figure_specs_dir = ensure_directory(output_dir / "figure_specs")
    manifest_rows: list[dict[str, Any]] = []
    for spec in specs:
        spec_path = figure_specs_dir / f"{spec['figure_id']}.json"
        write_json(spec_path, spec)
        output_files: list[str] = []
        for fmt in figure_formats:
            normalized = fmt.lower()
            figure_path = output_dir / f"{spec['figure_id']}.{normalized}"
            if normalized == "svg":
                figure_path.write_text(_render_svg(spec), encoding="utf-8")
            elif normalized == "png":
                _render_png_with_powershell(spec_path=spec_path, output_path=figure_path)
            else:
                raise ValueError(f"Unsupported figure export format: {fmt}")
            output_files.append(str(figure_path))
        manifest_rows.append(
            {
                "figure_id": spec["figure_id"],
                "title": spec["title"],
                "description": spec["description"],
                "caption": spec["caption"],
                "source_files": [
                    {
                        "path": str(path),
                        "sha256": _sha256(Path(path)),
                    }
                    for path in spec["source_files"]
                ],
                "spec_file": str(spec_path),
                "output_files": output_files,
            }
        )
    temp_dir = output_dir / "_tmp_figure_render"
    if temp_dir.exists() and not any(temp_dir.iterdir()):
        temp_dir.rmdir()
    return {
        "task": "figure_manifest",
        "figure_formats": [fmt.lower() for fmt in figure_formats],
        "figures": manifest_rows,
    }


def render_figure_manifest(report: dict[str, Any]) -> str:
    lines = [
        "# Figure Manifest",
        "",
        f"- Figure formats: `{report['figure_formats']}`",
        "",
    ]
    for figure in report.get("figures", []):
        lines.extend([f"## {figure['figure_id']}", ""])
        lines.append(f"- Title: {figure['title']}")
        lines.append(f"- Description: {figure['description']}")
        lines.append(f"- Spec file: `{figure['spec_file']}`")
        lines.append(f"- Caption: {figure['caption']}")
        lines.append("- Source files:")
        for source in figure.get("source_files", []):
            lines.append(f"  - `{source['path']}` sha256 `{source['sha256']}`")
        lines.append("- Output files:")
        for output_file in figure.get("output_files", []):
            lines.append(f"  - `{output_file}`")
        lines.append("")
    return "\n".join(lines).strip() + "\n"


def build_figure_captions(report: dict[str, Any]) -> dict[str, Any]:
    return {
        "task": "figure_captions",
        "captions": [
            {
                "figure_id": figure["figure_id"],
                "caption": figure["caption"],
            }
            for figure in report.get("figures", [])
        ],
    }


def render_figure_captions(report: dict[str, Any]) -> str:
    lines = ["# Figure Captions", ""]
    for row in report.get("captions", []):
        lines.append(f"## {row['figure_id']}")
        lines.append("")
        lines.append(row["caption"])
        lines.append("")
    return "\n".join(lines).strip() + "\n"


def _build_main_robustness_spec(
    *,
    chart_data_main_performance: dict[str, Any],
    chart_data_robustness_deltas: dict[str, Any],
    source_files: list[Path],
) -> dict[str, Any]:
    apa_row = _require_row(
        chart_data_main_performance.get("rows", []),
        "model_variant",
        "averaged_passive_aggressive::pseudo_all_sections",
    )
    delta_rows = [
        row
        for row in chart_data_robustness_deltas.get("rows", [])
        if row["model_variant"] == "averaged_passive_aggressive::pseudo_all_sections"
    ]
    reasoning_row = _require_row(delta_rows, "recipe_name", "keep_reasoning_only")
    precedents_row = _require_row(delta_rows, "recipe_name", "drop_precedents")

    unperturbed_macro_f1 = float(apa_row["macro_f1"])
    bars = [
        {"label": "UNPERTURBED", "value": unperturbed_macro_f1, "color": PALETTE["apa"]},
        {
            "label": "REASONING\nONLY",
            "value": unperturbed_macro_f1 + float(reasoning_row["macro_f1_delta"]),
            "color": PALETTE["reasoning_probe"],
        },
        {
            "label": "DROP\nPRECEDENTS",
            "value": unperturbed_macro_f1 + float(precedents_row["macro_f1_delta"]),
            "color": PALETTE["precedent_probe"],
        },
    ]
    spec = _new_figure(
        figure_id="fig_main_robustness",
        title="MAIN ROBUSTNESS EFFECTS (APA)",
        description="APA macro F1 under unperturbed input, reasoning-only ablation, and precedent removal.",
        caption=(
            "Figure 1. APA-centered robustness results on pseudo-sectioned CJPE. "
            "Keeping only reasoning causes the largest performance drop, while dropping precedents yields a smaller but still meaningful degradation."
        ),
        source_files=source_files,
    )
    _draw_vertical_bar_chart(
        spec,
        panel_title="APA MACRO F1 ON PSEUDO_ALL_SECTIONS",
        x=110,
        y=150,
        width=980,
        height=420,
        bars=bars,
        y_min=0.0,
        y_max=0.65,
        tick_values=[0.0, 0.2, 0.4, 0.6],
    )
    _add_text(spec, 600, 70, spec["title"], size=26, anchor="middle", weight="bold")
    _add_text(
        spec,
        600,
        620,
        "APA IS THE CENTRAL MODEL. CJPE SECTIONS ARE TRANSFERRED/PREDICTED PSEUDO-SECTIONS, NOT GOLD.",
        size=12,
        anchor="middle",
        fill=PALETTE["muted_text"],
    )
    return spec


def _build_model_comparison_spec(
    *,
    chart_data_main_performance: dict[str, Any],
    chart_data_robustness_deltas: dict[str, Any],
    source_files: list[Path],
) -> dict[str, Any]:
    model_rows = chart_data_main_performance.get("rows", [])
    delta_rows = chart_data_robustness_deltas.get("rows", [])
    spec = _new_figure(
        figure_id="fig_model_comparison",
        title="MODEL COMPARISON ON FOCUSED PROBES",
        description="Unperturbed APA, NB, logistic, and contextual approximation performance plus focused perturbation deltas.",
        caption=(
            "Figure 2. Four-model comparison on pseudo_all_sections. APA is the strongest unperturbed model, "
            "while focused perturbation deltas show that absolute strength, retention, and stability are distinct properties."
        ),
        source_files=source_files,
        width=1320,
        height=760,
    )
    _add_text(spec, 660, 60, spec["title"], size=26, anchor="middle", weight="bold")

    unperturbed_bars = [
        {
            "label": _short_model_label(row["model_label"]),
            "value": float(row["macro_f1"]),
            "color": _model_color(row["model_label"]),
        }
        for row in model_rows
    ]
    _draw_vertical_bar_chart(
        spec,
        panel_title="UNPERTURBED MACRO F1",
        x=80,
        y=150,
        width=510,
        height=420,
        bars=unperturbed_bars,
        y_min=0.0,
        y_max=0.65,
        tick_values=[0.0, 0.2, 0.4, 0.6],
    )

    grouped_rows: list[dict[str, Any]] = []
    for recipe_name in ("keep_reasoning_only", "drop_precedents"):
        for row in delta_rows:
            if row["recipe_name"] != recipe_name:
                continue
            grouped_rows.append(
                {
                    "group": _recipe_label(recipe_name),
                    "label": _short_model_label(row["model_label"]),
                    "value": float(row["macro_f1_delta"]),
                    "color": _model_color(row["model_label"]),
                }
            )
    _draw_grouped_signed_bar_chart(
        spec,
        panel_title="MACRO F1 DELTA VS UNPERTURBED",
        x=680,
        y=150,
        width=560,
        height=420,
        grouped_rows=grouped_rows,
        y_min=-0.07,
        y_max=0.02,
        tick_values=[-0.06, -0.03, 0.0, 0.02],
    )
    _draw_model_legend(spec, x=990, y=95)
    return spec


def _build_flip_rates_spec(
    *,
    chart_data_flip_rates: dict[str, Any],
    source_files: list[Path],
) -> dict[str, Any]:
    rows = chart_data_flip_rates.get("flip_rate_rows", [])
    spec = _new_figure(
        figure_id="fig_flip_rates",
        title="FLIP-RATE COMPARISON",
        description="Prediction flip rates for the focused perturbations across all four models.",
        caption=(
            "Figure 3. Flip-rate comparison for the focused perturbations. "
            "Lower flip rates do not necessarily coincide with the strongest absolute performance."
        ),
        source_files=source_files,
        width=1320,
        height=760,
    )
    _add_text(spec, 660, 60, spec["title"], size=26, anchor="middle", weight="bold")
    grouped_rows = [
        {
            "group": _recipe_label(row["recipe_name"]),
            "label": _short_model_label(row["model_label"]),
            "value": float(row["flip_rate"]),
            "color": _model_color(row["model_label"]),
        }
        for row in rows
    ]
    _draw_grouped_positive_bar_chart(
        spec,
        panel_title="PREDICTION FLIP RATE",
        x=90,
        y=150,
        width=1140,
        height=420,
        grouped_rows=grouped_rows,
        y_min=0.0,
        y_max=0.38,
        tick_values=[0.0, 0.1, 0.2, 0.3],
    )
    _draw_model_legend(spec, x=980, y=95)
    _add_text(
        spec,
        660,
        620,
        "LOGISTIC IS THE MOST STABLE MODEL BY FLIP RATE IN THE FROZEN FOCUSED PACKAGE.",
        size=13,
        anchor="middle",
        fill=PALETTE["muted_text"],
    )
    return spec


def _build_section_importance_spec(
    *,
    chart_data_section_importance_scores: dict[str, Any],
    source_files: list[Path],
) -> dict[str, Any]:
    rows = sorted(chart_data_section_importance_scores.get("rows", []), key=lambda row: row["rank"])
    spec = _new_figure(
        figure_id="fig_section_importance",
        title="SECTION IMPORTANCE RANKING (APA)",
        description="Composite section-importance ranking under the APA baseline.",
        caption=(
            "Figure 4. Coverage-aware section-importance ranking under APA. "
            "Precedents rank first, followed by facts and reasoning, while conclusion remains low confidence because coverage is sparse."
        ),
        source_files=source_files,
        width=1280,
        height=760,
    )
    _add_text(spec, 640, 60, spec["title"], size=26, anchor="middle", weight="bold")
    _draw_horizontal_bar_chart(
        spec,
        panel_title="COMPOSITE IMPORTANCE SCORE",
        x=180,
        y=140,
        width=960,
        height=470,
        rows=[
            {
                "label": row["section"].upper(),
                "value": float(row["composite_importance_score"]),
                "color": _confidence_color(row["confidence_label"]),
                "note": row["confidence_label"].replace("_importance_estimate", "").replace("_", " ").upper(),
            }
            for row in rows
        ],
        x_min=0.0,
        x_max=1.0,
        tick_values=[0.0, 0.25, 0.5, 0.75, 1.0],
    )
    _add_text(
        spec,
        640,
        650,
        "RANKING IS APA-CENTERED AND SHOULD BE DESCRIBED AS PILOT EVIDENCE OVER PREDICTED PSEUDO-SECTIONS.",
        size=13,
        anchor="middle",
        fill=PALETTE["muted_text"],
    )
    return spec


def _build_coverage_spec(
    *,
    chart_data_section_importance_coverage: dict[str, Any],
    source_files: list[Path],
) -> dict[str, Any]:
    rows = sorted(
        chart_data_section_importance_coverage.get("rows", []),
        key=lambda row: row["combined_effective_coverage"],
        reverse=True,
    )
    spec = _new_figure(
        figure_id="fig_coverage",
        title="SECTION COVERAGE FOR IMPORTANCE ESTIMATES",
        description="Combined effective coverage for each section-importance estimate under APA.",
        caption=(
            "Figure 5. Coverage behind the section-importance estimates. "
            "Conclusion coverage is extremely sparse, so conclusion-importance claims remain low confidence."
        ),
        source_files=source_files,
        width=1280,
        height=760,
    )
    _add_text(spec, 640, 60, spec["title"], size=26, anchor="middle", weight="bold")
    _draw_horizontal_bar_chart(
        spec,
        panel_title="COMBINED EFFECTIVE COVERAGE",
        x=180,
        y=140,
        width=960,
        height=470,
        rows=[
            {
                "label": row["section"].upper(),
                "value": float(row["combined_effective_coverage"]),
                "color": _confidence_color(row["confidence_label"]),
                "note": row["confidence_label"].replace("_importance_estimate", "").replace("_", " ").upper(),
            }
            for row in rows
        ],
        x_min=0.0,
        x_max=1.0,
        tick_values=[0.0, 0.25, 0.5, 0.75, 1.0],
    )
    _add_text(
        spec,
        640,
        650,
        "CONCLUSION COVERAGE = 0.026368, SO CONCLUSION IMPORTANCE SHOULD STAY CLEARLY CAVEATED IN THE MANUSCRIPT.",
        size=13,
        anchor="middle",
        fill=PALETTE["muted_text"],
    )
    return spec


def _new_figure(
    *,
    figure_id: str,
    title: str,
    description: str,
    caption: str,
    source_files: list[Path],
    width: int = 1200,
    height: int = 720,
) -> dict[str, Any]:
    return {
        "figure_id": figure_id,
        "title": title,
        "description": description,
        "caption": caption,
        "width": width,
        "height": height,
        "background": PALETTE["background"],
        "source_files": [str(path.resolve()) for path in source_files],
        "elements": [],
    }


def _draw_vertical_bar_chart(
    spec: dict[str, Any],
    *,
    panel_title: str,
    x: float,
    y: float,
    width: float,
    height: float,
    bars: list[dict[str, Any]],
    y_min: float,
    y_max: float,
    tick_values: list[float],
) -> None:
    _panel_background(spec, x=x, y=y, width=width, height=height)
    plot_x = x + 65
    plot_y = y + 40
    plot_width = width - 95
    plot_height = height - 120
    _add_text(spec, x + width / 2, y + 16, panel_title, size=16, anchor="middle", weight="bold")
    _draw_y_grid(spec, plot_x, plot_y, plot_width, plot_height, tick_values, y_min=y_min, y_max=y_max)
    _add_line(spec, plot_x, plot_y + plot_height, plot_x + plot_width, plot_y + plot_height, width=2)

    bar_gap = 28
    bar_width = (plot_width - bar_gap * (len(bars) + 1)) / max(len(bars), 1)
    for index, bar in enumerate(bars):
        bar_x = plot_x + bar_gap + index * (bar_width + bar_gap)
        bar_height = _scale_value(float(bar["value"]), y_min, y_max, plot_height)
        bar_top = plot_y + plot_height - bar_height
        _add_rect(spec, bar_x, bar_top, bar_width, bar_height, fill=bar["color"])
        _add_text(spec, bar_x + bar_width / 2, bar_top - 18, f"{float(bar['value']):.3f}", size=12, anchor="middle")
        _add_text(spec, bar_x + bar_width / 2, plot_y + plot_height + 12, bar["label"], size=12, anchor="middle")


def _draw_grouped_signed_bar_chart(
    spec: dict[str, Any],
    *,
    panel_title: str,
    x: float,
    y: float,
    width: float,
    height: float,
    grouped_rows: list[dict[str, Any]],
    y_min: float,
    y_max: float,
    tick_values: list[float],
) -> None:
    _panel_background(spec, x=x, y=y, width=width, height=height)
    plot_x = x + 70
    plot_y = y + 40
    plot_width = width - 100
    plot_height = height - 120
    _add_text(spec, x + width / 2, y + 16, panel_title, size=16, anchor="middle", weight="bold")
    _draw_y_grid(spec, plot_x, plot_y, plot_width, plot_height, tick_values, y_min=y_min, y_max=y_max)
    zero_y = plot_y + plot_height - _scale_value(0.0, y_min, y_max, plot_height)
    _add_line(spec, plot_x, zero_y, plot_x + plot_width, zero_y, width=2, stroke=PALETTE["axis"])

    groups = _group_rows(grouped_rows)
    outer_gap = 26
    inner_gap = 12
    bars_per_group = max(len(rows) for rows in groups.values())
    group_width = (plot_width - outer_gap * (len(groups) + 1)) / max(len(groups), 1)
    bar_width = (group_width - inner_gap * (bars_per_group + 1)) / max(bars_per_group, 1)
    for group_index, (group_name, rows) in enumerate(groups.items()):
        group_x = plot_x + outer_gap + group_index * (group_width + outer_gap)
        _add_text(spec, group_x + group_width / 2, plot_y + plot_height + 48, group_name, size=13, anchor="middle", weight="bold")
        for row_index, row in enumerate(rows):
            bar_x = group_x + inner_gap + row_index * (bar_width + inner_gap)
            value = float(row["value"])
            magnitude = _scale_value(abs(value), 0.0, max(abs(y_min), abs(y_max)), plot_height / 2)
            bar_y = zero_y - magnitude if value >= 0 else zero_y
            _add_rect(spec, bar_x, bar_y, bar_width, magnitude, fill=row["color"])
            _add_text(
                spec,
                bar_x + bar_width / 2,
                bar_y - 18 if value >= 0 else bar_y + magnitude + 4,
                f"{value:+.3f}",
                size=11,
                anchor="middle",
            )
            _add_text(spec, bar_x + bar_width / 2, plot_y + plot_height + 14, row["label"], size=11, anchor="middle")


def _draw_grouped_positive_bar_chart(
    spec: dict[str, Any],
    *,
    panel_title: str,
    x: float,
    y: float,
    width: float,
    height: float,
    grouped_rows: list[dict[str, Any]],
    y_min: float,
    y_max: float,
    tick_values: list[float],
) -> None:
    _panel_background(spec, x=x, y=y, width=width, height=height)
    plot_x = x + 70
    plot_y = y + 40
    plot_width = width - 100
    plot_height = height - 120
    _add_text(spec, x + width / 2, y + 16, panel_title, size=16, anchor="middle", weight="bold")
    _draw_y_grid(spec, plot_x, plot_y, plot_width, plot_height, tick_values, y_min=y_min, y_max=y_max)
    _add_line(spec, plot_x, plot_y + plot_height, plot_x + plot_width, plot_y + plot_height, width=2)

    groups = _group_rows(grouped_rows)
    outer_gap = 26
    inner_gap = 12
    bars_per_group = max(len(rows) for rows in groups.values())
    group_width = (plot_width - outer_gap * (len(groups) + 1)) / max(len(groups), 1)
    bar_width = (group_width - inner_gap * (bars_per_group + 1)) / max(bars_per_group, 1)
    for group_index, (group_name, rows) in enumerate(groups.items()):
        group_x = plot_x + outer_gap + group_index * (group_width + outer_gap)
        _add_text(spec, group_x + group_width / 2, plot_y + plot_height + 48, group_name, size=13, anchor="middle", weight="bold")
        for row_index, row in enumerate(rows):
            bar_x = group_x + inner_gap + row_index * (bar_width + inner_gap)
            bar_height = _scale_value(float(row["value"]), y_min, y_max, plot_height)
            bar_y = plot_y + plot_height - bar_height
            _add_rect(spec, bar_x, bar_y, bar_width, bar_height, fill=row["color"])
            _add_text(spec, bar_x + bar_width / 2, bar_y - 18, f"{float(row['value']):.3f}", size=11, anchor="middle")
            _add_text(spec, bar_x + bar_width / 2, plot_y + plot_height + 14, row["label"], size=11, anchor="middle")


def _draw_horizontal_bar_chart(
    spec: dict[str, Any],
    *,
    panel_title: str,
    x: float,
    y: float,
    width: float,
    height: float,
    rows: list[dict[str, Any]],
    x_min: float,
    x_max: float,
    tick_values: list[float],
) -> None:
    _panel_background(spec, x=x, y=y, width=width, height=height)
    plot_x = x + 180
    plot_y = y + 50
    plot_width = width - 230
    plot_height = height - 100
    _add_text(spec, x + width / 2, y + 16, panel_title, size=16, anchor="middle", weight="bold")
    _draw_x_grid(spec, plot_x, plot_y, plot_width, plot_height, tick_values, x_min=x_min, x_max=x_max)

    row_gap = 18
    bar_height = (plot_height - row_gap * (len(rows) + 1)) / max(len(rows), 1)
    for index, row in enumerate(rows):
        bar_y = plot_y + row_gap + index * (bar_height + row_gap)
        bar_width = _scale_value(float(row["value"]), x_min, x_max, plot_width)
        _add_text(spec, x + 150, bar_y + bar_height / 2 - 8, row["label"], size=13, anchor="end", weight="bold")
        _add_rect(spec, plot_x, bar_y, bar_width, bar_height, fill=row["color"])
        _add_text(spec, plot_x + bar_width + 12, bar_y + bar_height / 2 - 8, f"{float(row['value']):.3f}", size=12, anchor="start")
        if row.get("note"):
            _add_text(spec, plot_x + plot_width + 20, bar_y + bar_height / 2 - 8, row["note"], size=11, anchor="end", fill=PALETTE["muted_text"])


def _panel_background(spec: dict[str, Any], *, x: float, y: float, width: float, height: float) -> None:
    _add_rect(spec, x, y, width, height, fill=PALETTE["panel_background"], stroke=PALETTE["grid"], stroke_width=1)


def _draw_y_grid(
    spec: dict[str, Any],
    x: float,
    y: float,
    width: float,
    height: float,
    tick_values: list[float],
    *,
    y_min: float,
    y_max: float,
) -> None:
    for tick in tick_values:
        tick_y = y + height - _scale_value(tick, y_min, y_max, height)
        _add_line(spec, x, tick_y, x + width, tick_y, stroke=PALETTE["grid"], width=1)
        _add_text(spec, x - 10, tick_y - 8, f"{tick:.2f}", size=11, anchor="end", fill=PALETTE["muted_text"])


def _draw_x_grid(
    spec: dict[str, Any],
    x: float,
    y: float,
    width: float,
    height: float,
    tick_values: list[float],
    *,
    x_min: float,
    x_max: float,
) -> None:
    for tick in tick_values:
        tick_x = x + _scale_value(tick, x_min, x_max, width)
        _add_line(spec, tick_x, y, tick_x, y + height, stroke=PALETTE["grid"], width=1)
        _add_text(spec, tick_x, y + height + 12, f"{tick:.2f}", size=11, anchor="middle", fill=PALETTE["muted_text"])


def _draw_model_legend(spec: dict[str, Any], *, x: float, y: float) -> None:
    legend_rows = [
        ("APA", PALETTE["apa"]),
        ("NB", PALETTE["nb"]),
        ("LOGISTIC", PALETTE["logistic"]),
        ("CONTEXTUAL", PALETTE["contextual"]),
    ]
    for index, (label, color) in enumerate(legend_rows):
        row_y = y + index * 24
        _add_rect(spec, x, row_y, 16, 16, fill=color)
        _add_text(spec, x + 26, row_y + 1, label, size=12, anchor="start")


def _add_rect(
    spec: dict[str, Any],
    x: float,
    y: float,
    width: float,
    height: float,
    *,
    fill: str,
    stroke: str | None = None,
    stroke_width: float = 1.0,
) -> None:
    spec["elements"].append(
        {
            "type": "rect",
            "x": x,
            "y": y,
            "width": width,
            "height": height,
            "fill": fill,
            "stroke": stroke,
            "stroke_width": stroke_width,
        }
    )


def _add_line(
    spec: dict[str, Any],
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    *,
    stroke: str = PALETTE["axis"],
    width: float = 1.0,
) -> None:
    spec["elements"].append(
        {
            "type": "line",
            "x1": x1,
            "y1": y1,
            "x2": x2,
            "y2": y2,
            "stroke": stroke,
            "stroke_width": width,
        }
    )


def _add_text(
    spec: dict[str, Any],
    x: float,
    y: float,
    text: str,
    *,
    size: int = 14,
    anchor: str = "start",
    fill: str = PALETTE["text"],
    weight: str = "normal",
) -> None:
    spec["elements"].append(
        {
            "type": "text",
            "x": x,
            "y": y,
            "text": text,
            "font_size": size,
            "anchor": anchor,
            "fill": fill,
            "weight": weight,
        }
    )


def _render_svg(spec: dict[str, Any]) -> str:
    lines = [
        f"<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"{spec['width']}\" height=\"{spec['height']}\" viewBox=\"0 0 {spec['width']} {spec['height']}\">",
        f"<rect x=\"0\" y=\"0\" width=\"{spec['width']}\" height=\"{spec['height']}\" fill=\"{spec['background']}\" />",
    ]
    for element in spec.get("elements", []):
        if element["type"] == "rect":
            stroke = element.get("stroke")
            stroke_attr = ""
            if stroke:
                stroke_attr = f" stroke=\"{stroke}\" stroke-width=\"{element.get('stroke_width', 1)}\""
            lines.append(
                f"<rect x=\"{_fmt(element['x'])}\" y=\"{_fmt(element['y'])}\" width=\"{_fmt(element['width'])}\" "
                f"height=\"{_fmt(element['height'])}\" fill=\"{element['fill']}\"{stroke_attr} />"
            )
        elif element["type"] == "line":
            lines.append(
                f"<line x1=\"{_fmt(element['x1'])}\" y1=\"{_fmt(element['y1'])}\" x2=\"{_fmt(element['x2'])}\" y2=\"{_fmt(element['y2'])}\" "
                f"stroke=\"{element['stroke']}\" stroke-width=\"{_fmt(element.get('stroke_width', 1))}\" />"
            )
        elif element["type"] == "text":
            anchor = {"start": "start", "middle": "middle", "end": "end"}[element["anchor"]]
            lines.extend(
                _svg_text_lines(
                    x=float(element["x"]),
                    y=float(element["y"]),
                    text=str(element["text"]),
                    size=int(element["font_size"]),
                    anchor=anchor,
                    fill=str(element["fill"]),
                    weight=str(element["weight"]),
                )
            )
    lines.append("</svg>")
    return "\n".join(lines)


def _svg_text_lines(
    *,
    x: float,
    y: float,
    text: str,
    size: int,
    anchor: str,
    fill: str,
    weight: str,
) -> list[str]:
    rendered: list[str] = []
    for index, line in enumerate(text.split("\n")):
        line_y = y + index * size * 1.3
        rendered.append(
            f"<text x=\"{_fmt(x)}\" y=\"{_fmt(line_y)}\" fill=\"{fill}\" font-family=\"Arial, sans-serif\" "
            f"font-size=\"{size}\" font-weight=\"{weight}\" text-anchor=\"{anchor}\">{_escape_xml(line)}</text>"
        )
    return rendered


def _render_png_with_powershell(*, spec_path: Path, output_path: Path) -> None:
    script = r"""
param(
    [string]$SpecPath,
    [string]$OutputPath
)
$ErrorActionPreference = 'Stop'
Add-Type -AssemblyName System.Drawing
$spec = Get-Content -Raw -Path $SpecPath | ConvertFrom-Json
$bitmap = New-Object System.Drawing.Bitmap([int]$spec.width, [int]$spec.height)
$graphics = [System.Drawing.Graphics]::FromImage($bitmap)
$graphics.SmoothingMode = [System.Drawing.Drawing2D.SmoothingMode]::HighQuality
$graphics.TextRenderingHint = [System.Drawing.Text.TextRenderingHint]::AntiAliasGridFit
$graphics.Clear([System.Drawing.ColorTranslator]::FromHtml($spec.background))
function Get-Color([string]$Hex) {
    return [System.Drawing.ColorTranslator]::FromHtml($Hex)
}
foreach ($element in $spec.elements) {
    switch ($element.type) {
        "rect" {
            $brush = New-Object System.Drawing.SolidBrush (Get-Color $element.fill)
            $graphics.FillRectangle($brush, [float]$element.x, [float]$element.y, [float]$element.width, [float]$element.height)
            $brush.Dispose()
            if ($element.stroke) {
                $pen = New-Object System.Drawing.Pen((Get-Color $element.stroke), [float]$element.stroke_width)
                $graphics.DrawRectangle($pen, [float]$element.x, [float]$element.y, [float]$element.width, [float]$element.height)
                $pen.Dispose()
            }
        }
        "line" {
            $pen = New-Object System.Drawing.Pen((Get-Color $element.stroke), [float]$element.stroke_width)
            $graphics.DrawLine($pen, [float]$element.x1, [float]$element.y1, [float]$element.x2, [float]$element.y2)
            $pen.Dispose()
        }
        "text" {
            $fontStyle = [System.Drawing.FontStyle]::Regular
            if ($element.weight -eq "bold") {
                $fontStyle = [System.Drawing.FontStyle]::Bold
            }
            $font = New-Object System.Drawing.Font("Arial", [float]$element.font_size, $fontStyle, [System.Drawing.GraphicsUnit]::Pixel)
            $brush = New-Object System.Drawing.SolidBrush (Get-Color $element.fill)
            $lines = ($element.text -split "`n")
            $lineHeight = [math]::Max([float]$element.font_size * 1.3, 14.0)
            for ($i = 0; $i -lt $lines.Count; $i++) {
                $line = $lines[$i]
                $size = $graphics.MeasureString($line, $font)
                $drawX = [float]$element.x
                if ($element.anchor -eq "middle") {
                    $drawX = [float]$element.x - ($size.Width / 2.0)
                } elseif ($element.anchor -eq "end") {
                    $drawX = [float]$element.x - $size.Width
                }
                $drawY = [float]$element.y + ($i * $lineHeight)
                $graphics.DrawString($line, $font, $brush, [float]$drawX, [float]$drawY)
            }
            $brush.Dispose()
            $font.Dispose()
        }
    }
}
$bitmap.Save($OutputPath, [System.Drawing.Imaging.ImageFormat]::Png)
$graphics.Dispose()
$bitmap.Dispose()
"""
    temp_dir = ensure_directory(output_path.parent / "_tmp_figure_render")
    script_path = temp_dir / f"{output_path.stem}_{uuid.uuid4().hex[:8]}.ps1"
    script_path.write_text(script, encoding="utf-8")
    try:
        completed = subprocess.run(
            [
                "powershell",
                "-NoProfile",
                "-ExecutionPolicy",
                "Bypass",
                "-File",
                str(script_path),
                str(spec_path),
                str(output_path),
            ],
            check=False,
            capture_output=True,
            text=True,
        )
    finally:
        try:
            script_path.unlink()
        except FileNotFoundError:
            pass
    if completed.returncode != 0 or not output_path.exists():
        raise RuntimeError(
            "Failed to render PNG figure via PowerShell.\n"
            f"stdout:\n{completed.stdout}\n"
            f"stderr:\n{completed.stderr}\n"
            f"output_exists:\n{output_path.exists()}"
        )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def _scale_value(value: float, min_value: float, max_value: float, extent: float) -> float:
    if max_value <= min_value:
        return 0.0
    normalized = (value - min_value) / (max_value - min_value)
    return max(0.0, min(extent, normalized * extent))


def _group_rows(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row["group"]), []).append(row)
    return grouped


def _model_color(model_label: str) -> str:
    normalized = model_label.lower()
    if "apa" in normalized:
        return PALETTE["apa"]
    if normalized == "nb":
        return PALETTE["nb"]
    if "logistic" in normalized:
        return PALETTE["logistic"]
    return PALETTE["contextual"]


def _short_model_label(model_label: str) -> str:
    normalized = model_label.lower()
    if normalized == "contextual approx.":
        return "CONTEXTUAL"
    return model_label.upper()


def _recipe_label(recipe_name: str) -> str:
    mapping = {
        "keep_reasoning_only": "REASONING ONLY",
        "drop_precedents": "DROP PRECEDENTS",
    }
    return mapping.get(recipe_name, recipe_name.replace("_", " ").upper())


def _confidence_color(confidence_label: str) -> str:
    if confidence_label.startswith("high"):
        return PALETTE["importance_high"]
    if confidence_label.startswith("medium"):
        return PALETTE["importance_medium"]
    return PALETTE["importance_low"]


def _require_row(rows: list[dict[str, Any]], key: str, expected_value: str) -> dict[str, Any]:
    for row in rows:
        if row.get(key) == expected_value:
            return row
    raise KeyError(f"Missing required row where {key} == {expected_value!r}")


def _escape_xml(value: str) -> str:
    return (
        value.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace("\"", "&quot;")
    )


def _fmt(value: float) -> str:
    if value == int(value):
        return str(int(value))
    return f"{value:.3f}"
