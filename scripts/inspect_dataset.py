from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from legal_robustness.config.loader import load_app_config
from legal_robustness.data.inspection import DatasetInspector, render_markdown_report
from legal_robustness.utils.artifacts import create_stage_output_dir, write_json, write_text, write_yaml
from legal_robustness.utils.cli import build_common_parser
from legal_robustness.utils.logging import configure_logging, get_logger
from legal_robustness.utils.seeds import seed_everything


def parse_args() -> object:
    parser = build_common_parser("Inspect a local IL-TUR dataset directory and save a report.")
    return parser.parse_args()


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
        base_dir=config.output.dataset_inspection_dir,
        stage_name="dataset_inspection",
        run_name=config.runtime.run_name,
    )
    log_file = output_dir / "inspection.log" if config.logging.log_to_file else None
    configure_logging(level=config.logging.level, log_file=log_file)
    logger = get_logger("legal_robustness.scripts.inspect_dataset")
    seed_everything(config.runtime.seed, deterministic=config.runtime.deterministic)
    write_yaml(output_dir / "config_snapshot.yaml", config.to_dict())

    inspector = DatasetInspector(config=config, logger=logger)
    report = inspector.inspect()
    write_json(output_dir / "inspection_report.json", report.to_dict())
    write_text(output_dir / "inspection_report.md", render_markdown_report(report))

    logger.info("Dataset inspection artifacts written to %s", output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
