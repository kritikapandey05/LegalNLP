from __future__ import annotations

from pathlib import Path

from legal_robustness.config.loader import load_app_config
from legal_robustness.utils.artifacts import create_stage_output_dir, write_yaml
from legal_robustness.utils.logging import configure_logging, get_logger
from legal_robustness.utils.seeds import seed_everything


def run_placeholder_stage(stage_name: str, args: object, project_root: Path) -> int:
    config = load_app_config(
        config_path=Path(args.config),
        dataset_root_override=Path(args.dataset_root) if args.dataset_root else None,
        run_name_override=args.run_name,
        log_level_override=args.log_level,
        project_root=project_root,
    )
    output_dir = create_stage_output_dir(
        base_dir=config.output.reports_dir / stage_name,
        stage_name=stage_name,
        run_name=config.runtime.run_name,
    )
    configure_logging(
        level=config.logging.level,
        log_file=output_dir / f"{stage_name}.log" if config.logging.log_to_file else None,
    )
    logger = get_logger(f"legal_robustness.scripts.{stage_name}")
    seed_everything(config.runtime.seed, deterministic=config.runtime.deterministic)
    write_yaml(output_dir / "config_snapshot.yaml", config.to_dict())
    logger.info("%s is intentionally left as a placeholder in Phase 1.", stage_name)
    logger.info("This command currently validates configuration wiring only.")
    logger.info("Next implementation stages will replace this placeholder with functional code.")
    logger.info("Artifacts written to %s", output_dir)
    return 0
