from __future__ import annotations

import argparse


def build_common_parser(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--config",
        default="configs/base.yaml",
        help="Path to the YAML configuration file.",
    )
    parser.add_argument(
        "--dataset-root",
        default=None,
        help="Optional dataset root override. This takes precedence over config and environment values.",
    )
    parser.add_argument(
        "--run-name",
        default=None,
        help="Optional run-name suffix used for artifact directories.",
    )
    parser.add_argument(
        "--log-level",
        default=None,
        help="Optional log-level override, for example INFO or DEBUG.",
    )
    return parser
