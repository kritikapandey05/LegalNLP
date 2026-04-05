"""Configuration loading and schema definitions."""

from legal_robustness.config.loader import load_app_config
from legal_robustness.config.schema import AppConfig

__all__ = ["AppConfig", "load_app_config"]
