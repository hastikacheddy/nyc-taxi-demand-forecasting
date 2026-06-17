"""Shared configuration + provenance helpers (single source of truth)."""
import os
import subprocess

import yaml

_CONFIG_PATH = os.path.join(os.path.dirname(__file__), '..', '..', 'config.yaml')


def load_config(path: str = _CONFIG_PATH) -> dict:
    """Load config.yaml, expanding ${ENV_VAR} placeholders from the environment."""
    with open(path, 'r') as f:
        return yaml.safe_load(os.path.expandvars(f.read()))


def get_git_revision_hash() -> str:
    """Current git commit, logged with every training run for provenance."""
    try:
        return subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode('ascii').strip()
    except Exception:
        return "unknown"


def get_hparams(config: dict, model_key: str) -> dict:
    """LightGBM hyperparameters for 'forecast' or 'nowcast', from config.yaml."""
    return dict(config['training'][model_key])
