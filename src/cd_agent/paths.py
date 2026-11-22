"""Centralized path resolution for CD-Agent.

All models and datasets are accessed via repo-local symlinks under `models/` and `datasets/`
pointing to `/guisongxia01/pangchao/wangyihan/{models,datasets}`. Environment variables or
an optional gitignored `configs/paths.local.yaml` can override any location without editing code.
"""
from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
ACCOUNT_ROOT = Path(os.environ.get('CD_AGENT_ACCOUNT_ROOT', '/guisongxia01/pangchao/wangyihan'))
MODELS_DIR = Path(os.environ.get('CD_AGENT_MODELS', REPO_ROOT / 'models'))
DATASETS_DIR = Path(os.environ.get('CD_AGENT_DATASETS', REPO_ROOT / 'datasets'))
OUTPUTS_DIR = Path(os.environ.get('CD_AGENT_OUTPUTS', REPO_ROOT / 'outputs'))
RUNTIME_DIR = Path(os.environ.get('CD_AGENT_RUNTIME', REPO_ROOT / '.runtime'))
THIRD_PARTY_DIR = Path(os.environ.get('CD_AGENT_THIRD_PARTY', REPO_ROOT / 'third_party'))
CONDA_ENVS_DIR = Path(os.environ.get('CD_AGENT_CONDA_ENVS', ACCOUNT_ROOT / 'conda_envs'))


def model_path(*parts: str) -> Path:
    return MODELS_DIR.joinpath(*parts)


def dataset_path(*parts: str) -> Path:
    return DATASETS_DIR.joinpath(*parts)


def output_dir(track: str, run_name: str, *, create: bool = False) -> Path:
    path = OUTPUTS_DIR / track / run_name
    if create:
        (path / 'logs').mkdir(parents=True, exist_ok=True)
    return path


def smoke_dir(run_name: str, *, create: bool = False) -> Path:
    path = RUNTIME_DIR / 'smoke' / run_name
    if create:
        (path / 'logs').mkdir(parents=True, exist_ok=True)
    return path
