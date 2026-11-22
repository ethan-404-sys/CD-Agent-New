"""CD-Agent unified Python package.

Exposes centralized path helpers (`cd_agent.paths`) and wires the minimal `_compat`
helper directory so legacy helper imports resolve cleanly without external repo paths.
"""
import sys
from pathlib import Path

_compat = Path(__file__).resolve().parent / '_compat'
if _compat.is_dir() and str(_compat) not in sys.path:
    sys.path.append(str(_compat))

from .paths import (
    DATASETS_DIR,
    MODELS_DIR,
    OUTPUTS_DIR,
    REPO_ROOT,
    RUNTIME_DIR,
    THIRD_PARTY_DIR,
    dataset_path,
    model_path,
    output_dir,
    smoke_dir,
)

__all__ = [
    'REPO_ROOT',
    'MODELS_DIR',
    'DATASETS_DIR',
    'OUTPUTS_DIR',
    'RUNTIME_DIR',
    'THIRD_PARTY_DIR',
    'model_path',
    'dataset_path',
    'output_dir',
    'smoke_dir',
]
