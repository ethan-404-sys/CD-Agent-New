"""Offline, no-TTA SCanNet inference in canonical SECOND class order."""
from __future__ import annotations

import os
from pathlib import Path
import sys

import numpy as np
from PIL import Image

from cd_agent import paths

NATIVE_TO_SECOND = (0, 4, 2, 1, 3, 5, 6)
DEFAULT_CHECKPOINT = 'SCanNet_32e_mIoU73.37_Sek23.94_Fscd63.66_OA87.86.pth'
# RS_ST.py uses RGB on the original 0–255 scale, with separate date statistics.
MEANS = ((113.40, 114.08, 116.45), (111.07, 114.04, 118.18))
STDS = ((48.30, 46.27, 48.14), (49.41, 47.01, 47.94))
INPUT_SIZE = 512


def canonical_semantics(native: np.ndarray) -> np.ndarray:
    native = np.asarray(native)
    if native.dtype.kind not in 'iu' or ((native < 0) | (native >= len(NATIVE_TO_SECOND))).any():
        raise ValueError('SCanNet semantic IDs must be integers in 0..6')
    return np.asarray(NATIVE_TO_SECOND, dtype=np.uint8)[native]


def checkpoint_path(override: Path | None = None) -> Path:
    return Path(override or paths.model_path('SCanNet', DEFAULT_CHECKPOINT)).expanduser().resolve(strict=True)


class SCanNetAdapter:
    def __init__(self, checkpoint: Path | None = None):
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError('SCanNet inference requires CUDA')
        checkpoint = checkpoint_path(checkpoint)
        chang_root = Path(os.environ.get('CHANG_AGENT_ROOT', paths.ACCOUNT_ROOT / 'Chang-Agent'))
        vendor = (chang_root / 'third_party' / 'SCanNet').resolve(strict=True)
        # The upstream code uses absolute imports for its local models and utils.
        sys.path.insert(0, str(vendor))
        try:
            from models.SCanNet import SCanNet
        finally:
            sys.path.pop(0)
        model = SCanNet(in_channels=3, num_classes=7, input_size=INPUT_SIZE, pretrained=False)
        model.load_state_dict(torch.load(checkpoint, map_location='cpu', weights_only=True), strict=True)
        self.model = model.cuda().eval()

    def infer_pair(self, img_t1_path: Path | str, img_t2_path: Path | str) -> dict[str, np.ndarray]:
        import torch

        tensors = []
        for path, mean, std in zip((img_t1_path, img_t2_path), MEANS, STDS):
            with Image.open(path) as image:
                rgb = np.asarray(image.convert('RGB'))
            if rgb.shape != (INPUT_SIZE, INPUT_SIZE, 3):
                raise ValueError(f'{path}: expected native {INPUT_SIZE}x{INPUT_SIZE} RGB, got {rgb.shape}')
            normalized = ((rgb - np.asarray(mean)) / np.asarray(std)).astype(np.float32)
            tensors.append(torch.from_numpy(normalized.transpose(2, 0, 1).copy())[None].cuda())
        with torch.inference_mode():
            binary, semantic_t1, semantic_t2 = self.model(*tensors)
            return {
                'B': (binary.sigmoid()[0, 0] > 0.5).cpu().numpy(),
                'Y1': canonical_semantics(semantic_t1.argmax(dim=1)[0].cpu().numpy()),
                'Y2': canonical_semantics(semantic_t2.argmax(dim=1)[0].cpu().numpy()),
            }
