"""SAM3 construction shared by agent workers and baseline adapters."""
from __future__ import annotations

import os
import sys
from contextlib import nullcontext
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[3]
MODELS = Path(os.environ.get("CD_AGENT_MODELS", REPO / "models"))
THIRD_PARTY = Path(os.environ.get("CD_AGENT_THIRD_PARTY", REPO / "third_party"))
CHANG_AGENT = os.environ.get("CHANG_AGENT_ROOT", "/guisongxia01/pangchao/wangyihan/Chang-Agent")
CHECKPOINT = str(MODELS / "sam3/sam3.pt")
_local_bpe = THIRD_PARTY / "OmniOVCD/sam3/assets/bpe_simple_vocab_16e6.txt.gz"
BPE = str(_local_bpe if _local_bpe.is_file() else Path(CHANG_AGENT) / "third_party/OmniOVCD/sam3/assets/bpe_simple_vocab_16e6.txt.gz")
RESOLUTION = 1008


def build_processor(inst_interactivity: bool = False):
    omni = THIRD_PARTY / "OmniOVCD"
    if omni.is_dir() and str(omni) not in sys.path:
        sys.path.insert(0, str(omni))
    from sam3 import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor

    model = build_sam3_image_model(
        bpe_path=BPE,
        checkpoint_path=CHECKPOINT,
        device="cuda",
        load_from_HF=False,
        enable_inst_interactivity=inst_interactivity,
    )
    model.to("cuda")
    model._device = torch.device("cuda")
    return Sam3Processor(model, resolution=RESOLUTION, device="cuda")


def autocast():
    if not torch.cuda.is_available():
        return nullcontext()
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    return torch.autocast(device_type="cuda", dtype=dtype)


def text_adapter(processor):
    if CHANG_AGENT not in sys.path:
        sys.path.insert(0, CHANG_AGENT)
    from change_agent.adapters.sam3_adapter import SAM3ProcessorAdapter

    return SAM3ProcessorAdapter(processor)
