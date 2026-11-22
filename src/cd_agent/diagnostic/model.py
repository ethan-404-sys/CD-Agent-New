"""Directed bitemporal hypothesis verification and the additive ablation."""
from __future__ import annotations

import math
from collections.abc import Sequence

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class _QueryAttention(nn.Module):
    def __init__(self, channels: int, heads: int):
        super().__init__()
        self.norm = nn.LayerNorm(channels)
        self.attention = nn.MultiheadAttention(channels, heads, batch_first=True)

    def forward(self, query: Tensor, memory: Tensor, padding: Tensor | None = None) -> Tensor:
        return query + self.attention(
            self.norm(query), memory, memory, key_padding_mask=padding, need_weights=False,
        )[0]


class HVVerifier(nn.Module):
    def __init__(self, backbone: nn.Module, *, mode: str = 'hv', heads: int = 8):
        super().__init__()
        if mode not in ('hv', 'additive'):
            raise ValueError(f'Unknown verifier mode: {mode}')
        self.mode = mode
        self.backbone = backbone
        c = backbone.channels
        self.dbst = nn.ModuleList(nn.Conv2d(4 * c, c, 1) for _ in range(3))
        self.quality_head = nn.Sequential(nn.LayerNorm(c), nn.Linear(c, c), nn.GELU(), nn.Linear(c, 1))
        self.presence_head = nn.Sequential(nn.LayerNorm(c), nn.Linear(c, c), nn.GELU(), nn.Linear(c, 2))
        if mode == 'hv':
            self.prototype_slots = nn.Parameter(torch.randn(4, c) * 0.02)
            self.prototype_text = nn.Linear(c, c)
            self.prototype_key = nn.Linear(c, c)
            self.fp_queries = nn.Parameter(torch.randn(8, c) * 0.02)
            self.fn_queries = nn.Parameter(torch.randn(8, c) * 0.02)
            self.conv_query = nn.Parameter(torch.randn(1, c) * 0.02)
            self.fp_reader = _QueryAttention(c, heads)
            self.fn_reader = _QueryAttention(c, heads)
            self.conv_reader = _QueryAttention(c, heads)
            self.fp_pixel = nn.Linear(c, c)
            self.fn_pixel = nn.Linear(c, c)
        else:
            self.error_head = nn.Conv2d(c, 2, 1)

    def set_vision_lora_trainable(self, enabled: bool) -> None:
        self.backbone.set_vision_lora_trainable(enabled)

    def _prototypes(
        self, visual: Tensor, condition: Tensor, text: Tensor, padding: Tensor, inside: Tensor,
    ) -> tuple[Tensor, Tensor]:
        text_weight = (~padding).to(text.dtype).unsqueeze(-1)
        pooled = (text * text_weight).sum(1) / text_weight.sum(1).clamp_min(1)
        slots = self.prototype_text(pooled)[:, None] + self.prototype_slots
        keys = self.prototype_key(visual + condition)
        similarity = 8 * (F.normalize(slots.float(), dim=-1) @ F.normalize(keys.float(), dim=-1).transpose(1, 2))
        # Fractional area is a prior. A finite all-empty softmax followed by
        # exact zeroing keeps the prototypes finite and lets attention mask them.
        present = inside.gt(0)
        scores = similarity + inside.clamp_min(1e-8).log()[:, None]
        weights = scores.masked_fill(~present[:, None], torch.finfo(scores.dtype).min).softmax(-1)
        weights = weights * present[:, None]
        prototypes = weights.to(visual.dtype) @ visual
        missing = (~present.any(1, keepdim=True)).expand(-1, self.prototype_slots.shape[0])
        return prototypes, missing

    def forward(
        self, image_t1: Tensor, image_t2: Tensor, init_mask: Tensor, instruction: Sequence[str],
    ) -> dict[str, Tensor]:
        if image_t1.shape != image_t2.shape or image_t1.ndim != 4 or image_t1.shape[1] != 3:
            raise ValueError('Expected matching Bx3xHxW images')
        batch = image_t1.shape[0]
        if init_mask.ndim != 4 or init_mask.shape[:2] != (batch, 1) or len(instruction) != batch:
            raise ValueError('Expected Bx1xHxW masks and one instruction per pair')
        fpn = self.backbone.encode_pair(image_t1, image_t2)
        directed = [
            project(torch.cat((first, second, second - first, first * second), dim=1))
            for project, (first, second) in zip(self.dbst, fpn, strict=True)
        ]
        text, padding = self.backbone.encode_text(instruction, directed[-1].device)
        top = self.backbone.fuse_text(directed[-1], text, padding)
        mask_condition = self.backbone.embed_mask(init_mask.float(), top)
        if self.mode == 'additive':
            directed[-1] = top + mask_condition
            logits = self.error_head(self.backbone.decode_pixels(directed))
            global_token = directed[-1].mean(dim=(-2, -1))
        else:
            directed[-1] = top
            visual = top.flatten(2).transpose(1, 2)
            inside = F.interpolate(init_mask.float(), size=top.shape[-2:], mode='area').flatten(1)
            prototypes, missing = self._prototypes(
                visual, mask_condition.flatten(2).transpose(1, 2), text, padding, inside,
            )
            memory = torch.cat((visual, text, prototypes), dim=1)
            fp = self.fp_reader(self.fp_queries.expand(batch, -1, -1), memory,
                                torch.cat((inside <= 0, padding, missing), dim=1))
            fn = self.fn_reader(self.fn_queries.expand(batch, -1, -1), memory,
                                torch.cat((inside >= 1, padding, missing), dim=1))
            global_memory = torch.cat((fp, fn, visual.mean(1, keepdim=True)), dim=1)
            global_token = self.conv_reader(self.conv_query.expand(batch, -1, -1), global_memory)[:, 0]
            pixels = self.backbone.decode_pixels(directed)
            query = torch.stack((self.fp_pixel(fp.mean(1)), self.fn_pixel(fn.mean(1))), dim=1)
            # Linear interpolation commutes with the spatially constant dot product.
            logits = torch.einsum('bkc,bchw->bkhw', query, pixels) / math.sqrt(pixels.shape[1])
        logits = F.interpolate(logits.float(), size=init_mask.shape[-2:], mode='bilinear', align_corners=False)
        presence = self.presence_head(global_token).float()
        return {
            'logits_fp': logits[:, :1], 'logits_fn': logits[:, 1:],
            'logits_has_fp': presence[:, 0], 'logits_has_fn': presence[:, 1],
            'q_score': self.quality_head(global_token).float().squeeze(-1).sigmoid(),
        }


def gated_probabilities(outputs: dict[str, Tensor], init_mask: Tensor) -> dict[str, Tensor]:
    inside = init_mask.bool()
    return {
        f'prob_{kind}': outputs[f'logits_{kind}'].sigmoid() * support
        * outputs[f'logits_has_{kind}'].sigmoid()[:, None, None, None]
        for kind, support in (('fp', inside), ('fn', ~inside))
    }
