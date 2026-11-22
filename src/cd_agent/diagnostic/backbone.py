"""Strict SAM 3 submodule loader and a shape-compatible CPU test double."""
from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
import sys

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from cd_agent.paths import THIRD_PARTY_DIR, model_path


class LoRALinear(nn.Module):
    """Preserve the pretrained linear map and learn a zero-initialized residual."""

    def __init__(self, base: nn.Linear, rank: int, alpha: float):
        super().__init__()
        self.base = base.requires_grad_(False)
        self.lora_a = nn.Linear(base.in_features, rank, bias=False, device=base.weight.device)
        self.lora_b = nn.Linear(rank, base.out_features, bias=False, device=base.weight.device)
        nn.init.zeros_(self.lora_b.weight)
        self.scale = alpha / rank

    def forward(self, x: Tensor) -> Tensor:
        return self.base(x) + self.lora_b(self.lora_a(x)) * self.scale


def _load_retained(parts: dict[str, nn.Module], path: Path) -> None:
    state = torch.load(path, map_location='cpu', mmap=True, weights_only=True)
    state = state.get('model', state)
    consumed = set()
    for prefix, module in parts.items():
        retained = {}
        expected = module.state_dict()
        for name in expected:
            key = prefix + name
            if prefix.endswith('vision_backbone.') and name.endswith('.attn.freqs_cis'):
                # Resolution-specific complex buffers are built by ViT itself.
                retained[name] = expected[name]
                consumed.add(key)
                continue
            if key not in state or state[key].shape != expected[name].shape:
                found = None if key not in state else tuple(state[key].shape)
                raise ValueError(f'SAM 3 tensor {key}: expected {tuple(expected[name].shape)}, got {found}')
            retained[name] = state[key]
            consumed.add(key)
        module.load_state_dict(retained, strict=True, assign=True)
    unused_prefixes = (
        'detector.backbone.vision_backbone.convs.3.',  # original detector drops this level
        'detector.backbone.vision_backbone.sam2_convs.',
        'detector.transformer.decoder.', 'detector.geometry_encoder.', 'detector.dot_prod_scoring.',
        'detector.segmentation_head.cross_attend_prompt.', 'detector.segmentation_head.cross_attn_norm.',
        'detector.segmentation_head.instance_seg_head.', 'detector.segmentation_head.semantic_seg_head.',
        'detector.segmentation_head.mask_predictor.',
        'tracker.mask_downsample.', 'tracker.obj_ptr_proj.', 'tracker.obj_ptr_tpos_proj.',
        'tracker.sam_mask_decoder.', 'tracker.sam_prompt_encoder.', 'tracker.transformer.',
    )
    unused_exact = {
        'tracker.maskmem_tpos_enc', 'tracker.no_mem_embed', 'tracker.no_mem_pos_enc',
        'tracker.no_obj_ptr', 'tracker.no_obj_embed_spatial',
    }
    unexpected = sorted(k for k in state.keys() - consumed
                        if k not in unused_exact and not k.startswith(unused_prefixes))
    if unexpected:
        raise ValueError(f'Unexpected SAM 3 checkpoint keys: {unexpected[:12]}')


class SAM3Backbone(nn.Module):
    channels = 256

    def __init__(
        self, *, image_size: int = 672, lora_rank: int = 16, lora_alpha: float = 32,
        late_mlp_blocks: int = 0, fusion_top_k: int = 2, weights: Path | None = None,
    ):
        super().__init__()
        if image_size <= 0 or image_size % 14 or lora_rank < 1 or lora_alpha <= 0:
            raise ValueError('Use a positive image size divisible by 14, LoRA rank and alpha')
        if not 0 <= late_mlp_blocks <= 32 or not 0 <= fusion_top_k <= 6:
            raise ValueError('Invalid number of LoRA MLP blocks or unfrozen fusion layers')
        vendor = THIRD_PARTY_DIR / 'OmniOVCD'
        if not (vendor / 'sam3' / 'model_builder.py').is_file():
            raise FileNotFoundError(f'Vendored SAM 3 is unavailable: {vendor}')
        sys.path.insert(0, str(vendor))
        from sam3 import model_builder as sam

        if Path(sam.__file__).resolve().parent != (vendor / 'sam3').resolve():
            raise RuntimeError('SAM 3 was imported from outside the OmniOVCD vendor tree')
        self.image_size = image_size
        self.vision = sam.Sam3DualViTDetNeck(
            trunk=sam.ViT(
                img_size=image_size, pretrain_img_size=336, patch_size=14, embed_dim=1024,
                depth=32, num_heads=16, mlp_ratio=4.625, norm_layer='LayerNorm',
                drop_path_rate=0.1, qkv_bias=True, use_abs_pos=True, tile_abs_pos=True,
                global_att_blocks=(7, 15, 23, 31), rel_pos_blocks=(), use_rope=True,
                use_interp_rope=True, window_size=24, pretrain_use_cls_token=True,
                retain_cls_token=False, ln_pre=True, bias_patch_embed=False, use_act_checkpoint=True,
            ),
            position_encoding=sam.PositionEmbeddingSine(num_pos_feats=256, normalize=True),
            d_model=256, scale_factors=(4.0, 2.0, 1.0),
        )
        self.text = sam._create_text_encoder(str(vendor / 'sam3/assets/bpe_simple_vocab_16e6.txt.gz'))
        self.fusion = sam._create_transformer_encoder()
        self.pixel_decoder = sam.PixelDecoder(hidden_dim=256, num_upsampling_stages=3,
                                              interpolation_mode='nearest')
        self.mask_encoder = sam.SimpleMaskEncoder(
            out_dim=64, position_encoding=sam.PositionEmbeddingSine(num_pos_feats=64, normalize=True),
            mask_downsampler=sam.SimpleMaskDownSampler(
                kernel_size=3, stride=2, padding=1, interpol_size=[image_size // 14 * 16] * 2,
            ),
            fuser=sam.SimpleFuser(sam.CXBlock(dim=256, kernel_size=7, padding=3), num_layers=2),
        )
        _load_retained({
            'detector.backbone.vision_backbone.': self.vision,
            'detector.backbone.language_backbone.': self.text,
            'detector.transformer.encoder.': self.fusion,
            'detector.segmentation_head.pixel_decoder.': self.pixel_decoder,
            'tracker.maskmem_backbone.': self.mask_encoder,
        }, weights or model_path('sam3/sam3.pt'))

        self.vision.requires_grad_(False)
        self.text.requires_grad_(False).eval()
        self.mask_encoder.requires_grad_(False).eval()
        self.fusion.requires_grad_(False)
        if fusion_top_k:
            for layer in self.fusion.layers[-fusion_top_k:]:
                layer.requires_grad_(True)
        # The upstream decoder consumes two skips for a 3-level pyramid; the
        # checkpoint includes a dormant third block, which stays frozen.
        self.pixel_decoder.conv_layers[-1].requires_grad_(False)
        self.pixel_decoder.norms[-1].requires_grad_(False)
        self.mask_projection = nn.Conv2d(64, 256, 1)
        for index, block in enumerate(self.vision.trunk.blocks):
            for name in ('qkv', 'proj'):
                setattr(block.attn, name, LoRALinear(getattr(block.attn, name), lora_rank, lora_alpha))
            if index >= len(self.vision.trunk.blocks) - late_mlp_blocks:
                for name in ('fc1', 'fc2'):
                    setattr(block.mlp, name, LoRALinear(getattr(block.mlp, name), lora_rank, lora_alpha))
        self.set_vision_lora_trainable(True)
        self.train(self.training)

    def vision_adaptation_parameters(self):
        for module in self.vision.trunk.modules():
            if isinstance(module, LoRALinear):
                yield from module.lora_a.parameters()
                yield from module.lora_b.parameters()
            elif isinstance(module, nn.LayerNorm):
                yield from module.parameters(recurse=False)

    def set_vision_lora_trainable(self, enabled: bool) -> None:
        self._vision_trainable = enabled
        for parameter in self.vision_adaptation_parameters():
            parameter.requires_grad_(enabled)
        self.vision.train(self.training and enabled)

    def train(self, mode: bool = True):
        super().train(mode)
        self.vision.train(mode and self._vision_trainable)
        self.text.eval()
        self.mask_encoder.eval()
        return self

    def encode_pair(self, first: Tensor, second: Tensor) -> list[tuple[Tensor, Tensor]]:
        def encode(image: Tensor) -> list[Tensor]:
            image = F.interpolate(image, (self.image_size, self.image_size), mode='bilinear',
                                  align_corners=False, antialias=True)
            return self.vision(image * 2 - 1)[0]

        return list(zip(encode(first), encode(second), strict=True))

    @torch.no_grad()
    def encode_text(self, prompts: Sequence[str], device: torch.device) -> tuple[Tensor, Tensor]:
        lengths = [len(self.text.tokenizer.encode(prompt)) + 2 for prompt in prompts]
        for prompt, length in zip(prompts, lengths, strict=True):
            if not 2 < length <= self.text.context_length:
                raise ValueError(f'Expected a nonempty SAM 3 prompt of at most 32 tokens incl. BOS/EOS; got {length}: {prompt!r}')
        _, tokens, _ = self.text(list(prompts), device=device)
        padding = torch.arange(self.text.context_length, device=device)[None] >= torch.tensor(
            lengths, device=device,
        )[:, None]
        return tokens.transpose(0, 1), padding

    def fuse_text(self, top: Tensor, text: Tensor, padding: Tensor) -> Tensor:
        pos = self.vision.position_encoding(top).to(top.dtype)
        # Upstream TransformerEncoderFusion expects seq-first (H*W, B, C) with feat_sizes
        # (its feat_sizes=None branch has an upstream `x.dim == 4` typo).
        h, w = top.shape[-2:]
        src_flat = top.flatten(2).permute(2, 0, 1).contiguous()
        pos_flat = pos.flatten(2).permute(2, 0, 1).contiguous()
        memory = self.fusion(
            src=[src_flat],
            prompt=text.transpose(0, 1),
            src_pos=[pos_flat],
            prompt_key_padding_mask=padding,
            feat_sizes=[(h, w)],
        )['memory']
        return memory.permute(1, 2, 0).reshape_as(top)

    def embed_mask(self, mask: Tensor, top: Tensor) -> Tensor:
        embedded = self.mask_encoder(torch.zeros_like(top), mask, skip_mask_sigmoid=True)['vision_features']
        return self.mask_projection(embedded)

    def decode_pixels(self, pyramid: list[Tensor]) -> Tensor:
        if self.training:
            return checkpoint(self.pixel_decoder, pyramid, use_reentrant=False)
        return self.pixel_decoder(pyramid)


class DummyBackbone(nn.Module):
    """Same 3 FPN strides, channel width and 32 text tokens without vendor imports."""

    def __init__(self, *, image_size: int = 28, channels: int = 256):
        super().__init__()
        self.image_size, self.channels = image_size, channels
        self.vision = nn.Conv2d(3, channels, 1)
        self.text = nn.Embedding(259, channels, padding_idx=258).requires_grad_(False)
        self.fusion = nn.MultiheadAttention(channels, 4, batch_first=True)
        self.mask_projection = nn.Conv2d(1, channels, 1)
        self.pixel_decoder = nn.Conv2d(channels, channels, 1)

    def vision_adaptation_parameters(self):
        return self.vision.parameters()

    def set_vision_lora_trainable(self, enabled: bool) -> None:
        self.vision.requires_grad_(enabled)

    def encode_pair(self, first: Tensor, second: Tensor) -> list[tuple[Tensor, Tensor]]:
        def encode(image: Tensor) -> list[Tensor]:
            top = self.vision(F.adaptive_avg_pool2d(image, (self.image_size // 14,) * 2))
            return [F.interpolate(top, scale_factor=factor, mode='nearest') for factor in (4, 2, 1)]
        return list(zip(encode(first), encode(second), strict=True))

    def encode_text(self, prompts: Sequence[str], device: torch.device) -> tuple[Tensor, Tensor]:
        rows = [[[256], *[list(word.encode('utf-8')) for word in prompt.split()], [257]] for prompt in prompts]
        if any(not 2 < len(row) <= 32 for row in rows):
            raise ValueError('Dummy prompts must contain 1–30 words')
        width = max(len(word) for row in rows for word in row)
        tokens = torch.tensor([
            [word + [258] * (width - len(word)) for word in row + [[]] * (32 - len(row))]
            for row in rows
        ], device=device)
        padding = torch.arange(32, device=device)[None] >= torch.tensor(
            [len(row) for row in rows], device=device,
        )[:, None]
        return self.text(tokens).sum(2) / tokens.ne(258).sum(2, keepdim=True).clamp_min(1), padding

    def fuse_text(self, top: Tensor, text: Tensor, padding: Tensor) -> Tensor:
        visual = top.flatten(2).transpose(1, 2)
        fused = self.fusion(visual, text, text, key_padding_mask=padding, need_weights=False)[0]
        return top + fused.transpose(1, 2).reshape_as(top)

    def embed_mask(self, mask: Tensor, top: Tensor) -> Tensor:
        return self.mask_projection(F.interpolate(mask, size=top.shape[-2:], mode='area'))

    def decode_pixels(self, pyramid: list[Tensor]) -> Tensor:
        size = pyramid[0].shape[-2:]
        return self.pixel_decoder(sum(F.interpolate(f, size=size, mode='nearest') for f in pyramid))
