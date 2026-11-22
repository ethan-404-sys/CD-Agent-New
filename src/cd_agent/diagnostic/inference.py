"""Online entry point: project EpisodeInput onto a strictly read-only allowlist."""
from __future__ import annotations

import torch

from cd_agent.verify import EpisodeInput

from .data import load_mask, load_rgb
from .model import HVVerifier, gated_probabilities


@torch.inference_mode()
def predict_episode(model: HVVerifier, episode: EpisodeInput) -> dict[str, torch.Tensor]:
    if type(episode) is not EpisodeInput:
        raise TypeError('Online diagnostic inference requires an EpisodeInput')
    if not episode.init_mask:
        raise ValueError('EpisodeInput.init_mask is required for hypothesis verification')
    if model.training:
        raise ValueError('Set the diagnostic model to eval() before online inference')
    device = next(model.parameters()).device
    image_t1 = load_rgb(episode.image_t1).unsqueeze(0).to(device)
    image_t2 = load_rgb(episode.image_t2).unsqueeze(0).to(device)
    init_mask = load_mask(episode.init_mask).unsqueeze(0).to(device)
    if image_t1.shape[-2:] != init_mask.shape[-2:]:
        raise ValueError('EpisodeInput images and initial mask must be aligned')
    outputs = model(image_t1=image_t1, image_t2=image_t2, init_mask=init_mask, instruction=[episode.instruction])
    return {**outputs, **gated_probabilities(outputs, init_mask)}
