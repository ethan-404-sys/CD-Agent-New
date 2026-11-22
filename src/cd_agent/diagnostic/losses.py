"""Valid-region supervision, independent presence gates and optional pair ranking."""
from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import Tensor
from torch.nn import functional as F


def effective_iou(prediction: Tensor, target: Tensor, valid: Tensor) -> Tensor:
    """Per-image IoU, with empty union scored 1; callers may skip all-ignore images."""
    pred, truth = prediction.bool(), target.bool()
    intersection = (pred & truth & valid).flatten(1).sum(1)
    union = ((pred | truth) & valid).flatten(1).sum(1)
    return torch.where(union > 0, intersection / union.clamp_min(1), 1.0)


def _mean_present(values: Tensor, present: Tensor) -> Tensor:
    return (values * present).sum() / present.sum().clamp_min(1)


def _dense_loss(logits: Tensor, target: Tensor, support: Tensor, positive_weight: float) -> Tensor:
    x, y, mask = logits.float().flatten(1), target.float().flatten(1), support.flatten(1)
    count = mask.sum(1)
    bce = F.binary_cross_entropy_with_logits(x, y, reduction='none', pos_weight=x.new_tensor(positive_weight))
    bce = (bce * mask).sum(1) / count.clamp_min(1)
    probability = x.sigmoid() * mask
    pixels = (y * mask).sum(1)
    dice = 1 - (2 * (probability * y).sum(1) + 1e-6) / (probability.sum(1) + pixels + 1e-6)
    return _mean_present(bce + dice * (pixels > 0), count > 0)


def counterfactual_margin_loss(
    outputs: dict[str, Tensor], init_mask: Tensor, valid: Tensor,
    targets: dict[str, Tensor], pairs: Sequence[tuple[int, int]], *, margin: float = 1.0,
) -> Tensor:
    """Rank raw error logits at valid pixels whose labels flip under the same mask."""
    total = outputs['logits_fn'].float().sum() * 0
    count = init_mask.new_zeros((), dtype=torch.int64)
    for first, second in pairs:
        if first == second or not torch.equal(init_mask[first], init_mask[second]):
            raise ValueError('Counterfactual pairs must have distinct rows and an identical initial mask')
        shared_valid = valid[first] & valid[second]
        for kind, domain in (('fp', init_mask[first]), ('fn', ~init_mask[first])):
            y = targets[kind]
            sign = y[first].float() - y[second].float()
            support = shared_valid & domain & sign.ne(0)
            logits = outputs[f'logits_{kind}'].float()
            total = total + (F.relu(margin - sign * (logits[first] - logits[second])) * support).sum()
            count = count + support.sum()
    return total / count.clamp_min(1)


def diagnostic_loss(
    outputs: dict[str, Tensor], *, init_mask: Tensor, gt_clean: Tensor,
    ignore_mask: Tensor, fp_mask: Tensor, fn_mask: Tensor,
    positive_weight: float = 2.0, presence_weight: float = 1.0, quality_weight: float = 1.0,
    cf_pairs: Sequence[tuple[int, int]] = (), cf_weight: float = 0.0, cf_margin: float = 1.0,
) -> dict[str, Tensor]:
    inside, valid = init_mask.bool(), ~ignore_mask.bool()
    targets = {'fp': fp_mask.bool() & inside & valid, 'fn': fn_mask.bool() & ~inside & valid}
    usable = valid.flatten(1).any(1)
    parts = {}
    for kind, domain in (('fp', inside), ('fn', ~inside)):
        parts[kind] = _dense_loss(outputs[f'logits_{kind}'], targets[kind], domain & valid, positive_weight)
        has_error = targets[kind].flatten(1).any(1).float()
        parts[f'has_{kind}'] = _mean_present(F.binary_cross_entropy_with_logits(
            outputs[f'logits_has_{kind}'].float(), has_error, reduction='none',
        ), usable)
    parts['quality'] = _mean_present(F.huber_loss(
        outputs['q_score'].float(), effective_iou(inside, gt_clean, valid), delta=0.1, reduction='none',
    ), usable)
    parts['counterfactual'] = (
        counterfactual_margin_loss(outputs, inside, valid, targets, cf_pairs, margin=cf_margin)
        if cf_weight and cf_pairs else outputs['logits_fn'].new_zeros(())
    )
    parts['total'] = (
        parts['fp'] + parts['fn'] + presence_weight * (parts['has_fp'] + parts['has_fn'])
        + quality_weight * parts['quality'] + cf_weight * parts['counterfactual']
    )
    return parts
