"""Offline valid-region spatial, counterfactual and hypothesis-use metrics."""
from __future__ import annotations

from collections import Counter, defaultdict
from itertools import combinations

import torch
from torch import Tensor

from .model import gated_probabilities

TASK_TYPES = ('related', 'into', 'outof', 'transition')
IOU_BANDS = ('[0,0.3)', '[0.3,0.7)', '[0.7,1.0]')


def _counts(prediction: Tensor, truth: Tensor, support: Tensor) -> tuple[int, int, int]:
    return tuple(int(x.sum()) for x in (
        prediction & truth & support, prediction & ~truth & support, ~prediction & truth & support,
    ))


def _scores(tp: int, fp: int, fn: int) -> tuple[float, float]:
    return (2 * tp / (2 * tp + fp + fn), tp / (tp + fp + fn)) if tp + fp + fn else (1.0, 1.0)


def predict_errors(outputs: dict[str, Tensor], init_mask: Tensor, *, threshold: float = 0.25) -> dict[str, Tensor]:
    probabilities = gated_probabilities(outputs, init_mask)
    return {kind: probabilities[f'prob_{kind}'].gt(threshold) for kind in ('fp', 'fn')}


def hypothesis_utilization_gain(
    real_fn: Tensor, zeroed_fn: Tensor, gt: Tensor, init_mask: Tensor, ignore: Tensor,
) -> dict[str, float | int | None]:
    """Compare binary FN predictions against identical original outer-domain targets."""
    outer_misses = gt.bool() & ~init_mask.bool() & ~ignore.bool()
    count = int(outer_misses.sum())
    real = int((real_fn.bool() & outer_misses).sum())
    zero = int((zeroed_fn.bool() & outer_misses).sum())
    return {
        'support': count, 'real_tp': real, 'zeroed_tp': zero,
        'real_recall': real / count if count else None, 'zeroed_recall': zero / count if count else None,
        'gain': (real - zero) / count if count else None,
    }


class DiagnosticMetrics:
    """Micro FP/FN scores, macro refined scores and pixel-weighted CF flip rates.

    For CF evaluation, send rows ordered by shared-mask group. A flip is counted
    only if both predictions are correct at a valid pixel where the task GT flips.
    An anchor is flagged if it contains any predicted valid-region FP or FN.
    """

    def __init__(self):
        self.buckets = defaultdict(Counter)
        self.seen = 0
        self._active_group = None
        self._pending = []

    @staticmethod
    def _keys(task: str, init_iou: float):
        return ('overall', task, IOU_BANDS[0 if init_iou < 0.3 else 1 if init_iou < 0.7 else 2])

    def update(
        self, pred_fp: Tensor, pred_fn: Tensor, init: Tensor, gt: Tensor, ignore: Tensor, *,
        task: str, is_anchor: bool = False, cf_group=None, instruction: str = '',
        zeroed_fn: Tensor | None = None,
    ) -> dict[str, float]:
        if task not in TASK_TYPES:
            raise ValueError(f'Unknown diagnostic task: {task}')
        self.seen += 1
        if cf_group != self._active_group:
            self._flush_pairs()
            self._active_group = cf_group
        fp, fn, init, gt, ignore = (t.detach().cpu().bool() for t in (pred_fp, pred_fn, init, gt, ignore))
        valid = ~ignore
        if not valid.any():
            return {}
        fp, fn = fp & init & valid, fn & ~init & valid
        refined = (init & ~fp) | fn
        init_iou = _scores(*_counts(init, gt, valid))[1]
        refined_f1, refined_iou = _scores(*_counts(refined, gt, valid))
        keys = self._keys(task, init_iou)
        values = dict(init_iou=init_iou, refined_f1=refined_f1,
                      refined_iou=refined_iou, delta_iou=refined_iou - init_iou)
        counts = {}
        for kind, prediction, truth, support in (
            ('fp', fp, init & ~gt, init & valid), ('fn', fn, ~init & gt, ~init & valid),
        ):
            counts.update(zip((f'{kind}_tp', f'{kind}_fp', f'{kind}_fn'), _counts(prediction, truth, support)))
        if zeroed_fn is not None:
            gain = hypothesis_utilization_gain(fn, zeroed_fn.detach().cpu(), gt, init, ignore)
            counts.update(hug_tp=gain['real_tp'], hug_zero=gain['zeroed_tp'], hug_support=gain['support'])
        if is_anchor or init_iou == 1.0:
            counts.update(anchor_count=1, anchor_false_alarm=int((fp | fn).any()))
        for key in keys:
            self.buckets[key].update(dict(count=1, **values, **counts))
        if cf_group is not None:
            self._pending.append((keys, instruction, init, gt, valid, refined))
        return values

    def _flush_pairs(self):
        for (keys_a, text_a, init_a, gt_a, valid_a, pred_a), (keys_b, text_b, init_b, gt_b, valid_b, pred_b) in combinations(self._pending, 2):
            if text_a == text_b:
                continue
            if not torch.equal(init_a, init_b):
                raise ValueError('Counterfactual evaluation group does not share the same initial mask')
            flips = (gt_a ^ gt_b) & valid_a & valid_b
            total = int(flips.sum())
            if total:
                correct = int(((pred_a == gt_a) & (pred_b == gt_b) & flips).sum())
                for key in set(keys_a + keys_b):
                    self.buckets[key].update(cf_pairs=1, cf_pixels=total, cf_correct=correct)
        self._pending.clear()

    def _summary(self, key: str) -> dict:
        stats = self.buckets[key]
        count = stats['count']
        result = {'count': count}
        for kind in ('fp', 'fn'):
            f1, iou = _scores(*(stats[f'{kind}_{n}'] for n in ('tp', 'fp', 'fn')))
            result.update({f'{kind}_f1': f1 if count else None, f'{kind}_iou': iou if count else None})
        for name in ('init_iou', 'refined_f1', 'refined_iou', 'delta_iou'):
            result[name] = stats[name] / count if count else None
        for name, numerator, denominator in (
            ('anchor_fpr', 'anchor_false_alarm', 'anchor_count'),
            ('cf_flip_rate', 'cf_correct', 'cf_pixels'),
            ('fn_recall_real_init', 'hug_tp', 'hug_support'),
            ('fn_recall_zero_init', 'hug_zero', 'hug_support'),
        ):
            result[name] = stats[numerator] / stats[denominator] if stats[denominator] else None
        result.update(
            hypothesis_utilization_gain=(stats['hug_tp'] - stats['hug_zero']) / stats['hug_support']
            if stats['hug_support'] else None,
            anchor_count=stats['anchor_count'], cf_pairs=stats['cf_pairs'], cf_pixels=stats['cf_pixels'],
            hypothesis_utilization_support=stats['hug_support'],
        )
        return result

    def summary(self) -> dict:
        self._flush_pairs()
        return {
            'count': self.seen, 'failed': 0, 'overall': self._summary('overall'),
            'by_task': {task: self._summary(task) for task in TASK_TYPES},
            'by_init_iou': {band: self._summary(band) for band in IOU_BANDS},
        }
