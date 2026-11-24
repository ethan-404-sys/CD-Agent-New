"""Shared explicit-image cache address used by preparation, workers and readers."""
from pathlib import Path
from urllib.parse import quote
import numpy as np
from . import instances


def manifest_cache_path(run, image_path, text, tile, conf, pixel_thr):
    source = Path(image_path).resolve()
    slug = quote(' '.join(text.lower().split()), safe='')
    return (Path(run) / 'sam3_manifest_cache' / f'{slug}_t{tile}_c{conf:g}_p{pixel_thr:g}' /
            source.relative_to(source.anchor) / 'instances.npz')


def read_instances(path, score_thr):
    with np.load(path) as z:
        h, w = map(int, z['shape'])
        keep = z['scores'] >= score_thr
        dense = np.unpackbits(z['masks'][keep], axis=1, count=h*w).reshape(-1, h, w).astype(bool)
        return instances.from_dense(dense, z['scores'][keep])


def _components_as_instances(mask, prob_map):
    from common import components
    if not mask.any():
        return []
    labels, comps = components(mask)
    rows = []
    for comp in sorted(comps, key=lambda c: (-c['area'], c['id'])):
        region = labels == comp['id']
        idx = np.flatnonzero(region).astype(np.int32)
        score = float(prob_map[region].mean()) if prob_map is not None else 1.0
        rows.append(dict(id=len(rows) + 1, idx=idx, area=int(idx.size),
                         score=score, bbox=list(comp['bbox'])))
    return rows


def read_segment_result(path, score_thr, mode='instance'):
    """Read SAM3 cached output in instance, semantic, or fused mode with presence metadata."""
    if mode not in ('instance', 'semantic', 'fused'):
        raise ValueError(f'Unsupported SAM3 segment mode: {mode}')
    with np.load(path) as z:
        h, w = map(int, z['shape'])
        keep = z['scores'] >= score_thr
        inst_scores = z['scores'][keep]
        inst_dense = np.unpackbits(z['masks'][keep], axis=1, count=h*w).reshape(-1, h, w).astype(bool)
        has_sem = 'sem_prob' in z.files
        presence = float(z['presence_score']) if 'presence_score' in z.files else None
        sem_eff = None
        if has_sem:
            sem_raw = z['sem_prob'].astype(np.float32) / 255.0
            sem_eff = sem_raw * (presence if presence is not None else 1.0)
        sem_area = int((sem_eff >= score_thr).sum()) if sem_eff is not None else None
        extra = dict(mode=mode, presence_score=presence, semantic_area=sem_area, has_semantic=has_sem)
        if mode == 'instance':
            return instances.from_dense(inst_dense, inst_scores), extra
        if mode == 'semantic':
            if sem_eff is None:
                return [], extra
            return _components_as_instances(sem_eff >= score_thr, sem_eff), extra
        # mode == 'fused': combine instance decoder and semantic head (max probability, OmniOVCD style)
        inst_prob = np.zeros((h, w), np.float32)
        inst_union = np.zeros((h, w), bool)
        for m, sc in zip(inst_dense, inst_scores):
            inst_union |= m
            inst_prob = np.maximum(inst_prob, m.astype(np.float32) * float(sc))
        if sem_eff is None:
            return _components_as_instances(inst_union, inst_prob), extra
        fused_prob = np.maximum(inst_prob, sem_eff)
        fused_mask = inst_union | (sem_eff >= score_thr)
        return _components_as_instances(fused_mask, fused_prob), extra
