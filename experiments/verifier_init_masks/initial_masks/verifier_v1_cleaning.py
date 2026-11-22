"""Offline FP/FN supervision cleaning; never edit predictions or raw task truth.

Only NumPy/SciPy are needed. RGB inputs may be arrays or image paths. The legacy
three-argument call retains tiny-component and matched-boundary behavior.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy import ndimage as ndi


DEFAULTS = {
    "min_area": 16,
    "ignore_radius": 2,
    "ignore_border_black": True,
    "whu_strip_enabled": True,
    "whu_strip_max_width": 3.0,
    "whu_strip_min_length": 12.0,
    "whu_strip_min_aspect": 5.0,
    "whu_strip_max_gt_distance": 8.0,
}
REASON_ORDER = ("no_data", "tiny", "whu_strip", "boundary")
STRUCTURE = np.ones((3, 3), dtype=bool)


def tiny_components(mask, min_area):
    labels, count = ndi.label(mask, STRUCTURE)
    sizes = np.bincount(labels.ravel(), minlength=count + 1)
    ids = np.flatnonzero((sizes < min_area) & (np.arange(len(sizes)) > 0))
    return np.isin(labels, ids), len(ids)


def _rgb(values, shape):
    if isinstance(values, (str, Path)):
        from PIL import Image

        with Image.open(values) as image:
            values = np.asarray(image.convert("RGB"))
    else:
        values = np.asarray(values)
    if values.shape != (*shape, 3):
        raise ValueError(f"Expected RGB image with shape {(*shape, 3)}, got {values.shape}")
    return values


def border_black_mask(shape, image_t1=None, image_t2=None):
    """Union of each date's *own* border-connected exact RGB(0,0,0) domains.

    Label dates separately: a black patch in T2 cannot connect an interior T1
    black object to the border. Interior black roofs/shadows remain valid.
    """
    result = np.zeros(shape, dtype=bool)
    counts = []
    for image in (image_t1, image_t2):
        if image is None:
            counts.append(0)
            continue
        black = np.all(_rgb(image, shape) == 0, axis=2)
        labels, _ = ndi.label(black, STRUCTURE)
        ids = np.unique(np.concatenate((labels[0], labels[-1], labels[:, 0], labels[:, -1])))
        ids = ids[ids > 0]
        result |= np.isin(labels, ids)
        counts.append(len(ids))
    return result, counts


def _strip_geometry(ys, xs):
    """Whole-component local chord width, allowing bends in a roof outline.

    At each pixel take the shortest foreground chord in four lattice directions
    (diagonal lengths in Euclidean pixels). The maximum over the entire body
    rejects a thick independent object, while a one-pixel L does not become
    falsely 'wide' merely because its two arms have different directions.
    """
    xy = np.column_stack((xs, ys)).astype(float)
    centered = xy - xy.mean(axis=0)
    _, vectors = np.linalg.eigh(centered.T @ centered / max(1, len(xy)))
    major = vectors[:, 1]
    span = float(np.ptp(centered @ major) + np.abs(major).sum())
    region = np.zeros((int(np.ptp(ys)) + 1, int(np.ptp(xs)) + 1), dtype=bool)
    region[ys - ys.min(), xs - xs.min()] = True
    local_width = np.full(region.shape, np.inf)
    for dy, dx in ((0, 1), (1, 0), (1, 1), (1, -1)):
        direction = np.zeros((3, 3), dtype=bool)
        direction[1, 1] = direction[1 + dy, 1 + dx] = direction[1 - dy, 1 - dx] = True
        runs, count = ndi.label(region, direction)
        sizes = np.bincount(runs.ravel(), minlength=count + 1)
        local_width = np.minimum(local_width, sizes[runs] * np.hypot(dx, dy))
    width = float(local_width[region].max())
    length = max(span, len(xs) / float(local_width[region].mean()))
    return length, width, length / max(width, 1.0)


def whu_strip_mask(residual_fp, matched_gt, cfg):
    """Select entire residual FP bodies after the legacy ignore stage.

    Do not clip components to a GT-neighborhood band: a large independent
    building cannot donate its near-GT edge to the thin-strip rule.
    """
    result = np.zeros_like(residual_fp)
    details = []
    if not matched_gt.any():
        return result, details
    gt_boundary = matched_gt & ~ndi.binary_erosion(matched_gt, structure=STRUCTURE, border_value=0)
    distance = ndi.distance_transform_edt(~gt_boundary)
    labels, _ = ndi.label(residual_fp, STRUCTURE)
    for component_id, slices in enumerate(ndi.find_objects(labels), 1):
        if slices is None:
            continue
        region = labels[slices] == component_id
        ys, xs = np.nonzero(region)
        ys, xs = ys + slices[0].start, xs + slices[1].start
        if len(xs) < 2:
            continue
        length, width, aspect = _strip_geometry(ys, xs)
        if width > cfg["whu_strip_max_width"] + 1e-6:
            continue
        if length < cfg["whu_strip_min_length"] or aspect < cfg["whu_strip_min_aspect"]:
            continue
        max_distance = float(distance[ys, xs].max())
        if max_distance > cfg["whu_strip_max_gt_distance"]:
            continue
        result[ys, xs] = True
        details.append(dict(component_id=component_id, area=len(xs), length=length,
                            width=width, aspect=aspect, max_gt_distance=max_distance,
                            bbox_xyxy=[int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]))
    return result, details


def clean_diagnostic(pred, raw, cfg, dataset=None, image_t1=None, image_t2=None):
    """Return clean/fp/fn/ignore/protected/metrics/cleaning plus reason masks.

    Masks are 2-D; positive values mean foreground. ``clean`` removes only small
    raw GT components. Every removed GT pixel is ignored, including pixels in M,
    preventing artificial FP labels. Inputs are never written to.
    """
    cfg = {**DEFAULTS, **cfg}
    if cfg["min_area"] < 1 or cfg["ignore_radius"] < 0:
        raise ValueError("min_area must be >= 1 and ignore_radius >= 0")
    pred, raw = np.asarray(pred) > 0, np.asarray(raw) > 0
    if pred.ndim != 2 or pred.shape != raw.shape:
        raise ValueError("pred and raw must be equally shaped 2-D masks")
    small_gt, ng = tiny_components(raw, cfg["min_area"])
    clean = raw & ~small_gt
    raw_fp, raw_fn = pred & ~raw, ~pred & raw
    small_fp, nfp = tiny_components(raw_fp, cfg["min_area"])
    small_fn, nfn = tiny_components(raw_fn, cfg["min_area"])
    tiny = small_gt | small_fp | small_fn

    if cfg["ignore_border_black"]:
        no_data, no_data_components = border_black_mask(raw.shape, image_t1, image_t2)
    else:
        no_data, no_data_components = np.zeros_like(raw), [0, 0]
    # Match using observable pixels only; a match in padding must not license
    # suppressing a separate visible false positive near that GT component.
    pl, _ = ndi.label(pred & ~no_data, STRUCTURE)
    gl, _ = ndi.label(clean & ~no_data, STRUCTURE)
    overlap = pred & clean & ~no_data
    ip, ig = np.unique(pl[overlap]), np.unique(gl[overlap])
    ip, ig = ip[ip > 0], ig[ig > 0]
    matched_p, matched_g = pred & np.isin(pl, ip), clean & np.isin(gl, ig)
    independent_p, independent_g = pred & ~matched_p, clean & ~matched_g
    baseline_protected = (independent_p | independent_g) & ~tiny & ~no_data
    boundary = ((matched_p ^ ndi.binary_erosion(matched_p, structure=STRUCTURE, border_value=0)) |
                (matched_g ^ ndi.binary_erosion(matched_g, structure=STRUCTURE, border_value=0)))
    tolerance = (ndi.distance_transform_edt(~boundary) <= cfg["ignore_radius"]
                 if boundary.any() and cfg["ignore_radius"] > 0 else np.zeros_like(raw))
    tolerance &= ~baseline_protected
    whu_enabled = dataset == "WHU-CD" and cfg["whu_strip_enabled"]
    if whu_enabled:
        residual_fp = raw_fp & ~tiny & ~no_data & ~tolerance
        strip, strip_details = whu_strip_mask(residual_fp, matched_g & ~no_data, cfg)
    else:
        strip, strip_details = np.zeros_like(raw), []
    protected = baseline_protected & ~strip

    ignore = np.zeros_like(raw)
    reasons = {}
    for reason, domain in zip(REASON_ORDER, (no_data, tiny, strip, tolerance)):
        reasons[reason] = domain & ~ignore
        ignore |= domain
    valid = ~ignore
    fp, fn = pred & ~clean & valid, ~pred & clean & valid
    raw_union = int((pred | raw).sum())
    effective_union = int(((pred | clean) & valid).sum())
    effective_iou = float((pred & clean & valid).sum() / effective_union) if effective_union else None
    stats = dict(
        min_area=cfg["min_area"], ignore_radius=cfg["ignore_radius"], removed_gt_components=ng,
        removed_gt_pixels=int(small_gt.sum()), tiny_raw_fp_components=nfp, tiny_raw_fn_components=nfn,
        raw_fp_area=int(raw_fp.sum()), raw_fn_area=int(raw_fn.sum()),
        matched_tolerance_pixels=int(reasons["boundary"].sum()), ignored_pixels=int(ignore.sum()),
        ignore_fraction=float(ignore.mean()),
        ignored_residual_fraction=float(((raw_fp | raw_fn) & ignore).sum() / max(1, (raw_fp | raw_fn).sum())),
        independent_fp_preserved_pixels=int((independent_p & raw_fp & valid).sum()),
        independent_fn_preserved_pixels=int((independent_g & raw_fn & valid).sum()),
        matched_prediction_components=len(ip), matched_gt_components=len(ig),
        no_data_components_t1=no_data_components[0], no_data_components_t2=no_data_components[1],
        no_data_pixels=int(no_data.sum()), whu_strip_enabled=bool(whu_enabled),
        whu_strip_components=len(strip_details), whu_strip_details=strip_details,
        whu_strip_geometry="maximum local four-direction chord width on complete post-boundary FP component",
        whu_strip_config={key: cfg[key] for key in DEFAULTS if key.startswith("whu_strip_")},
        ignore_reason_order=list(REASON_ORDER), ignore_reason_pixels={key: int(mask.sum()) for key, mask in reasons.items()},
    )
    for reason, domain in reasons.items():
        stats[f"{reason}_ignored_fp_area"] = int((raw_fp & domain).sum())
        stats[f"{reason}_ignored_fn_area"] = int((raw_fn & domain).sum())
    metrics = dict(
        raw_iou=float((pred & raw).sum() / raw_union) if raw_union else 1.0,
        effective_iou=effective_iou, iou=effective_iou,
        raw_foreground_union=raw_union, effective_foreground_union=effective_union,
        fp_area=int(fp.sum()), fn_area=int(fn.sum()), mask_area=int(pred.sum()),
        target_area=int(clean.sum()), raw_target_area=int(raw.sum()), valid_area=int(valid.sum()),
        ignore_area=int(ignore.sum()), ignore_fraction=float(ignore.mean()),
        raw_fp_area=stats["raw_fp_area"], raw_fn_area=stats["raw_fn_area"],
        empty_effective_foreground=effective_union == 0,
    )
    metrics.update({key: value for key, value in stats.items() if "_ignored_" in key})
    return dict(clean=clean, fp=fp, fn=fn, ignore=ignore, protected=protected,
                metrics=metrics, cleaning=stats, ignore_reasons=reasons)
