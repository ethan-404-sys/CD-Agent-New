"""Minimal Visual Evidence Pack generator for CD-Agent experiments.

Automatically selects 6-8 high-signal cases (best F1, mid-band F1~0.5, largest FP, largest FN)
and renders a compact offline HTML review page so the researcher can inspect real visual quality
in ~5 minutes instead of manually digging through hundreds of outputs.
"""
from __future__ import annotations

from html import escape
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from PIL import Image

from .verify import check_binary_mask


def _load_rgb(src: Any) -> np.ndarray:
    if isinstance(src, (str, Path)):
        with Image.open(src) as im:
            return np.asarray(im.convert('RGB'), dtype=np.uint8)
    arr = np.asarray(src)
    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=-1)
    return arr.astype(np.uint8)


def _load_mask(src: Any, shape: tuple[int, int]) -> np.ndarray:
    if isinstance(src, (str, Path)):
        with Image.open(src) as im:
            arr = np.asarray(im.convert('L'))
        return check_binary_mask(arr > 127, expected_shape=shape, name=str(src))
    return check_binary_mask(src, expected_shape=shape, name='mask')


def compute_case_stats(pred: np.ndarray, gt: np.ndarray) -> dict[str, Any]:
    tp = int((pred & gt).sum())
    fp = int((pred & ~gt).sum())
    fn = int((~pred & gt).sum())
    union = tp + fp + fn
    if union == 0:
        f1 = iou = 1.0
    else:
        f1 = 2.0 * tp / (2.0 * tp + fp + fn)
        iou = tp / union
    return {'tp': tp, 'fp': fp, 'fn': fn, 'f1': round(f1, 4), 'iou': round(iou, 4)}


def select_evidence_cases(
    cases: Sequence[Mapping[str, Any]],
    *,
    per_bucket: int = 2,
) -> list[dict[str, Any]]:
    """Deterministically pick representative cases across 4 diagnostic buckets."""
    enriched: list[dict[str, Any]] = []
    for row in cases:
        item = dict(row)
        item.setdefault('f1', 0.0)
        item.setdefault('fp', 0)
        item.setdefault('fn', 0)
        enriched.append(item)

    chosen: list[dict[str, Any]] = []
    seen: set[str] = set()

    def take(ordered: list[dict[str, Any]], bucket: str) -> None:
        added = 0
        for item in ordered:
            cid = str(item['id'])
            if cid in seen:
                continue
            seen.add(cid)
            copy_item = dict(item)
            copy_item['bucket'] = bucket
            chosen.append(copy_item)
            added += 1
            if added >= per_bucket:
                break

    take(sorted(enriched, key=lambda r: (-float(r['f1']), str(r['id']))), 'high_f1')
    take(sorted(enriched, key=lambda r: (abs(float(r['f1']) - 0.5), str(r['id']))), 'mid_f1')
    take(sorted(enriched, key=lambda r: (-int(r['fp']), str(r['id']))), 'high_fp')
    take(sorted(enriched, key=lambda r: (-int(r['fn']), str(r['id']))), 'high_fn')
    return chosen


def _error_overlay(base_rgb: np.ndarray, pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    """Overlay TP (green), FP (red), FN (blue) onto a temporal RGB image."""
    out = base_rgb.astype(np.float32) * 0.55
    tp = pred & gt
    fp = pred & ~gt
    fn = ~pred & gt
    out[tp] = out[tp] + np.array([20.0, 210.0, 70.0]) * 0.45
    out[fp] = out[fp] + np.array([235.0, 45.0, 45.0]) * 0.45
    out[fn] = out[fn] + np.array([45.0, 120.0, 245.0]) * 0.45
    return np.clip(out, 0, 255).astype(np.uint8)


def _save_thumb(arr: np.ndarray, path: Path, max_side: int) -> None:
    im = Image.fromarray(arr)
    if max(im.size) > max_side:
        im.thumbnail((max_side, max_side), Image.Resampling.BILINEAR)
    path.parent.mkdir(parents=True, exist_ok=True)
    im.save(path)


def build_evidence_page(
    cases: Sequence[Mapping[str, Any]],
    out_dir: Path | str,
    *,
    title: str = 'CD-Agent Visual Evidence Pack',
    metrics: Mapping[str, Any] | None = None,
    per_bucket: int = 2,
    max_side: int = 384,
) -> Path:
    """Generate `index.html` + `assets/` + `selection.json` inside `out_dir`."""
    out = Path(out_dir)
    assets = out / 'assets'
    assets.mkdir(parents=True, exist_ok=True)

    prepared: list[dict[str, Any]] = []
    for row in cases:
        t1 = _load_rgb(row['image_t1'])
        t2 = _load_rgb(row['image_t2'])
        shape = t1.shape[:2]
        pred = _load_mask(row['pred_mask'], shape)
        gt = _load_mask(row['gt_mask'], shape)
        stats = compute_case_stats(pred, gt)
        item = dict(row)
        item.update(stats)
        item['_t1'] = t1
        item['_t2'] = t2
        item['_pred'] = pred
        item['_gt'] = gt
        if row.get('init_mask') is not None:
            item['_init'] = _load_mask(row['init_mask'], shape)
        if row.get('ref_image') is not None:
            item['_ref'] = _load_rgb(row['ref_image'])
        prepared.append(item)

    selected = select_evidence_cases(prepared, per_bucket=per_bucket)
    cards_html: list[str] = []
    manifest_rows: list[dict[str, Any]] = []

    for idx, item in enumerate(selected):
        cid = str(item['id'])
        safe = ''.join(ch if ch.isalnum() or ch in '-_' else '_' for ch in cid)
        prefix = f'{idx:02d}_{safe}'
        panels: list[tuple[str, str]] = []

        _save_thumb(item['_t1'], assets / f'{prefix}_t1.png', max_side)
        panels.append(('T1', f'assets/{prefix}_t1.png'))

        _save_thumb(item['_t2'], assets / f'{prefix}_t2.png', max_side)
        panels.append(('T2', f'assets/{prefix}_t2.png'))

        if '_ref' in item:
            _save_thumb(item['_ref'], assets / f'{prefix}_ref.png', max_side)
            panels.append(('Ref', f'assets/{prefix}_ref.png'))
        elif '_init' in item:
            _save_thumb(item['_init'].astype(np.uint8) * 255, assets / f'{prefix}_init.png', max_side)
            panels.append(('Init Mask', f'assets/{prefix}_init.png'))

        _save_thumb(item['_pred'].astype(np.uint8) * 255, assets / f'{prefix}_pred.png', max_side)
        panels.append(('Pred', f'assets/{prefix}_pred.png'))

        _save_thumb(item['_gt'].astype(np.uint8) * 255, assets / f'{prefix}_gt.png', max_side)
        panels.append(('GT', f'assets/{prefix}_gt.png'))

        overlay_t1 = _error_overlay(item['_t1'], item['_pred'], item['_gt'])
        _save_thumb(overlay_t1, assets / f'{prefix}_err_t1.png', max_side)
        panels.append(('Err@T1 (TP=G FP=R FN=B)', f'assets/{prefix}_err_t1.png'))

        overlay_t2 = _error_overlay(item['_t2'], item['_pred'], item['_gt'])
        _save_thumb(overlay_t2, assets / f'{prefix}_err_t2.png', max_side)
        panels.append(('Err@T2 (TP=G FP=R FN=B)', f'assets/{prefix}_err_t2.png'))

        note = str(item.get('summary') or item.get('instruction') or '')
        panel_tags = ''.join(
            f'<figure><img src="{escape(src)}" alt="{escape(lbl)}"><figcaption>{escape(lbl)}</figcaption></figure>'
            for lbl, src in panels
        )
        cards_html.append(
            f'<section class="card">'
            f'<div class="head"><span class="badge">{escape(item["bucket"])}</span>'
            f'<strong>{escape(cid)}</strong>'
            f'<span class="stats">F1={item["f1"]:.4f} | IoU={item["iou"]:.4f} | '
            f'FP={item["fp"]}px | FN={item["fn"]}px</span></div>'
            f'{f"<p class=note>{escape(note)}</p>" if note else ""}'
            f'<div class="grid">{panel_tags}</div></section>'
        )
        manifest_rows.append({
            'id': cid,
            'bucket': item['bucket'],
            'f1': item['f1'],
            'iou': item['iou'],
            'tp': item['tp'],
            'fp': item['fp'],
            'fn': item['fn'],
        })

    metrics_block = ''
    if metrics:
        metrics_block = '<pre class="metrics">' + escape(json.dumps(metrics, ensure_ascii=False, indent=2)) + '</pre>'

    html = f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><title>{escape(title)}</title>
<style>
body{{font-family:system-ui,sans-serif;margin:20px;background:#f6f7f9;color:#1d232a}}
h1{{margin:0 0 8px 0;font-size:20px}}
.sub{{color:#556;font-size:13px;margin-bottom:14px}}
.metrics{{background:#fff;border:1px solid #dce1e8;padding:10px 12px;border-radius:6px;font-size:12px;overflow:auto}}
.card{{background:#fff;border:1px solid #dce1e8;border-radius:8px;padding:12px;margin:14px 0}}
.head{{display:flex;gap:10px;align-items:center;flex-wrap:wrap;margin-bottom:8px}}
.badge{{background:#1f6feb;color:#fff;padding:2px 8px;border-radius:10px;font-size:12px}}
.stats{{margin-left:auto;font-family:monospace;font-size:12px;color:#334}}
.note{{margin:0 0 8px 0;font-size:13px;color:#445}}
.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(160px,1fr));gap:8px}}
figure{{margin:0;text-align:center}}
img{{width:100%;height:auto;border:1px solid #e1e5eb;border-radius:4px;background:#111}}
figcaption{{font-size:11px;color:#556;margin-top:3px}}
</style></head><body>
<h1>{escape(title)}</h1>
<div class="sub">Selected {len(selected)} representative cases from {len(prepared)} total (TP=Green, FP=Red, FN=Blue).</div>
{metrics_block}
{''.join(cards_html)}
</body></html>"""

    page = out / 'index.html'
    page.write_text(html, encoding='utf-8')
    (out / 'selection.json').write_text(json.dumps(manifest_rows, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    return page
