#!/usr/bin/env python3
"""Compare two T3.2 Dev-32 runs (baseline vs current) and build an offline A/B report + HTML page."""
from __future__ import annotations

import argparse
import html
import json
import os
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding='utf-8'))


def _trace_stats(run_dir: Path, cases: list[dict]) -> dict:
    tot_calls = 0
    zero_seg = 0
    tot_seg = 0
    modes = Counter()
    for c in cases:
        steps_path = run_dir / 'traces' / c['case'] / 'steps.jsonl'
        if not steps_path.is_file():
            continue
        for line in steps_path.read_text(encoding='utf-8').splitlines():
            if not line.strip():
                continue
            s = json.loads(line)
            for call in s.get('calls') or []:
                tot_calls += 1
                tool = call.get('tool')
                args = call.get('args') or {}
                res = json.loads(call.get('result_text') or '{}')
                if tool == 'segment_text':
                    tot_seg += 1
                    modes[args.get('mode', 'instance')] += 1
                    if res.get('n', 0) == 0:
                        zero_seg += 1
                elif tool == 'segment_text_pair':
                    modes[args.get('mode', 'instance')] += 2
                    for sub in (res.get('results') or {}).values():
                        tot_seg += 1
                        if sub.get('n', 0) == 0:
                            zero_seg += 1
    avg_rounds = sum(c.get('planning_rounds', 0) for c in cases) / max(1, len(cases))
    return dict(
        avg_rounds=round(avg_rounds, 2),
        total_tool_calls=tot_calls,
        segment_calls=tot_seg,
        zero_detections=zero_seg,
        zero_rate=round(zero_seg / max(1, tot_seg), 4),
        segment_modes=dict(modes),
    )


def _render_error_overlay(query_t2: Path, pred_path: Path, gt_path: Path, ign_path: Path | None) -> Image.Image:
    base = np.asarray(Image.open(query_t2).convert('RGB'), dtype=np.float32)
    pred = np.asarray(Image.open(pred_path)) > 0
    gt = np.asarray(Image.open(gt_path)) > 0
    valid = ~(np.asarray(Image.open(ign_path)) > 0) if (ign_path and ign_path.is_file()) else np.ones_like(gt)
    out = base.copy()
    for region, color in (
        (pred & gt & valid, np.array([0.0, 230.0, 118.0], dtype=np.float32)),
        (pred & (~gt) & valid, np.array([255.0, 68.0, 68.0], dtype=np.float32)),
        ((~pred) & gt & valid, np.array([64.0, 156.0, 255.0], dtype=np.float32)),
    ):
        if region.any():
            out[region] = out[region] * 0.50 + color * 0.50
    return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8))


def _pct(v: float | None) -> str:
    return '—' if v is None else f'{v * 100:.1f}%'


def _delta_str(v: float) -> str:
    return f'{v * 100:+.1f}%'


def build_comparison(baseline_dir: Path, current_dir: Path, out_dir: Path, top_k: int = 5) -> dict:
    base_m = _load_json(baseline_dir / 'metrics.json')
    curr_m = _load_json(current_dir / 'metrics.json')

    base_by_case = {c['case']: c for c in base_m['cases']}
    deltas = []
    for nc in curr_m['cases']:
        oc = base_by_case.get(nc['case'])
        if oc is None:
            continue
        deltas.append(dict(
            case=nc['case'],
            dataset=nc['dataset'],
            target_class_name=nc['target_class_name'],
            is_negative=bool(nc['is_negative']),
            gt_pixels=int(nc['gt_pixels']),
            old_f1=float(oc['f1']),
            new_f1=float(nc['f1']),
            delta_f1=float(nc['f1'] - oc['f1']),
            old_iou=float(oc['iou']),
            new_iou=float(nc['iou']),
            old_p=float(oc['precision']),
            new_p=float(nc['precision']),
            old_r=float(oc['recall']),
            new_r=float(nc['recall']),
            old_sub=oc['submitted_by'],
            new_sub=nc['submitted_by'],
            old_rounds=int(oc['planning_rounds']),
            new_rounds=int(nc['planning_rounds']),
            old_fp=int(oc['fp']),
            new_fp=int(nc['fp']),
        ))

    top_improved = sorted([d for d in deltas if d['delta_f1'] > 0], key=lambda x: (-x['delta_f1'], x['case']))[:top_k]
    top_regressed = sorted([d for d in deltas if d['delta_f1'] < 0], key=lambda x: (x['delta_f1'], x['case']))[:top_k]

    base_stats = _trace_stats(baseline_dir, base_m['cases'])
    curr_stats = _trace_stats(current_dir, curr_m['cases'])

    summary = dict(
        baseline_name=baseline_dir.name,
        current_name=current_dir.name,
        overall_old=base_m['overall'],
        overall_new=curr_m['overall'],
        datasets_old=base_m['datasets'],
        datasets_new=curr_m['datasets'],
        trace_stats_old=base_stats,
        trace_stats_new=curr_stats,
        top_improved=top_improved,
        top_regressed=top_regressed,
        cases=deltas,
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    assets_dir = out_dir / 'assets'
    assets_dir.mkdir(parents=True, exist_ok=True)

    # Render compact visual A/B thumbnails for top improved + top regressed cases
    featured = {d['case']: d for d in [*top_improved, *top_regressed]}
    for cid in featured:
        c_assets = assets_dir / cid
        c_assets.mkdir(parents=True, exist_ok=True)
        q1 = current_dir / 'inputs' / cid / 'query-a.jpg'
        q2 = current_dir / 'inputs' / cid / 'query-b.jpg'
        ref_new = current_dir / 'inputs' / cid / 'reference-view.jpg'
        gt_p = current_dir / 'traces' / cid / 'offline-target.png'
        ign_p = current_dir / 'traces' / cid / 'offline-ignore.png'
        old_pred = baseline_dir / 'traces' / cid / 'prediction.png'
        new_pred = current_dir / 'traces' / cid / 'prediction.png'
        if ref_new.is_file():
            Image.open(ref_new).convert('RGB').save(c_assets / 'ref_new.jpg', quality=85)
        if q1.is_file():
            Image.open(q1).convert('RGB').save(c_assets / 'query_t1.jpg', quality=85)
        if q2.is_file():
            Image.open(q2).convert('RGB').save(c_assets / 'query_t2.jpg', quality=85)
        if q2.is_file() and gt_p.is_file() and old_pred.is_file():
            _render_error_overlay(q2, old_pred, gt_p, ign_p).save(c_assets / 'old_err.jpg', quality=85)
        if q2.is_file() and gt_p.is_file() and new_pred.is_file():
            _render_error_overlay(q2, new_pred, gt_p, ign_p).save(c_assets / 'new_err.jpg', quality=85)

    (out_dir / 'compare.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    (out_dir / 'compare.md').write_text(_render_markdown(summary, current_dir, out_dir), encoding='utf-8')
    (out_dir / 'compare.html').write_text(_render_html(summary, current_dir, out_dir), encoding='utf-8')
    return summary


def _replay_href(current_dir: Path, out_dir: Path, cid: str) -> str:
    replay_file = current_dir / 'replay' / f'{cid}.html'
    if replay_file.is_file():
        return Path(os.path.relpath(replay_file, out_dir)).as_posix()
    return ''


def _render_markdown(summary: dict, current_dir: Path, out_dir: Path) -> str:
    oo, on = summary['overall_old'], summary['overall_new']
    to, tn = summary['trace_stats_old'], summary['trace_stats_new']
    lines = [
        f"# T3.2 Dev-32 对比简报：{summary['baseline_name']} → {summary['current_name']}",
        '',
        '## 一、总体指标对比',
        '',
        '| 指标 | 基线 | 当前 | 变化 |',
        '| :--- | :---: | :---: | :---: |',
    ]
    for label, key in (('Macro IoU', 'macro/iou'), ('Macro F1', 'macro/f1'),
                       ('Positive Macro F1', 'positive_macro/f1'),
                       ('Micro IoU', 'micro/iou'), ('Micro F1', 'micro/f1'),
                       ('Micro Precision', 'micro/precision'), ('Micro Recall', 'micro/recall')):
        sec, metric = key.split('/')
        ov, nv = oo[sec][metric], on[sec][metric]
        lines.append(f'| {label} | {_pct(ov)} | {_pct(nv)} | {_delta_str(nv - ov)} |')
    lines.extend([
        f"| 负例虚警像素 / 全空数 | {oo['negative_false_alarms']} ({oo['negative_empty_predictions']}/{oo['negatives']}) | {on['negative_false_alarms']} ({on['negative_empty_predictions']}/{on['negatives']}) | {on['negative_false_alarms'] - oo['negative_false_alarms']:+d} px |",
        f"| 平均轮次 | {to['avg_rounds']} | {tn['avg_rounds']} | {tn['avg_rounds'] - to['avg_rounds']:+.2f} |",
        f"| 文本分割零检率 | {to['zero_detections']}/{to['segment_calls']} ({_pct(to['zero_rate'])}) | {tn['zero_detections']}/{tn['segment_calls']} ({_pct(tn['zero_rate'])}) | {_delta_str(tn['zero_rate'] - to['zero_rate'])} |",
        '',
        '## 二、Top-5 涨分样本',
        '',
        '| 样本 | 数据集 | 隐藏类名 | 旧 F1 | 新 F1 | 变化 | 轮次 | 回放 |',
        '| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :--- |',
    ])
    for d in summary['top_improved']:
        href = _replay_href(current_dir, out_dir, d['case'])
        link = f"[回放]({href})" if href else '—'
        lines.append(f"| {d['case']} | {d['dataset']} | {d['target_class_name']} | {_pct(d['old_f1'])} | {_pct(d['new_f1'])} | {_delta_str(d['delta_f1'])} | {d['old_rounds']}→{d['new_rounds']} | {link} |")
    lines.extend([
        '',
        '## 三、Top-5 掉分样本',
        '',
        '| 样本 | 数据集 | 隐藏类名 | 旧 F1 | 新 F1 | 变化 | 轮次 | 回放 |',
        '| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :--- |',
    ])
    for d in summary['top_regressed']:
        href = _replay_href(current_dir, out_dir, d['case'])
        link = f"[回放]({href})" if href else '—'
        lines.append(f"| {d['case']} | {d['dataset']} | {d['target_class_name']} | {_pct(d['old_f1'])} | {_pct(d['new_f1'])} | {_delta_str(d['delta_f1'])} | {d['old_rounds']}→{d['new_rounds']} | {link} |")
    return '\n'.join(lines) + '\n'


def _render_cards(items: list[dict], current_dir: Path, out_dir: Path, tone: str) -> str:
    cards = []
    for d in items:
        cid = d['case']
        href = _replay_href(current_dir, out_dir, cid)
        replay_btn = f'<a class="btn" href="{html.escape(href)}">查看完整轨迹回放</a>' if href else ''
        cards.append(f"""
        <section class="card {tone}">
          <div class="card-head">
            <div>
              <strong>{html.escape(cid)}</strong> · {html.escape(d['dataset'])} · 隐藏类：<code>{html.escape(d['target_class_name'])}</code>
              · F1: <b>{_pct(d['old_f1'])} → {_pct(d['new_f1'])} ({_delta_str(d['delta_f1'])})</b>
              · 新 P/R: {_pct(d['new_p'])} / {_pct(d['new_r'])} · 轮次 {d['old_rounds']}→{d['new_rounds']} ({html.escape(d['old_sub'])}→{html.escape(d['new_sub'])})
            </div>
            <div>{replay_btn}</div>
          </div>
          <div class="ref-row"><img src="assets/{cid}/ref_new.jpg" alt="ref" /></div>
          <div class="grid4">
            <figure><img src="assets/{cid}/query_t1.jpg" /><figcaption>Query T1</figcaption></figure>
            <figure><img src="assets/{cid}/query_t2.jpg" /><figcaption>Query T2</figcaption></figure>
            <figure><img src="assets/{cid}/old_err.jpg" /><figcaption>旧基线 (绿TP/红FP/蓝FN) · F1 {_pct(d['old_f1'])}</figcaption></figure>
            <figure><img src="assets/{cid}/new_err.jpg" /><figcaption>当前版 (绿TP/红FP/蓝FN) · F1 {_pct(d['new_f1'])}</figcaption></figure>
          </div>
        </section>""")
    return '\n'.join(cards) if cards else '<p class="muted">无此类样本。</p>'


def _render_html(summary: dict, current_dir: Path, out_dir: Path) -> str:
    oo, on = summary['overall_old'], summary['overall_new']
    to, tn = summary['trace_stats_old'], summary['trace_stats_new']
    ds_rows = []
    for ds in summary['datasets_new']:
        od, nd = summary['datasets_old'][ds], summary['datasets_new'][ds]
        of1, nf1 = od['positive_macro']['f1'], nd['positive_macro']['f1']
        om, nm = od['micro']['f1'], nd['micro']['f1']
        ds_rows.append(
            f"<tr><th>{html.escape(ds)}</th><td>{_pct(om)}</td><td>{_pct(nm)}</td><td>{_delta_str(nm - om)}</td>"
            f"<td>{_pct(of1)}</td><td>{_pct(nf1)}</td><td>{_delta_str(nf1 - of1)}</td></tr>"
        )
    replay_hub = current_dir / 'replay' / 'index.html'
    hub_link = f'<a class="btn" href="{html.escape(Path(os.path.relpath(replay_hub, out_dir)).as_posix())}">打开 32 题完整回放总览</a>' if replay_hub.is_file() else ''
    return f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8" />
<title>T3.2 Dev-32 A/B 对比审阅页</title>
<style>
  body {{ margin: 0; background: #0f141b; color: #e6edf3; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", "PingFang SC", sans-serif; line-height: 1.5; }}
  header {{ padding: 16px 24px; background: #161d27; border-bottom: 1px solid #2b3648; display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 12px; }}
  main {{ max-width: 1460px; margin: 20px auto 60px; padding: 0 20px; }}
  h1, h2 {{ margin: 0 0 12px; }}
  .sub {{ color: #9aa7b8; font-size: 13px; }}
  .btn {{ display: inline-block; padding: 6px 12px; background: #0e7490; color: #fff; text-decoration: none; border-radius: 6px; font-size: 13px; font-weight: 600; }}
  table {{ width: 100%; border-collapse: collapse; margin-bottom: 22px; background: #18202b; border: 1px solid #2b3648; }}
  th, td {{ padding: 8px 12px; border: 1px solid #2b3648; text-align: center; font-size: 13px; }}
  th {{ background: #1f2937; }}
  .card {{ background: #18202b; border: 1px solid #2b3648; border-radius: 10px; margin-bottom: 18px; overflow: hidden; }}
  .card.up {{ border-left: 4px solid #22c55e; }}
  .card.down {{ border-left: 4px solid #ef4444; }}
  .card-head {{ padding: 10px 14px; background: #1f2937; display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 8px; font-size: 13px; }}
  .ref-row {{ padding: 10px 14px 4px; }}
  .ref-row img {{ max-width: 100%; max-height: 220px; border-radius: 6px; border: 1px solid #2b3648; }}
  .grid4 {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 10px; padding: 10px 14px 14px; }}
  figure {{ margin: 0; background: #0b0f14; border: 1px solid #2b3648; border-radius: 6px; overflow: hidden; }}
  figure img {{ width: 100%; display: block; }}
  figcaption {{ font-size: 11px; color: #9aa7b8; text-align: center; padding: 4px; }}
  .muted {{ color: #9aa7b8; }}
</style>
</head>
<body>
<header>
  <div>
    <h1>T3.2 Dev-32 A/B 对比审阅页</h1>
    <div class="sub">基线：{html.escape(summary['baseline_name'])} → 当前：{html.escape(summary['current_name'])} · Macro F1 {_pct(oo['macro']['f1'])} → {_pct(on['macro']['f1'])} ({_delta_str(on['macro']['f1'] - oo['macro']['f1'])}) · 平均轮次 {to['avg_rounds']} → {tn['avg_rounds']}</div>
  </div>
  <div>{hub_link}</div>
</header>
<main>
  <h2>分数据集对比</h2>
  <table>
    <thead><tr><th>数据集</th><th>旧 Micro F1</th><th>新 Micro F1</th><th>Micro 变化</th><th>旧正例 Macro F1</th><th>新正例 Macro F1</th><th>正例 Macro 变化</th></tr></thead>
    <tbody>{''.join(ds_rows)}</tbody>
  </table>
  <h2>Top-5 涨分样本</h2>
  {_render_cards(summary['top_improved'], current_dir, out_dir, 'up')}
  <h2>Top-5 掉分样本（优先人工排查）</h2>
  {_render_cards(summary['top_regressed'], current_dir, out_dir, 'down')}
</main>
</body>
</html>
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', type=Path, required=True, help='Baseline Dev-32 run directory')
    parser.add_argument('--current', type=Path, required=True, help='Current Dev-32 run directory')
    parser.add_argument('--out', type=Path, help='Output directory (default: <current>/compare)')
    parser.add_argument('--top-k', type=int, default=5)
    args = parser.parse_args()
    out_dir = args.out.resolve() if args.out else (args.current.resolve() / 'compare')
    summary = build_comparison(args.baseline.resolve(), args.current.resolve(), out_dir, top_k=args.top_k)
    oo, on = summary['overall_old'], summary['overall_new']
    print(
        f"COMPARE_DONE macro_f1={oo['macro']['f1']:.4f}->{on['macro']['f1']:.4f} "
        f"pos_macro_f1={oo['positive_macro']['f1']:.4f}->{on['positive_macro']['f1']:.4f} "
        f"micro_f1={oo['micro']['f1']:.4f}->{on['micro']['f1']:.4f} out={out_dir}",
        flush=True,
    )


if __name__ == '__main__':
    main()
