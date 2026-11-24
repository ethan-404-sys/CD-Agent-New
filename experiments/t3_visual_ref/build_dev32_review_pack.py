#!/usr/bin/env python3
"""Build a self-contained offline manual review & annotation pack for T3.2 Dev-32."""
from __future__ import annotations

import argparse
import csv
import html
import json
import shutil
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter


THING_CLASSES = frozenset({
    'building', 'buildings', 'structure', 'structures', 'playground',
})

STUFF_CLASSES = frozenset({
    'bare land', 'bareland', 'n.v.g surface', 'farmland', 'agriculture land',
    'planting land', 'grass', 'low vegetation', 'vegetation', 'tree',
    'forest and grass cover', 'water', 'road', 'railway and road',
    'artificial excavation', 'developed space',
})


def classify_thing_or_stuff(class_name: str) -> str:
    key = ' '.join((class_name or '').strip().lower().split())
    if key in THING_CLASSES:
        return 'thing'
    if key in STUFF_CLASSES:
        return 'stuff'
    return ''


def render_clean_3panel(reference: Image.Image, mask: Image.Image, tile_size: int = 384, header_h: int = 28) -> Image.Image:
    ref_rgb = reference.convert('RGB')
    mask_l = mask.convert('L')
    mask_bool = np.asarray(mask_l) != 0
    ref_arr = np.asarray(ref_rgb, dtype=np.float32)

    overlay = ref_arr.copy()
    tint = np.array([0.0, 230.0, 118.0], dtype=np.float32)
    overlay[mask_bool] = overlay[mask_bool] * 0.60 + tint * 0.40
    eroded = np.asarray(mask_l.filter(ImageFilter.MinFilter(3))) != 0
    inner_edge = mask_bool & ~eroded
    overlay[inner_edge] = np.array([0.0, 255.0, 140.0], dtype=np.float32)
    overlay_img = Image.fromarray(np.clip(overlay, 0, 255).astype(np.uint8))

    binary_arr = np.where(mask_bool, 255, 0).astype(np.uint8)
    binary_img = Image.fromarray(np.stack([binary_arr] * 3, axis=-1))

    panels = [
        ('1. Original Reference', ref_rgb.resize((tile_size, tile_size), Image.Resampling.BILINEAR)),
        ('2. Green = TARGET (unshaded = NOT)', overlay_img.resize((tile_size, tile_size), Image.Resampling.BILINEAR)),
        ('3. Mask: WHITE = TARGET, BLACK = NOT', binary_img.resize((tile_size, tile_size), Image.Resampling.NEAREST)),
    ]
    gap = 6
    canvas = Image.new('RGB', (tile_size * 3 + gap * 2, header_h + tile_size), (24, 28, 34))
    draw = ImageDraw.Draw(canvas)
    for idx, (title, img) in enumerate(panels):
        x0 = idx * (tile_size + gap)
        draw.rectangle([x0, 0, x0 + tile_size - 1, header_h - 1], fill=(32, 38, 46))
        draw.text((x0 + 8, 8), title, fill=(240, 245, 250))
        canvas.paste(img, (x0, header_h))
    return canvas


def render_gt_overlay(query_t2: Image.Image, gt_mask: np.ndarray, ignore_mask: np.ndarray | None = None) -> Image.Image:
    base = np.asarray(query_t2.convert('RGB'), dtype=np.float32)
    out = base.copy()
    if ignore_mask is not None and ignore_mask.any():
        ign_color = np.array([255.0, 214.0, 0.0], dtype=np.float32)
        out[ignore_mask] = out[ignore_mask] * 0.65 + ign_color * 0.35
    if gt_mask.any():
        gt_color = np.array([0.0, 230.0, 118.0], dtype=np.float32)
        out[gt_mask] = out[gt_mask] * 0.55 + gt_color * 0.45
        gt_img = Image.fromarray(gt_mask.astype(np.uint8) * 255)
        edge = gt_mask & (np.asarray(gt_img.filter(ImageFilter.MinFilter(3))) == 0)
        out[edge] = np.array([0.0, 255.0, 128.0], dtype=np.float32)
    return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8))


def render_pred_error_overlay(query_t2: Image.Image, pred_mask: np.ndarray, gt_mask: np.ndarray, ignore_mask: np.ndarray | None = None) -> Image.Image:
    base = np.asarray(query_t2.convert('RGB'), dtype=np.float32)
    out = base.copy()
    valid = ~ignore_mask if ignore_mask is not None else np.ones_like(gt_mask, dtype=bool)
    tp = pred_mask & gt_mask & valid
    fp = pred_mask & (~gt_mask) & valid
    fn = (~pred_mask) & gt_mask & valid
    for region, color in (
        (tp, np.array([0.0, 230.0, 118.0], dtype=np.float32)),   # green = TP
        (fp, np.array([255.0, 68.0, 68.0], dtype=np.float32)),   # red = FP
        (fn, np.array([64.0, 156.0, 255.0], dtype=np.float32)),  # blue = FN
    ):
        if region.any():
            out[region] = out[region] * 0.50 + color * 0.50
    return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8))


README_MD = """# T3.2 Dev-32 视觉参考开发集 · 人工审批与标注说明

> 说明：已核查仓库与历史产物，当前单期跨图视觉参考（Form 1）开发集共 **32 题（Dev-32，`case-01` ~ `case-32`）**，不存在 `dev36`；本压缩包已完整打包全部 32 题的图像、基线预测对比与离线交互打标页。

## 一、怎么用（推荐方式一）

### 方式一：离线网页直接打标（最省事）
1. 解压本压缩包后，直接用浏览器（Chrome / Edge / Safari）打开根目录下的 `index.html`（**完全离线，无需联网**）。
2. 建议先打开顶部的 **「盲审模式」开关**：此时会隐藏真实类别名、GT 叠加图与基线分数；先只看「修复后 3 联参考图 + Query T1/T2」判断参考图清晰度与潜在歧义，再点击单卡上的 **「揭晓真值与基线」** 按钮完成其余字段打标。
3. 点击任意缩略图可全屏放大查看（按 `Esc` 或点击空白关闭）。
4. 每次勾选都会自动保存在浏览器本地 `localStorage`（刷新不丢）；打完后点击顶部 **「导出 JSON」** 或 **「导出 CSV」**，把导出的文件发回即可。

### 方式二：用 Excel / Numbers 直接编辑表格
- 打开根目录下的 `annotations_template.csv`（UTF-8 带 BOM，Excel 打开不会中文乱码），对照 `assets/case-XX/` 中的图片填写右侧 6 个标注列并保存。

---

## 二、图像说明（每题 `assets/case-XX/` 下 6 张图）

| 文件 | 含义 |
| :--- | :--- |
| `ref_clean_3panel.jpg` | **新版无歧义 3 联参考图**（左：原图；中：仅前景内部绿色半透明着色，未着色区域与孔洞均非目标；右：纯黑白掩码，白色=目标类，黑色=非目标） |
| `ref_old_spotlight.jpg` | **旧版青蓝聚光灯参考图**（供复盘旧版“图底反转”现象，如 `case-20/23/29/32`） |
| `query_t1.jpg` / `query_t2.jpg` | 查询双时相原图（512×512） |
| `query_t2_gt_overlay.jpg` | 查询真值 GT 叠加在 T2 上（绿色=前景变化，黄色=ignore 255 忽略像素） |
| `query_t2_pred_overlay.jpg` | 旧基线预测错漏叠加在 T2 上（**绿色=TP 命中，红色=FP 误检，蓝色=FN 漏检**） |

---

## 三、核心标注字段与取值口径

### 1. 只读元数据（已自动填充，无需修改）
- `case`：样本编号（`case-01` ~ `case-32`）。
- `dataset`：来源数据集（`HIUCDMINI_tile512` / `SECOND` / `HRMS_SCD` / `SCSCD7` / `lsSCD`）。
- `target_class_name`：原始数据集隐藏类名（推理时模型看不到，仅用于事后审计）。
- `area_bin` / `gt_pixels` / `is_negative`：真值前景面积档、前景像素数、是否为空真值负例（共 3 题：`case-02 / 22 / 25`）。
- `baseline_f1 / precision / recall / submitted_by`：旧版 Dev-32 基线实跑得分与提交状态。

### 2. 需审批/标注的 7 个核心字段

| 字段名 | 含义 | 可选值与判定标准 |
| :--- | :--- | :--- |
| **`thing_or_stuff`** | 地物形态二分（**已按类名自动预填，仅需抽查微调**） | - `thing`：可数离散实体，有明确闭合个体边界（建筑、构筑物、操场等）<br>- `stuff`：不可数连续铺展或网状地物（裸地、农田、草地/植被、水体、道路、开挖区等） |
| **`ref_clarity`** | **参考图前景清晰度**（只看 3 联参考图，能否唯一认出目标是什么地类） | - `clear`：参考前景清晰，人眼一眼能认出属于哪类地物<br>- `ambiguous`：前景纹理模糊、面积过碎或混杂多种地物，较难单靠参考图确认<br>- `bad_mask`：参考掩码本身疑似错标/错位/异常 |
| **`appearance_gap`** | **参考图与查询真值目标的跨场景外观跨度** | - `low`：外观高度同质（同材质/同风格，如住宅→住宅、标准跑道→跑道）<br>- `medium`：有一定色差、季节或尺度差异，但仍明显属同类<br>- `high`：大类相同但子类外观迥异（如高层住宅参考 → 蓝顶工业厂房查询）<br>- `na`：空真值负例（查询图无真实变化目标） |
| **`granularity_ambiguity`** | **单示例粒度歧义**（是否存在合理的“多种粒度理解”） | - `none`：按土地覆盖大类口径目标明确无歧义<br>- `subtype_vs_category`：存在“只找同外观子类 vs 找整个大类”的合理分歧（如住宅 vs 工业厂房、大棚 vs 厂房）<br>- `multi_interpretation`：不看隐藏标签，人类也会基于参考图选出截然不同的查询变化区 |
| **`gt_quality`** | **查询真值 GT 标注质量** | - `ok`：真值准确可靠（含正确空真值负例）<br>- `minor_noise`：主体变化正确，但边缘粗糙或有少量碎斑/小漏标<br>- `problematic`：明显漏标主体、大面积错标、或与参考类别明显不符 |
| **`keep_in_dev`** | **开发集准入裁决（最终结论）** | - `keep_clean`：质量过关，收入干净核心开发集<br>- `keep_hard`：真值无误但属高难/跨外观/多粒度难例，保留并归入困难分层<br>- `drop`：数据本身质量有问题（参考错标或 GT 明显错误），建议剔除 |
| **`notes`** | **自由备注（可选）** | 记录具体错标位置、建议替换理由、或典型错误观察 |
"""


def build_html(records: list[dict]) -> str:
    data_json = json.dumps(records, ensure_ascii=False)
    return f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>T3.2 Dev-32 视觉参考开发集 · 人工审批与标注台</title>
<style>
  :root {{
    --bg: #0f141b; --card: #18202b; --panel: #1f2937; --border: #2e3b4e;
    --text: #e6edf3; --muted: #9aa7b8; --accent: #22d3ee; --green: #22c55e;
    --red: #ef4444; --amber: #f59e0b; --blue: #60a5fa;
  }}
  * {{ box-sizing: border-box; }}
  body {{ margin: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", "PingFang SC", "Microsoft YaHei", sans-serif; background: var(--bg); color: var(--text); line-height: 1.5; }}
  header {{ position: sticky; top: 0; z-index: 50; background: rgba(15, 20, 27, 0.96); backdrop-filter: blur(8px); border-bottom: 1px solid var(--border); padding: 12px 20px; }}
  .topbar {{ display: flex; flex-wrap: wrap; align-items: center; justify-content: space-between; gap: 12px; max-width: 1560px; margin: 0 auto; }}
  h1 {{ margin: 0; font-size: 18px; font-weight: 600; }}
  .sub {{ font-size: 12px; color: var(--muted); }}
  .controls {{ display: flex; flex-wrap: wrap; align-items: center; gap: 8px; }}
  button, .file-btn {{ background: var(--panel); color: var(--text); border: 1px solid var(--border); border-radius: 6px; padding: 6px 12px; font-size: 13px; cursor: pointer; }}
  button:hover, .file-btn:hover {{ border-color: var(--accent); }}
  button.primary {{ background: #0e7490; border-color: #06b6d4; color: #fff; font-weight: 600; }}
  button.active {{ background: #1e3a5f; border-color: var(--accent); color: #fff; }}
  .badge {{ display: inline-block; padding: 2px 8px; border-radius: 999px; font-size: 12px; font-weight: 600; border: 1px solid var(--border); background: var(--panel); }}
  .badge.low {{ color: #fca5a5; border-color: #7f1d1d; background: #2b1414; }}
  .badge.mid {{ color: #fcd34d; border-color: #78350f; background: #291e0d; }}
  .badge.high {{ color: #86efac; border-color: #14532d; background: #11261a; }}
  .badge.neg {{ color: #c4b5fd; border-color: #4c1d95; background: #1f1633; }}
  .badge.done {{ color: #86efac; border-color: #16a34a; background: #0f291a; }}
  main {{ max-width: 1560px; margin: 16px auto 60px; padding: 0 20px; }}
  .guide {{ background: var(--card); border: 1px solid var(--border); border-radius: 10px; padding: 12px 16px; margin-bottom: 16px; font-size: 13px; color: var(--muted); }}
  .guide summary {{ cursor: pointer; color: var(--text); font-weight: 600; }}
  .guide ul {{ margin: 8px 0 0; padding-left: 18px; }}
  .card {{ background: var(--card); border: 1px solid var(--border); border-radius: 12px; margin-bottom: 18px; overflow: hidden; }}
  .card.complete {{ border-left: 4px solid var(--green); }}
  .card-head {{ display: flex; flex-wrap: wrap; align-items: center; justify-content: space-between; gap: 10px; padding: 10px 16px; background: var(--panel); border-bottom: 1px solid var(--border); }}
  .card-title {{ display: flex; flex-wrap: wrap; align-items: center; gap: 8px; font-size: 14px; font-weight: 600; }}
  .card-body {{ display: grid; grid-template-columns: 1.65fr 1fr; gap: 16px; padding: 14px 16px; }}
  @media (max-width: 1120px) {{ .card-body {{ grid-template-columns: 1fr; }} }}
  .img-section {{ display: flex; flex-direction: column; gap: 10px; }}
  .ref-banner img {{ width: 100%; border-radius: 8px; border: 1px solid var(--border); cursor: zoom-in; display: block; }}
  .img-grid {{ display: grid; grid-template-columns: repeat(5, 1fr); gap: 8px; }}
  @media (max-width: 900px) {{ .img-grid {{ grid-template-columns: repeat(2, 1fr); }} }}
  figure {{ margin: 0; background: #0b0f14; border: 1px solid var(--border); border-radius: 8px; overflow: hidden; }}
  figure img {{ width: 100%; aspect-ratio: 1 / 1; object-fit: contain; display: block; cursor: zoom-in; }}
  figcaption {{ font-size: 11px; color: var(--muted); padding: 4px 6px; text-align: center; border-top: 1px solid var(--border); }}
  .blind-mask {{ display: flex; align-items: center; justify-content: center; aspect-ratio: 1 / 1; font-size: 12px; color: var(--muted); background: #101620; padding: 8px; text-align: center; }}
  .form-panel {{ display: flex; flex-direction: column; gap: 10px; background: #121821; border: 1px solid var(--border); border-radius: 10px; padding: 12px 14px; }}
  .field {{ display: flex; flex-direction: column; gap: 5px; }}
  .field-label {{ font-size: 12px; font-weight: 600; color: #cbd5e1; }}
  .field-hint {{ font-weight: 400; color: var(--muted); font-size: 11px; margin-left: 4px; }}
  .pills {{ display: flex; flex-wrap: wrap; gap: 6px; }}
  .pill {{ padding: 4px 9px; border-radius: 6px; border: 1px solid var(--border); background: var(--card); color: var(--muted); font-size: 12px; cursor: pointer; user-select: none; }}
  .pill:hover {{ border-color: var(--accent); color: var(--text); }}
  .pill.selected {{ background: #0e7490; border-color: #22d3ee; color: #fff; font-weight: 600; }}
  .pill.selected.warn {{ background: #92400e; border-color: #f59e0b; }}
  .pill.selected.danger {{ background: #991b1b; border-color: #ef4444; }}
  .pill.selected.ok {{ background: #166534; border-color: #22c55e; }}
  textarea {{ width: 100%; min-height: 54px; resize: vertical; background: var(--bg); color: var(--text); border: 1px solid var(--border); border-radius: 6px; padding: 6px 8px; font-size: 12px; font-family: inherit; }}
  .modal {{ position: fixed; inset: 0; background: rgba(0,0,0,0.88); display: none; align-items: center; justify-content: center; flex-direction: column; z-index: 100; padding: 20px; }}
  .modal.open {{ display: flex; }}
  .modal img {{ max-width: 95vw; max-height: 86vh; border: 1px solid var(--border); border-radius: 8px; }}
  .modal-cap {{ margin-top: 10px; font-size: 13px; color: #e2e8f0; }}
</style>
</head>
<body>
<header>
  <div class="topbar">
    <div>
      <h1>T3.2 Dev-32 视觉参考开发集 · 人工审批与标注台</h1>
      <div class="sub">共 32 题（Form 1 单期跨图参考）· 标注自动保存至本机浏览器 · 支持盲审与导出 CSV/JSON</div>
    </div>
    <div class="controls">
      <span id="progressBadge" class="badge">已标注 0 / 32</span>
      <button id="blindToggle" onclick="toggleBlindMode()">盲审模式：关</button>
      <button class="filter-btn active" data-filter="all" onclick="setFilter('all')">全部 (32)</button>
      <button class="filter-btn" data-filter="low" onclick="setFilter('low')">极低分 F1&lt;0.1</button>
      <button class="filter-btn" data-filter="midhigh" onclick="setFilter('midhigh')">中高分 F1≥0.4</button>
      <button class="filter-btn" data-filter="neg" onclick="setFilter('neg')">负例 (3)</button>
      <button class="filter-btn" data-filter="todo" onclick="setFilter('todo')">未完成</button>
      <label class="file-btn">导入 JSON<input type="file" accept=".json" style="display:none" onchange="importJSON(event)" /></label>
      <button class="primary" onclick="exportJSON()">导出 JSON</button>
      <button class="primary" onclick="exportCSV()">导出 CSV</button>
    </div>
  </div>
</header>
<main>
  <details class="guide" open>
    <summary>标注口径速查（点击折叠；完整说明见 README_标注说明.md）</summary>
    <ul>
      <li><b>thing_or_stuff</b>（已按类名预填）：<code>thing</code>=可数离散实体（建筑/构筑物/操场）；<code>stuff</code>=连续铺展或网状地物（裸地/农田/草地植被/水体/道路/开挖区）。</li>
      <li><b>ref_clarity</b>：<code>clear</code>=参考前景清晰可唯一识别；<code>ambiguous</code>=含混/混杂多种地物；<code>bad_mask</code>=参考掩码疑似错标。</li>
      <li><b>appearance_gap</b>：<code>low</code>=同质外观；<code>medium</code>=中等色差/季节差；<code>high</code>=大类同但子类外观迥异（如住宅→蓝顶厂房）；<code>na</code>=负例无变化目标。</li>
      <li><b>granularity_ambiguity</b>：<code>none</code>=大类口径下无歧义；<code>subtype_vs_category</code>=子类 vs 大类合理分歧；<code>multi_interpretation</code>=不看标签人也会选不同区域。</li>
      <li><b>gt_quality</b>：<code>ok</code>=准确；<code>minor_noise</code>=边缘粗糙/小碎斑；<code>problematic</code>=明显错标或大面积漏标。</li>
      <li><b>keep_in_dev</b>：<code>keep_clean</code>=收入干净核心集；<code>keep_hard</code>=保留入高难/跨外观分层；<code>drop</code>=建议剔除。</li>
    </ul>
  </details>
  <div id="cards"></div>
</main>
<div id="modal" class="modal" onclick="closeModal()">
  <img id="modalImg" src="" alt="" />
  <div id="modalCap" class="modal-cap"></div>
</div>
<script>
const CASES = {data_json};
const STORAGE_KEY = 't32_dev32_manual_review_v1';
const FIELDS = [
  {{ key: 'thing_or_stuff', label: '1. thing_or_stuff', hint: '已按类名预填，可微调', options: [
    ['thing', 'thing（离散实体）', ''], ['stuff', 'stuff（连续地物）', '']
  ]}},
  {{ key: 'ref_clarity', label: '2. ref_clarity', hint: '参考前景清晰度', options: [
    ['clear', 'clear（清晰）', 'ok'], ['ambiguous', 'ambiguous（含混）', 'warn'], ['bad_mask', 'bad_mask（错标）', 'danger']
  ]}},
  {{ key: 'appearance_gap', label: '3. appearance_gap', hint: '参考 vs 查询外观跨度', options: [
    ['low', 'low（同质）', 'ok'], ['medium', 'medium（中等）', ''], ['high', 'high（跨子类迥异）', 'warn'], ['na', 'na（负例）', '']
  ]}},
  {{ key: 'granularity_ambiguity', label: '4. granularity_ambiguity', hint: '单示例粒度歧义', options: [
    ['none', 'none（无歧义）', 'ok'], ['subtype_vs_category', 'subtype_vs_category（子类/大类分歧）', 'warn'], ['multi_interpretation', 'multi_interpretation（多解）', 'danger']
  ]}},
  {{ key: 'gt_quality', label: '5. gt_quality', hint: '查询真值 GT 质量', options: [
    ['ok', 'ok（准确）', 'ok'], ['minor_noise', 'minor_noise（边缘/碎斑噪点）', 'warn'], ['problematic', 'problematic（明显错漏）', 'danger']
  ]}},
  {{ key: 'keep_in_dev', label: '6. keep_in_dev', hint: '开发集准入裁决', options: [
    ['keep_clean', 'keep_clean（干净核心）', 'ok'], ['keep_hard', 'keep_hard（困难分层）', 'warn'], ['drop', 'drop（建议剔除）', 'danger']
  ]}}
];

let state = {{}};
let currentFilter = 'all';
let blindMode = false;
let revealed = {{}};

function initState() {{
  try {{ state = JSON.parse(localStorage.getItem(STORAGE_KEY) || '{{}}'); }} catch (e) {{ state = {{}}; }}
  for (const c of CASES) {{
    if (!state[c.case]) {{
      state[c.case] = {{
        thing_or_stuff: c.thing_or_stuff || '',
        ref_clarity: '',
        appearance_gap: c.is_negative ? 'na' : '',
        granularity_ambiguity: '',
        gt_quality: '',
        keep_in_dev: '',
        notes: ''
      }};
    }}
  }}
  saveState();
}}

function saveState() {{
  localStorage.setItem(STORAGE_KEY, JSON.stringify(state));
  updateProgress();
}}

function isComplete(cid) {{
  const s = state[cid] || {{}};
  return FIELDS.every(f => Boolean(s[f.key]));
}}

function updateProgress() {{
  const done = CASES.filter(c => isComplete(c.case)).length;
  const el = document.getElementById('progressBadge');
  el.textContent = `已标注 ${{done}} / ${{CASES.length}}`;
  el.className = 'badge' + (done === CASES.length ? ' done' : '');
}}

function setField(cid, key, val) {{
  state[cid][key] = val;
  saveState();
  render();
}}

function setNotes(cid, val) {{
  state[cid].notes = val;
  localStorage.setItem(STORAGE_KEY, JSON.stringify(state));
}}

function toggleBlindMode() {{
  blindMode = !blindMode;
  if (!blindMode) revealed = {{}};
  document.getElementById('blindToggle').textContent = blindMode ? '盲审模式：开（隐藏GT）' : '盲审模式：关';
  document.getElementById('blindToggle').className = blindMode ? 'active' : '';
  render();
}}

function revealCase(cid) {{
  revealed[cid] = !revealed[cid];
  render();
}}

function setFilter(f) {{
  currentFilter = f;
  document.querySelectorAll('.filter-btn').forEach(b => b.classList.toggle('active', b.dataset.filter === f));
  render();
}}

function matchesFilter(c) {{
  if (currentFilter === 'low') return !c.is_negative && c.baseline_f1 < 0.10;
  if (currentFilter === 'midhigh') return c.baseline_f1 >= 0.40;
  if (currentFilter === 'neg') return c.is_negative;
  if (currentFilter === 'todo') return !isComplete(c.case);
  return true;
}}

function openModal(src, cap) {{
  document.getElementById('modalImg').src = src;
  document.getElementById('modalCap').textContent = cap;
  document.getElementById('modal').classList.add('open');
}}

function closeModal() {{
  document.getElementById('modal').classList.remove('open');
}}

document.addEventListener('keydown', e => {{ if (e.key === 'Escape') closeModal(); }});

function scoreBadge(c) {{
  if (c.is_negative) return `<span class="badge neg">空真值负例 · 虚警 ${{c.false_alarm_pixels || 0}}px</span>`;
  const cls = c.baseline_f1 < 0.1 ? 'low' : (c.baseline_f1 < 0.5 ? 'mid' : 'high');
  return `<span class="badge ${{cls}}">基线 F1=${{c.baseline_f1.toFixed(2)}} (P=${{c.baseline_precision.toFixed(2)}} R=${{c.baseline_recall.toFixed(2)}}) · ${{c.baseline_submitted_by}}</span>`;
}}

function render() {{
  const container = document.getElementById('cards');
  container.innerHTML = '';
  for (const c of CASES) {{
    if (!matchesFilter(c)) continue;
    const s = state[c.case];
    const done = isComplete(c.case);
    const hideGT = blindMode && !revealed[c.case];
    const card = document.createElement('section');
    card.className = 'card' + (done ? ' complete' : '');

    const fieldsHtml = FIELDS.map(f => `
      <div class="field">
        <div class="field-label">${{f.label}}<span class="field-hint">${{f.hint}}</span></div>
        <div class="pills">
          ${{f.options.map(([val, txt, tone]) => `
            <span class="pill ${{s[f.key] === val ? 'selected ' + tone : ''}}" onclick="setField('${{c.case}}','${{f.key}}','${{val}}')">${{txt}}</span>
          `).join('')}}
        </div>
      </div>
    `).join('');

    const gtTile = hideGT
      ? `<div class="blind-mask">盲审隐藏 GT<br/>点击右上角「揭晓真值」查看</div>`
      : `<img src="assets/${{c.case}}/query_t2_gt_overlay.jpg" onclick="openModal(this.src,'${{c.case}} GT 叠加 (T2) · 类名: ${{c.target_class_name}}')" />`;

    const predTile = hideGT
      ? `<div class="blind-mask">盲审隐藏基线错漏图</div>`
      : `<img src="assets/${{c.case}}/query_t2_pred_overlay.jpg" onclick="openModal(this.src,'${{c.case}} 基线预测错漏 (绿TP / 红FP / 蓝FN) · F1=${{c.baseline_f1.toFixed(3)}}')" />`;

    card.innerHTML = `
      <div class="card-head">
        <div class="card-title">
          <span>${{c.case}}</span>
          <span class="badge">${{c.dataset}}</span>
          ${{hideGT ? '<span class="badge">类名已盲审隐藏</span>' : `<span class="badge">隐藏类名：${{c.target_class_name}}</span>`}}
          <span class="badge">GT面积：${{c.gt_pixels}}px (${{c.area_bin}})</span>
          ${{hideGT ? '' : scoreBadge(c)}}
          ${{done ? '<span class="badge done">已完成</span>' : ''}}
        </div>
        <div>
          ${{blindMode ? `<button onclick="revealCase('${{c.case}}')">${{hideGT ? '揭晓真值与基线' : '重新隐藏'}}</button>` : ''}}
        </div>
      </div>
      <div class="card-body">
        <div class="img-section">
          <div class="ref-banner">
            <img src="assets/${{c.case}}/ref_clean_3panel.jpg" alt="clean 3-panel ref" onclick="openModal(this.src,'${{c.case}} 新版无歧义 3 联参考图')" />
          </div>
          <div class="img-grid">
            <figure>
              <img src="assets/${{c.case}}/ref_old_spotlight.jpg" onclick="openModal(this.src,'${{c.case}} 旧版青蓝聚光灯参考图')" />
              <figcaption>旧版聚光灯参考</figcaption>
            </figure>
            <figure>
              <img src="assets/${{c.case}}/query_t1.jpg" onclick="openModal(this.src,'${{c.case}} Query T1')" />
              <figcaption>Query T1</figcaption>
            </figure>
            <figure>
              <img src="assets/${{c.case}}/query_t2.jpg" onclick="openModal(this.src,'${{c.case}} Query T2')" />
              <figcaption>Query T2</figcaption>
            </figure>
            <figure>
              ${{gtTile}}
              <figcaption>GT 叠加 (绿=变化)</figcaption>
            </figure>
            <figure>
              ${{predTile}}
              <figcaption>基线错漏 (绿TP/红FP/蓝FN)</figcaption>
            </figure>
          </div>
        </div>
        <div class="form-panel">
          ${{fieldsHtml}}
          <div class="field">
            <div class="field-label">7. notes<span class="field-hint">备注（可选）</span></div>
            <textarea placeholder="记录错标位置、歧义原因或处理建议..." oninput="setNotes('${{c.case}}', this.value)">${{(s.notes || '').replace(/</g, '&lt;')}}</textarea>
          </div>
        </div>
      </div>
    `;
    container.appendChild(card);
  }}
  updateProgress();
}}

function mergedRows() {{
  return CASES.map(c => Object.assign({{}}, c, state[c.case] || {{}}));
}}

function downloadFile(filename, content, mime) {{
  const blob = new Blob([content], {{ type: mime }});
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = filename;
  a.click();
  URL.revokeObjectURL(a.href);
}}

function exportJSON() {{
  downloadFile('dev32_annotations.json', JSON.stringify({{ version: 't32_dev32_review_v1', count: CASES.length, cases: mergedRows() }}, null, 2) + '\\n', 'application/json;charset=utf-8');
}}

function exportCSV() {{
  const cols = ['case','dataset','target_class_name','area_bin','gt_pixels','is_negative','baseline_f1','baseline_precision','baseline_recall','baseline_submitted_by','thing_or_stuff','ref_clarity','appearance_gap','granularity_ambiguity','gt_quality','keep_in_dev','notes'];
  const esc = v => '"' + String(v ?? '').replace(/"/g, '""') + '"';
  const lines = [cols.join(','), ...mergedRows().map(r => cols.map(k => esc(r[k])).join(','))];
  downloadFile('dev32_annotations.csv', '\\uFEFF' + lines.join('\\n') + '\\n', 'text/csv;charset=utf-8');
}}

function importJSON(ev) {{
  const file = ev.target.files[0];
  if (!file) return;
  const reader = new FileReader();
  reader.onload = () => {{
    try {{
      const parsed = JSON.parse(reader.result);
      const rows = Array.isArray(parsed) ? parsed : (parsed.cases || []);
      for (const r of rows) {{
        if (r.case && state[r.case]) {{
          for (const k of ['thing_or_stuff','ref_clarity','appearance_gap','granularity_ambiguity','gt_quality','keep_in_dev','notes']) {{
            if (r[k] !== undefined) state[r.case][k] = r[k];
          }}
        }}
      }}
      saveState();
      render();
      alert('导入成功！');
    }} catch (err) {{
      alert('JSON 解析失败：' + err.message);
    }}
  }};
  reader.readAsText(file, 'utf-8');
}}

initState();
render();
</script>
</body>
</html>
"""


def build_pack(source: Path, out: Path, zip_path: Path | None = None) -> None:
    metrics = json.loads((source / 'metrics.json').read_text(encoding='utf-8'))
    cases = metrics['cases']
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)
    assets_root = out / 'assets'
    assets_root.mkdir(parents=True, exist_ok=True)

    records = []
    for row in cases:
        cid = row['case']
        in_dir = source / 'inputs' / cid
        tr_dir = source / 'traces' / cid
        dst = assets_root / cid
        dst.mkdir(parents=True, exist_ok=True)

        ref_img = Image.open(in_dir / 'reference.png').convert('RGB')
        ref_mask = Image.open(in_dir / 'reference-mask.png').convert('L')
        t1_img = Image.open(in_dir / 'query-a.png').convert('RGB')
        t2_img = Image.open(in_dir / 'query-b.png').convert('RGB')
        gt_mask = np.asarray(Image.open(tr_dir / 'offline-target.png')) > 0
        ign_path = tr_dir / 'offline-ignore.png'
        ign_mask = (np.asarray(Image.open(ign_path)) > 0) if ign_path.exists() else None
        pred_mask = np.asarray(Image.open(tr_dir / 'prediction.png')) > 0

        shutil.copyfile(in_dir / 'reference-view.jpg', dst / 'ref_old_spotlight.jpg')
        render_clean_3panel(ref_img, ref_mask).save(dst / 'ref_clean_3panel.jpg', quality=88)
        t1_img.save(dst / 'query_t1.jpg', quality=88)
        t2_img.save(dst / 'query_t2.jpg', quality=88)
        render_gt_overlay(t2_img, gt_mask, ign_mask).save(dst / 'query_t2_gt_overlay.jpg', quality=88)
        render_pred_error_overlay(t2_img, pred_mask, gt_mask, ign_mask).save(dst / 'query_t2_pred_overlay.jpg', quality=88)

        records.append(dict(
            case=cid,
            dataset=row['dataset'],
            target_class_name=row['target_class_name'],
            area_bin=row['area_bin'],
            gt_pixels=int(row['gt_pixels']),
            is_negative=bool(row['is_negative']),
            false_alarm_pixels=row.get('false_alarm_pixels'),
            baseline_f1=round(float(row['f1']), 4),
            baseline_precision=round(float(row['precision']), 4),
            baseline_recall=round(float(row['recall']), 4),
            baseline_submitted_by=row['submitted_by'],
            thing_or_stuff=classify_thing_or_stuff(row['target_class_name']),
            ref_clarity='',
            appearance_gap='na' if row['is_negative'] else '',
            granularity_ambiguity='',
            gt_quality='',
            keep_in_dev='',
            notes='',
        ))

    (out / 'README_标注说明.md').write_text(README_MD.strip() + '\n', encoding='utf-8')
    (out / 'index.html').write_text(build_html(records), encoding='utf-8')
    (out / 'annotations_template.json').write_text(
        json.dumps(dict(version='t32_dev32_review_v1', count=len(records), cases=records), ensure_ascii=False, indent=2) + '\n',
        encoding='utf-8',
    )
    csv_cols = [
        'case', 'dataset', 'target_class_name', 'area_bin', 'gt_pixels', 'is_negative',
        'baseline_f1', 'baseline_precision', 'baseline_recall', 'baseline_submitted_by',
        'thing_or_stuff', 'ref_clarity', 'appearance_gap', 'granularity_ambiguity',
        'gt_quality', 'keep_in_dev', 'notes',
    ]
    with (out / 'annotations_template.csv').open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=csv_cols)
        writer.writeheader()
        for r in records:
            writer.writerow({k: r.get(k, '') for k in csv_cols})

    print(f'Built Dev-32 review pack: {len(records)} cases -> {out}', flush=True)

    if zip_path is not None:
        zip_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_zip = zip_path.with_suffix('.zip.tmp')
        file_count = 0
        with zipfile.ZipFile(tmp_zip, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
            for item in sorted(out.rglob('*')):
                if item.is_file():
                    arcname = str(Path(out.name) / item.relative_to(out))
                    zf.write(item, arcname)
                    file_count += 1
        tmp_zip.replace(zip_path)
        size_mb = zip_path.stat().st_size / (1024 * 1024)
        print(f'Compressed {file_count} files -> {zip_path} ({size_mb:.2f} MB)', flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True, help='Existing t32_dev32 run directory')
    parser.add_argument('--out', type=Path, required=True, help='Output review pack directory')
    parser.add_argument('--zip', type=Path, help='Optional output .zip archive path')
    args = parser.parse_args()
    build_pack(args.source.resolve(), args.out.resolve(), args.zip.resolve() if args.zip else None)


if __name__ == '__main__':
    main()
