#!/usr/bin/env python3
"""Run an independent thumbnail baseline and a bounded zoom agent on XLRS VQA."""

from __future__ import annotations

import argparse
import base64
from collections import Counter
from contextlib import ExitStack
from dataclasses import asdict, dataclass, field
import html
import json
import os
from pathlib import Path
import re
from typing import Callable

import numpy as np
from PIL import Image, ImageDraw
import requests

MODEL = "qwen3.7-flash"
BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
MAX_SIDE = 1024
REPLIES = 8
PRICING = "https://help.aliyun.com/zh/model-studio/qwen3-7-flash"


@dataclass(frozen=True)
class Question:
    id: str
    category: str
    question: str
    options: tuple[str, ...]
    images: tuple[Path, ...]
    image_sizes: tuple[tuple[int, int], ...]

    @property
    def letters(self) -> str:
        return "ABCD"[:len(self.options)]

    def prompt(self) -> str:
        return f"{self.question}\n" + "\n".join(
            f"{letter}. {option}" for letter, option in zip(self.letters, self.options))


def parse_choice(value, letters: str) -> str | None:
    if not isinstance(value, str) or not re.fullmatch(f"[{letters}]+", value.strip()):
        return None
    value = value.strip()
    return "".join(sorted(value)) if len(set(value)) == len(value) else None


def read_questions(manifest: Path) -> tuple[list[Question], dict[str, str]]:
    root = manifest.parent.resolve()
    questions, answers = [], {}
    for line in manifest.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        sid = row["id"]
        if not re.fullmatch(r"[a-zA-Z0-9_-]+", sid) or sid in answers:
            raise ValueError("invalid or repeated VQA ID")
        images = tuple((root / path).resolve() for path in row["images"])
        sizes = tuple(tuple(pair) for pair in row["image_sizes"])
        if (len(images) not in (1, 2) or len(images) != len(sizes)
            or any(not path.is_relative_to(root) or not path.is_file() for path in images)):
            raise ValueError(f"{sid}: expected one or two in-subset image files")
        if any(len(pair) != 2 or any(type(n) is not int or n < 1 for n in pair) for pair in sizes):
            raise ValueError(f"{sid}: invalid native size")
        raw_options = row["options"]
        if not isinstance(raw_options, list) or not 2 <= len(raw_options) <= 4:
            raise ValueError(f"{sid}: invalid options")
        opts = []
        for letter, value in zip("ABCD", raw_options):
            value = value.strip()
            match = re.match(r"^(?:\(([A-D])\)|([A-D])[.):])\s*", value)
            if match:
                if (match[1] or match[2]) != letter:
                    raise ValueError(f"{sid}: unexpected option label")
                value = value[match.end():]
            if not value:
                raise ValueError(f"{sid}: empty option")
            opts.append(value)
        question = Question(sid, row["category"], row["question"], tuple(opts), images, sizes)
        answer = parse_choice(row["answer"].upper(), question.letters)
        if not answer:
            raise ValueError(f"{sid}: invalid answer label")
        questions.append(question)
        answers[sid] = answer
    return questions, answers


def thumbnail(image: Image.Image) -> Image.Image:
    image = image.copy()
    image.thumbnail((MAX_SIDE, MAX_SIDE), Image.Resampling.LANCZOS)
    return image.convert("RGB")


def with_ticks(image: Image.Image, region: tuple[int, int, int, int]) -> Image.Image:
    image = image.copy()
    draw = ImageDraw.Draw(image)
    x0, y0, x1, y1 = region
    w, h = image.size
    for n in range(5):
        x, y = (w - 1) * n // 4, (h - 1) * n // 4
        if w >= 80:
            draw.line((x, 0, x, min(6, h - 1)), fill="yellow", width=2)
            label = str(x0 + (x1 - x0) * n // 4)
            draw.text((max(0, min(x + 2, w - 7 * len(label))), 1), label,
                      fill="yellow", stroke_width=1, stroke_fill="black")
        if h >= 80:
            draw.line((0, y, min(6, w - 1), y), fill="cyan", width=2)
            if n:
                draw.text((2, max(0, min(y, h - 12))), str(y0 + (y1 - y0) * n // 4),
                          fill="cyan", stroke_width=1, stroke_fill="black")
    return image


def validate_box(box, size: tuple[int, int]) -> tuple[int, int, int, int]:
    if not isinstance(box, list) or len(box) != 4 or any(type(x) is not int for x in box):
        raise ValueError("region must be four native pixel integers")
    x0, y0, x1, y1 = box
    w, h = size
    if not (0 <= x0 < x1 <= w and 0 <= y0 < y1 <= h):
        raise ValueError(f"region must be nonempty and inside {w}x{h}")
    return x0, y0, x1, y1


def grid_boxes(size: tuple[int, int], rows: int, cols: int) -> list[list[int]]:
    w, h = size
    if (type(rows) is not int or type(cols) is not int
        or not 1 <= rows <= min(8, h) or not 1 <= cols <= min(8, w)):
        raise ValueError("rows and cols must be integers 1–8 within the image dimensions")
    return [[c * w // cols, r * h // rows, (c + 1) * w // cols, (r + 1) * h // rows]
            for r in range(rows) for c in range(cols)]


def diff_hotspots(overview: list[Image.Image], size: tuple[int, int], top_k: int) -> list[list[int]]:
    if len(overview) != 2 or overview[0].size != overview[1].size:
        raise ValueError("difference scan requires two same-sized temporal views")
    if type(top_k) is not int or not 1 <= top_k <= 12:
        raise ValueError("top_k must be an integer 1–12")
    small = [i.copy() for i in overview]
    for im in small:
        im.thumbnail((256, 256), Image.Resampling.BILINEAR)
    a, b = (np.asarray(im, dtype=np.float32) for im in small)
    diff = np.abs(a - b).mean(axis=2)
    dh, dw = diff.shape
    rows, cols = min(8, dh), min(8, dw)
    boxes = grid_boxes(size, rows, cols)
    low_boxes = grid_boxes((dw, dh), rows, cols)
    scores = [float(diff[y0:y1, x0:x1].mean()) for x0, y0, x1, y1 in low_boxes]
    return [boxes[i] for i in sorted(range(len(boxes)), key=lambda i: -scores[i])[:top_k]]


@dataclass
class Tokens:
    prompt: int = 0
    completion: int = 0
    cached: int = 0
    cost_cny: float = 0
    missing_usage_replies: int = 0

    def add(self, usage: dict | None) -> None:
        if not usage or "prompt_tokens" not in usage or "completion_tokens" not in usage:
            self.missing_usage_replies += 1
            return
        inp, out = usage["prompt_tokens"], usage["completion_tokens"]
        cache = (usage.get("prompt_tokens_details") or {}).get("cached_tokens") or 0
        self.prompt += inp
        self.completion += out
        self.cached += cache
        in_rate, out_rate = (0.2, 0.8) if inp <= 32768 else (
            (0.6, 2.4) if inp <= 262144 else (1.2, 4.8))
        self.cost_cny += ((inp - cache) * in_rate + cache * in_rate * 0.2 + out * out_rate) / 1e6


@dataclass
class Result:
    choice: str | None = None
    replies: int = 0
    tokens: Tokens = field(default_factory=Tokens)
    zooms: list[dict] = field(default_factory=list)
    evidence: list[dict] = field(default_factory=list)
    rationale: str = ""
    trace: list[dict] = field(default_factory=list)


def message(role: str, text: str, images: list[Path] | tuple[Path, ...] = ()) -> dict:
    content = [{"type": "text", "text": text}]
    for path in images:
        data = base64.b64encode(path.read_bytes()).decode("ascii")
        content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{data}"}})
    return {"role": role, "content": content}


class Bailian:
    def __init__(self, session: requests.Session):
        self.session = session
        self.session.trust_env = False
        self.key = os.environ.get("DASHSCOPE_API_KEY")
        if not self.key:
            raise ValueError("Bailian key not loaded; use run_with_bailian_csv.py")
        if os.environ.get("DASHSCOPE_BASE_URL") != BASE_URL:
            raise ValueError("unexpected Bailian endpoint")

    def __call__(self, messages: list[dict], *, max_tokens: int) -> tuple[str, dict | None]:
        try:
            with self.session.post(
                f"{BASE_URL}/chat/completions",
                headers={"Authorization": f"Bearer {self.key}"},
                json={"model": MODEL, "messages": messages, "max_tokens": max_tokens,
                      "temperature": 0, "enable_thinking": False},
                timeout=(12, 90), allow_redirects=False,
            ) as response:
                if response.status_code != 200:
                    # Do not log provider payloads, URLs with queries, request headers or messages.
                    details = ""
                    try:
                        code = response.json().get("error", {}).get("code")
                        if isinstance(code, str) and re.fullmatch(r"[a-zA-Z0-9_.-]{1,48}", code):
                            details = f" ({code})"
                    except (ValueError, AttributeError):
                        pass
                    raise ValueError(f"Bailian returned HTTP {response.status_code}{details}")
                payload = response.json()
                return payload["choices"][0]["message"]["content"] or "", payload.get("usage")
        except requests.RequestException as error:
            raise ValueError(f"Bailian request failed ({type(error).__name__})") from None


def answer_once(q: Question, global_files: list[Path], call: Callable) -> Result:
    text = ("Answer the multiple-choice visual question using the image(s) in source order. "
            "Reply with only one option letter, or concatenated letters if the question calls for "
            "more than one option. No separators or prose.\n\n" + q.prompt())
    reply, usage = call([message("user", text, global_files)], max_tokens=24)
    result = Result(choice=parse_choice(reply, q.letters), replies=1)
    result.tokens.add(usage)
    result.trace.append({"reply": reply})
    return result


def target_region(spec: dict, sizes):
    index = spec.get("image")
    if index is None:
        indices = list(range(len(sizes)))
    elif type(index) is int and 1 <= index <= len(sizes):
        indices = [index - 1]
    else:
        raise ValueError("image must be a valid 1-based image index")
    box = validate_box(spec.get("region"), tuple(map(min, zip(*(sizes[j] for j in indices)))))
    return box, indices


def answer_zoom(q: Question, native: list[Image.Image], overview: list[Image.Image],
                global_files: list[Path], asset_dir: Path, call: Callable) -> Result:
    sizes = q.image_sizes
    common_size = tuple(map(min, zip(*sizes)))
    paired = len(sizes) == 2 and sizes[0] == sizes[1]
    instructions = (
        'Reply only with one JSON tool call per turn. Allowed calls:\n'
        '{"tool":"view","region":[x0,y0,x1,y1]} (native crop of every image; add "image":1 or 2 '
        'to choose one available image).\n'
        '{"tool":"grid_overview","rows":4,"cols":4} (native grid boxes of the shared canvas).\n'
        '{"tool":"submit_answer","choice":"A","evidence_regions":[[x0,y0,x1,y1]],'
        '"rationale":"brief visual justification"}. Select one letter unless the question calls for '
        'multiple options, in which case concatenate the letters. '
        'use an empty evidence list only if no specific region supports the answer. For evidence in '
        'just one image you can use {"image":1,"region":[x0,y0,x1,y1]}.\n'
    )
    if paired:
        instructions += ('{"tool":"scan_diff_hotspots","top_k":6} (coarse apparent differences; '
                         'may include illumination or alignment artifacts).\n')
    text = (f"{instructions}Images are numbered in source order; native sizes: {sizes}. "
            "Coordinates use top-left origin, half-open boxes; yellow ticks are x and cyan ticks are y. "
            "Use the overview to target helpful details; inspect native crops if needed. "
            f"Budget: {REPLIES} replies including submit_answer.\n\n{q.prompt()}")
    messages = [message("user", text, global_files)]
    result = Result()
    for step in range(1, REPLIES + 1):
        if step == REPLIES:
            messages[-1]["content"].append({"type": "text", "text": "Final reply: submit_answer now."})
        reply, usage = call(messages, max_tokens=384)
        result.replies = step
        result.tokens.add(usage)
        event = {"step": step, "reply": reply}
        result.trace.append(event)
        crops = []
        try:
            raw = reply.strip()
            if fenced := re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", raw, re.S):
                raw = fenced[1]
            cmd = json.loads(raw)
            if not isinstance(cmd, dict):
                raise ValueError("expected a JSON object")
            tool = cmd.get("tool")
            if tool == "submit_answer":
                choice = parse_choice(cmd.get("choice"), q.letters)
                regions, rationale = cmd.get("evidence_regions"), cmd.get("rationale")
                if not choice or not isinstance(regions, list) or len(regions) > 12:
                    raise ValueError("submit a valid choice and up to 12 evidence regions")
                if not isinstance(rationale, str) or not rationale.strip():
                    raise ValueError("submit a brief rationale")
                evidence = []
                for spec in regions:
                    box, indices = target_region(spec if isinstance(spec, dict) else {"region": spec}, sizes)
                    evidence.append({"region": box, "images": [i + 1 for i in indices]})
                result.choice, result.evidence, result.rationale = choice, evidence, rationale.strip()
                event["tool"] = tool
                break
            if step == REPLIES:
                raise ValueError("reply budget exhausted without submit_answer")
            if tool == "view":
                box, indices = target_region(cmd, sizes)
                for i in indices:
                    crop = thumbnail(native[i].crop(box))
                    path = asset_dir / f"step-{step}-image-{i + 1}.jpg"
                    with_ticks(crop, box).save(path, quality=92)
                    crops.append(path)
                result.zooms.append({"step": step, "region": box, "images": [i + 1 for i in indices],
                                     "files": [p.name for p in crops]})
                observation = f"Native region {box}, images {[i + 1 for i in indices]} in that order."
            elif tool == "grid_overview":
                observation = "Native grid boxes, row-major: " + json.dumps(
                    grid_boxes(common_size, cmd.get("rows"), cmd.get("cols")))
            elif tool == "scan_diff_hotspots" and paired:
                observation = "Candidate native boxes ranked by coarse RGB difference: " + json.dumps(
                    diff_hotspots(overview, common_size, cmd.get("top_k")))
            else:
                raise ValueError("unknown or unavailable tool")
            event.update(tool=tool, observation=observation)
        except (json.JSONDecodeError, ValueError) as error:
            observation = f"Tool request rejected: {error}"
            event["error"] = observation
        if step < REPLIES:
            messages.extend([{"role": "assistant", "content": reply},
                             message("user", f"{observation}\n{REPLIES - step} replies remain.", crops)])
    return result


def run_question(q: Question, folder: Path, modes: tuple[str, ...], call: Callable) -> dict[str, Result]:
    folder.mkdir(parents=True, exist_ok=False)
    with ExitStack() as stack:
        native = [stack.enter_context(Image.open(path)) for path in q.images]
        if tuple(im.size for im in native) != q.image_sizes:
            raise ValueError(f"{q.id}: image headers do not match the manifest")
        overview = [thumbnail(im) for im in native]
        plain, ticks = [], []
        for i, (im, (w, h)) in enumerate(zip(overview, q.image_sizes), 1):
            path = folder / f"global-{i}.jpg"
            im.save(path, quality=92)
            plain.append(path)
            path = folder / f"ticks-{i}.jpg"
            with_ticks(im, (0, 0, w, h)).save(path, quality=92)
            ticks.append(path)
        results = {}
        if "global_only" in modes:
            results["global_only"] = answer_once(q, plain, call)
        if "zoom_agent" in modes:
            results["zoom_agent"] = answer_zoom(q, native, overview, ticks, folder, call)
        return results


def write_viz(viz: Path, rows: list[dict], metrics: dict) -> None:
    def ranked(row):
        return len((row["runs"].get("zoom_agent") or {}).get("zooms", []))

    ordered = sorted(rows, key=ranked, reverse=True)
    representatives = []
    patterns = set()
    for row in ordered:
        pattern = tuple((row["runs"].get(name) or {}).get("choice") == row["answer"]
                        for name in ("global_only", "zoom_agent"))
        if pattern not in patterns:
            patterns.add(pattern)
            representatives.append(row)
    representatives = (representatives + [r for r in ordered if r not in representatives])[:6]
    esc = lambda value: html.escape(str(value), quote=True)
    page = ['<!doctype html><html lang="en"><meta charset="utf-8"><title>XLRS VQA probe</title>',
            '<meta name="viewport" content="width=device-width, initial-scale=1">',
            '<style>body{font:16px system-ui;margin:2rem auto;max-width:1250px;padding:0 1rem}'
            'figure{display:inline-block;margin:8px;max-width:95%;vertical-align:top}'
            'img{max-width:min(100%,550px);height:auto} section{border-top:1px solid #aaa;padding:1em 0}'
            'figcaption{font:14px monospace} pre{white-space:pre-wrap}</style><body><h1>XLRS VQA probe</h1>',
            '<p>Representative cases. Red boxes are viewed crops; green boxes are submitted evidence.</p>',
            f'<details><summary>Metrics</summary><pre>{esc(json.dumps(metrics, indent=2))}</pre></details>']
    for row in representatives:
        sid = row["id"]
        source = viz / "assets" / sid
        agent = row["runs"].get("zoom_agent", {})
        baseline = row["runs"].get("global_only", {})
        page.append(f'<section><h2>{esc(sid)} · {esc(row["category"])}</h2><p>{esc(row["question"])}</p>')
        page.append('<ol type="A">' + "".join(f"<li>{esc(x)}</li>" for x in row["options"]) + "</ol>")
        page.append(f'<p>GT: <b>{esc(row["answer"])}</b> · Baseline: {esc(baseline.get("choice") or "—")}'
                    f' · Agent: {esc(agent.get("choice") or "—")}</p>')
        for i, (w, h) in enumerate(row["image_sizes"], 1):
            with Image.open(source / f"ticks-{i}.jpg") as im:
                montage = im.convert("RGB")
            draw = ImageDraw.Draw(montage)
            for entries, color in ((agent.get("zooms", []), "#ff3030"), (agent.get("evidence", []), "#54ff63")):
                for entry in entries:
                    if i not in entry["images"]:
                        continue
                    x0, y0, x1, y1 = entry["region"]
                    box = (x0 * (montage.width - 1) / w, y0 * (montage.height - 1) / h,
                           x1 * (montage.width - 1) / w, y1 * (montage.height - 1) / h)
                    draw.rectangle(box, outline=color, width=3)
                    if "step" in entry:
                        draw.text(box[:2], str(entry["step"]), fill=color, stroke_width=1, stroke_fill="black")
            montage.save(source / f"overlay-{i}.jpg", quality=91)
            page.append(f'<figure><img loading="lazy" src="assets/{sid}/overlay-{i}.jpg" '
                        f'alt="Global image {i}"><figcaption>Image {i} · native {w}×{h}</figcaption></figure>')
        if agent.get("rationale"):
            page.append(f'<p>Agent rationale: {esc(agent["rationale"])}</p>')
        for zoom in agent.get("zooms", []):
            for i, name in zip(zoom["images"], zoom["files"]):
                page.append(f'<figure><img loading="lazy" src="assets/{sid}/{name}" alt="Native zoom crop">'
                            f'<figcaption>Reply {zoom["step"]} · image {i} · {esc(zoom["region"])}'
                            '</figcaption></figure>')
        page.append('</section>')
    (viz / "index.html").write_text("\n".join(page + ["</body></html>"]), encoding="utf-8")


def score(rows: list[dict], modes: tuple[str, ...]) -> dict:
    metrics = {"model": MODEL, "samples": len(rows), "category_counts": dict(Counter(r["category"] for r in rows)),
               "reply_budget": REPLIES, "pricing_source": PRICING, "cost_currency": "CNY",
               "cost_note": "Beijing published on-demand prices with reported cache hits; excludes free quota.",
               "modes": {}}
    for mode in modes:
        results = [r["runs"][mode] for r in rows]
        correct = sum(p["choice"] == r["answer"] for p, r in zip(results, rows))
        tokens = {key: sum(p["tokens"][key] for p in results) for key in asdict(Tokens())}
        metrics["modes"][mode] = {
            "accuracy": correct / len(rows), "correct": correct,
            "invalid": sum(p["choice"] is None for p in results),
            "average_zoom_steps": sum(len(p["zooms"]) for p in results) / len(rows),
            "average_replies": sum(p["replies"] for p in results) / len(rows), **tokens,
        }
    metrics["total_cost_cny"] = sum(m["cost_cny"] for m in metrics["modes"].values())
    if len(modes) == 2:
        metrics["accuracy_delta"] = (metrics["modes"]["zoom_agent"]["accuracy"]
                                     - metrics["modes"]["global_only"]["accuracy"])
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("datasets/XLRS-Bench-lite/cd_subset/dev_cd_vqa.jsonl"))
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--mode", choices=("both", "global_only", "zoom_agent"), default="both")
    parser.add_argument("--smoke", action="store_true", help="evaluate the first two real questions")
    args = parser.parse_args()
    modes = ("global_only", "zoom_agent") if args.mode == "both" else (args.mode,)
    try:
        questions, answers = read_questions(args.manifest)
        if not 20 <= len(questions) <= 30:
            raise ValueError("expected a 20–30 question development subset")
        if args.smoke:
            questions = questions[:2]
        Image.MAX_IMAGE_PIXELS = 150_000_000
        rows = []
        viz = args.out / "viz"
        viz.mkdir(parents=True, exist_ok=False)
        with requests.Session() as session, (args.out / "predictions.jsonl").open("x", encoding="utf-8") as predictions:
            call = Bailian(session)
            for q in questions:
                inferred = run_question(q, viz / "assets" / q.id, modes, call)
                # The scoring join is outside both inference procedures.
                row = dict(id=q.id, category=q.category, question=q.question, options=q.options,
                           image_sizes=q.image_sizes, answer=answers[q.id],
                           runs={mode: asdict(result) for mode, result in inferred.items()})
                rows.append(row)
                predictions.write(json.dumps(row, ensure_ascii=False) + "\n")
                predictions.flush()
                print(f"{q.id}: " + " ".join(f"{m}={r.choice or 'invalid'} zooms={len(r.zooms)}"
                                              for m, r in inferred.items()), flush=True)
        metrics = score(rows, modes)
        write_viz(viz, rows, metrics)
        (args.out / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(metrics), flush=True)
        if args.smoke:
            if any(m["invalid"] or m["missing_usage_replies"] for m in metrics["modes"].values()):
                raise ValueError("smoke failed: invalid answers or missing API usage")
            if "zoom_agent" in modes and not metrics["modes"]["zoom_agent"]["average_zoom_steps"]:
                raise ValueError("smoke failed: no real zoom crop exercised")
            print("XLRS_SMOKE_PASSED", flush=True)
    except (ValueError, KeyError, OSError) as error:
        raise SystemExit(f"XLRS probe stopped: {error}") from None


if __name__ == "__main__":
    main()
