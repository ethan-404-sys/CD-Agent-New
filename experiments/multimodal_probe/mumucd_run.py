"""Run and score the Tirana probe in separate, label-blind and offline processes."""
from __future__ import annotations

import argparse
import ast
from collections import Counter, defaultdict
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image

from cd_agent.verify import EpisodeInput, check_binary_mask

from .mumucd_prepare import GSD_M, INSTRUCTION, RUN_NAME, T1_CHANNELS, T2_CHANNELS, TILE, TRACK, tiles

MODEL = 'qwen3.7-flash'
TASK = 'T1.3'
TARGET = 'any change'
BUDGET = 14
MAX_COST = 20.0
SPLIT = 'tirana-probe'
SEED = 20261123
METRIC_KEYS = ('f1', 'iou', 'precision', 'recall')
COUNT_KEYS = ('tp', 'fp', 'fn')
TEMPORAL_TOOLS = {'segment_text', 'segment_prompt', 'refine_region'}


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def load_inputs(inputs):
    """Only an exact allowlist of fields and relative image paths enters inference."""
    root = inputs.resolve()
    records = read_json(root / 'cases.json')
    if [row.get('id') for row in records] != [case for case, _ in tiles()]:
        raise ValueError('Input manifest must contain all nine grid tiles in row-major order')
    for row in records:
        case = row['id']
        expected = dict(id=case, image_t1=f'{case}/t1.png', image_t2=f'{case}/t2.png', instruction=INSTRUCTION)
        if row != expected:
            raise ValueError(f'{case}: nonstandard inference manifest entry')
        for field in ('image_t1', 'image_t2'):
            path = root / row[field]
            if path.resolve() != path:
                raise ValueError(f'{case}: image path may not use a symlink')
        yield EpisodeInput.from_mapping(row)


def read_png(path, mode):
    with Image.open(path) as image:
        if image.format != 'PNG' or image.size != (TILE, TILE) or image.mode not in mode:
            raise ValueError(f'{path}: unexpected PNG mode or size ({image.mode}, {image.size})')
        return np.asarray(image)


def infer(inputs, traces, worker, limit, max_cost, smoke=False):
    from cd_agent.backends import ApiAsk, SegmentBackend, SharedPrompter
    from cd_agent.loop_json import run_session
    from cd_agent.planner import Planner, RunGuard
    from cd_agent.recorder import Recorder
    from cd_agent.state import Session
    from cd_agent.tasks import seed_for
    from cd_agent.tools import Toolbox

    episodes = list(load_inputs(inputs))[:limit]
    guard = RunGuard(traces / 'STOP.json', budget_file=traces / 'cost.txt', max_cost=max_cost)
    prompter = SharedPrompter()
    for episode in episodes:
        guard.check()
        case = episode.id
        directory = traces / case
        directory.mkdir(parents=True, exist_ok=False)
        paths = [str(inputs / episode.image_t1), str(inputs / episode.image_t2)]
        rgb = [read_png(path, ('RGB',)) for path in paths]
        session = Session((TILE, TILE), target=TARGET, instruction=episode.instruction, gsd_m=GSD_M)
        backend = SegmentBackend(SPLIT, case, worker, target=TARGET, image_paths=paths)
        verifier = ApiAsk(guard=guard, model=MODEL, correspondence=True, session=session)
        recorder = Recorder(directory, SPLIT, case, evaluate=False)
        planner = Planner(guard=guard, parameters={'model': MODEL}, seed=seed_for(case, SEED))
        toolbox = Toolbox(rgb, directory, backend, prompter.for_image(SPLIT, case, rgb=rgb),
                          verifier, session=session, python=True)
        if smoke:
            # Connectivity check only, excluded from measured routing and session metrics.
            backend('t1', 'building', 0, .5)
            pair = [Image.fromarray(image) for image in rgb]
            answer, _, _, cost = verifier.one(pair * 2, '', 0)
            if answer['answer'] not in ('yes', 'no'):
                raise RuntimeError('Smoke verifier returned no yes/no decision')
            print(f'{case}: SAM3 and frozen verifier preflight passed (verifier estimated CNY {cost:.6f})', flush=True)
        result = run_session(toolbox, planner, recorder, task=TASK, budget=BUDGET, verify=True, status=True,
                             metadata=dict(session_id=case, source_images=paths,
                                           backends=dict(sam3='local_worker', sam_prompt='shared_prompter',
                                                         ask='api', ask_model=MODEL, verify_stage=True)))
        if result['error'] or not result['planning_rounds']:
            raise RuntimeError(f'{case}: session did not complete; inspect {directory / "final.json"}')
        binary_png(directory / result['mask_path'])
        print(case, result['submitted_by'], 'rounds', result['planning_rounds'], flush=True)


def binary_png(path):
    return check_binary_mask(read_png(path, ('L', '1')), expected_shape=(TILE, TILE), name=str(path))


def pixel_metrics(tp, fp, fn):
    union = tp + fp + fn
    return dict(tp=tp, fp=fp, fn=fn,
                f1=2 * tp / (2 * tp + fp + fn) if union else 1.0,
                iou=tp / union if union else 1.0,
                precision=tp / (tp + fp) if tp + fp else 0.0,
                recall=tp / (tp + fn) if tp + fn else 0.0)


def python_route(code):
    """Classify explicit literal references, not actual Python runtime reads."""
    try:
        tree = ast.parse(code)
    except (SyntaxError, TypeError, ValueError):
        return 'unspecified'
    times = {node.slice.value for node in ast.walk(tree)
             if isinstance(node, ast.Subscript) and isinstance(node.value, ast.Name) and node.value.id == 'images'
             and isinstance(node.slice, ast.Constant) and node.slice.value in ('t1', 't2')}
    return '+'.join(sorted(times)) or 'unspecified'


def tool_routing(steps):
    calls, outcomes, by_time, python = Counter(), defaultdict(Counter), defaultdict(Counter), Counter()
    for step in steps:
        for call in step['calls']:
            name = call['tool']
            calls[name] += 1
            status = call.get('status', 'executed')
            outcome = status if status != 'executed' else ('ok' if call['ok'] else 'error')
            outcomes[name][outcome] += 1
            args = call.get('args') or {}
            if name in TEMPORAL_TOOLS:
                when = args.get('time')
                by_time[name][when if when in ('t1', 't2') else 'unspecified'] += 1
            elif name == 'python':
                python[python_route(args.get('code'))] += 1
    return dict(calls=dict(calls), outcomes={k: dict(v) for k, v in outcomes.items()},
                segmentation_by_requested_time={k: dict(v) for k, v in by_time.items()},
                python_literal_image_mentions=dict(python))


def report(out, limit, max_cost):
    from cd_agent.evaluation import pixel_counts
    from cd_agent.evidence import build_evidence_page
    from cd_agent.verify import verify_run_dir, write_run_contract

    audit = read_json(out / 'sample_audit.json')
    labels = {row['id']: row for row in audit['cases']}
    inputs = out / 'inputs'
    episodes = list(load_inputs(inputs))[:limit]
    rows, evidence, steps = [], [], []
    for episode in episodes:
        case = episode.id
        label, trace = labels[case], out / 'traces' / case
        final = read_json(trace / 'final.json')
        if final['error'] or not final['planning_rounds']:
            raise ValueError(f'{case}: cannot score a failed inference')
        gt = binary_png(out / label['gt_mask'])
        ignore = label['gt_ignore_mask']
        valid = ~binary_png(out / ignore) if ignore else np.ones_like(gt)
        if int(gt.sum()) != label['gt_positive_pixels'] or int((~valid).sum()) != label['gt_nodata_pixels']:
            raise ValueError(f'{case}: staged labels disagree with the audit')
        pred = binary_png(trace / final['mask_path'])
        score = pixel_metrics(**pixel_counts(pred, gt, valid))
        case_steps = [json.loads(line) for line in (trace / 'steps.jsonl').read_text(encoding='utf-8').splitlines()]
        steps.extend(case_steps)
        rows.append(dict(id=case, **score, gt_positive_pixels=label['gt_positive_pixels'],
                         ignored_pixels=label['gt_nodata_pixels'], tool_routing=tool_routing(case_steps),
                         **{k: final[k] for k in ('submitted_by', 'termination_reason', 'planning_rounds',
                                                  'planner_api_attempts', 'ask_api_attempts', 'cost_cny')}))
        evidence.append(dict(id=case, image_t1=inputs / episode.image_t1, image_t2=inputs / episode.image_t2,
                             pred_mask=pred & valid, gt_mask=gt & valid,
                             summary=f'{final["summary"] or ""} (Nodata masked out: {(~valid).sum()} pixels.)'))
    metrics = dict(count=len(rows), micro=pixel_metrics(**{k: sum(row[k] for row in rows) for k in COUNT_KEYS}),
                   macro={k: sum(row[k] for row in rows) / len(rows) for k in METRIC_KEYS},
                   metric_convention='Empty union: F1 and IoU = 1. Undefined precision or recall = 0.',
                   tool_routing=tool_routing(steps), submissions=dict(Counter(row['submitted_by'] for row in rows)),
                   planner_api_attempts=sum(row['planner_api_attempts'] for row in rows),
                   verifier_api_attempts=sum(row['ask_api_attempts'] for row in rows),
                   estimated_session_cost_cny=sum(row['cost_cny'] for row in rows), cases=rows)
    page = build_evidence_page(evidence, out / 'viz', title='MUMUCD Tirana · Optical S2 × SAR S1',
                               metrics={k: metrics[k] for k in ('count', 'micro', 'macro', 'tool_routing')})
    write_run_contract(out, track=TRACK, run_name=RUN_NAME,
                       config=dict(planner_model=MODEL, verifier_model=MODEL, target=TARGET, gsd_m=GSD_M,
                                   t1_rgb=T1_CHANNELS, t2_rgb=T2_CHANNELS, budget=BUDGET, max_cost_cny=max_cost,
                                   python=True, smoke=limit == 1),
                       metrics=metrics)
    verify_run_dir(out, expected_count=limit)
    print(f'{limit} scored; micro F1={metrics["micro"]["f1"]:.4f}; evidence: {page}', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    inference = sub.add_parser('infer')
    inference.add_argument('--inputs', type=Path, required=True)
    inference.add_argument('--traces', type=Path, required=True)
    inference.add_argument('--sam3-worker', required=True)
    inference.add_argument('--smoke', action='store_true')
    scoring = sub.add_parser('report')
    scoring.add_argument('--out', type=Path, required=True)
    for stage in (inference, scoring):
        stage.add_argument('--limit', type=int, choices=(1, 9), default=9)
        stage.add_argument('--max-cost-cny', type=float, default=MAX_COST)
    args = parser.parse_args()
    if not math.isfinite(args.max_cost_cny) or not 0 < args.max_cost_cny <= MAX_COST:
        parser.error(f'--max-cost-cny must be in (0, {MAX_COST:g}]')
    if args.command == 'infer':
        if args.smoke and args.limit != 1:
            parser.error('--smoke requires --limit 1')
        infer(args.inputs.resolve(), args.traces.resolve(), args.sam3_worker, args.limit, args.max_cost_cny, args.smoke)
    else:
        report(args.out.resolve(), args.limit, args.max_cost_cny)


if __name__ == '__main__':
    main()
