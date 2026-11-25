"""Run the paired-modality probe, then score and render it in a separate process."""

from __future__ import annotations

import argparse
from collections import Counter
import json
import math
from pathlib import Path
import shutil

import numpy as np
from PIL import Image

from cd_agent.backends import ApiAsk, SegmentBackend, SharedPrompter
from cd_agent.evaluation import pixel_counts
from cd_agent.evidence import build_evidence_page
from cd_agent.loop_json import run_session
from cd_agent.planner import Planner, RunGuard
from cd_agent.recorder import Recorder
from cd_agent.scope_t13 import session_prompts
from cd_agent.state import Session
from cd_agent.tasks import seed_for
from cd_agent.tools import Toolbox
from cd_agent.verify import check_binary_mask, verify_run_dir, write_run_contract

from .bright_xbd_prepare import (
    COUNTS, DAMAGE_TASK, MODALITIES, RUN_NAME, SEED, SIZE, TRACK, load, load_inputs, read_labels, read_rgb,
)

MODEL = 'qwen3.7-flash'
BUDGET = 16
TARGET = ('disaster-related building damage or destruction',)
ROUTING = ('t1_optical_seg_calls', 't2_optical_seg_calls', 'python_calls', 'ask_calls', 'rounds', 'cost_cny')
SCORES = ('f1', 'iou', 'precision', 'recall')


class DamageAsk(ApiAsk):
    """Frozen ApiAsk protocol and model; give its question the experiment's task wording."""

    def __init__(self, dataset, **kwargs):
        super().__init__(model=MODEL, **kwargs)
        self.task_context = f'{MODALITIES[dataset]} {DAMAGE_TASK}'

    def one(self, images, question, number, level='L1', **kwargs):
        return super().one(images, f'{self.task_context} {question}', number, level, **kwargs)


def prediction(path):
    with Image.open(path) as im:
        if im.format != 'PNG' or im.mode not in ('L', '1') or im.size != SIZE:
            raise ValueError(f'{path}: expected a 1024×1024 single-channel PNG')
        return check_binary_mask(np.asarray(im), expected_shape=SIZE, name=str(path))


def infer_case(dataset, ep, inputs, traces, worker, prompter, guard, *, smoke=False):
    folder = traces / ep.id
    sources = [inputs / ep.image_t1, inputs / ep.image_t2]
    rgb = [read_rgb(path, staged=True, sar=dataset == 'bright' and n == 1) for n, path in enumerate(sources)]
    guard.check()
    if folder.exists():
        raise FileExistsError(f'Refusing to overwrite an existing session: {folder}')
    recorder = Recorder(folder, dataset, ep.id, evaluate=False)
    # Use the generic scoped prompt rather than the built-in new/demolished/rebuilt-only building scope.
    session = Session(SIZE, target=TARGET, instruction=ep.instruction)
    session.visual_query = {}
    for time, array in enumerate(rgb, 1):
        path = folder / 'images' / f'native_t{time}.jpg'
        Image.fromarray(array).save(path, quality=95)
        session.visual_query[f'T{time}'] = path
    planner = Planner(guard=guard, parameters={'model': MODEL}, seed=seed_for(ep.id, SEED))
    segment = SegmentBackend(dataset, ep.id, worker, target='building', image_paths=list(map(str, sources)))
    prompt = prompter.for_image(dataset, ep.id, rgb=rgb)
    verifier = DamageAsk(dataset, guard=guard, session=session)
    toolbox = Toolbox(rgb, folder, segment, prompt, verifier, session=session, python=True)

    if smoke:
        # Ensure these providers really ran even if the model does not choose them in the smoke trace.
        segment('t1', 'building', 0, .5)
        answer, _, _, _ = verifier.one([Image.fromarray(im) for im in rgb],
                                       session_prompts(session)['VERIFY'], 0)
        if answer['answer'] not in ('yes', 'no'):
            raise RuntimeError(f'{ep.id}: smoke verifier did not return a parseable decision')

    result = run_session(
        toolbox, planner, recorder, task='disaster_building_damage', budget=BUDGET, prompt_version='v4',
        verify=True, status=True,
        metadata=dict(session_id=ep.id, source_images=list(map(str, sources)),
                      first_images={label: str(path) for label, path in session.visual_query.items()},
                      backends=dict(ask='api', ask_model=MODEL, verify_stage=True, segment='sam3_worker',
                                    prompt='shared_prompter')),
    )
    source = folder / result['mask_path']
    prediction(source)
    shutil.copyfile(source, folder / 'prediction.png')
    print(f'{ep.id}: {result["submitted_by"]}, {result["planning_rounds"]} rounds', flush=True)
    if smoke and (result['error'] or not result['planning_rounds'] or result['submitted_by'] != 'model'):
        raise RuntimeError(f'{ep.id}: smoke did not complete an explicit model submission; see {folder}')
    guard.check()


def infer(inputs, work, worker, max_cost, *, smoke=False):
    # This entry point never reads the offline audit manifest or GT raster directory.
    guard = RunGuard(work / 'STOP.json', budget_file=work / 'budget.txt', max_cost=max_cost)
    prompter = SharedPrompter()
    episodes = load_inputs(inputs)
    for dataset, cases in episodes.items():
        if smoke and dataset != 'bright':
            continue
        for ep in cases[:1] if smoke else cases:
            infer_case(dataset, ep, inputs, work / 'traces', worker, prompter, guard, smoke=smoke)


def routing(steps, final):
    """Count Planner tool attempts recorded by the loop, including failed routes."""
    totals = dict.fromkeys(ROUTING, 0)
    for step in steps:
        for call in step['calls']:
            if call.get('status') == 'skipped_due_to_submit':
                continue
            tool, args = call['tool'], call.get('args')
            if tool == 'segment_text' and isinstance(args, dict) and args.get('time') in ('t1', 't2'):
                totals[f'{args["time"]}_optical_seg_calls'] += 1
            elif tool in ('python', 'ask'):
                totals[f'{tool}_calls'] += 1
    totals.update(rounds=final['planning_rounds'], cost_cny=final['cost_cny'])
    return totals


def scores(tp, fp, fn):
    union = tp + fp + fn
    return dict(tp=tp, fp=fp, fn=fn, f1=2 * tp / (2 * tp + fp + fn) if union else 1.,
                iou=tp / union if union else 1.,
                precision=tp / (tp + fp) if tp + fp else 0.,
                recall=tp / (tp + fn) if tp + fn else 0.)


def aggregate(rows):
    n = len(rows)
    sums = {key: sum(row['routing'][key] for row in rows) for key in ROUTING}
    seg_calls = sums['t1_optical_seg_calls'] + sums['t2_optical_seg_calls']
    return dict(
        count=n, failed=sum(row['failed'] for row in rows),
        positive=sum(row['tp'] + row['fn'] > 0 for row in rows),
        negative=sum(row['tp'] + row['fn'] == 0 for row in rows),
        events=dict(sorted(Counter(row['event'] for row in rows).items())),
        micro=scores(*(sum(row[key] for row in rows) for key in ('tp', 'fp', 'fn'))),
        macro={key: sum(row[key] for row in rows) / n for key in SCORES},
        tool_routing=dict(totals=sums, mean_per_case={key: total / n for key, total in sums.items()},
                          cases_with_t2_optical_seg=sum(row['routing']['t2_optical_seg_calls'] > 0 for row in rows),
                          t2_optical_seg_share=sums['t2_optical_seg_calls'] / seg_calls if seg_calls else None,
                          submissions=dict(Counter(row['submitted_by'] for row in rows))),
    )


def report(data, work, max_cost, *, smoke=False):
    episodes = load_inputs(data / 'inputs')
    audit = {row['id']: row for row in load(data / 'sample_audit.json')['cases']}
    summaries, all_rows = {}, []
    for dataset, cases in episodes.items():
        if smoke and dataset != 'bright':
            continue
        rows, evidence = [], []
        for ep in cases[:1] if smoke else cases:
            entry = audit[ep.id]
            folder = work / 'traces' / ep.id
            pred_path = folder / 'prediction.png'
            gt_path = data / entry['gt_mask']
            gt = check_binary_mask(read_labels(gt_path), expected_shape=SIZE, name=ep.id)
            if int(gt.sum()) != entry['damage_pixels']:
                raise ValueError(f'{ep.id}: offline GT does not match the audit')
            final = load(folder / 'final.json')
            with (folder / 'steps.jsonl').open(encoding='utf-8') as f:
                tool_stats = routing((json.loads(line) for line in f), final)
            row = dict(id=ep.id, dataset=dataset, event=entry['event'], failed=bool(final['error']),
                       submitted_by=final['submitted_by'], termination_reason=final['termination_reason'],
                       routing=tool_stats, **scores(**pixel_counts(prediction(pred_path), gt)))
            rows.append(row)
            evidence.append(dict(id=ep.id, image_t1=data / 'inputs' / ep.image_t1, image_t2=data / 'inputs' / ep.image_t2,
                                 pred_mask=pred_path, gt_mask=gt_path, summary=f'{entry["event"]}: {final["summary"]}'))

        summaries[dataset] = aggregate(rows)
        name = 'BRIGHT' if dataset == 'bright' else 'xBD'
        build_evidence_page(evidence, work / 'viz' / dataset, title=f'{name} disaster building damage probe',
                            metrics=summaries[dataset])
        all_rows.extend(rows)
    expected = 1 if smoke else sum(COUNTS.values())
    result = dict(count=len(all_rows), failed=sum(row['failed'] for row in all_rows), datasets=summaries,
                  cases=all_rows,
                  conventions=dict(empty_union_f1_iou=1, undefined_precision_recall=0,
                                   routing='Planner tool attempts (includes failures; excludes post-submit skips). '
                                           'Cost includes Planner and Verifier session requests, excluding smoke preflight.'))
    write_run_contract(work, track=TRACK, run_name=f'{RUN_NAME}_smoke' if smoke else RUN_NAME,
                       config=dict(model=MODEL, verifier=MODEL, reply_budget=BUDGET, max_cost_cny=max_cost),
                       metrics=result)
    verify_run_dir(work, expected_count=expected)
    print(f'Run contract passed: {len(all_rows)} cases; metrics and evidence written to {work}', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='action', required=True)
    online, offline = sub.add_parser('infer'), sub.add_parser('report')
    for cmd in (online, offline):
        cmd.add_argument('--work', type=Path, required=True)
        cmd.add_argument('--smoke', action='store_true')
        cmd.add_argument('--max-cost-cny', type=float, default=40.)
    online.add_argument('--inputs', type=Path, required=True)
    online.add_argument('--sam3-worker', required=True)
    offline.add_argument('--data', type=Path, required=True)
    args = parser.parse_args()
    if not math.isfinite(args.max_cost_cny) or not 0 < args.max_cost_cny <= 40:
        parser.error('--max-cost-cny must be finite and in (0, 40]')
    if args.action == 'infer':
        infer(args.inputs.absolute(), args.work.absolute(), args.sam3_worker, args.max_cost_cny, smoke=args.smoke)
    else:
        report(args.data.absolute(), args.work.absolute(), args.max_cost_cny, smoke=args.smoke)


if __name__ == '__main__':
    main()
