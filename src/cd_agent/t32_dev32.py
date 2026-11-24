"""Stage, run and score T3.2 single-date reference Dev-32."""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
import json
from pathlib import Path
import random
import shutil

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from .evaluation import pixel_counts

DATASETS = ('HIUCDMINI_tile512', 'SECOND', 'HRMS_SCD', 'SCSCD7', 'lsSCD')
SEED = 321118
MODEL = 'qwen3.7-flash'
SIZE = (512, 512)
SPLIT = 'dev'
INSTRUCTION = (
    'The Visual reference image has 3 panels from left to right: (1) original reference scene, '
    '(2) green-tinted TARGET land-cover category (unshaded regions and unshaded holes are NOT the target), '
    'and (3) binary mask where WHITE = TARGET category and BLACK = NOT target. '
    'The reference and query are from different places and have no spatial correspondence. '
    'Match at the broad land-cover category level (e.g., buildings regardless of roof color or subtype, '
    'bare land, farmland, grass/vegetation, road, water, playground). '
    'Find all real changes between Query T1 and Query T2 involving that same category, including appearance, '
    'disappearance, or conversion from or to another category. If there is no such change, submit an all-zero mask. '
    'The submit tool writes your final 512×512 single-channel binary PNG to prediction.png in the session output directory.'
)


def render_reference_panel(reference, mask, tile_size=384, header_h=28):
    """Render an unambiguous 3-panel visual reference: Original | Green-tinted TARGET | Binary White=TARGET."""
    ref_rgb = reference.convert('RGB')
    mask_l = mask.convert('L')
    mask_bool = np.asarray(mask_l) != 0
    ref_arr = np.asarray(ref_rgb, dtype=np.float32)

    # Shade strictly inside foreground pixels; draw 1px inner edge so excluded holes never get enclosed.
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
    width = tile_size * 3 + gap * 2
    height = header_h + tile_size
    canvas = Image.new('RGB', (width, height), (24, 28, 34))
    draw = ImageDraw.Draw(canvas)
    for idx, (title, img) in enumerate(panels):
        x0 = idx * (tile_size + gap)
        draw.rectangle([x0, 0, x0 + tile_size - 1, header_h - 1], fill=(32, 38, 46))
        draw.text((x0 + 8, 8), title, fill=(240, 245, 250))
        canvas.paste(img, (x0, header_h))
    return canvas


def load(path):
    return json.loads(path.read_text())


def save(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(path)


def read_image(path, mode=None):
    with Image.open(path) as image:
        if image.size != SIZE:
            raise ValueError(f'Unexpected image dimensions: {path}')
        return image.convert(mode) if mode else image.copy()


def class_key(row):
    return row['meta']['source_class_name'] or row['meta']['target_class_name']


def area_bin(pixels):
    if pixels == 0:
        return 'empty'
    if pixels < 4096:
        return '<4096'
    if pixels < 16384:
        return '4096–16383'
    if pixels < 65536:
        return '16384–65535'
    return '≥65536'


def sample(rows):
    rng = random.Random(SEED)
    by_dataset = {dataset: [] for dataset in DATASETS}
    for index, row in enumerate(rows):
        meta = row['meta']
        ref = row['reference']
        if (meta['dataset'] in by_dataset and meta['form'] == 1 and meta['prompt_mode'] == 'na'
                and meta['image_pixels'] == 512**2 and len(ref['image']) == 1 and ref.get('mask')
                and ref['image'][0] not in row['image']):
            by_dataset[meta['dataset']].append((index, row))

    selected, groups = [], set()
    negatives = sorted(DATASETS, key=lambda ds: -sum(row['meta']['is_negative'] for _, row in by_dataset[ds]))[:3]
    for dataset, count in zip(DATASETS, (7, 7, 6, 6, 6)):
        pool = by_dataset[dataset]
        rng.shuffle(pool)
        classes, areas, strata = Counter(), Counter(), Counter()
        for slot in range(count):
            is_negative = dataset in negatives and slot == count - 1
            eligible = [entry for entry in pool if entry[1]['meta']['is_negative'] == is_negative
                        and tuple(entry[1]['meta']['identity_group']) not in groups]
            def rank(entry):
                row = entry[1]
                cls, area = class_key(row), area_bin(row['meta']['positive_pixels'])
                return (classes[cls] > 0, areas[area] > 0, strata[cls, area], classes[cls], areas[area])
            index, row = min(eligible, key=rank)
            cls, area = class_key(row), area_bin(row['meta']['positive_pixels'])
            groups.add(tuple(row['meta']['identity_group']))
            classes[cls] += 1
            areas[area] += 1
            strata[cls, area] += 1
            selected.append((index, row))
    rng.shuffle(selected)
    return selected


def prepare(manifest, out, reuse_from=None):
    if reuse_from is not None:
        source_dir = Path(reuse_from).resolve()
        sample_data = load(source_dir / 'sample.json')
        out.mkdir(parents=True, exist_ok=True)
        public = []
        for record in sample_data['cases']:
            case = record['case']
            src_folder = source_dir / 'inputs' / case
            dst_folder = out / 'inputs' / case
            dst_folder.mkdir(parents=True, exist_ok=True)
            for fname in ('query-a.png', 'query-b.png', 'query-a.jpg', 'query-b.jpg', 'reference.png', 'reference-mask.png'):
                shutil.copyfile(src_folder / fname, dst_folder / fname)
            reference = read_image(dst_folder / 'reference.png', 'RGB')
            mask = read_image(dst_folder / 'reference-mask.png', 'L')
            render_reference_panel(reference, mask).save(dst_folder / 'reference-view.jpg', quality=92)
            public.append(case)
        save(out / 'inputs' / 'cases.json', public)
        save(out / 'sample.json', sample_data)
        print('Reused and refreshed', len(public), 'cases into', out, flush=True)
        return
    if (out / 'sample.json').exists():
        print('Using the existing fixed sample', flush=True)
        return
    rows = sample(load(manifest))
    out.mkdir(parents=True, exist_ok=True)
    public, audit = [], []
    for number, (index, row) in enumerate(rows, 1):
        case = f'case-{number:02d}'
        folder = out / 'inputs' / case
        folder.mkdir(parents=True, exist_ok=True)
        ref, meta = row['reference'], row['meta']
        for suffix, path in zip(('a', 'b'), row['image']):
            image = read_image(path, 'RGB')
            image.save(folder / f'query-{suffix}.png')
            image.save(folder / f'query-{suffix}.jpg', quality=95)
        reference = read_image(ref['image'][0], 'RGB')
        mask = Image.fromarray((np.asarray(read_image(ref['mask'])) != 0).astype(np.uint8) * 255)
        if mask.mode != 'L' or not mask.getbbox():
            raise ValueError(f'{case}: expected a nonempty reference mask')
        reference.save(folder / 'reference.png')
        mask.save(folder / 'reference-mask.png')
        render_reference_panel(reference, mask).save(folder / 'reference-view.jpg', quality=92)
        public.append(case)
        audit.append(dict(case=case, manifest_row=index, gt=row['label'][0], dataset=meta['dataset'],
                          gt_pixels=meta['positive_pixels'], ignore_pixels=meta['ignore_pixels'],
                          area_bin=area_bin(meta['positive_pixels']), is_negative=meta['is_negative'],
                          source_class_name=meta['source_class_name'], target_class_name=meta['target_class_name']))
    save(out / 'inputs' / 'cases.json', public)
    save(out / 'sample.json', dict(seed=SEED, source=str(manifest), count=len(rows), cases=audit))
    print('Prepared', len(rows), 'cases', Counter(row['dataset'] for row in audit), flush=True)


def infer(inputs, traces, worker, concurrency, limit, smoke=False):
    from .backends import ApiAsk, SegmentBackend, SharedPrompter
    from .loop_json import run_session
    from .planner import Planner, RunGuard, stop_path
    from .recorder import Recorder
    from .state import Session
    from .tasks import seed_for
    from .tools import Toolbox

    guard = RunGuard(stop_path(traces / 'STOP.json'))
    prompter = SharedPrompter()
    cases = load(inputs / 'cases.json')[:limit]

    def process(case):
        directory = traces / case
        if guard.stopped():
            raise RuntimeError('The API run guard stopped evaluation')
        if (directory / 'prediction.png').exists():
            return
        folder = inputs / case
        sources = [folder / 'query-a.png', folder / 'query-b.png']
        rgb = [np.asarray(read_image(path, 'RGB')) for path in sources]
        reference = folder / 'reference-view.jpg'
        session = Session(rgb[0].shape[:2], target='any change', instruction=INSTRUCTION)
        session.visual_query = {'Query T1': folder / 'query-a.jpg', 'Query T2': folder / 'query-b.jpg',
                                'Visual reference': reference}
        session.reference_view = reference
        session.reference_paths = (folder / 'reference.png', folder / 'reference-mask.png')
        recorder = Recorder(directory, SPLIT, case, evaluate=False)
        planner = Planner(guard=guard, parameters={'model': MODEL}, seed=seed_for(case, SEED))
        backend = SegmentBackend(SPLIT, case, worker, target=session.target, image_paths=list(map(str, sources)))
        verifier = ApiAsk(guard=guard, model=MODEL, correspondence=True, session=session)
        toolbox = Toolbox(rgb, directory, backend, prompter.for_image(SPLIT, case, rgb=rgb), verifier,
                          session=session, python=True, big_box_share=0.5, answer_state=True,
                          explicit_answer=True, segment_pair=True, sam3_semantic=True)
        if smoke:
            # Exercise both providers (instance + semantic head) before starting the ordinary session.
            backend('t1', 'building', 0, .5, mode='fused')
            answer, _, _, _ = verifier.one([read_image(path, 'RGB') for path in sources], '', 0)
            if answer['answer'] not in ('yes', 'no'):
                raise RuntimeError('Verifier returned no decision')
        result = run_session(toolbox, planner, recorder, task='T3.2', prompt_version='refcd',
                             verify=True, gate3=True, status=True, answer_state=True,
                             explicit_answer=True, segment_pair=True, sam3_semantic=True,
                             metadata=dict(session_id=case, source_images=list(map(str, sources)),
                                           reference_view=str(reference), reference_image=str(folder / 'reference.png'),
                                           reference_mask=str(folder / 'reference-mask.png'),
                                           first_images={label: str(path) for label, path in session.visual_query.items()},
                                           backends=dict(ask='api', ask_model=MODEL, verify_stage=True,
                                                         ask_correspondence=True, sam3_semantic=True,
                                                         explicit_answer=True, segment_pair=True)))
        final_mask = directory / result['mask_path']
        with Image.open(final_mask) as image:
            if image.size != SIZE or image.format != 'PNG' or image.mode not in ('L', '1'):
                raise ValueError(f'{case}: invalid prediction format')
        temporary = directory / 'prediction.tmp.png'
        shutil.copyfile(final_mask, temporary)
        temporary.replace(directory / 'prediction.png')
        print(case, result['submitted_by'], 'rounds', result['planning_rounds'], flush=True)
        if smoke and (result['error'] or not result['planning_rounds']):
            raise RuntimeError(f'{case}: smoke session failed before producing a valid model trace')

    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        for _ in pool.map(process, cases):
            pass


def metrics(tp, fp, fn):
    return dict(tp=tp, fp=fp, fn=fn, iou=tp / (tp + fp + fn) if tp + fp + fn else 1.,
                f1=2 * tp / (2 * tp + fp + fn) if tp + fp + fn else 1.,
                precision=tp / (tp + fp) if tp + fp else 0.,
                recall=tp / (tp + fn) if tp + fn else 0.)


def score_case(record, directory):
    raw = np.asarray(read_image(record['gt']))
    gt, valid = raw == 1, raw != 255
    if raw.ndim != 2 or not np.isin(raw, [0, 1, 255]).all():
        raise ValueError(f'{record["case"]}: unexpected GT encoding')
    if gt.sum() != record['gt_pixels'] or (~valid).sum() != record['ignore_pixels']:
        raise ValueError(f'{record["case"]}: GT pixel counts differ from the official manifest')
    Image.fromarray(gt.astype(np.uint8) * 255).save(directory / 'offline-target.png')
    if (~valid).any():
        Image.fromarray((~valid).astype(np.uint8) * 255).save(directory / 'offline-ignore.png')

    def measure(path):
        return metrics(**pixel_counts(np.asarray(read_image(path)) != 0, gt, valid))

    @lru_cache(maxsize=None)
    def handle_metric(handle):
        return measure(directory / 'masks' / (handle.replace('#', '_') + '.png')) if handle else None

    def edge_value(after, before):
        current, previous = handle_metric(after), handle_metric(before)
        return dict(current, delta_f1=current['f1'] - previous['f1'] if previous else None) if current else None

    info = load(directory / 'input.json')
    info.update(gt_mask_path='offline-target.png', gt_ignore_mask_path='offline-ignore.png' if (~valid).any() else None,
                reference_reveal={key: record[key] for key in ('dataset', 'source_class_name', 'target_class_name', 'is_negative')})
    save(directory / 'input.json', info)
    steps = [json.loads(line) for line in (directory / 'steps.jsonl').read_text().splitlines()]
    previous = None
    for step in steps:
        if 'current_mask' in step:
            step['eval'] = edge_value(step['current_mask'], previous)
            previous = step['current_mask']
        for call in step['calls']:
            call['eval'] = edge_value(call.get('current_after'), call.get('current_before'))
    temporary = directory / 'steps.jsonl.tmp'
    temporary.write_text(''.join(json.dumps(step, ensure_ascii=False) + '\n' for step in steps))
    temporary.replace(directory / 'steps.jsonl')
    final = load(directory / 'final.json')
    metric = measure(directory / 'prediction.png')
    final.update(eval=metric, precision=metric['precision'], recall=metric['recall'],
                 final_delta=edge_value(final['final_handle'], previous),
                 false_alarm_pixels=metric['fp'] if record['is_negative'] else None)
    save(directory / 'final.json', final)
    return dict(**{key: value for key, value in record.items() if key != 'gt'}, **metric,
                submitted_by=final['submitted_by'], termination_reason=final['termination_reason'],
                planning_rounds=final['planning_rounds'], tool_calls=final['tool_calls'],
                false_alarm_pixels=final['false_alarm_pixels'])


def aggregate(rows):
    negatives = [row for row in rows if row['is_negative']]
    positives = [row for row in rows if not row['is_negative']]
    keys = ('iou', 'f1', 'precision', 'recall')
    def mean(group):
        return {key: sum(row[key] for row in group) / len(group) for key in keys} if group else None
    return dict(count=len(rows), positives=len(positives), negatives=len(negatives),
                submissions=dict(Counter(row['submitted_by'] for row in rows)),
                micro=metrics(**{key: sum(row[key] for row in rows) for key in ('tp', 'fp', 'fn')}),
                macro=mean(rows), positive_macro=mean(positives),
                negative_false_alarms=sum(row['false_alarm_pixels'] for row in negatives),
                negative_empty_predictions=sum(row['false_alarm_pixels'] == 0 for row in negatives))


def report(out):
    sample_data = load(out / 'sample.json')
    traces = out / 'traces'
    records = sample_data['cases']
    missing = [row['case'] for row in records if not (traces / row['case'] / 'prediction.png').is_file()]
    if missing:
        raise ValueError('Missing predictions: ' + ', '.join(missing))
    rows = [score_case(row, traces / row['case']) for row in records]
    summary = dict(overall=aggregate(rows),
                   datasets={ds: aggregate([row for row in rows if row['dataset'] == ds]) for ds in DATASETS},
                   cases=rows)
    save(out / 'metrics.json', summary)
    from .t32_replay import build_replays
    build_replays(out, summary)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    stage = sub.add_parser('prepare')
    stage.add_argument('--manifest', type=Path)
    stage.add_argument('--reuse-from', type=Path)
    stage.add_argument('--out', type=Path, required=True)
    run = sub.add_parser('infer')
    run.add_argument('--inputs', type=Path, required=True)
    run.add_argument('--traces', type=Path, required=True)
    run.add_argument('--sam3-worker', required=True)
    run.add_argument('--concurrency', type=int, default=3)
    run.add_argument('--limit', type=int)
    run.add_argument('--smoke', action='store_true')
    finish = sub.add_parser('report')
    finish.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if args.command == 'prepare':
        if not args.reuse_from and not args.manifest:
            parser.error('prepare requires either --manifest or --reuse-from')
        prepare(args.manifest.resolve() if args.manifest else None, args.out.resolve(),
                reuse_from=args.reuse_from.resolve() if args.reuse_from else None)
    elif args.command == 'infer':
        infer(args.inputs.resolve(), args.traces.resolve(), args.sam3_worker, args.concurrency, args.limit, args.smoke)
    else:
        report(args.out.resolve())


if __name__ == '__main__':
    main()
