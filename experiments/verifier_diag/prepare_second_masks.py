"""Freeze SECOND requests, generate masks without labels, then export supervision."""
from __future__ import annotations

import argparse
from collections import Counter
import importlib.util
import json
import os
from pathlib import Path
import re
import time

import numpy as np
from PIL import Image

from cd_agent import evidence, paths, verify

# Support both python -m experiments.verifier_diag.prepare_second_masks and direct use.
if __package__:
    from .scannet_adapter import SCanNetAdapter, checkpoint_path
else:
    from scannet_adapter import SCanNetAdapter, checkpoint_path


SPLIT_DIR = paths.REPO_ROOT / 'configs' / 'splits'
SPLIT_COUNTS = {'train': 2568, 'val': 400, 'test': 1694}
METHODS = ('scannet', 'omniovcd', 'rea_ovcd')
CLASS_NAMES = ('unchanged', 'low vegetation', 'non-vegetated ground', 'trees',
               'water bodies', 'buildings', 'sports fields')
CLEAR = (3, 4, 5, 6)
ALL_CLASSES = (1, 2, 3, 4, 5, 6)
CLASS_MODES = {'clear4': CLEAR, 'all6': ALL_CLASSES}
DEFAULT_CLASS_MODE = 'clear4'
TASK_TYPES = ('related', 'into', 'outof', 'transition')
MIN_AREA, IGNORE_RADIUS = 16, 2
TRANSITION_COVERAGE, MAX_TRANSITION_FRAGMENTS = .95, 1
CONNECTIVITY = np.ones((3, 3), dtype=bool)
BANDS = ('[0,0.3)', '[0.3,0.7)', '[0.7,0.9)', '[0.9,1.0]')
PLAN = 'inference_plan.json'
VERSION = 1


def official_names(split: str, count: int) -> list[str]:
    return [f'{split}_{i}.png' for i in range(count)]


def frozen_splits(directory: Path = SPLIT_DIR) -> dict[str, list[str]]:
    splits = {key: (directory / f'second_{key}.txt').read_text().split() for key in SPLIT_COUNTS}
    report = verify.check_split_isolation(splits)
    if report['counts'] != SPLIT_COUNTS or any(len(splits[k]) != n for k, n in SPLIT_COUNTS.items()):
        raise ValueError(f'Wrong frozen split counts or duplicate IDs: {report["counts"]}')
    if (set(splits['train'] + splits['val']) != set(official_names('train', 2968))
            or splits['test'] != official_names('test', 1694)):
        raise ValueError('Frozen SECOND splits must exactly cover the official source IDs')
    return splits


def task_catalog(class_mode: str = DEFAULT_CLASS_MODE) -> list[dict]:
    """Enumerate without labels; eligibility/transition completeness is measured in export."""
    if class_mode not in CLASS_MODES:
        raise ValueError(f'Unknown SECOND class_mode: {class_mode}')
    active = CLASS_MODES[class_mode]
    active_set = set(active)
    tasks = []
    for c in active:
        name = CLASS_NAMES[c]
        prompts = (f'Find all changes involving {name}.',
                   f'Find areas that became {name}.',
                   f'Find areas where {name} changed into other land cover.')
        for kind, prompt in zip(TASK_TYPES[:3], prompts):
            tasks.append(dict(id=f'{kind}_{c}', task_type=kind, task_params={'class_id': c},
                              instruction=prompt, prompt_en=prompt))
    for a in range(1, len(CLASS_NAMES)):
        for b in range(1, len(CLASS_NAMES)):
            if a == b or not ({a, b} & active_set):
                continue
            prompt = f'Find areas that changed from {CLASS_NAMES[a]} to {CLASS_NAMES[b]}.'
            tasks.append(dict(id=f'transition_{a}_{b}', task_type='transition',
                              task_params={'source_class': a, 'target_class': b},
                              instruction=prompt, prompt_en=prompt))
    return tasks


def compile_target(change: np.ndarray, t1: np.ndarray, t2: np.ndarray, task: dict) -> np.ndarray:
    """Apply the same task algebra to GT and to the three SCanNet predictions."""
    kind, p = task['task_type'], task['task_params']
    if kind == 'transition':
        return change & (t1 == p['source_class']) & (t2 == p['target_class'])
    c = p['class_id']
    src, dst = t1 == c, t2 == c
    if kind == 'related':
        return change & (src | dst)
    if kind == 'into':
        return change & ~src & dst
    if kind == 'outof':
        return change & src & ~dst
    raise ValueError(f'Unknown SECOND task: {kind}')


def complete_transition(raw: np.ndarray, proxy: np.ndarray) -> bool:
    """Require >=95% coverage and at most one fragment in every touched proxy entity."""
    if int(raw.sum()) < MIN_AREA:
        return False
    from scipy import ndimage as ndi

    labels, _ = ndi.label(proxy, CONNECTIVITY)
    ids, counts = np.unique(labels[raw], return_counts=True)
    sizes = np.bincount(labels.ravel())
    if not len(ids) or 0 in ids or np.any(counts / sizes[ids] < TRANSITION_COVERAGE):
        return False
    boxes = ndi.find_objects(labels)
    for ident in ids:
        box = boxes[ident - 1]
        if ndi.label(raw[box] & (labels[box] == ident), CONNECTIVITY)[1] > MAX_TRANSITION_FRAGMENTS:
            return False
    return True


def eligible_targets(change: np.ndarray, t1: np.ndarray, t2: np.ndarray, tasks: list[dict]) -> dict:
    """Compile eligible targets with C-domain entity proxy for transitions."""
    targets = {}
    for task in tasks:
        p = task['task_params']
        raw = compile_target(change, t1, t2, task)
        if task['task_type'] == 'transition':
            a, b = p['source_class'], p['target_class']
            # Prefer a clear-class endpoint as entity proxy; for 1<->2 use destination b.
            proxy = change & ((t2 == b) if (b in CLEAR or a not in CLEAR) else (t1 == a))
            if not complete_transition(raw, proxy):
                continue
        else:
            c = p['class_id']
            src, dst = t1 == c, t2 == c
            if int((change & (src | dst)).sum()) < MIN_AREA or not (change & (src ^ dst)).any():
                continue
        targets[task['id']] = raw
    return targets


def read_json(path: Path):
    return json.loads(path.read_text(encoding='utf-8'))


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f'.{path.name}.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False) + '\n', encoding='utf-8')
    tmp.replace(path)


def prepare(out: Path, split: str, *, limit: int | None = None, shard: int = 0,
            num_shards: int = 1, dataset: Path | None = None,
            class_mode: str = DEFAULT_CLASS_MODE) -> dict:
    if not 0 <= shard < num_shards:
        raise ValueError('Expected 0 <= shard < num_shards')
    if limit is not None and limit < 1:
        raise ValueError('limit must be positive')
    if class_mode not in CLASS_MODES:
        raise ValueError(f'Unknown SECOND class_mode: {class_mode}')
    names = frozen_splits()[split][shard::num_shards][:limit]
    if not names:
        raise ValueError('The requested SECOND shard/limit has no pairs')
    dataset = (dataset or paths.dataset_path('SECOND')).absolute()
    pairs = [dict(id=name, image_t1=str(dataset / 'img1' / name),
                  image_t2=str(dataset / 'img2' / name)) for name in names]
    for pair in pairs:
        for key in ('image_t1', 'image_t2'):
            if not Path(pair[key]).is_file():
                raise FileNotFoundError(pair[key])
    plan = dict(version=VERSION, split=split, shard=shard, num_shards=num_shards,
                tasks=task_catalog(class_mode), pairs=pairs)
    if class_mode != DEFAULT_CLASS_MODE:
        plan['class_mode'] = class_mode
    path = out / PLAN
    if path.exists():
        if read_json(path) != plan:
            raise ValueError(f'{path} already freezes a different request; use a new output directory')
    else:
        write_json(path, plan)
    return plan


def load_plan(out: Path) -> dict:
    plan = read_json(out / PLAN)
    class_mode = plan.get('class_mode', DEFAULT_CLASS_MODE)
    if plan.get('version') != VERSION or class_mode not in CLASS_MODES or plan.get('tasks') != task_catalog(class_mode):
        raise ValueError('Unknown inference plan version or unexpected task fields')
    splits = frozen_splits()
    shard, num_shards = plan['shard'], plan['num_shards']
    if not 0 <= shard < num_shards or not plan['pairs']:
        raise ValueError('Invalid inference plan shard')
    names = splits[plan['split']][shard::num_shards][:len(plan['pairs'])]
    if [p['id'] for p in plan['pairs']] != names:
        raise ValueError('Inference plan pairs do not match the frozen split')
    for pair in plan['pairs']:
        if set(pair) != {'id', 'image_t1', 'image_t2'}:
            raise ValueError('Inference plan pairs may contain only an ID and two images')
        for key, folder in (('image_t1', 'img1'), ('image_t2', 'img2')):
            image = Path(pair[key])
            if not image.is_absolute() or image.name != pair['id'] or image.parent.name != folder:
                raise ValueError(f'Unexpected SECOND image path: {image}')
        if Path(pair['image_t1']).parents[1] != Path(pair['image_t2']).parents[1]:
            raise ValueError('SECOND dates must share a dataset root')
    return plan


def save_mask(path: Path, mask: np.ndarray) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f'.{path.name}.tmp')
    Image.fromarray(mask.astype(bool)).save(tmp, format='PNG')
    tmp.replace(path)
    return str(path)


def read_pixels(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.array(image)


def mask_name(task: dict) -> str:
    return f'{task["id"]}.png'


def generation_dir(out: Path, method: str, pair: dict) -> Path:
    return out / 'generation' / method / Path(pair['id']).stem


def model_settings(method: str, override: Path | None) -> dict:
    if method == 'scannet':
        return {'checkpoint': str(checkpoint_path(override))}
    return {'checkpoint': str(paths.model_path('sam3', 'sam3.pt').resolve(strict=True)),
            'source': str((paths.THIRD_PARTY_DIR / ('OmniOVCD' if method == 'omniovcd' else 'ReA-OVCD'))
                          .resolve(strict=True))}


def build_adapter(method: str, work: Path, settings: dict):
    if method == 'scannet':
        adapter = SCanNetAdapter(Path(settings['checkpoint']))

        def predict(t1, t2, tasks):
            maps = adapter.infer_pair(t1, t2)
            return [compile_target(maps['B'], maps['Y1'], maps['Y2'], task) for task in tasks]

        return predict

    source = (paths.REPO_ROOT / 'experiments' / 'verifier_init_masks' / 'initial_masks'
              / 'verifier_v1_ovcd_methods.py')
    spec = importlib.util.spec_from_file_location('second_ovcd_methods', source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    work.mkdir(parents=True, exist_ok=True)
    adapter = (module.OmniAdapter if method == 'omniovcd' else module.ReAAdapter)(work)

    def predict(t1, t2, tasks):
        # The legacy adapter has a wider request schema; send only image/task data.
        row = dict(id=Path(t1).stem, dataset='SECOND', image_t1=t1, image_t2=t2, pilot_tasks=tasks)
        masks, _ = adapter.infer(row)
        return masks

    return predict


def receipt_for(pair: dict, tasks: list[dict], method: str, settings: dict) -> dict:
    return dict(version=VERSION, method=method, settings=settings, pair=pair,
                tasks=tasks, masks=[mask_name(task) for task in tasks])


def completed_receipt(directory: Path, pair: dict, tasks: list[dict], method: str,
                      settings: dict | None = None) -> dict | None:
    path = directory / 'receipt.json'
    if not path.is_file():
        return None
    receipt = read_json(path)
    if settings is None:
        settings = receipt['settings']
    expected = receipt_for(pair, tasks, method, settings)
    if any(receipt.get(key) != value for key, value in expected.items()):
        raise ValueError(f'{path} is for different inputs or model settings; use a new output directory')
    if not all((directory / name).is_file() for name in expected['masks']):
        raise ValueError(f'Completed receipt {path} is missing task masks')
    return receipt


def infer(out: Path, method: str, *, limit: int | None = None, scannet_ckpt: Path | None = None) -> None:
    if not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('GPU inference must run inside Slurm (SLURM_JOB_ID is required)')
    if limit is not None and limit < 1:
        raise ValueError('limit must be positive')
    if scannet_ckpt is not None and method != 'scannet':
        raise ValueError('--scannet-ckpt is only valid for --method scannet')
    os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
    plan = load_plan(out)
    tasks = plan['tasks']
    settings = model_settings(method, scannet_ckpt)
    predictor = None
    for pair in plan['pairs'][:limit]:
        directory = generation_dir(out, method, pair)
        if completed_receipt(directory, pair, tasks, method, settings) is not None:
            print(f'{method} {pair["id"]}: already complete', flush=True)
            continue
        if predictor is None:
            predictor = build_adapter(method, out / 'generation' / method / 'work', settings)
        started = time.monotonic()
        with Image.open(pair['image_t1']) as t1, Image.open(pair['image_t2']) as t2:
            if t1.size != t2.size:
                raise ValueError(f'{pair["id"]}: mismatched image sizes')
            shape = (t1.height, t1.width)
        masks = predictor(pair['image_t1'], pair['image_t2'], tasks)
        if len(masks) != len(tasks):
            raise ValueError(f'{method} returned {len(masks)} masks for {len(tasks)} tasks')
        for task, mask in zip(tasks, masks):
            binary = verify.check_binary_mask(mask, expected_shape=shape, name=task['id'])
            save_mask(directory / mask_name(task), binary)
        receipt = receipt_for(pair, tasks, method, settings)
        receipt.update(seconds=round(time.monotonic() - started, 3), slurm_job_id=os.environ['SLURM_JOB_ID'])
        write_json(directory / 'receipt.json', receipt)
        print(f'{method} {pair["id"]}: {receipt["seconds"]}s', flush=True)


def load_truth(pair: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Read labels only in export; inference never calls this function."""
    t1_path = Path(pair['image_t1'])
    dataset = t1_path.parents[1]
    name = pair['id']
    with Image.open(t1_path) as image:
        shape = (image.height, image.width)
    change = verify.check_binary_mask(read_pixels(dataset / 'label_unch0chi1' / name),
                                      expected_shape=shape, name=f'{name}: binary GT')
    t1, t2 = (read_pixels(dataset / folder / name) for folder in ('slabel1', 'slabel2'))
    for label in (t1, t2):
        if label.shape != shape or label.dtype.kind not in 'iu' or ((label < 0) | (label > 6)).any():
            raise ValueError(f'{name}: invalid SECOND semantic labels; expected IDs 0..6 at shape {shape}')
    if ((t1 == 0) ^ (t2 == 0)).any():
        raise ValueError(f'{name}: one-sided non-change semantic labels')
    return change, t1, t2


def tiny_components(mask: np.ndarray) -> np.ndarray:
    from scipy import ndimage as ndi

    labels, _ = ndi.label(mask, CONNECTIVITY)
    counts = np.bincount(labels.ravel())
    small = counts < MIN_AREA
    small[0] = False
    return small[labels]


def clean_diagnostic(pred: np.ndarray, raw: np.ndarray) -> dict:
    """Reuse pilot cleaning: tiny components and matched boundaries are ignored.

    Non-tiny unmatched bodies remain supervised, even when near a matched boundary.
    Candidate pixels are never modified.
    """
    from scipy import ndimage as ndi

    small_gt = tiny_components(raw)
    clean = raw & ~small_gt
    tiny = small_gt | tiny_components(pred & ~raw) | tiny_components(~pred & raw)
    intersection = pred & clean
    matched = []
    for mask in (pred, clean):
        labels, n = ndi.label(mask, CONNECTIVITY)
        ids = np.zeros(n + 1, dtype=bool)
        ids[labels[intersection]] = True
        matched.append(ids[labels])
    matched_pred, matched_gt = matched
    boundary = np.zeros(raw.shape, dtype=bool)
    for mask in matched:
        boundary |= mask & ~ndi.binary_erosion(mask, CONNECTIVITY)
    ignore = tiny.copy()
    if boundary.any():
        independent = (pred & ~matched_pred) | (clean & ~matched_gt)
        ignore |= (ndi.distance_transform_edt(~boundary) <= IGNORE_RADIUS) & ~independent
    valid = ~ignore
    return dict(gt=clean, ignore=ignore, fp=pred & ~clean & valid, fn=~pred & clean & valid)


def effective_difference(base: dict, switched: dict) -> int:
    return int(((base['gt'] ^ switched['gt']) & ~(base['ignore'] | switched['ignore'])).sum())


def label_metrics(pred: np.ndarray, labels: dict) -> dict:
    gt, ignore, fp, fn = (labels[key] for key in ('gt', 'ignore', 'fp', 'fn'))
    stats = verify.check_diagnostic_labels(pred, gt, fp, fn, ignore)
    tp = int((pred & gt & ~ignore).sum())
    union = tp + stats['fp_area'] + stats['fn_area']
    return dict(**stats, tp=tp, fp=stats['fp_area'], fn=stats['fn_area'],
                effective_foreground_union=union,
                iou=tp / union if union else None, f1=2 * tp / (tp + union) if union else None)


def difficulty(iou: float | None) -> str:
    if iou is None:
        return 'no_effective_foreground'
    return BANDS[0 if iou < .3 else 1 if iou < .7 else 2 if iou < .9 else 3]


def emit_record(out: Path, plan: dict, pair: dict, task: dict, method: str, pred: np.ndarray,
                labels: dict, init_path: Path | None, *, base: dict | None = None,
                diff: int = 0) -> dict:
    metrics = label_metrics(pred, labels)
    ident = f'{Path(pair["id"]).stem}__{method}__{task["id"]}'
    if base is not None:
        if task['id'] == base['task_id'] or diff <= 0 or str(init_path) != base['init_mask']:
            raise ValueError('Counterfactual must switch tasks on the same M_init with positive GT difference')
        ident = f'{base["id"]}__switch__{task["id"]}'
    folder = out / 'cases' / ident
    dirs = {k: save_mask(folder / f'{k}.png', labels[key])
            for k, key in (('gt_clean', 'gt'), ('fp', 'fp'), ('fn', 'fn'), ('ignore_mask', 'ignore'))}
    return dict(
        id=ident, dataset='SECOND', split=plan['split'], internal_split=plan['split'],
        official_split='test' if plan['split'] == 'test' else 'train',
        source_image_id=pair['id'], source_group_id=f'SECOND/{pair["id"]}',
        image_t1=pair['image_t1'], image_t2=pair['image_t2'],
        task_id=task['id'], task_type=task['task_type'], task_params=task['task_params'],
        instruction=task['instruction'], prompt_en=task['prompt_en'],
        generator=method, init_mask=str(init_path) if init_path else dirs['gt_clean'],
        gt_mask=dirs['gt_clean'], labels={k: dirs[k] for k in ('fp', 'fn', 'ignore_mask')},
        metrics=metrics, quality_band=difficulty(metrics['iou']), is_anchor=method == 'gt_anchor',
        is_counterfactual=base is not None, base_task_id=base['id'] if base else None,
        counterfactual_group_id=base['id'] if base else ident,
        effective_gt_diff_area=diff if base else None,
    )


def switch_task(pred: np.ndarray, base_task: dict, base_labels: dict,
                tasks: list[dict], targets: dict) -> tuple[dict, dict, int] | None:
    # Prefer a complete transition, then a different directional type, then another class.
    order = sorted(tasks, key=lambda t: (t['task_type'] != 'transition',
                                        t['task_type'] == base_task['task_type']))
    for task in order:
        key = task['id']
        if key == base_task['id'] or key not in targets:
            continue
        raw = targets[key]
        # The clean target is independent of the prediction; a raw match is never a switch.
        if np.array_equal(raw, targets[base_task['id']]):
            continue
        switched = clean_diagnostic(pred, raw)
        diff = effective_difference(base_labels, switched)
        if diff:
            return task, switched, diff
    return None


def histograms() -> dict:
    return {band: 0 for band in (*BANDS, 'no_effective_foreground')}


def show_evidence(out: Path, cases: list[dict], metrics: dict) -> Path:
    # The evidence helper decodes every supplied image; preselect from scalar stats
    # so full runs only decode the small review set, not the entire manifest again.
    selected = evidence.select_evidence_cases(cases)
    for row in selected:
        valid = ~read_pixels(Path(row.pop('ignore_mask'))).astype(bool)
        row['pred_mask'] = read_pixels(Path(row['pred_mask'])).astype(bool) & valid
        row['gt_mask'] = read_pixels(Path(row['gt_mask'])).astype(bool) & valid
    return evidence.build_evidence_page(selected, out / 'viz',
                                        title='SECOND initial masks (valid pixels only)', metrics=metrics)


def export(out: Path) -> dict:
    out = out.absolute()
    plan = load_plan(out)
    tasks = plan['tasks']
    settings = {}
    for pair in plan['pairs']:
        for method in METHODS:
            folder = generation_dir(out, method, pair)
            receipt = completed_receipt(folder, pair, tasks, method, settings.get(method))
            if receipt is None:
                raise FileNotFoundError(f'Missing generation receipt: {folder / "receipt.json"}')
            settings[method] = receipt['settings']

    metrics = dict(count=0, failed=0, pairs=len(plan['pairs']), pairs_without_tasks=0,
                   record_types=dict(generated=0, counterfactual=0, anchor=0),
                   by_generator={m: histograms() for m in METHODS},
                   by_task={t: histograms() for t in TASK_TYPES},
                   by_generator_task={m: {t: histograms() for t in TASK_TYPES} for m in METHODS})
    review = []
    manifest_tmp = out / '.manifest.jsonl.tmp'
    with manifest_tmp.open('w', encoding='utf-8') as stream:
        def record(row: dict, kind: str) -> None:
            stream.write(json.dumps(row, ensure_ascii=False) + '\n')
            metrics['count'] += 1
            metrics['record_types'][kind] += 1
            if kind == 'generated':
                method, task, band = row['generator'], row['task_type'], row['quality_band']
                metrics['by_generator'][method][band] += 1
                metrics['by_task'][task][band] += 1
                metrics['by_generator_task'][method][task][band] += 1
                stats = row['metrics']
                if stats['f1'] is not None:
                    review.append(dict(id=row['id'], image_t1=row['image_t1'], image_t2=row['image_t2'],
                                       pred_mask=row['init_mask'], gt_mask=row['gt_mask'],
                                       ignore_mask=row['labels']['ignore_mask'], instruction=row['instruction'],
                                       f1=stats['f1'], fp=stats['fp'], fn=stats['fn']))

        for pair in plan['pairs']:
            change, t1, t2 = load_truth(pair)
            targets = eligible_targets(change, t1, t2, tasks)
            if not targets:
                metrics['pairs_without_tasks'] += 1
                continue
            for method in METHODS:
                switched = False
                for task in tasks:
                    key = task['id']
                    if key not in targets:
                        continue
                    init_path = generation_dir(out, method, pair) / mask_name(task)
                    pred = verify.check_binary_mask(read_pixels(init_path), expected_shape=change.shape,
                                                    name=str(init_path))
                    labels = clean_diagnostic(pred, targets[key])
                    row = emit_record(out, plan, pair, task, method, pred, labels, init_path)
                    record(row, 'generated')
                    if not switched and pred.any():
                        alternate = switch_task(pred, task, labels, tasks, targets)
                        if alternate is not None:
                            alt_task, alt_labels, diff = alternate
                            row = emit_record(out, plan, pair, alt_task, method, pred,
                                              alt_labels, init_path, base=row, diff=diff)
                            record(row, 'counterfactual')
                            switched = True
            for task in tasks:
                if task['id'] not in targets:
                    continue
                raw = targets[task['id']]
                clean = raw & ~tiny_components(raw)
                labels = clean_diagnostic(clean, raw)
                if (clean & ~labels['ignore']).any():
                    record(emit_record(out, plan, pair, task, 'gt_anchor', clean, labels, None), 'anchor')
                    break
    if not metrics['record_types']['generated']:
        raise ValueError('Export found no pilot-eligible SECOND tasks')
    show_evidence(out, review, metrics)
    manifest_tmp.replace(out / 'manifest.jsonl')
    verify.write_run_contract(
        out, track='verifier_diag', run_name=out.name, metrics=metrics,
        config=dict(split=plan['split'], shard=plan['shard'], num_shards=plan['num_shards'],
                    class_mode=plan.get('class_mode', DEFAULT_CLASS_MODE),
                    task_catalog_version=VERSION, generators=settings, min_area=MIN_AREA,
                    ignore_radius=IGNORE_RADIUS, transition_coverage=TRANSITION_COVERAGE,
                    max_transition_fragments=MAX_TRANSITION_FRAGMENTS,
                    histograms_include='generated records only; no GT anchors or task switches'),
    )
    return metrics


def positive(value: str) -> int:
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError('must be positive')
    return number


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='stage', required=True)
    prep = sub.add_parser('prepare', help='Freeze image pairs and the label-free candidate catalog')
    prep.add_argument('--split', choices=SPLIT_COUNTS, required=True)
    prep.add_argument('--limit', type=positive)
    prep.add_argument('--shard', type=int, default=0)
    prep.add_argument('--num-shards', type=positive, default=1)
    prep.add_argument('--class-mode', choices=tuple(CLASS_MODES), default=DEFAULT_CLASS_MODE)
    inference = sub.add_parser('infer', help='Generate/resume task masks inside a GPU Slurm job')
    inference.add_argument('--method', choices=METHODS, required=True)
    inference.add_argument('--limit', type=positive)
    inference.add_argument('--scannet-ckpt', type=Path)
    exporting = sub.add_parser('export', help='Read GT and build diagnostic labels and evidence')
    for command in (prep, inference, exporting):
        command.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    out = args.out.absolute()
    if args.stage == 'prepare':
        plan = prepare(out, args.split, limit=args.limit, shard=args.shard, num_shards=args.num_shards,
                       class_mode=args.class_mode)
        print(f'Frozen {len(plan["pairs"])} {args.split} pairs and {len(plan["tasks"])} tasks ({args.class_mode}) in {out / PLAN}')
    elif args.stage == 'infer':
        infer(out, args.method, limit=args.limit, scannet_ckpt=args.scannet_ckpt)
    else:
        print(json.dumps(export(out), ensure_ascii=False))


if __name__ == '__main__':
    main()
