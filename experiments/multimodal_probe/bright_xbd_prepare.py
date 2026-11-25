"""Freeze an event-stratified damage sample and stage GT-free agent inputs."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict, deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import fcntl
import json
from pathlib import Path
import random

import numpy as np
from PIL import Image

from cd_agent.paths import DATASETS_DIR, output_dir
from cd_agent.verify import EpisodeInput, check_binary_mask

TRACK, RUN_NAME = 'multimodal_probe', '20261123_bright_xbd_probe'
SIZE = (1024, 1024)
SEED = 20261123
COUNTS = {'bright': 24, 'xbd': 12}
EVENT_COUNTS = {'bright': 9, 'xbd': 10}
DAMAGE = {'bright': (2, 3), 'xbd': (2, 3, 4)}
MODALITIES = {
    'bright': 'T1 is pre-disaster optical RGB imagery. T2 is post-disaster SAR intensity imagery, shown in grayscale.',
    'xbd': 'T1 is pre-disaster optical RGB imagery. T2 is post-disaster optical RGB imagery.',
}
DAMAGE_TASK = (
    'Detect only buildings damaged or destroyed by the disaster, including damaged buildings still standing. '
    'Do not include intact buildings or non-building changes. Use the original building footprints for destroyed '
    'buildings. Appearance differences alone are not evidence of damage.'
)
INSTRUCTIONS = {
    name: f'{modalities} {DAMAGE_TASK} '
    + ('segment_text is an optical segmentation model suited for T1 optical imagery; T2 is SAR, not optical. '
       if name == 'bright' else '')
    + 'Submit a single binary mask in the original 1024 by 1024 coordinates. If there is no damage, submit an empty mask.'
    for name, modalities in MODALITIES.items()
}


def load(path):
    return json.loads(path.read_text(encoding='utf-8'))


def save(path, value):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    temp.replace(path)


@dataclass(frozen=True)
class Source:
    key: str
    event: str
    t1: Path
    t2: Path
    target: Path


def discover(root, dataset):
    if dataset == 'bright':
        base = root / 'BRIGHT'
        val = base / 'dfc25_track2_trainval' / 'dfc25_track2_trainval' / 'val'
        t1, t2 = val / 'pre-event', val / 'post-event'
        targets = base / 'dfc25_track2_val_labels' / 'dfc25_track2_val_labels' / 'target'
        suffix, ext = '_building_damage.tif', '.tif'
    else:
        t1 = t2 = root / 'xBD' / 'hold' / 'images'
        targets = root / 'xBD' / 'hold' / 'targets'
        suffix, ext = '_post_disaster_target.png', '.png'
    groups = defaultdict(list)
    for label in sorted(targets.glob('*' + suffix)):
        key = label.name.removesuffix(suffix)
        event, tile = key.rsplit('_', 1)
        source = Source(key, event, t1 / f'{key}_pre_disaster{ext}', t2 / f'{key}_post_disaster{ext}', label)
        if not tile.isdecimal():
            raise ValueError(f'{dataset}: malformed tile name {key}')
        groups[event].append(source)
    if len(groups) != EVENT_COUNTS[dataset]:
        raise ValueError(f'{dataset}: expected {EVENT_COUNTS[dataset]} events, found {sorted(groups)}')
    return groups


def read_labels(path):
    with Image.open(path) as im:
        labels = np.array(im)
        if im.size != SIZE or labels.ndim != 2 or labels.dtype != np.uint8:
            raise ValueError(f'{path}: expected a {SIZE} single-channel uint8 label raster')
    return labels


def damage_mask(labels, dataset):
    return np.isin(labels, DAMAGE[dataset])


def select(groups, dataset, *, count, negatives, seed):
    """Read seeded candidate labels lazily; balance counts with event round-robin."""
    rng = random.Random(seed)
    events = sorted(groups)
    rng.shuffle(events)
    pools = {event: iter(rng.sample(groups[event], len(groups[event]))) for event in events}
    negative_pool = defaultdict(list)
    positive_buffer = defaultdict(deque)

    def classify(source):
        hist = np.bincount(read_labels(source.target).ravel(), minlength=5)
        return source, int(sum(hist[k] for k in DAMAGE[dataset])), int(hist[1])

    def take_positive(event):
        if positive_buffer[event]:
            return positive_buffer[event].popleft()
        for source in pools[event]:
            row = classify(source)
            if row[1]:
                return row
            negative_pool[event].append(row)
        return None

    rows = [take_positive(event) for event in events]
    if any(row is None for row in rows):
        missing = [event for event, row in zip(events, rows) if row is None]
        raise ValueError(f'{dataset}: no positive tile for events {missing}')
    smoke_row = rows[0]

    if negatives:
        # Prefer intact buildings over pure background and differing events over duplicates.
        # The initial positive sweep usually discovers enough intact negatives.
        for event in events:
            if sum(any(row[2] for row in pool) for pool in negative_pool.values()) >= negatives:
                break
            if any(row[2] for row in negative_pool[event]):
                continue
            for source in pools[event]:
                row = classify(source)
                if row[1]:
                    positive_buffer[event].append(row)
                    continue
                negative_pool[event].append(row)
                if row[2]:
                    break
        def ranked(row):
            return (not row[2], row[0].key)

        distinct = [min(pool, key=ranked) for pool in negative_pool.values() if pool]
        picked = sorted(distinct, key=ranked)[:negatives]
        if len(picked) < negatives:
            rest = [row for pool in negative_pool.values() for row in pool if row not in picked]
            picked.extend(sorted(rest, key=ranked)[:negatives - len(picked)])
        rows.extend(picked)

    event_size = Counter(row[0].event for row in rows)
    active = events.copy()
    while len(rows) < count and active:
        event = min(active, key=event_size.get)
        row = take_positive(event)
        if row is None:
            active.remove(event)
            continue
        rows.append(row)
        event_size[event] += 1
    if len(rows) != count:
        raise ValueError(f'{dataset}: could not select {count} eligible cases')
    rest = [row for row in rows if row != smoke_row]
    rng.shuffle(rest)
    return [smoke_row, *rest]


def read_rgb(path, *, sar=False, staged=False):
    with Image.open(path) as im:
        expected_mode = 'RGB' if staged or not sar else 'L'
        if im.size != SIZE or im.mode != expected_mode or (staged and im.format != 'PNG'):
            raise ValueError(f'{path}: expected {SIZE} {expected_mode} imagery')
        if not any(lo != hi for lo, hi in (im.getextrema() if im.mode == 'RGB' else [im.getextrema()])):
            raise ValueError(f'{path}: constant image')
        rgb = np.array(im.convert('RGB'))
    if sar and not (np.array_equal(rgb[..., 0], rgb[..., 1]) and np.array_equal(rgb[..., 0], rgb[..., 2])):
        raise ValueError(f'{path}: SAR channels differ')
    return rgb


def stage(out, dataset, number, selected):
    source, pixels, intact = selected
    case = f'{dataset}-{number:02d}'
    folder = out / 'inputs' / case
    folder.mkdir(parents=True, exist_ok=True)
    for time, path in (('t1', source.t1), ('t2', source.t2)):
        rgb = read_rgb(path, sar=dataset == 'bright' and time == 't2')
        Image.fromarray(rgb).save(folder / f'{time}.png', compress_level=1)
    labels = read_labels(source.target)
    mask = damage_mask(labels, dataset)
    if int(mask.sum()) != pixels:
        raise ValueError(f'{case}: damage pixel count changed during staging')
    gt = out / 'offline_gt' / f'{case}.png'
    Image.fromarray(mask.astype(np.uint8) * 255).save(gt)
    online = dict(id=case, image_t1=f'{case}/t1.png', image_t2=f'{case}/t2.png',
                  instruction=INSTRUCTIONS[dataset])
    audit = dict(id=case, dataset=dataset, event=source.event, source_id=source.key,
                 source_label=str(source.target), gt_mask=str(gt.relative_to(out)),
                 damage_pixels=pixels, intact_pixels=intact)
    if dataset == 'xbd':
        audit['severe_pixels'] = int(np.count_nonzero((labels == 3) | (labels == 4)))
    return online, audit


def load_inputs(inputs):
    rows = load(inputs / 'cases.json')
    if set(rows) != set(COUNTS):
        raise ValueError('Online manifest must contain exactly BRIGHT and xBD')
    episodes = {}
    for dataset, entries in rows.items():
        if len(entries) != COUNTS[dataset]:
            raise ValueError(f'{dataset}: invalid online sample size')
        episodes[dataset] = []
        for number, entry in enumerate(entries, 1):
            case = f'{dataset}-{number:02d}'
            if entry != dict(id=case, image_t1=f'{case}/t1.png', image_t2=f'{case}/t2.png',
                             instruction=INSTRUCTIONS[dataset]):
                raise ValueError(f'{case}: unexpected online manifest fields or values')
            episode = EpisodeInput.from_mapping(entry)
            for name in (episode.image_t1, episode.image_t2):
                path = inputs / name
                if path.resolve() != inputs.resolve() / name or not path.is_file():
                    raise ValueError(f'{case}: staged image is missing or resolves outside its expected path')
            episodes[dataset].append(episode)
    return episodes


def verify_sample(out):
    episodes = load_inputs(out / 'inputs')
    audit = load(out / 'sample_audit.json')['cases']
    indexed = {row['id']: row for row in audit}
    if len(audit) != sum(COUNTS.values()) or set(indexed) != {ep.id for eps in episodes.values() for ep in eps}:
        raise ValueError('Public and audit case IDs do not match')

    def check(entry):
        dataset, ep = entry
        row = indexed[ep.id]
        if row['dataset'] != dataset or row['gt_mask'] != f'offline_gt/{ep.id}.png':
            raise ValueError(f'{ep.id}: invalid audit mapping')
        for time, name in (('t1', ep.image_t1), ('t2', ep.image_t2)):
            read_rgb(out / 'inputs' / name, sar=dataset == 'bright' and time == 't2', staged=True)
        mask = check_binary_mask(read_labels(out / row['gt_mask']), expected_shape=SIZE, name=ep.id)
        if int(mask.sum()) != row['damage_pixels']:
            raise ValueError(f'{ep.id}: staged GT pixel count mismatch')
        if dataset == 'xbd' and not 0 <= row['severe_pixels'] <= row['damage_pixels']:
            raise ValueError(f'{ep.id}: invalid severe-only statistics')

    with ThreadPoolExecutor(max_workers=4) as workers:
        list(workers.map(check, [(dataset, ep) for dataset, eps in episodes.items() for ep in eps]))
    summary = {}
    for dataset in COUNTS:
        records = [row for row in audit if row['dataset'] == dataset]
        counts = Counter(row['event'] for row in records)
        negatives = sum(row['damage_pixels'] == 0 for row in records)
        if (len(counts) != EVENT_COUNTS[dataset] or len({row['source_id'] for row in records}) != COUNTS[dataset]
                or negatives > (2 if dataset == 'bright' else 0)):
            raise ValueError(f'{dataset}: incorrect event coverage, duplicate scenes or excess negatives')
        summary[dataset] = dict(events=dict(sorted(counts.items())), positive=len(records) - negatives, negative=negatives)
    return summary


def prepare(root, out):
    out.mkdir(parents=True, exist_ok=True)
    # A CPU-only prep job and the smoke job can reach this stage in either order.
    with (out / '.prepare.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        return prepare_locked(root, out)


def prepare_locked(root, out):
    if (out / 'inputs' / 'cases.json').exists():
        return verify_sample(out)
    out.joinpath('offline_gt').mkdir(parents=True, exist_ok=True)
    staged = {}
    for offset, (dataset, count) in enumerate(COUNTS.items()):
        groups = discover(root, dataset)
        selected = select(groups, dataset, count=count, negatives=2 if dataset == 'bright' else 0, seed=SEED + offset)
        with ThreadPoolExecutor(max_workers=4) as workers:
            staged[dataset] = list(workers.map(lambda item: stage(out, dataset, *item), enumerate(selected, 1)))
    save(out / 'sample_audit.json', dict(seed=SEED, splits={'bright': 'val', 'xbd': 'hold'},
                                        cases=[audit for pairs in staged.values() for _, audit in pairs]))
    save(out / 'inputs' / 'cases.json', {dataset: [row for row, _ in pairs] for dataset, pairs in staged.items()})
    return verify_sample(out)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--datasets', type=Path, default=DATASETS_DIR)
    parser.add_argument('--out', type=Path, default=output_dir(TRACK, RUN_NAME))
    parser.add_argument('--verify-only', action='store_true')
    args = parser.parse_args()
    summary = verify_sample(args.out) if args.verify_only else prepare(args.datasets, args.out)
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
