"""SECOND diagnostic manifests. Supervision never appears in the model input map."""
from __future__ import annotations

from collections import defaultdict
from itertools import combinations
import json
from pathlib import Path
import random

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, default_collate
from torch.nn import functional as F

from cd_agent.paths import REPO_ROOT, dataset_path
from cd_agent.verify import check_binary_mask, check_diagnostic_labels, check_split_isolation


MODEL_INPUT_KEYS = ('image_t1', 'image_t2', 'init_mask', 'instruction')
SUPERVISION_KEYS = ('gt_clean', 'ignore_mask', 'fp_mask', 'fn_mask')


def load_rgb(path: str | Path) -> torch.Tensor:
    with Image.open(path) as image:
        array = np.array(image.convert('RGB'))
    return torch.from_numpy(array).permute(2, 0, 1).float().div_(255)


def load_mask(path: str | Path) -> torch.Tensor:
    with Image.open(path) as image:
        array = np.array(image)
    return torch.from_numpy(check_binary_mask(array, name=str(path)).copy()).unsqueeze(0)


def _field(record: dict, *names: str):
    for name in names:
        if record.get(name) is not None:
            return record[name]
        if isinstance(record.get('paths'), dict) and record['paths'].get(name) is not None:
            return record['paths'][name]
    return None


def _finish(
    first: torch.Tensor, second: torch.Tensor, initial: torch.Tensor,
    gt: torch.Tensor, ignore: torch.Tensor, prompt: str, meta: dict,
) -> dict:
    if first.shape != second.shape or initial.shape != gt.shape or initial.shape != ignore.shape or (
        first.shape[-2:] != initial.shape[-2:]
    ):
        raise ValueError(f'Unaligned diagnostic inputs for {meta["id"]}')
    fp, fn = initial & ~gt & ~ignore, ~initial & gt & ~ignore
    return {
        'inputs': dict(zip(MODEL_INPUT_KEYS, (first, second, initial, prompt), strict=True)),
        'targets': dict(zip(SUPERVISION_KEYS, (gt, ignore, fp, fn), strict=True)),
        'meta': meta,
    }


class SecondDiagnosticDataset(Dataset):
    """One or more JSONL manifests with explicit split and source-image identity.

    Relative image/mask paths resolve next to their manifest. Counterfactual
    membership is explicit or inferred from a shared initial-mask path on the
    same source. Grouped rows receive the exact same spatial/style augmentation.
    """

    def __init__(
        self, manifests: str | Path | list[Path], *, split: str,
        spatial_augment: bool = False, contour_randomize: bool = False,
        seed: int = 42,
    ):
        if split not in ('train', 'val', 'test'):
            raise ValueError(f'Unknown diagnostic split {split!r}')
        if isinstance(manifests, (str, Path)):
            manifests = [Path(manifests)]
        self.rows: list[dict] = []
        self.groups: dict[tuple[str, str], list[int]] = defaultdict(list)
        self.spatial_augment, self.contour_randomize = spatial_augment, contour_randomize
        self.seed, self.epoch = seed, 0
        split_sources: dict[str, set[str]] = defaultdict(set)
        ids = set()
        for manifest in map(Path, manifests):
            with manifest.open(encoding='utf-8') as stream:
                for line_number, line in enumerate(stream, 1):
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    if str(row.get('dataset', 'SECOND')).upper() != 'SECOND':
                        raise ValueError(f'SECOND-only manifest expected: {manifest}:{line_number}')
                    row_split = {'fit': 'train', 'validation': 'val'}.get(
                        _field(row, 'split', 'internal_split'), _field(row, 'split', 'internal_split'),
                    )
                    source = _field(row, 'source_image_id', 'source_id', 'source_group_id')
                    if row_split not in ('train', 'val', 'test') or not source:
                        raise ValueError(f'Missing split or source identity: {manifest}:{line_number}')
                    split_sources[row_split].add(str(source))
                    if row_split != split:
                        continue
                    self._register(row, str(source), manifest, line_number, ids)
        check_split_isolation(split_sources)
        if not self.rows:
            raise ValueError(f'No SECOND diagnostic records for {split}')
        self.source_ids = frozenset(split_sources[split])

    def _register(self, row: dict, source: str, manifest: Path, line: int, ids: set) -> None:
        def path(*names: str, required: bool = True) -> Path | None:
            value = _field(row, *names)
            if not value:
                if required:
                    raise ValueError(f'Missing {names[0]}: {manifest}:{line}')
                return None
            return manifest.parent / str(value)

        task_type = _field(row, 'task_type', 'direction', 'task')
        prompt = _field(row, 'prompt_en')
        initial = path('init_mask', 'init_mask_path', 'mask_path')
        identity = str(_field(row, 'id', 'sample_id') or f'{manifest.name}:{line}')
        if identity in ids:
            raise ValueError(f'Duplicate diagnostic record ID: {identity}')
        ids.add(identity)
        if not isinstance(prompt, str) or not prompt.strip() or task_type not in ('related', 'into', 'outof', 'transition'):
            raise ValueError(f'Expected prompt_en and a recognized task type: {manifest}:{line}')
        group = (source, str(_field(row, 'counterfactual_group_id', 'cf_group_id', 'cf_pair_id',
                                    'shared_mask_id') or initial.resolve()))
        indices = self.groups[group]
        index = len(self.rows)
        labels = row.get('labels') or {}
        if not isinstance(labels, dict):
            raise ValueError(f'Expected a labels mapping: {manifest}:{line}')
        self.rows.append({
            'first': path('image_t1', 'image_t1_path', 't1_path'),
            'second': path('image_t2', 'image_t2_path', 't2_path'),
            'initial': initial, 'gt': path('gt_clean', 'gt_clean_path', 'gt_mask', 'gt_mask_path'),
            'ignore': path('ignore_mask', 'ignore_mask_path', required=False)
            or (manifest.parent / labels['ignore_mask'] if labels.get('ignore_mask') else None),
            'fp': path('fp_mask', 'fp_mask_path', required=False)
            or (manifest.parent / labels['fp'] if labels.get('fp') else None),
            'fn': path('fn_mask', 'fn_mask_path', required=False)
            or (manifest.parent / labels['fn'] if labels.get('fn') else None),
            'instruction': prompt.strip(), 'augmentation_index': indices[0] if indices else index,
            'meta': {
                'id': identity, 'source_id': source, 'task_type': task_type,
                'cf_group': group, 'is_anchor': bool(row.get('is_anchor', row.get('anchor', False))),
            },
        })
        indices.append(index)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict:
        row = self.rows[index]
        first, second = load_rgb(row['first']), load_rgb(row['second'])
        init, gt = load_mask(row['initial']), load_mask(row['gt'])
        ignore = load_mask(row['ignore']) if row['ignore'] else torch.zeros_like(gt)
        if row['fp'] or row['fn']:
            if not (row['fp'] and row['fn']):
                raise ValueError(f'Both FP/FN paths are required when supplying error labels: {row["meta"]["id"]}')
            check_diagnostic_labels(*(value[0].numpy() for value in (
                init, gt, load_mask(row['fp']), load_mask(row['fn']), ignore,
            )))
        rng = random.Random(self.seed + self.epoch * len(self) + row['augmentation_index'])
        if self.spatial_augment:
            axes = tuple(axis for axis in (-2, -1) if rng.random() < 0.5)
            turns = rng.randrange(4)
            first, second, init, gt, ignore = (
                torch.rot90(value.flip(axes), turns, (-2, -1))
                for value in (first, second, init, gt, ignore)
            )
        if self.contour_randomize and rng.random() < 0.3:
            size = init.shape[-2:]
            reduced = F.interpolate(init[None].float(), size=[max(1, d // 2) for d in size], mode='area')
            init = F.interpolate(reduced, size=size, mode='bilinear', align_corners=False)[0].ge(0.5)
        return _finish(first, second, init, gt, ignore, row['instruction'], row['meta'])


def collate_diagnostics(samples: list[dict]) -> dict:
    inputs = default_collate([sample['inputs'] for sample in samples])
    targets = default_collate([sample['targets'] for sample in samples])
    metadata = [sample['meta'] for sample in samples]
    groups = defaultdict(list)
    for i, meta in enumerate(metadata):
        if meta.get('cf_group') is not None:
            groups[meta['cf_group']].append(i)
    pairs = [(a, b) for group in groups.values() for a, b in combinations(group, 2)
             if inputs['instruction'][a] != inputs['instruction'][b]]
    return {'inputs': inputs, 'targets': targets, 'meta': metadata, 'cf_pairs': pairs}


class SecondSmokeDataset(Dataset):
    """Fixed real SECOND source pairs for plumbing only, not an evaluation split.

    A GT-independent radiometric change heuristic supplies an initial mask. The
    official binary SECOND mask is read separately and used only as supervision.
    """

    def __init__(self, split: str):
        self.root = dataset_path('SECOND')
        if split not in ('train', 'val'):
            raise ValueError('Smoke supports only train and val')
        inventory = REPO_ROOT / 'configs' / 'splits' / f'second_{split}.txt'
        if inventory.is_file():
            self.names = tuple(inventory.read_text(encoding='utf-8').split()[:2])
        else:
            self.names = ('train_0.png', 'train_1.png') if split == 'train' else ('train_2.png', 'train_3.png')
        if len(set(self.names)) != 2 or any(not name.startswith('train_') or Path(name).name != name
                                             for name in self.names):
            raise ValueError(f'Invalid smoke source inventory: {inventory}')
        self.source_ids = frozenset(self.names)
        self.groups = {(name, name): [i] for i, name in enumerate(self.names)}

    def __len__(self) -> int:
        return len(self.names)

    def __getitem__(self, index: int) -> dict:
        name = self.names[index]
        first, second = (load_rgb(self.root / part / name) for part in ('img1', 'img2'))
        signal = F.avg_pool2d((second - first).abs().mean(0, keepdim=True)[None], 5, 1, 2)[0]
        initial = signal > signal.mean()
        gt = load_mask(self.root / 'label_unch0chi1' / name)
        return _finish(first, second, initial, gt, torch.zeros_like(gt),
                       'Find all land cover transitions from the first image to the second image.',
                       {'id': name, 'source_id': name, 'task_type': 'related', 'is_anchor': False})
