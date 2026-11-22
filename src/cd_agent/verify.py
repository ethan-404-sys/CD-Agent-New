"""Lightweight research verification primitives for CD-Agent.

Covers the high-impact failure modes observed in change-detection agent experiments:
1. Physical GT isolation on online inference inputs (`EpisodeInput`).
2. Dataset split leakage checks (`check_split_isolation`).
3. Binary mask & diagnostic supervision set axioms (`check_binary_mask`, `check_diagnostic_labels`).
4. Standard experiment output contract (`write_run_contract`, `verify_run_dir`).
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np

FORBIDDEN_INFERENCE_KEYS = frozenset({
    'gt',
    'gt_mask',
    'gt_path',
    'raw_task_gt',
    'gt_clean',
    'label',
    'labels',
    'label_path',
    'original_change_gt',
    'semantic_t1',
    'semantic_t2',
})


@dataclass(frozen=True)
class EpisodeInput:
    """GT-isolated input contract for online Planner / Verifier inference."""

    id: str
    image_t1: str
    image_t2: str
    instruction: str
    init_mask: str | None = None
    ref_image: str | None = None
    ref_mask: str | None = None
    meta: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, record: Mapping[str, Any]) -> 'EpisodeInput':
        leaked = sorted(k for k in record.keys() if k.lower() in FORBIDDEN_INFERENCE_KEYS)
        if leaked:
            raise ValueError(f'GT/label fields are forbidden in online inference input: {leaked}')
        meta = dict(record.get('meta') or {})
        leaked_meta = sorted(k for k in meta.keys() if k.lower() in FORBIDDEN_INFERENCE_KEYS)
        if leaked_meta:
            raise ValueError(f'GT/label fields are forbidden in inference meta: {leaked_meta}')
        for req in ('id', 'image_t1', 'image_t2', 'instruction'):
            if not record.get(req):
                raise ValueError(f'Missing required inference field: {req}')
        return cls(
            id=str(record['id']),
            image_t1=str(record['image_t1']),
            image_t2=str(record['image_t2']),
            instruction=str(record['instruction']),
            init_mask=str(record['init_mask']) if record.get('init_mask') else None,
            ref_image=str(record['ref_image']) if record.get('ref_image') else None,
            ref_mask=str(record['ref_mask']) if record.get('ref_mask') else None,
            meta=meta,
        )


def check_split_isolation(splits: Mapping[str, Iterable[str]]) -> dict[str, Any]:
    """Verify that named dataset splits share zero source IDs."""
    sets = {name: set(ids) for name, ids in splits.items()}
    names = sorted(sets.keys())
    overlaps: dict[str, list[str]] = {}
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            inter = sorted(sets[a] & sets[b])
            if inter:
                overlaps[f'{a}&{b}'] = inter
    if overlaps:
        preview = {k: v[:5] for k, v in overlaps.items()}
        raise ValueError(f'Dataset split leakage detected: {preview}')
    return {'ok': True, 'counts': {k: len(v) for k, v in sets.items()}, 'overlaps': {}}


def check_binary_mask(
    mask: Any,
    *,
    expected_shape: tuple[int, int] | None = None,
    name: str = 'mask',
) -> np.ndarray:
    """Validate a 2D binary mask (bool, {0,1}, or {0,255}) and return bool array."""
    arr = np.asarray(mask)
    if arr.ndim != 2:
        raise ValueError(f'{name}: expected 2D mask, got shape {arr.shape}')
    if expected_shape is not None and arr.shape != expected_shape:
        raise ValueError(f'{name}: shape {arr.shape} != expected {expected_shape}')
    if arr.dtype == np.bool_:
        return arr
    if np.issubdtype(arr.dtype, np.floating) and not np.isfinite(arr).all():
        raise ValueError(f'{name}: contains NaN or Inf')
    uniq = set(np.unique(arr).tolist())
    if not (uniq <= {0, 1} or uniq <= {0, 255}):
        raise ValueError(f'{name}: non-binary values {sorted(uniq)[:8]}')
    return arr > 0


def check_diagnostic_labels(
    init_mask: Any,
    gt_clean: Any,
    fp_mask: Any,
    fn_mask: Any,
    ignore_mask: Any = None,
) -> dict[str, int]:
    """Verify Direction-3 spatial error diagnostic supervision axioms."""
    m = check_binary_mask(init_mask, name='init_mask')
    shape = m.shape
    g = check_binary_mask(gt_clean, expected_shape=shape, name='gt_clean')
    fp = check_binary_mask(fp_mask, expected_shape=shape, name='fp_mask')
    fn = check_binary_mask(fn_mask, expected_shape=shape, name='fn_mask')
    ign = (
        check_binary_mask(ignore_mask, expected_shape=shape, name='ignore_mask')
        if ignore_mask is not None
        else np.zeros(shape, dtype=bool)
    )
    valid = ~ign

    errors: list[str] = []
    if (fp & ~m).any():
        errors.append('FP contains pixels outside init_mask')
    if (fn & m).any():
        errors.append('FN contains pixels inside init_mask')
    if (fp & fn).any():
        errors.append('FP and FN overlap')
    if ((fp | fn) & ign).any():
        errors.append('FP or FN overlaps ignore_mask')
    reconstructed = (m & ~fp) | fn
    if not np.array_equal(reconstructed[valid], g[valid]):
        errors.append('Valid-region reconstruction ((M \\ FP) | FN) != gt_clean')
    if errors:
        raise ValueError('Diagnostic label invariant failed: ' + '; '.join(errors))

    return {
        'mask_area': int(m.sum()),
        'gt_area': int(g.sum()),
        'fp_area': int(fp.sum()),
        'fn_area': int(fn.sum()),
        'ignore_area': int(ign.sum()),
        'valid_area': int(valid.sum()),
    }


def _check_finite_numbers(obj: Any, prefix: str = '') -> list[str]:
    bad: list[str] = []
    if isinstance(obj, bool) or obj is None or isinstance(obj, str):
        return bad
    if isinstance(obj, (int, float)):
        if not math.isfinite(float(obj)):
            bad.append(f'{prefix or "root"}={obj}')
        return bad
    if isinstance(obj, Mapping):
        for k, v in obj.items():
            key = f'{prefix}.{k}' if prefix else str(k)
            bad.extend(_check_finite_numbers(v, key))
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            bad.extend(_check_finite_numbers(v, f'{prefix}[{i}]'))
    return bad


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name('.' + path.name + '.tmp')
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    tmp.replace(path)


def write_run_contract(
    run_dir: Path | str,
    *,
    track: str,
    run_name: str,
    config: Mapping[str, Any],
    metrics: Mapping[str, Any],
    status: str = 'complete',
    extra_meta: Mapping[str, Any] | None = None,
) -> tuple[Path, Path]:
    """Atomically write standardized `run_meta.json` and `metrics.json` for a formal run."""
    non_finite = _check_finite_numbers(metrics)
    if non_finite:
        raise ValueError(f'Refusing to write non-finite metrics: {non_finite[:8]}')
    root = Path(run_dir)
    (root / 'logs').mkdir(parents=True, exist_ok=True)
    meta: dict[str, Any] = {
        'track': track,
        'run_name': run_name,
        'status': status,
        'updated_utc': datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ'),
        'config': dict(config),
    }
    if extra_meta:
        meta.update(extra_meta)
    meta_path = root / 'run_meta.json'
    metrics_path = root / 'metrics.json'
    _atomic_write_json(meta_path, meta)
    _atomic_write_json(metrics_path, dict(metrics))
    return meta_path, metrics_path


def verify_run_dir(
    run_dir: Path | str,
    *,
    expected_count: int | None = None,
    max_failed: int = 0,
    require_logs: bool = True,
) -> dict[str, Any]:
    """Verify that an experiment output directory satisfies the CD-Agent Run Contract."""
    root = Path(run_dir)
    errors: list[str] = []
    meta_path = root / 'run_meta.json'
    metrics_path = root / 'metrics.json'

    if not root.is_dir():
        raise FileNotFoundError(f'Run directory does not exist: {root}')
    if require_logs and not (root / 'logs').is_dir():
        errors.append('Missing logs/ directory')
    if not meta_path.is_file():
        errors.append('Missing run_meta.json')
    if not metrics_path.is_file():
        errors.append('Missing metrics.json')
    if errors:
        raise ValueError('; '.join(errors))

    meta = json.loads(meta_path.read_text(encoding='utf-8'))
    metrics = json.loads(metrics_path.read_text(encoding='utf-8'))

    for key in ('track', 'run_name', 'status'):
        if not meta.get(key):
            errors.append(f'run_meta.json missing field: {key}')
    if meta.get('status') != 'complete':
        errors.append(f'run_meta.json status is {meta.get("status")!r}, expected "complete"')

    bad_nums = _check_finite_numbers(metrics)
    if bad_nums:
        errors.append(f'Non-finite numbers in metrics.json: {bad_nums[:8]}')

    count = metrics.get('count')
    if count is None and isinstance(metrics.get('overall'), Mapping):
        count = metrics['overall'].get('count')
    if expected_count is not None and count != expected_count:
        errors.append(f'Sample count mismatch: got {count}, expected {expected_count}')

    failed = metrics.get('failed', 0)
    if isinstance(failed, int) and failed > max_failed:
        errors.append(f'Failed sample count {failed} exceeds max_failed={max_failed}')

    if errors:
        raise ValueError('Run contract check failed: ' + '; '.join(errors))
    return {'ok': True, 'run_dir': str(root), 'track': meta['track'], 'run_name': meta['run_name'], 'count': count}


def main() -> None:
    parser = argparse.ArgumentParser(description='Verify CD-Agent run directories or split files.')
    sub = parser.add_subparsers(dest='cmd', required=True)

    p_run = sub.add_parser('run', help='Verify a formal run output directory.')
    p_run.add_argument('run_dir', type=Path)
    p_run.add_argument('--expected-count', type=int, default=None)
    p_run.add_argument('--max-failed', type=int, default=0)
    p_run.add_argument('--no-require-logs', action='store_true')

    p_splits = sub.add_parser('splits', help='Verify zero source overlap across split .txt files.')
    p_splits.add_argument('files', nargs='+', type=Path)

    args = parser.parse_args()
    if args.cmd == 'run':
        report = verify_run_dir(
            args.run_dir,
            expected_count=args.expected_count,
            max_failed=args.max_failed,
            require_logs=not args.no_require_logs,
        )
        print(json.dumps(report, ensure_ascii=False))
    else:
        splits = {}
        for path in args.files:
            lines = [ln.strip() for ln in path.read_text(encoding='utf-8').splitlines() if ln.strip()]
            splits[path.stem] = lines
        report = check_split_isolation(splits)
        print(json.dumps(report, ensure_ascii=False))


if __name__ == '__main__':
    main()
