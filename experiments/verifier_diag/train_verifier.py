"""Train SAM 3 HV-Verifier and evaluate the reloaded best weights in one job."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import io
from itertools import islice
import json
import math
import os
from pathlib import Path
import random

import torch
from torch.utils.data import DataLoader

from cd_agent.diagnostic.backbone import SAM3Backbone
from cd_agent.diagnostic.data import SecondDiagnosticDataset, SecondSmokeDataset, collate_diagnostics
from cd_agent.diagnostic.losses import diagnostic_loss
from cd_agent.diagnostic.metrics import DiagnosticMetrics, predict_errors
from cd_agent.diagnostic.model import HVVerifier
from cd_agent.evidence import build_evidence_page, compute_case_stats
from cd_agent.paths import OUTPUTS_DIR, output_dir, smoke_dir
from cd_agent.verify import check_split_isolation, verify_run_dir, write_run_contract

TRACK = 'verifier_diag'
WARMUP_EPOCHS = 2


def _options(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--manifest', type=Path, action='append', help='Repeat for split/shard files')
    parser.add_argument('--run-dir', type=Path)
    parser.add_argument('--mode', choices=('hv', 'additive'), default='hv')
    parser.add_argument('--image-size', type=int, default=672)
    parser.add_argument('--lora-rank', type=int, default=16)
    parser.add_argument('--lora-alpha', type=float, default=32)
    parser.add_argument('--late-mlp-blocks', type=int, default=0)
    parser.add_argument('--fusion-top-k', type=int, default=2)
    parser.add_argument('--epochs', type=int)
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--workers', type=int, default=0)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--vision-lr', type=float, default=3e-5)
    parser.add_argument('--positive-weight', type=float, default=2.0)
    parser.add_argument('--threshold', type=float, default=0.25)
    parser.add_argument('--cf-weight', type=float, default=0.0)
    parser.add_argument('--cf-margin', type=float, default=1.0)
    parser.add_argument('--contour-randomize', action='store_true')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args(argv)
    if not args.smoke and (args.epochs is None or args.epochs < WARMUP_EPOCHS + 1):
        parser.error('Formal training requires an explicit --epochs of at least 3')
    if args.batch_size < 1 or args.workers < 0 or not 0 < args.vision_lr < args.lr:
        parser.error('Use a positive batch size and a positive vision LR lower than the head LR')
    if (not 0 <= args.threshold <= 1 or args.positive_weight <= 0
            or args.cf_weight < 0 or args.cf_margin <= 0):
        parser.error('Invalid threshold or loss weights')
    if args.smoke and (args.batch_size != 1 or args.fusion_top_k < 1 or args.cf_weight):
        parser.error('Smoke needs batch size 1, trainable fusion, and no pair-ranking loss')
    if args.cf_weight and args.batch_size < 2:
        parser.error('Counterfactual training requires --batch-size ≥ 2')
    return args


def _datasets(args):
    default = OUTPUTS_DIR / TRACK / 'second_full_v1'
    ready = [(default / split / 'manifest.jsonl') for split in ('train', 'val', 'test')]
    manifests = args.manifest or ([p for p in ready if p.is_file()] if all(p.is_file() for p in ready[:2]) else [])
    if manifests:
        train = SecondDiagnosticDataset(manifests, split='train', spatial_augment=not args.smoke,
                                        contour_randomize=args.contour_randomize and not args.smoke, seed=args.seed)
        val = SecondDiagnosticDataset(manifests, split='val')
        flavor = 'SECOND diagnostic manifest'
    elif args.smoke:
        train, val = SecondSmokeDataset('train'), SecondSmokeDataset('val')
        flavor = 'real SECOND image-only radiometric init (smoke plumbing, not a benchmark)'
    else:
        raise FileNotFoundError('No train/val SECOND manifest; pass --manifest for each file')
    check_split_isolation({'train': train.source_ids, 'val': val.source_ids})
    if args.smoke and len(train) < 2:
        raise ValueError('Smoke requires two real training rows')
    return train, val, manifests, flavor


def _loader(data, args, *, train: bool, epoch: int = 0):
    kwargs = dict(collate_fn=collate_diagnostics, num_workers=args.workers, pin_memory=True)
    if not train:
        order = [index for group in data.groups.values() for index in group]
        return DataLoader(data, batch_size=1, sampler=order, **kwargs)
    if not args.cf_weight:
        rng = torch.Generator().manual_seed(args.seed + epoch)
        return DataLoader(data, batch_size=args.batch_size, shuffle=True, generator=rng, **kwargs)
    groups = [group for group in data.groups.values() if len(group) > 1]
    if not groups:
        raise ValueError('A counterfactual loss was requested but no shared-mask pairs exist')
    # Keep grouped task switches together while shuffling groups each epoch.
    if any(len(group) > args.batch_size for group in groups):
        raise ValueError('Counterfactual group exceeds the requested batch size')
    groups = list(data.groups.values())
    random.Random(args.seed + epoch).shuffle(groups)
    batches, current = [], []
    for group in groups:
        if len(current) + len(group) > args.batch_size:
            batches.append(current)
            current = []
        current.extend(group)
    if current:
        batches.append(current)
    return DataLoader(data, batch_sampler=batches, **kwargs)


def _cuda(values):
    return {name: value.to('cuda', non_blocking=True) if isinstance(value, torch.Tensor) else value
            for name, value in values.items()}


def _counts(model):
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {'trainable': trainable, 'frozen': sum(p.numel() for p in model.parameters()) - trainable}


def _capture(model, names, arch, *, epoch):
    state = model.state_dict()
    return {'state_dict': {key: state[key].detach().cpu() for key in names}, 'architecture': arch, 'epoch': epoch}


def _restore(model, payload, names):
    changed = payload['state_dict']
    if changed.keys() != set(names):
        raise ValueError('Checkpoint has missing or unexpected adaptation parameters')
    state = model.state_dict()
    state.update(changed)
    model.load_state_dict(state, strict=True)


def _grad_checks(model, *, vision: bool):
    groups = {'fusion': [], 'pixel_decoder': [], 'diagnostic': []}
    if vision:
        groups.update(lora=[], vision_layernorm=[])
    for key, param in model.named_parameters():
        if not param.requires_grad:
            if param.grad is not None:
                raise RuntimeError(f'Unexpected gradient on frozen parameter {key}')
            continue
        if key.startswith('backbone.vision.'):
            group = 'lora' if '.lora_' in key else 'vision_layernorm'
        elif key.startswith('backbone.fusion.'):
            group = 'fusion'
        elif key.startswith('backbone.pixel_decoder.'):
            group = 'pixel_decoder'
        else:
            group = 'diagnostic'
        if param.grad is None:
            raise RuntimeError(f'Missing smoke gradient for {key}')
        groups[group].append(param.grad.detach().float().norm())
    norms = {key: torch.stack(values).norm().item() if values else 0.0 for key, values in groups.items()}
    if any(not math.isfinite(value) or value == 0.0 for value in norms.values()):
        raise RuntimeError(f'Missing, nonfinite or zero smoke gradient groups: {norms}')
    return norms


class _Evidence:
    """Keep bounded candidate images while scoring every validation episode."""
    def __init__(self):
        self.buckets = [[] for _ in range(4)]

    def add(self, batch, predictions, index, info):
        inputs, targets, meta = (batch[key] for key in ('inputs', 'targets', 'meta'))
        init = inputs['init_mask'][index, 0]
        valid = ~targets['ignore_mask'][index, 0]
        pred = ((init & ~predictions['fp'][index, 0]) | predictions['fn'][index, 0]) & valid
        gt = targets['gt_clean'][index, 0] & valid
        stats = compute_case_stats(pred.numpy(), gt.numpy())
        identity = meta[index]['id']
        ranks = (-stats['f1'], abs(stats['f1'] - 0.5), -stats['fp'], -stats['fn'])
        item = None
        for bucket, rank in zip(self.buckets, ranks, strict=True):
            pair = (rank, identity)
            # At most six identities can be claimed by higher-priority buckets.
            if len(bucket) == 8 and pair >= bucket[-1][0]:
                continue
            if item is None:
                item = dict(
                    id=identity, pred_mask=pred.numpy(), gt_mask=gt.numpy(), init_mask=(init & valid).numpy(),
                    instruction=inputs['instruction'][index],
                    summary=f'{inputs["instruction"][index]} | initial IoU={info["init_iou"]:.4f}',
                )
                for when in ('image_t1', 'image_t2'):
                    item[when] = inputs[when][index].permute(1, 2, 0).mul(255).round().byte().numpy()
            bucket.append((pair, item))
            bucket.sort(key=lambda entry: entry[0])
            del bucket[8:]

    def cases(self):
        return list({item['id']: item for bucket in self.buckets for _, item in bucket}.values())


@torch.no_grad()
def _evaluate(model, loader, args, dtype, *, evidence=None, limit=None):
    model.eval()
    meter = DiagnosticMetrics()
    stream = loader if limit is None else islice(loader, limit)
    for batch in stream:
        inputs = _cuda(batch['inputs'])
        init = inputs['init_mask']
        with torch.autocast('cuda', dtype=dtype):
            outputs = model(**inputs)
            prediction = predict_errors(outputs, init, threshold=args.threshold)
            zeroed_fn = None
            if evidence is not None:
                empty = torch.zeros_like(init)
                zeroed = model(**{**inputs, 'init_mask': empty})
                zeroed_fn = predict_errors(zeroed, empty, threshold=args.threshold)['fn'].cpu()
        prediction = {key: value.cpu() for key, value in prediction.items()}
        for i, meta in enumerate(batch['meta']):
            info = meter.update(
                prediction['fp'][i], prediction['fn'][i], batch['inputs']['init_mask'][i],
                batch['targets']['gt_clean'][i], batch['targets']['ignore_mask'][i],
                task=meta['task_type'], cf_group=meta.get('cf_group'), is_anchor=meta['is_anchor'],
                instruction=inputs['instruction'][i], zeroed_fn=None if zeroed_fn is None else zeroed_fn[i],
            )
            if info and evidence is not None:
                evidence.add(batch, prediction, i, info)
    result = meter.summary()
    if not result['overall']['count']:
        raise ValueError('Validation produced no valid-region examples')
    return result


@torch.no_grad()
def _smoke_roundtrip(model, loader, names, arch, dtype):
    model.eval()
    inputs = _cuda(next(iter(loader))['inputs'])
    with torch.autocast('cuda', dtype=dtype):
        before = model(**inputs)
    with io.BytesIO() as checkpoint:
        torch.save(_capture(model, names, arch, epoch=0), checkpoint)
        model.presence_head[-1].bias.add_(1.0)
        checkpoint.seek(0)
        _restore(model, torch.load(checkpoint, weights_only=True), names)
    with torch.autocast('cuda', dtype=dtype):
        after = model(**inputs)
    for key in before:
        torch.testing.assert_close(before[key], after[key], rtol=1e-4, atol=1e-4)


def main(argv=None) -> None:
    args = _options(argv)
    if not torch.cuda.is_available():
        raise RuntimeError('Train and smoke must be run on a Slurm GPU node')
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported(including_emulation=False) else torch.float16
    scaler = torch.amp.GradScaler('cuda', enabled=dtype == torch.float16, init_scale=128)
    day = datetime.now(timezone.utc).strftime('%Y%m%d')
    job = os.environ.get('SLURM_JOB_ID', 'interactive')
    root = args.run_dir or (smoke_dir(f'verifier-train-{job}') if args.smoke
                            else output_dir(TRACK, f'{day}_sam3-{args.mode}-{job}'))
    (root / 'logs').mkdir(parents=True, exist_ok=True)
    best_path = root / 'best.pt'
    if best_path.exists():
        raise FileExistsError(f'Refusing to overwrite pre-existing checkpoint: {best_path}')
    train, val, manifests, flavor = _datasets(args)
    val_loader = _loader(val, args, train=False)
    config = {k: getattr(args, k) for k in (
        'image_size', 'lora_rank', 'lora_alpha', 'late_mlp_blocks', 'fusion_top_k',
    )}
    arch = dict(mode=args.mode, **config)
    model = HVVerifier(SAM3Backbone(**config), mode=args.mode).cuda()
    print(f'Strict SAM 3 load complete. Data: {flavor}. Train={len(train)}, validation={len(val)}. AMP={dtype}', flush=True)
    names = [key for key, param in model.named_parameters() if param.requires_grad]
    vision = list(model.backbone.vision_adaptation_parameters())
    vision_ids = {id(p) for p in vision}
    heads = [p for p in model.parameters() if p.requires_grad and id(p) not in vision_ids]
    optimizer = torch.optim.AdamW([dict(params=heads, lr=args.lr), dict(params=vision, lr=args.vision_lr)])
    stage2 = _counts(model)
    model.set_vision_lora_trainable(False)
    parameter_counts = {'warmup': _counts(model), 'adaptation': stage2}
    print(f'{args.mode} parameters: {json.dumps(parameter_counts)}', flush=True)
    torch.cuda.reset_peak_memory_stats()

    best, best_epoch, history, norms = -math.inf, None, [], []
    for epoch in range(1, 2 if args.smoke else args.epochs + 1):
        model.train()
        if hasattr(train, 'epoch'):
            train.epoch = epoch
        model.set_vision_lora_trainable(not args.smoke and epoch > WARMUP_EPOCHS)
        loader = _loader(train, args, train=True, epoch=epoch)
        running = []
        for step, batch in enumerate(islice(loader, 2) if args.smoke else loader, 1):
            if args.smoke and step == 2:
                model.set_vision_lora_trainable(True)
            inputs = _cuda(batch['inputs'])
            targets = _cuda(batch['targets'])
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast('cuda', dtype=dtype):
                outputs = model(**inputs)
                loss = diagnostic_loss(
                    outputs, init_mask=inputs['init_mask'], **targets, positive_weight=args.positive_weight,
                    cf_pairs=batch['cf_pairs'], cf_weight=args.cf_weight, cf_margin=args.cf_margin,
                )['total']
            if not torch.isfinite(loss):
                raise FloatingPointError(f'Nonfinite training loss at epoch {epoch}, step {step}')
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            if args.smoke:
                norms.append(_grad_checks(model, vision=(step == 2)))
                print(f'Smoke step {step}: loss={loss.item():.6f}, gradients={norms[-1]}', flush=True)
            torch.nn.utils.clip_grad_norm_(heads + vision, max_norm=1.0, error_if_nonfinite=True)
            scaler.step(optimizer)
            scaler.update()
            running.append(loss.item())
        if args.smoke and len(running) != 2:
            raise RuntimeError(f'Expected exactly two optimizer steps, got {len(running)}')
        optimizer.zero_grad(set_to_none=True)
        entry = {'epoch': epoch, 'steps': len(running), 'training_loss': sum(running) / len(running)}
        if not args.smoke:
            interim = _evaluate(model, val_loader, args, dtype)
            entry.update(val_refined_iou=interim['overall']['refined_iou'], val_delta_iou=interim['overall']['delta_iou'])
            if entry['val_refined_iou'] > best:
                best, best_epoch = entry['val_refined_iou'], epoch
                staging = best_path.with_suffix('.pt.tmp')
                torch.save(_capture(model, names, arch, epoch=epoch), staging)
                staging.replace(best_path)
        history.append(entry)
        print(f'Epoch {epoch}: {entry}', flush=True)
    if args.smoke:
        _smoke_roundtrip(model, val_loader, names, arch, dtype)
    else:
        checkpoint = torch.load(best_path, map_location='cpu', weights_only=True)
        _restore(model, checkpoint, names)
        del checkpoint

    evidence = _Evidence()
    metrics = _evaluate(model, val_loader, args, dtype, evidence=evidence, limit=2 if args.smoke else None)
    metrics.update(history=history, best_epoch=best_epoch, parameters=parameter_counts,
                   peak_vram_allocated_mib=torch.cuda.max_memory_allocated() / 2**20,
                   peak_vram_reserved_mib=torch.cuda.max_memory_reserved() / 2**20)
    if args.smoke:
        metrics.update(gradient_group_norms=norms, roundtrip_equivalent=True, optimizer_steps=2)
    build_evidence_page(evidence.cases(), root / 'viz', title=f'SAM 3 {args.mode} SECOND diagnostics', metrics=metrics)
    write_run_contract(
        root, track=TRACK, run_name=root.name, metrics=metrics,
        config={**arch, 'training': {k: v for k, v in vars(args).items() if k not in ('manifest', 'run_dir')},
                'manifests': [str(p) for p in manifests], 'data_source': flavor, 'amp_dtype': str(dtype)},
    )
    verify_run_dir(root, expected_count=min(2, len(val)) if args.smoke else len(val))
    print(f'Complete: peak allocated {metrics["peak_vram_allocated_mib"]:.1f} MiB; {root}', flush=True)


if __name__ == '__main__':
    main()
