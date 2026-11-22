"""Independent Agent-semantic generation QC. Never opens any GT or label file."""
from __future__ import annotations

import argparse
from collections import Counter
import json
import os
from pathlib import Path
import tempfile

import numpy as np
from PIL import Image

METHODS = ('dynamic_earth_mci', 'omniovcd', 'rea_ovcd')
ROW_KEYS = {'id', 'dataset', 'dataset_slug', 'source_image_id', 'source_group_id', 'image_t1', 'image_t2', 'pilot_tasks'}
TASK_KEYS = {'task_type', 'task_params', 'prompt_en', 'prompt_zh', 'prompt_template_id', 'prompt_templates_en', 'prompt_templates_zh'}


def recipe(spec, basis=None):
    if basis and basis.get("recipe_type") in ("proposal_labels", "component_labels"):
        unit="proposal" if basis["recipe_type"]=="proposal_labels" else "component"
        return f"Select each {unit} by its two model phase labels for {spec['task_type']}; apply native gate; union selected units"
    kind, p = spec['task_type'], spec['task_params']
    if kind == 'transition':
        return f"B & (Y1 == {p['source_class']}) & (Y2 == {p['target_class']})"
    c = p['class_id']
    if kind == 'related': return f'B & ((Y1 == {c}) | (Y2 == {c}))'
    if kind == 'into': return f'B & (Y2 == {c}) & (Y1 != {c})'
    if kind == 'outof': return f'B & (Y1 == {c}) & (Y2 != {c})'
    raise ValueError(f'Unknown task type: {kind}')


def recompute(spec, b, y1, y2):
    """Intentionally independent of adapter task-composition functions."""
    kind, p = spec['task_type'], spec['task_params']
    if kind == 'transition': return b & (y1 == p['source_class']) & (y2 == p['target_class'])
    c = p['class_id']
    if kind == 'related': return b & ((y1 == c) | (y2 == c))
    if kind == 'into': return b & (y2 == c) & (y1 != c)
    if kind == 'outof': return b & (y1 == c) & (y2 != c)
    raise ValueError(f'Unknown task type: {kind}')


def choose(spec, y1, y2):
    kind,p=spec['task_type'],spec['task_params']
    if kind=='transition': return (y1==p['source_class']) & (y2==p['target_class'])
    c=p['class_id']
    if kind=='related': return (y1==c)|(y2==c)
    if kind=='into': return (y2==c)&(y1!=c)
    if kind=='outof': return (y1==c)&(y2!=c)
    raise ValueError(kind)


def independent_compile(spec, ref, arrays, adapted):
    typ=ref.get('recipe_type','dense_boolean')
    if adapted and ref['route'] != ('official_related' if spec['task_type']=='related' else 'task_adapted'):
        raise ValueError('Native related and task-adapted routes must be declared separately')
    if typ=='dense_boolean':
        b,y1,y2=(arrays[ref[key]] for key in ('B_key','Y1_key','Y2_key'))
        if b.dtype!=np.bool_ or y1.dtype!=np.uint8 or y2.dtype!=np.uint8:
            raise ValueError('Expected bool B and uint8 phase labels')
        if b.ndim!=2 or b.shape!=y1.shape or b.shape!=y2.shape:
            raise ValueError('Incompatible dense state shapes')
        if y1.max()>6 or y2.max()>6: raise ValueError('Semantic label outside canonical 0..6')
        return recompute(spec,b,y1,y2)
    if not adapted or typ not in ('proposal_labels','component_labels'):
        raise ValueError(f'Unsupported recipe type: {typ}')
    if ref['selection_mode']!=spec['task_type']: raise ValueError('Selection mode differs from task')
    y1,y2=(arrays[ref[key]] for key in ('phase1_labels_key','phase2_labels_key'))
    if y1.dtype!=np.uint8 or y2.dtype!=np.uint8 or y1.ndim!=1 or y1.shape!=y2.shape:
        raise ValueError('Expected paired one-dimensional canonical unit labels')
    if (y1.size and y1.max()>6) or (y2.size and y2.max()>6): raise ValueError('Unit label outside canonical 0..6')
    for phase,labels in ((1,y1),(2,y2)):
        key=ref.get(f'phase{phase}_scores_key')
        if key:
            scores=arrays[key]
            if scores.shape!=(len(labels),7): raise ValueError('Unit scores must be canonical Nx7')
            invalid=np.flatnonzero(~np.isfinite(scores).all(1))
            if len(invalid):
                # Native SegEarth mean(empty) is NaN and argmax remains background.
                # Permit it only when nearest projection independently has no pixels.
                if typ!='proposal_labels' or not ref.get(f'phase{phase}_support_key'):
                    raise ValueError('Nonfinite unit scores lack native empty-proposal evidence')
                support=arrays[ref[f'phase{phase}_support_key']]
                grid=arrays[ref[f'phase{phase}_grid_key']]
                if support.shape!=(len(labels),) or grid.shape!=(2,) or (grid<=0).any():
                    raise ValueError('Invalid native proposal support metadata')
                if support[invalid].any() or labels[invalid].any() or np.isinf(scores[invalid]).any():
                    raise ValueError('Nonfinite score on supported or foreground proposal')
                masks=arrays[ref['proposal_masks_key']]
                ys=np.arange(int(grid[0]))*masks.shape[1]//int(grid[0])
                xs=np.arange(int(grid[1]))*masks.shape[2]//int(grid[1])
                for index in invalid:
                    if masks[index][ys[:,None],xs[None,:]].any():
                        raise ValueError('Nonfinite proposal scores have actual sampled pixels')
            expected=scores.argmax(1).astype(np.uint8)
            expected[scores.max(1)<ref.get('conf_threshold',0.)]=0
            if typ=='component_labels' and len(expected): expected[0]=0
            if not np.array_equal(labels,expected): raise ValueError('Unit labels differ from model scores/threshold')
    selected=choose(spec,y1,y2)
    if typ=='proposal_labels':
        masks=arrays[ref['proposal_masks_key']]
        if masks.dtype!=np.bool_ or masks.ndim!=3 or len(masks)!=len(selected):
            raise ValueError('Expected bool proposal stack NxHxW')
        if ref.get('proposal_gate_key'):
            gate=arrays[ref['proposal_gate_key']]
            if gate.dtype!=np.bool_ or gate.shape!=selected.shape: raise ValueError('Invalid native proposal gate')
            selected &= gate
        return masks[selected].any(0) if selected.any() else np.zeros(masks.shape[1:],bool)
    component=arrays[ref['component_map_key']]
    if component.ndim!=2 or not np.issubdtype(component.dtype,np.integer) or component.min()<0 or component.max()>=len(selected):
        raise ValueError('Invalid component index map')
    if ref.get('B_key') and not np.array_equal(component>0,arrays[ref['B_key']]):
        raise ValueError('Component support differs from native change domain B')
    count=len(selected)-1
    sizes=np.bincount(component.ravel(),minlength=count+1)
    for phase in (1,2):
        dense_key=ref.get(f'dense_phase{phase}_scores_key')
        if dense_key:
            dense=arrays[dense_key]
            if dense.shape!=(7,*component.shape): raise ValueError('Dense phase scores must be 7xHxW')
            means=np.zeros((count+1,7),np.float32)
            for category in range(7):
                totals=np.bincount(component.ravel(),weights=dense[category].ravel(),minlength=count+1)
                means[1:,category]=totals[1:]/sizes[1:]
            if not np.allclose(arrays[ref[f'phase{phase}_scores_key']],means,rtol=1e-5,atol=1e-6):
                raise ValueError('Component scores differ from native dense score means')
    selected[0]=False
    return selected[component]


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    content = json.dumps(value, ensure_ascii=False, indent=2) + '\n'
    with tempfile.NamedTemporaryFile('w', dir=path.parent, prefix='.' + path.name + '.', delete=False) as handle:
        handle.write(content)
        temporary = handle.name
    os.replace(temporary, path)
    if path.read_text() != content: raise IOError(f'Incomplete write: {path}')


def current_path(value, out):
    path = Path(value)
    if not path.is_absolute() or not path.resolve().is_relative_to(out):
        raise ValueError(f'Artifact is outside fresh run: {path}')
    if not path.is_file(): raise FileNotFoundError(path)
    return path


def check(out):
    out = out.resolve()
    cfg, rows = read(out / 'config.json'), read(out / 'generator_inputs.json')
    failures, methods = [], {}
    adapted=cfg.get('task_adapted',False)
    baseline=Path(cfg['source_run']) if adapted else None
    related_tasks=sum(t['task_type']=='related' for row in rows for t in row['pilot_tasks'])
    if not cfg.get('agent_semantics'): failures.append('Agent semantics is not enabled')
    if len(rows) != 28 or sum(len(r['pilot_tasks']) for r in rows) != 74:
        failures.append('Frozen inventory differs from 28 sources / 74 instructions')
    if read(out / 'gt_anchor_tasks.json'): failures.append('GT anchors are forbidden')
    for row in rows:
        if set(row) != ROW_KEYS or any(set(t) != TASK_KEYS for t in row['pilot_tasks']):
            failures.append(f"{row['id']}: generation input schema contains unexpected context")
    for method in METHODS:
        counts = Counter(); modes = Counter(); details = []
        for row in rows:
            counts['sources'] += 1
            try:
                receipt_path = out / 'generation' / method / f"{row['id']}.json"
                receipt = read(receipt_path)
                if not receipt.get('generated_in_this_run') or not receipt.get('agent_semantics'):
                    raise ValueError('Receipt does not declare current Agent generation')
                if adapted and not receipt.get('task_adapted'): raise ValueError('Missing task-adapted receipt mode')
                if receipt['slurm_job_id'] != cfg['creation_slurm_job']:
                    raise ValueError('Receipt is from a different Slurm generation job')
                if receipt.get('ground_truth_inputs') is not False:
                    raise ValueError('Receipt must declare no GT input')
                images = {key: row[key] for key in ('image_t1', 'image_t2')}
                if receipt.get('input_images') != images:
                    raise ValueError('Receipt RGB inputs differ from frozen source pair')
                task_keys = [json.dumps({key: t[key] for key in ('task_type', 'task_params', 'prompt_en')}, sort_keys=True) for t in row['pilot_tasks']]
                if receipt['task_keys'] != task_keys: raise ValueError('Receipt tasks differ from frozen instructions')
                metadata = receipt['metadata']; basis = metadata['task_basis']
                if metadata.get('gt_used') is not False or metadata.get('input_source_paths') != list(images.values()):
                    raise ValueError('Adapter evidence must declare exactly the two RGB inputs and no GT')
                if not (metadata.get('config') or metadata.get('configs')):
                    raise ValueError('Adapter evidence is missing model configuration')
                state = current_path(metadata['model_state'], out)
                if len(basis) != len(row['pilot_tasks']) or len(receipt['predictions']) != len(basis):
                    raise ValueError('State basis / prediction count differs from frozen tasks')
                with np.load(state, allow_pickle=False) as arrays:
                    for index, (spec, ref, prediction) in enumerate(zip(row['pilot_tasks'], basis, receipt['predictions'])):
                        counts['tasks'] += 1
                        mode = ref['semantic_mode']; modes[mode] += 1
                        params = spec['task_params']
                        ambiguous = spec['task_type'] == 'transition' and {params['source_class'], params['target_class']} & {1, 2}
                        if not adapted and ambiguous and mode != 'fullclassargmax':
                            raise ValueError(f'Task {index}: ambiguous transition needs fullclassargmax')
                        expected = independent_compile(spec, ref, arrays, adapted)
                        path = current_path(prediction, out)
                        with Image.open(path) as image: actual = np.asarray(image) > 0
                        if actual.shape != expected.shape: raise ValueError(f'Task {index}: prediction shape differs from state')
                        mismatches = int(np.count_nonzero(actual != expected))
                        if mismatches: raise ValueError(f'Task {index}: {mismatches} formula mismatch pixels')
                        if adapted and spec['task_type']=='related':
                            prior=read(baseline/'generation'/method/f"{row['id']}.json")
                            if prior['task_keys'][index] != task_keys[index]: raise ValueError('Related baseline task differs')
                            with Image.open(prior['predictions'][index]) as image: old=np.asarray(image)>0
                            different=int(np.count_nonzero(actual!=old)) if actual.shape==old.shape else -1
                            if different: raise ValueError(f'Related native regression differs by {different} pixels')
                            counts['related_regression_passed'] += 1
                        counts['formula_passed'] += 1
                        details.append(dict(source_id=row['id'], task_index=index, recipe=recipe(spec,ref), recipe_type=ref.get('recipe_type','dense_boolean'),
                                            route=ref.get('route','agent_semantic'),semantic_mode=mode,
                                            model_state=str(state), task_basis=ref, mismatch_pixels=0,
                                            related_regression_mismatch_pixels=0 if adapted and spec['task_type']=='related' else None))
                counts['provenance_passed'] += 1
            except (OSError, ValueError, KeyError, TypeError, IndexError) as error:
                failures.append(f"{method}/{row['id']}: {error}")
        if counts['formula_passed'] != 74 or counts['provenance_passed'] != 28:
            failures.append(f'{method}: incomplete formula/provenance checks')
        if adapted and counts['related_regression_passed']!=related_tasks:
            failures.append(f'{method}: incomplete related regression')
        methods[method] = dict(counts, semantic_modes=dict(modes), tasks_checked=details)
    report = dict(status='failed' if failures else 'passed', task_adapted=adapted,agent_semantics=True, sources=len(rows), tasks_per_method=74,
                  normal_records=sum(v.get('formula_passed', 0) for v in methods.values()), gt_anchors=0,
                  methods=methods, failures=failures, ground_truth_read=False,
                  counterfactual_policy='Excluded: later Q swaps retain the exact nonempty normal mask; GT used only for offline pair QC',
                  scope='Independent pure-model dense Boolean / proposal-label union / component-label selection; RGB and same-run provenance checks',
                  related_regression=dict(status='passed' if adapted and not failures else 'not_applicable' if not adapted else 'failed',
                    baseline_run=str(baseline) if adapted else None,tasks_per_method=related_tasks if adapted else None))
    write(out / 'generation_qc.json', report)
    print(json.dumps({key: value for key, value in report.items() if key != 'methods'}, ensure_ascii=False), flush=True)
    if failures: raise RuntimeError('Agent-semantic generation QC failed')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if not os.environ.get('SLURM_JOB_ID'): raise RuntimeError('Slurm only')
    check(args.output)


if __name__ == '__main__': main()
