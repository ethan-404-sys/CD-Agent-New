"""Frozen-source pilot with official OVCD generators; all execution via Slurm."""
from __future__ import annotations
import argparse
from collections import Counter
import json
import os
from pathlib import Path
import sys
import time
import tempfile
import numpy as np
from PIL import Image

REPO=Path(__file__).resolve().parents[3]
OLD=REPO/'results/verifier_v1_pilot/20261118_fpfn_joint'
AGENT_SOURCE=REPO/'results/verifier_v1_pilot/20261118_official_ovcd_v2'
ADAPTED_SOURCE=REPO/'results/verifier_v1_pilot/20261118_agent_semantic_v3'
METHODS=('dynamic_earth_mci','omniovcd','rea_ovcd')
SPEC_KEYS=('task_type','task_params','prompt_en','prompt_zh','prompt_template_id','prompt_templates_en','prompt_templates_zh')


def read(path): return json.loads(path.read_text())
def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    text=json.dumps(value,ensure_ascii=False,indent=2)+'\n'
    with tempfile.NamedTemporaryFile('w',dir=path.parent,prefix='.'+path.name+'.',delete=False) as handle:
        handle.write(text); temporary=handle.name
    os.replace(temporary,path)
    if path.read_text()!=text: raise IOError(f'Incomplete write: {path}')
def pixels(path,rgb=False):
    with Image.open(path) as im: return np.asarray(im.convert('RGB') if rgb else im).copy()
def save(path,mask):
    path.parent.mkdir(parents=True,exist_ok=True)
    Image.fromarray(np.asarray(mask,np.uint8)*255).save(path)
    return str(path.resolve())
def task_key(spec): return json.dumps({k:spec[k] for k in ('task_type','task_params','prompt_en')},sort_keys=True)
def truth_key(spec): return json.dumps({k:spec[k] for k in ('task_type','task_params')},sort_keys=True)


def prepare_legacy(out):
    from verifier_v1_cleaning import DEFAULTS
    if (out/'generator_inputs.json').exists(): raise FileExistsError('A fresh version directory is required')
    original=read(OLD/'generator_inputs.json')
    old_rows=[json.loads(line) for line in (OLD/'manifest.jsonl').read_text().splitlines() if line.strip()]
    origins={r['source_group_id']:r for r in original}
    grouped={k:[] for k in origins}; anchors=[]
    for r in old_rows:
        if r['generator']=='gt_anchor':
            anchors.append(dict(source_group_id=r['source_group_id'],spec={k:r[k] for k in SPEC_KEYS}))
            continue
        specs=grouped[r['source_group_id']]
        spec={k:r[k] for k in SPEC_KEYS}
        if not any(task_key(v)==task_key(spec) for v in specs):
            spec['previous_manifest_ids']=[v['id'] for v in old_rows if v['source_group_id']==r['source_group_id'] and v['generator']!='gt_anchor' and task_key(v)==task_key(spec)]
            if r['task_type']=='transition':
                spec['transition_qc']={k:v for k,v in r['cleaning'].items() if k in ('transition_completeness','max_fragments','min_completeness','allowed_fragments','proxy_phase','proxy_class','matched_entities')}
            specs.append(spec)
    inputs=[]
    for row in original:
        value={k:row[k] for k in ('id','dataset','dataset_slug','source_image_id','source_group_id','image_t1','image_t2')}
        value['pilot_tasks']=grouped[row['source_group_id']]
        assert value['pilot_tasks']; inputs.append(value)
    assert len(inputs)==28 and len(anchors)==4
    cfg=dict(DEFAULTS,methods=list(METHODS),counterfactual_fraction=.2,transition_coverage=.95,max_transition_fragments=1,
        source_manifest=str(OLD/'manifest.jsonl'),source_policy='Exact previous 28 sources and exported task instructions; no source reselection',
        selection_policy='Keep every natural prediction, including empty masks; no IoU-based selection',review_status='pending_review')
    write(out/'config.json',cfg); write(out/'generator_inputs.json',inputs); write(out/'gt_anchor_tasks.json',anchors)
    for name in ('split_audit.json','source_splits.json'):
        write(out/name,read(OLD/name))
    write(out/'frozen_task_inventory.json',dict(sources=28,tasks=sum(len(x['pilot_tasks']) for x in inputs),
        tasks_by_type=dict(Counter(t['task_type'] for row in inputs for t in row['pilot_tasks'])),gt_anchors=4,
        generator_independent=True,source_manifest=str(OLD/'manifest.jsonl'),source_pairs_unchanged=True,official_val_test_used=False))
    print('frozen',read(out/'frozen_task_inventory.json'),flush=True)


def prepare(out,agent_semantics=False,source_run=None,task_adapted=False):
    agent_semantics = agent_semantics or task_adapted
    if not agent_semantics:
        if source_run is not None: raise ValueError('--source-run requires --agent-semantics')
        return prepare_legacy(out)
    from verifier_v1_cleaning import DEFAULTS
    source=(source_run or (ADAPTED_SOURCE if task_adapted else AGENT_SOURCE)).resolve()
    if out.resolve()==source: raise ValueError('New generation must use a fresh namespace')
    if any((out/name).exists() for name in ('generator_inputs.json','generation','predictions','cases')):
        raise FileExistsError('Agent generation requires a fresh namespace; only logs/runtime may preexist')
    original=read(source/'generator_inputs.json'); inputs=[]; annotations={}
    for row in original:
        value={key:row[key] for key in ('id','dataset','dataset_slug','source_image_id','source_group_id','image_t1','image_t2')}
        # Frozen instructions only: old masks, GT statistics and task-selection
        # annotations never enter adapters or generation receipts.
        value['pilot_tasks']=[{key:spec[key] for key in SPEC_KEYS} for spec in row['pilot_tasks']]
        for index,spec in enumerate(row['pilot_tasks']):
            if spec.get('transition_qc'): annotations[f'{row["id"]}:{index}']=spec['transition_qc']
        inputs.append(value)
    if len(inputs)!=28 or sum(len(row['pilot_tasks']) for row in inputs)!=74:
        raise ValueError('Agent generation requires the frozen 28 sources / 74 instructions')
    if task_adapted:
        annotations=read(source/'frozen_task_annotations.json')
    cfg=dict(DEFAULTS,methods=list(METHODS),counterfactual_fraction=.2,transition_coverage=.95,max_transition_fragments=1,
        task_adapted=task_adapted,agent_semantics=True,source_run=str(source),creation_slurm_job=os.environ['SLURM_JOB_ID'],gt_anchors=0,
        source_policy='Exact frozen RGB pairs and 74 instructions; baseline masks used only by independent regression QC',
        selection_policy='Keep all 222 normal predictions, including empty masks; no GT or IoU filtering',
        generation_policy='Only RGB, frozen Q, model weights/configuration; no GT crop, intersection or GT anchors',
        review_status='pending_review')
    write(out/'config.json',cfg); write(out/'generator_inputs.json',inputs)
    write(out/'gt_anchor_tasks.json',[]); write(out/'frozen_task_annotations.json',annotations)
    for name in ('split_audit.json','source_splits.json'): write(out/name,read(source/name))
    inventory=dict(sources=28,tasks=74,normal_records=222,counterfactual_target_per_method=18,gt_anchors=0,
        tasks_by_type=dict(Counter(t['task_type'] for row in inputs for t in row['pilot_tasks'])),
        generator_independent=True,source_run=str(source),source_pairs_unchanged=True,official_val_test_used=False,
        agent_semantics=True,task_adapted=task_adapted)
    write(out/'frozen_task_inventory.json',inventory); print('frozen',inventory,flush=True)


def infer(out,method,smoke=False):
    import torch
    from verifier_v1_ovcd_methods import DynamicAdapter,OmniAdapter,ReAAdapter
    assert torch.cuda.is_available() and torch.cuda.device_count()==1
    torch.manual_seed(20261118)
    cfg=read(out/'config.json'); agent=cfg.get('agent_semantics',False)
    if agent and cfg['creation_slurm_job']!=os.environ['SLURM_JOB_ID']:
        raise RuntimeError('Agent generation namespace belongs to a previous Slurm job')
    rows=read(out/'generator_inputs.json')
    if smoke:
        selected=[]
        for dataset in ('LEVIR-CD','WHU-CD','SECOND'):
            selected.append(next(r for r in rows if r['dataset']==dataset))
        if cfg.get('task_adapted'):
            second=[r for r in rows if r['dataset']=='SECOND']
            transition=next((r for r in second if any(t['task_type']=='transition' for t in r['pilot_tasks'])),None)
            building=next((r for r in second if any(t['task_type']=='related' and t['task_params']['class_id']==5 for t in r['pilot_tasks'])),None)
            selected=selected[:2]+[r for r in (transition,building) if r is not None]
            selected=list({r['id']:r for r in selected}.values())
        rows=selected
    work=out/'generation'/method; work.mkdir(parents=True,exist_ok=True)
    classes={'dynamic_earth_mci':DynamicAdapter,'omniovcd':OmniAdapter,'rea_ovcd':ReAAdapter}
    if cfg.get('task_adapted'):
        from verifier_v1_adapted_native import DynamicAdaptedAdapter, ReAAdaptedAdapter
        from verifier_v1_adapted_omni import OmniAdaptedAdapter
        classes={'dynamic_earth_mci':DynamicAdaptedAdapter,'omniovcd':OmniAdaptedAdapter,'rea_ovcd':ReAAdaptedAdapter}
    cls=classes[method]
    model=None
    for row in rows:
        receipt=work/f'{row["id"]}.json'
        paths=[out/'predictions'/method/row['id']/f'task_{i:03}.png' for i in range(len(row['pilot_tasks']))]
        if receipt.exists():
            previous=read(receipt)
            assert previous['generated_in_this_run'] and previous['task_keys']==[task_key(t) for t in row['pilot_tasks']]
            if agent: assert previous['slurm_job_id']==os.environ['SLURM_JOB_ID']
            assert all(p.is_file() for p in paths)
            print(method,row['id'],'reuse fresh predictions from this run smoke',flush=True); continue
        if model is None:
            print(method,'loading official model',flush=True)
            model=cls(work,read(REPO/'tools/verifier_v1_dynamic_configs.json')) if method=='dynamic_earth_mci' else cls(work)
        start=time.monotonic(); print(method,row['id'],'start',flush=True)
        with torch.inference_mode(): values,metadata=model.infer({**row,'agent_semantics':True} if agent else row)
        if cfg.get('task_adapted') and smoke:
            from verifier_v1_generation_qc import independent_compile
            with np.load(metadata['model_state'],allow_pickle=False) as state:
                for index,(spec,basis) in enumerate(zip(row['pilot_tasks'],metadata['task_basis'])):
                    expected=independent_compile(spec,basis,state,True)
                    if not np.array_equal(np.asarray(values[index],bool),expected):
                        raise ValueError(f"Smoke independent composition failed: {row['id']}/{index}")
            baseline=read(Path(cfg['source_run'])/'generation'/method/f"{row['id']}.json")
            for index,spec in enumerate(row['pilot_tasks']):
                if spec['task_type']=='related':
                    prior=pixels(baseline['predictions'][index])>0
                    if not np.array_equal(np.asarray(values[index],bool),prior):
                        raise ValueError(f"Smoke native related regression failed: {row['id']}/{index}")
        shape=pixels(row['image_t1'],True).shape[:2]
        assert len(values)==len(paths)
        for path,mask in zip(paths,values):
            assert mask.shape==shape and np.isfinite(mask).all()
            save(path,mask)
        if agent:
            if not Path(metadata['model_state']).is_absolute() or len(metadata['task_basis'])!=len(paths):
                raise ValueError('Missing pure-model state or per-task semantic basis')
        record=dict(source_id=row['id'],dataset=row['dataset'],method=method,generated_in_this_run=True,
            task_keys=[task_key(t) for t in row['pilot_tasks']],predictions=[str(p.resolve()) for p in paths],
            seconds=time.monotonic()-start,mask_areas=[int(np.asarray(v,bool).sum()) for v in values],metadata=metadata)
        if agent:
            record.update(task_adapted=cfg.get('task_adapted',False),agent_semantics=True,slurm_job_id=os.environ['SLURM_JOB_ID'],
                input_images={key:row[key] for key in ('image_t1','image_t2')},ground_truth_inputs=False)
        write(receipt,record)
        print(method,row['id'],'complete',round(record['seconds'],1),'areas',record['mask_areas'],flush=True)
    if smoke:
        write(work/'smoke_qc.json',dict(status='passed',sources=[r['id'] for r in rows],methods_core_called=True,ground_truth_inputs=False,
            independent_composition='passed' if cfg.get('task_adapted') else 'not_applicable',
            related_native_regression='passed' if cfg.get('task_adapted') else 'not_applicable'))


def export(out):
    from verifier_v1_pilot_generate import raw_target,quality,FORBIDDEN
    from verifier_v1_cleaning import clean_diagnostic
    cfg=read(out/'config.json'); inputs=read(out/'generator_inputs.json')
    agent=cfg.get('agent_semantics',False)
    if agent:
        generation_qc=read(out/'generation_qc.json')
        if generation_qc['status']!='passed' or not generation_qc.get('agent_semantics'):
            raise RuntimeError('Generation provenance/formula QC must pass before loading GT')
        annotations=read(out/'frozen_task_annotations.json')
        for row in inputs:
            for index,spec in enumerate(row['pilot_tasks']):
                if f'{row["id"]}:{index}' in annotations:
                    spec['transition_qc']=annotations[f'{row["id"]}:{index}']
    allocation=read(out/'source_splits.json')['datasets']; records=[]; checks=Counter(); cf=Counter(); rgb_cache={}; raw_cache={}
    def raw(row,spec):
        key=(row['id'],truth_key(spec))
        if key not in raw_cache: raw_cache[key]=raw_target(row,spec)
        return raw_cache[key]
    def diagnostics(row,spec,mask):
        if row['id'] not in rgb_cache:
            rgb_cache[row['id']]=[pixels(row[f'image_t{i}'],True) for i in (1,2)]
        t1,t2=rgb_cache[row['id']]
        return clean_diagnostic(mask,raw(row,spec),cfg,dataset=row['dataset'],image_t1=t1,image_t2=t2)
    def add(row,method,spec,mask_path,marker,base=None,prepared=None,generation=None):
        pred=pixels(mask_path)>0; gt=raw(row,spec); d=prepared if prepared is not None else diagnostics(row,spec,pred)
        ident=f'{row["id"]}_{method}_{spec["task_type"]}_{marker}'; folder=out/'cases'/ident
        label={key:save(folder/f'{key}.png',d[field]) for key,field in [('fp','fp'),('fn','fn'),('ignore_mask','ignore')]}
        raw_path=save(folder/'raw_task_gt.png',gt); clean_path=save(folder/'gt_clean.png',d['clean'])
        valid=~d['ignore']; stats=d['cleaning']; stats.update(spec.get('transition_qc',{})); metrics=d['metrics']
        if not metrics['raw_foreground_union']: metrics['raw_iou']=None
        clean_spec={k:spec[k] for k in SPEC_KEYS}
        tags=[]
        if metrics['fp_area']: tags.append('false_positive')
        if metrics['fn_area']: tags.append('false_negative')
        if metrics['empty_effective_foreground']: tags.append('no_effective_foreground')
        if method=='gt_anchor': tags.append('gt_anchor')
        elif metrics['iou'] is not None and metrics['iou']>=.9: tags.append('natural_high_score')
        if stats['removed_gt_components']: tags.append('tiny_gt_ignored')
        if stats['no_data_pixels']: tags.append('no_data_ignored')
        if stats['whu_strip_components']: tags.append('whu_strip_ignored')
        source_root=Path(row['image_t1']).parents[1]
        if row['dataset']=='LEVIR-CD': origin=source_root/'label'/row['source_image_id']
        else: origin=source_root/'label_unch0chi1'/row['source_image_id']
        receipt=out/'generation'/method/f'{row["id"]}.json'
        item=dict(id=ident,**{k:row[k] for k in ('dataset','source_group_id','source_image_id','image_t1','image_t2')},
            official_split='train',internal_split='fit',review_status='pending_review',**clean_spec,generator=method,
            init_mask=str(mask_path.resolve()),gt_mask=clean_path,raw_task_gt=raw_path,labels=label,metrics=metrics,cleaning=stats,
            quality_band=quality(metrics['iou']),error_tags=tags,is_counterfactual=base is not None,
            base_task_id=base['id'] if base else None,counterfactual_group_id=base['counterfactual_group_id'] if base else ident,
            source=dict(dataset_path=str(source_root),original_change_gt=str(origin),generator_receipt=str(receipt) if method!='gt_anchor' else None,
                gt_lineage='Unaltered raw task truth retained; GT fragments only removed in separate cleaned target and fully ignored'),
            isolation=dict(all_descendants_grouped=True,known_actual_exclusions_applied=True,source_unit='native_complete_pair'),
            previous_task_manifest_ids=spec.get('previous_manifest_ids',[]))
        if agent:
            if base:
                item['generation']=dict(base['generation'],formula_verified=False,counterfactual_same_mask=True,
                    base_task_id=base['id'],note='Q changed after model generation; normal-task formula intentionally not reapplied')
            else:
                item['generation']=generation
        invariant=dict(fp_inside_m=not (d['fp']&~pred).any(),fn_outside_m=not (d['fn']&pred).any(),
            disjoint=not (d['fp']&d['fn']).any(),ignore_excluded=not ((d['fp']|d['fn'])&d['ignore']).any(),
            fp_formula=np.array_equal(d['fp'],pred&~d['clean']&valid),fn_formula=np.array_equal(d['fn'],~pred&d['clean']&valid),
            valid_reconstruction=np.array_equal(((pred&~d['fp'])|d['fn'])[valid],d['clean'][valid]),
            no_false_fp_from_gt_cleaning=not ((gt&~d['clean'])&~d['ignore']).any(),
            candidate_unchanged=np.array_equal(pred,pixels(mask_path)>0),raw_gt_unchanged=np.array_equal(gt,pixels(raw_path)>0),
            independent_bodies_protected=not (d['protected']&d['ignore']).any(),
            ignore_reasons_disjoint=sum(int(v.sum()) for v in d['ignore_reasons'].values())==int(d['ignore'].sum()),
            source_in_fit=row['source_image_id'] in allocation[row['dataset']]['fit'],
            prompt_clean=not any(FORBIDDEN.search(v) for v in [spec['prompt_en'],spec['prompt_zh']]),
            gt_anchor_zero=method!='gt_anchor' or np.array_equal(pred,gt) and not (d['fp']|d['fn']).any())
        for residual in ('fp','fn'):
            keys=[k for k in stats if k.endswith(f'_ignored_{residual}_area')]
            invariant[f'{residual}_funnel']=stats[f'raw_{residual}_area']-sum(stats[k] for k in keys)==metrics[f'{residual}_area']
        if base:
            bg=pixels(base['gt_mask'])>0; bv=~(pixels(base['labels']['ignore_mask'])>0)
            invariant.update(cf_nonempty=pred.any(),cf_same_m=np.array_equal(pred,pixels(base['init_mask'])>0),
                cf_gt_diff=((bg^d['clean'])&bv&valid).any(),cf_changed_q=base['prompt_en']!=spec['prompt_en'])
        if spec['task_type']=='transition':
            invariant['transition_complete']=stats['transition_completeness']>=cfg['transition_coverage'] and stats['max_fragments']<=cfg['max_transition_fragments']
        for key,passed in invariant.items(): checks[key+('_passed' if passed else '_failed')]+=1
        records.append(item); return item
    base_rows={}
    for method in METHODS:
        for row in inputs:
            receipt=read(out/'generation'/method/f'{row["id"]}.json')
            assert receipt['generated_in_this_run']
            for index,spec in enumerate(row['pilot_tasks']):
                path=Path(receipt['predictions'][index]); provenance=None
                if agent:
                    from verifier_v1_generation_qc import recipe
                    basis=receipt['metadata']['task_basis'][index]
                    provenance=dict(semantics='task_adapted' if cfg.get('task_adapted') else 'agent',recipe=recipe(spec,basis),semantic_mode=basis['semantic_mode'],
                        route=basis.get('route','agent_semantic'),recipe_type=basis.get('recipe_type','dense_boolean'),
                        model_state=receipt['metadata']['model_state'],task_basis=basis,
                        receipt=str((out/'generation'/method/f'{row["id"]}.json').resolve()),task_index=index,
                        formula_verified=True,ground_truth_inputs=False)
                base_rows[(method,row['id'],index)]=add(row,method,spec,path,f'normal_{index}',generation=provenance)
    # Deterministic task/source order. Empty masks and semantically identical Q
    # pairs are ineligible CF; all normal model outputs remain in the package.
    for method in METHODS:
        total=sum(len(row['pilot_tasks']) for row in inputs); target=round(total*cfg['counterfactual_fraction']/(1-cfg['counterfactual_fraction']))
        accepted=0; seen=[]
        for slot in range(max(len(row['pilot_tasks']) for row in inputs)):
            for row in inputs:
                if row['dataset']!='SECOND' or slot>=len(row['pilot_tasks']) or accepted>=target: continue
                base=base_rows[(method,row['id'],slot)]; mask=pixels(base['init_mask'])>0
                if not mask.any(): cf[f'{method}_empty_m_ineligible']+=1; continue
                bg=pixels(base['gt_mask'])>0; bv=~(pixels(base['labels']['ignore_mask'])>0)
                for j,spec in enumerate(row['pilot_tasks']):
                    if spec['prompt_en']==base['prompt_en']: continue
                    if any(rid==row['id'] and q==spec['prompt_en'] and np.array_equal(mask,m) for rid,q,m in seen): continue
                    d=diagnostics(row,spec,mask)
                    if not ((bg^d['clean'])&bv&~d['ignore']).any(): continue
                    add(row,method,spec,Path(base['init_mask']),f'cf_{accepted}',base=base,prepared=d)
                    seen.append((row['id'],spec['prompt_en'],mask)); accepted+=1; break
        cf[f'{method}_target']=target; cf[f'{method}_accepted']=accepted
    origins={r['source_group_id']:r for r in inputs}
    anchors=read(out/'gt_anchor_tasks.json')
    if agent and anchors: raise RuntimeError('GT anchors are forbidden in agent generation')
    for i,anchor in enumerate(anchors):
        row=origins[anchor['source_group_id']]; spec=anchor['spec']; path=out/'predictions/gt_anchor'/f'{row["id"]}_{i}.png'
        save(path,raw(row,spec)); add(row,'gt_anchor',spec,path,f'anchor_{i}')
    failures={k:v for k,v in checks.items() if k.endswith('_failed')}
    if cfg.get('task_adapted'):
        for method in METHODS:
            if cf[f'{method}_accepted'] != 18: failures[f'{method}_cf_count']=cf[f'{method}_accepted']
        if len(records)!=276: failures['total_records']=len(records)
    expected={r['id'] for r in records}; actual={p.name for p in (out/'cases').iterdir() if p.is_dir()}
    if actual!=expected: failures['case_directory_mismatch']=len(actual^expected)
    qc=dict(status='passed' if not failures else 'failed',records=len(records),invariants=dict(checks),failures=failures,
        task_adapted=cfg.get('task_adapted',False),agent_semantics=agent,gt_anchors=len(anchors),normal_records=sum(not r['is_counterfactual'] and r['generator']!='gt_anchor' for r in records),
        counterfactual_records=sum(r['is_counterfactual'] for r in records),
        generation_qc='generation_qc.json' if agent else None,fp_fn_only=True,cleaning_config=cfg,counterfactual=dict(cf),official_methods=list(METHODS),automatic_only=True)
    write(out/'label_qc.json',qc)
    (out/'manifest.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in records))
    summary=dict(status='pending_review',independent_source_pairs=len(inputs),diagnostic_records=len(records),
        datasets=dict(Counter(r['dataset'] for r in records)),sources_by_dataset=dict(Counter(r['dataset'] for r in inputs)),
        generators=dict(Counter(r['generator'] for r in records)),tasks=dict(Counter(r['task_type'] for r in records)),
        quality_bands=dict(Counter(r['quality_band'] for r in records)),counterfactual=dict(cf),
        task_adapted=cfg.get('task_adapted',False),agent_semantics=agent,gt_anchors=len(anchors),normal_records=qc['normal_records'],counterfactual_records=qc['counterfactual_records'],
        normal_quality_bands=dict(Counter(r['quality_band'] for r in records if not r['is_counterfactual'])),
        counterfactual_quality_bands=dict(Counter(r['quality_band'] for r in records if r['is_counterfactual'])),
        cohort_metrics={name:dict(records=len(cohort),mean_effective_iou=float(np.mean(scores)) if scores else None,
            fp_area=sum(r['metrics']['fp_area'] for r in cohort),fn_area=sum(r['metrics']['fn_area'] for r in cohort),
            mean_ignore_fraction=float(np.mean([r['metrics']['ignore_fraction'] for r in cohort])) if cohort else None)
            for name,cohort in [('normal',[r for r in records if not r['is_counterfactual'] and r['generator']!='gt_anchor']),
                                ('counterfactual',[r for r in records if r['is_counterfactual']])]
            for scores in [[r['metrics']['effective_iou'] for r in cohort if r['metrics']['effective_iou'] is not None]]},
        qc_status=qc['status'],invariant_failures=failures,selection_policy=cfg['selection_policy'],
        mean_ignore_fraction=float(np.mean([r['metrics']['ignore_fraction'] for r in records])),
        cleaning_totals={k:sum((r['metrics'] if k in ('fp_area','fn_area') else r['cleaning']).get(k,0) for r in records) for k in ('raw_fp_area','raw_fn_area','fp_area','fn_area','no_data_pixels','whu_strip_components')})
    if agent:
        summary.pop('mean_ignore_fraction')
        summary.pop('quality_bands')
    # Preserve two specifically requested regression artifacts, not a project copy.
    import shutil
    regression=REPO/'.tmp/verifier_v1_official_cleaning_check_v2'
    evidence=out/'cleaning_validation'
    if not agent: evidence.mkdir(exist_ok=True)
    for source,name in (() if agent else ((regression/'cleaning_comparison.json','cleaning_comparison.json'),
                        (regression/'whu_train_658_sam3_imc_related_0/changed_labels_crop.png','whu658_before_after.png'))):
        shutil.copyfile(source,evidence/name)
    if not agent: summary['cleaning_validation']='cleaning_validation/cleaning_comparison.json'
    if cfg.get('task_adapted'):
        comparison=compare_versions(Path(cfg['source_run']), records)
        write(out/'v3_v4_comparison.json',comparison)
        summary['v3_v4_comparison']='v3_v4_comparison.json'
    write(out/'summary.json',summary); print(json.dumps(summary,ensure_ascii=False),flush=True)
    if failures: raise RuntimeError(f'Label QC failed: {failures}')


def compare_versions(source, records):
    """Offline descriptive comparison; normal performance never pools CF."""
    previous=[json.loads(line) for line in (source/'manifest.jsonl').read_text().splitlines() if line.strip()]
    def statistics(rows):
        groups={}
        for row in rows:
            if row['generator']=='gt_anchor': continue
            key=(row['generator'],row['dataset'],row['task_type'],'CF' if row.get('is_counterfactual') else 'normal')
            groups.setdefault(key,[]).append(row)
        result={}
        for key,items in groups.items():
            raw=[r['metrics'].get('raw_iou') for r in items]
            eff=[r['metrics'].get('effective_iou',r['metrics'].get('iou')) for r in items]
            areas=[int((pixels(r['init_mask'])>0).sum()) for r in items]
            fractions=[area/(pixels(r['init_mask']).size) for area,r in zip(areas,items)]
            result[key]=dict(records=len(items),mean_raw_iou=float(np.mean([v for v in raw if v is not None])) if any(v is not None for v in raw) else None,
                mean_effective_iou=float(np.mean([v for v in eff if v is not None])) if any(v is not None for v in eff) else None,
                empty_mask_records=sum(a==0 for a in areas),empty_mask_fraction=sum(a==0 for a in areas)/len(items),
                mean_mask_area=float(np.mean(areas)),mean_mask_fraction=float(np.mean(fractions)))
        return result
    before,after=statistics(previous),statistics(records)
    rows=[]
    for key in sorted(before.keys()|after.keys()):
        old,new=before.get(key),after.get(key)
        delta={field:new[field]-old[field] if old[field] is not None and new[field] is not None else None
               for field in new if field!='records'} if old and new else None
        rows.append(dict(zip(('method','dataset','task_type','cohort'),key),v3=old,v4=new,delta=delta))
    return dict(status='descriptive',baseline_run=str(source),normal_performance_includes_cf=False,
                grouping=['method','dataset','task_type','cohort'],rows=rows,
                counterfactual_note='CF strata compare exported condition-audit cohorts; they are not model inference performance')


def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('stage',choices=['prepare','infer','export'])
    p.add_argument('--output',type=Path,required=True); p.add_argument('--method',choices=METHODS); p.add_argument('--smoke',action='store_true')
    mode=p.add_mutually_exclusive_group()
    mode.add_argument('--agent-semantics',action='store_true')
    mode.add_argument('--task-adapted',action='store_true')
    p.add_argument('--source-run',type=Path)
    args=p.parse_args()
    if not os.environ.get('SLURM_JOB_ID'): raise RuntimeError('Slurm only')
    args.output.mkdir(parents=True,exist_ok=True)
    if args.stage=='prepare': prepare(args.output,args.agent_semantics,args.source_run,args.task_adapted)
    elif args.stage=='infer':
        if args.method is None: p.error('--method required')
        infer(args.output,args.method,args.smoke)
    else: export(args.output)

if __name__=='__main__': main()
