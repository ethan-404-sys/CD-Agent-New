"""Frozen 600/60/60 source split and real-model extension of the approved pilot."""
from __future__ import annotations
import argparse
from collections import Counter
import copy
import json
import os
from pathlib import Path
import random
import time

REPO = Path(__file__).resolve().parents[3]
ROOT = Path(os.environ.get('CD_AGENT_ACCOUNT_ROOT', '/guisongxia01/pangchao/wangyihan'))
DATASETS = Path(os.environ.get('CD_AGENT_DATASETS', REPO / 'datasets'))
RS = DATASETS
PILOT = Path(os.environ.get('VERIFIER_PILOT_DIR', ROOT / 'wyh/verifier-v1-pilot-20261117/results/verifier_v1_pilot/20261118_verifier_v1_final'))
METHODS = ('dynamic_earth_mci', 'omniovcd', 'rea_ovcd')
DATASETS = ('LEVIR-CD', 'WHU-CD', 'SECOND')
SLUGS = dict(zip(DATASETS, ('levir', 'whu', 'second')))
SEED = 20261119


def read(path): return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name('.' + path.name + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(path)


def rows(path): return [json.loads(s) for s in Path(path).read_text().splitlines() if s.strip()]


def write_rows(path, records):
    temporary = path.with_name('.' + path.name + '.tmp')
    temporary.write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in records))
    temporary.replace(path)


def shuffled(values, salt):
    result = sorted(set(values)); random.Random(SEED + salt).shuffle(result); return result


def source(dataset, name, split='train'):
    root = DATASETS / 'LEVIR-CD' / f'{split}_original' if dataset == 'LEVIR-CD' else RS / dataset
    dirs = ('A', 'B') if dataset == 'LEVIR-CD' else ('img1', 'img2')
    return dict(id=f'{SLUGS[dataset]}_{Path(name).stem}', dataset=dataset, dataset_slug=SLUGS[dataset],
                source_image_id=name, source_group_id=f'{dataset}/{split}/{name}',
                image_t1=str(root / dirs[0] / name), image_t2=str(root / dirs[1] / name))


def approve_pilot():
    records = rows(PILOT / 'manifest.jsonl')
    if len(records) != 276: raise ValueError('Expected 276 manually approved pilot records')
    for row in records: row.update(review_status='approved', review_basis='User confirmed full pilot review on 2026-11-19')
    write_rows(PILOT / 'manifest.jsonl', records)
    for name in ('summary.json', 'config.json'):
        value = read(PILOT / name)
        value.update(review_status='approved', review_basis='User confirmed full pilot review on 2026-11-19')
        if name == 'summary.json': value['status'] = 'approved'
        write(PILOT / name, value)
    # Only generated status text/metadata; never touch image or label assets.
    for name in ('audit.html', 'summary.html', 'summary.md'):
        path = PILOT / name; text = path.read_text().replace('pending_review', 'approved')
        for old, new in (('待人工审核', '人工审核通过'), ('全部候选等待人工审核。', '全部候选已由用户完成人工审核。'),
                         ('病例仍为 approved。', '病例已人工审核通过（approved）。'),
                         ('每条病例保持 approved。', '每条病例已人工审核通过（approved）。')):
            text = text.replace(old, new)
        path.write_text(text)
    from build_verifier_v1_summary import EmbeddedData
    from build_verifier_v1_review import display_metadata
    path=PILOT/'audit.html'; html=path.read_text(); parser=EmbeddedData(); parser.feed(html)
    original=''.join(parser.parts); data=json.loads(original)
    data.update(status='approved',frozen_config=display_metadata(read(PILOT/'config.json')))
    for record in data['records']: record['review_status']='approved'
    html=html.replace(original,json.dumps(data,ensure_ascii=False).replace('<','\\u003c'),1)
    path.write_text(html)
    write(PILOT / 'human_review.json', dict(status='approved', records=276, independent_source_pairs=28,
          basis='Explicit user approval on 2026-11-19', masks_and_supervision_changed=False))


def prepare(out):
    import numpy as np
    from verifier_v1_pilot_generate import image, task, semantic_truth, transition_filter, CLEAR
    from verifier_v1_cleaning import clean_diagnostic
    if (out / 'split_report.json').exists(): raise FileExistsError('Do not resample a prepared dataset')
    out.mkdir(parents=True, exist_ok=True)
    old = read(PILOT / 'generator_inputs.json'); prior = read(PILOT / 'source_splits.json')['datasets']
    cfg = read(PILOT / 'config.json')
    cfg.update(final_package=False, package_status='dataset_extension', stage='data_generation',
               task_adapted=False, agent_semantics=True, seed=SEED, review_status='pending_review',
               selection_policy='Frozen train sources; retain all model outputs, including empty M; no IoU selection',
               creation_slurm_job=os.environ['SLURM_JOB_ID'], approved_pilot=str(PILOT), expected_sources=572)
    old_names = {d: {r['source_image_id'] for r in old if r['dataset'] == d} for d in DATASETS}
    train, val, test, additions, inspected = {}, {}, {}, [], {}
    exclusions = set((REPO / 'doc/second_active_dev_ids.txt').read_text().split())
    inventory = []
    for number, dataset in enumerate(DATASETS):
        registered = prior[dataset]
        pool = set().union(*(registered[k] for k in ('fit', 'val', 'offline_eval')))
        if dataset == 'SECOND': pool -= exclusions
        if not old_names[dataset] <= pool: raise ValueError(f'Approved sources absent from admitted pool: {dataset}')
        exposed = set(registered.get('label_inspected_fit_ids', [])) | old_names[dataset]
        candidates = set(registered['val']) | set(registered['offline_eval'])
        val[dataset] = shuffled(candidates - exposed - (exclusions if dataset == 'SECOND' else set()), 10 + number)[:20]
        if len(val[dataset]) != 20: raise ValueError(f'Insufficient validation sources: {dataset}')
        if dataset == 'LEVIR-CD':
            names = [p.name for p in (DATASETS / 'LEVIR-CD/test_original/A').iterdir() if p.suffix == '.png']
        else:
            names = [Path(x[0]).name for x in read(RS / dataset / 'dataset_index.json') if Path(x[0]).name.startswith('test_')]
        test[dataset] = shuffled(names, 20 + number)[:20]
        needed = 200 - len(old_names[dataset]); chosen = []; inspected[dataset] = []
        class_counts = Counter(r['pilot_tasks'][0]['task_params']['class_id'] for r in old if r['dataset'] == dataset)
        for name in shuffled(pool - old_names[dataset] - set(val[dataset]), 30 + number):
            record = source(dataset, name); index = len(chosen)
            if dataset == 'WHU-CD':
                raw = image(RS / dataset / 'label_unch0chi1' / name) > 0; inspected[dataset].append(name)
                if int(raw.sum()) < 256: continue
                if not clean_diagnostic(raw, raw, cfg)['metrics']['effective_foreground_union']: continue
            if dataset == 'SECOND':
                cmap, l1, l2 = semantic_truth(name); inspected[dataset].append(name)
                if not np.isin(l1, range(7)).all() or not np.isin(l2, range(7)).all():
                    raise ValueError(f'Invalid SECOND labels: {name}')
                available = [c for c in CLEAR if int((cmap & ((l1 == c) | (l2 == c))).sum()) >= cfg['min_area']]
                if not available: continue
                category = min(available, key=lambda c: (class_counts[c], c)); class_counts[category] += 1
                tasks = [task(kind, category, index=index) for kind in ('related', 'into', 'outof')]
                hist = np.bincount((l1[cmap].astype(int)*7+l2[cmap]).ravel(), minlength=49).reshape(7, 7)
                transitions = [(int(hist[a,b]), int(a), int(b)) for a,b in zip(*np.nonzero(hist >= cfg['min_area']))
                               if a and b and a != b and category in (a,b)]
                for _, a, b in sorted(transitions, reverse=True):
                    okay, _ = transition_filter(cmap, l1, l2, a, b, cfg)
                    if okay: tasks.append(task('transition', category, a, b, index=index)); break
            else: tasks = [task('related', 5, index=index)]
            record['pilot_tasks'] = tasks; additions.append(record); chosen.append(name)
            if len(chosen) == needed: break
        if len(chosen) != needed: raise ValueError(f'{dataset}: found {len(chosen)} / {needed} usable new sources')
        train[dataset] = sorted(old_names[dataset] | set(chosen))
        for split, mapping in (('train', train), ('val', val), ('test', test)):
            official = 'test' if split == 'test' else 'train'
            for name in mapping[dataset]:
                record = source(dataset, name, official)
                if not all(Path(record[k]).is_file() for k in ('image_t1', 'image_t2')): raise FileNotFoundError(record['image_t1'])
                inventory.append(dict(split=split, official_split=official, reviewed_pilot=name in old_names[dataset] and split == 'train', **record))
        print('SELECTED', dataset, dict(train=200, val=20, test=20, new=len(chosen), inspected=len(inspected[dataset])), flush=True)
    for dataset in DATASETS:
        a,b,c = map(set, (train[dataset], val[dataset], test[dataset]))
        if a & b or a & c or b & c: raise ValueError('Source split overlap')
    for split, mapping in (('train', train), ('val', val), ('test', test)):
        (out / f'{split}.txt').write_text(''.join(f'{d}\t{n}\n' for d in DATASETS for n in mapping[d]))
    (out / 'train_new.txt').write_text(''.join(f'{r["dataset"]}\t{r["source_image_id"]}\n' for r in additions))
    write(out/'sources.json', inventory); write(out/'generator_inputs.json', additions); write(out/'config.json', cfg)
    write(out/'gt_anchor_tasks.json', []); write(out/'frozen_task_annotations.json', {})
    write(out/'source_splits.json', dict(seed=SEED, datasets={d:dict(fit=train[d], val=val[d], offline_eval=[], test=test[d]) for d in DATASETS}))
    report = dict(status='passed', seed=SEED, train_source_pairs=600, val_source_pairs=60, test_source_pairs=60,
                  reviewed_pilot_source_pairs=28, reviewed_pilot_records=276, new_train_source_pairs=len(additions), overlap_pairs=0,
                  sources_by_dataset={d:dict(train=200, val=20, test=20, approved=len(old_names[d]), new=200-len(old_names[d])) for d in DATASETS},
                  validation_source='Held out from local train; no generation in this task', test_source='Local official test; no generation in this task',
                  source_unit='Complete dataset image pair; WHU uses standard 512px tiles',
                  inspected_training_candidates={d:len(v) for d,v in inspected.items()},
                  second_exclusion_file=str(REPO/'doc/second_active_dev_ids.txt'),
                  task_counts=dict(Counter(t['task_type'] for r in additions for t in r['pilot_tasks'])))
    write(out/'split_report.json', report); write(out/'split_audit.json', report); approve_pilot()
    (out/'README.md').write_text('# Verifier dataset\n\n训练600对、验证60对、测试60对；每数据集200/20/20对。\n\n'
        '`train.txt`、`val.txt`、`test.txt`：每行数据集名、制表符、原始样本文件名。\n'
        '`sources.json`：两期完整路径与官方/本次划分。WHU单位为标准切片。\n'
        '`train_new.txt`：新增572对；`manifest.jsonl`：完成后全部训练病例。\n'
        '`new_manifest.jsonl`：新增病例；旧276条为approved，新病例为pending_review。\n'
        '`split_report.json`：划分验收；`dataset_report.json`：最终验收；`logs/`：日志。\n'
        '验证、测试本轮只冻结名单，不运行模型。\n')
    print(json.dumps(report, ensure_ascii=False), flush=True)


def fill_omni(adapter, row, masks, metadata):
    """Frozen A: keep native B and all original nonzero phase labels."""
    import numpy as np
    import torch
    from PIL import Image
    from verifier_v1_ovcd_methods import planner_mask
    path = Path(metadata['model_state'])
    with np.load(path, allow_pickle=False) as archive: state = {k:archive[k] for k in archive.files}
    model = adapter.model; cfg = adapter.config('SECOND', 5)
    groups = ['background'] + [Path(adapter.config('SECOND', c)['classname_path']).read_text().splitlines()[1] for c in range(1,7)]
    words, indices = [], []
    for c, group in enumerate(groups):
        for word in group.split(','): words.append(word.strip()); indices.append(c)
    for key, value in cfg.items():
        if key != 'classname_path': setattr(model, key, value)
    model.processor.confidence_threshold = cfg['confidence_threshold']
    model.query_words = words; model.query_idx = torch.tensor(indices, dtype=torch.long, device='cuda')
    model.num_cls, model.num_queries = 7, len(words)
    for phase in (1,2):
        with Image.open(row[f'image_t{phase}']) as image: raw = model._inference_single_view(image.convert('RGB'))
        index = torch.nn.functional.one_hot(model.query_idx).T.view(7,len(words),1,1)
        state[f'fill_full_scores{phase}'] = (raw.unsqueeze(0)*index).max(1)[0].float().cpu().numpy()
    for number, (spec, basis) in enumerate(zip(row['pilot_tasks'], metadata['task_basis'])):
        if spec['task_type'] == 'related': continue
        b = state[basis['B_key']]
        basis.update(fill_rule='original_zero_positive_six_foreground_argmax', original_B_key=basis['B_key'])
        for phase in (1,2):
            original_key = basis[f'Y{phase}_key']; original = state[original_key]
            scores = state[f'fill_full_scores{phase}'][1:]; filled = original.copy()
            domain = b & (original == 0) & (scores.max(0) > 0)
            filled[domain] = (scores.argmax(0)+1)[domain]
            key = f'task_{number}_Y{phase}_filled'; state[key] = filled
            basis.update({f'original_Y{phase}_key':original_key, f'Y{phase}_key':key,
                          f'fill_scores{phase}_key':f'fill_full_scores{phase}'})
        basis['semantic_mode'] = 'native_task_B_foreground_completion_A'
        masks[number] = planner_mask(spec,b,state[basis['Y1_key']],state[basis['Y2_key']])
    np.savez_compressed(path, **state)
    metadata['foreground_completion'] = 'Frozen A; unchanged native B and original nonzero labels'
    return masks, metadata


def reconstruct(spec,basis,state):
    """Check native state without importing the legacy pilot's sys.path setup."""
    import numpy as np
    from verifier_v1_generation_qc import independent_compile
    if basis.get('fill_rule'):
        b=state[basis['B_key']]
        if not np.array_equal(b,state[basis['original_B_key']]): raise ValueError('Native B changed')
        for phase in (1,2):
            original=state[basis[f'original_Y{phase}_key']]
            fore=state[basis[f'fill_scores{phase}_key']][1:]
            expected=original.copy(); domain=b & (original==0) & (fore.max(0)>0)
            expected[domain]=(fore.argmax(0)+1)[domain]
            if not np.array_equal(expected,state[basis[f'Y{phase}_key']]): raise ValueError('Frozen foreground A differs')
    return independent_compile(spec,basis,state,True)


def infer(out, method, smoke, source_ids=None):
    print('INFER_PROCESS',method,'load dependencies',flush=True)
    import numpy as np
    import torch
    from verifier_v1_official_generate import pixels, save, task_key
    from verifier_v1_ovcd_methods import OmniAdapter, ReAAdapter
    from verifier_v1_adapted_native import DynamicAdaptedAdapter
    policies=read(out/'config.json')['methods']
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1: raise RuntimeError('Exactly one GPU required')
    torch.manual_seed(20261118)
    inputs = read(out/'generator_inputs.json')
    if source_ids:
        inputs=[row for row in inputs if row['id'] in set(source_ids)]
        if not inputs: raise ValueError('Requested source IDs absent from frozen inputs')
    if smoke: inputs = [next(r for r in inputs if r['dataset']==d) for d in DATASETS]
    work = out/'generation'/method; work.mkdir(parents=True,exist_ok=True); adapter = None
    for row in inputs:
        receipt = work/f'{row["id"]}.json'
        if receipt.exists():
            previous = read(receipt)
            if previous['task_keys'] != [task_key(t) for t in row['pilot_tasks']]: raise ValueError('Frozen tasks changed')
            if not all(Path(p).is_file() for p in previous['predictions']): raise FileNotFoundError(receipt)
            print('REUSE_CURRENT_DATASET',method,row['id'],flush=True); continue
        if adapter is None:
            print('LOAD_MODEL',method,flush=True)
            adapter = DynamicAdaptedAdapter(work,read(Path(__file__).with_name('verifier_v1_dynamic_configs.json'))) if method=='dynamic_earth_mci' else (OmniAdapter if method=='omniovcd' else ReAAdapter)(work)
        started = time.monotonic(); print('GENERATE',method,row['id'],'start',flush=True)
        with torch.inference_mode():
            masks, metadata = adapter.infer(dict(row,agent_semantics=True))
            if method=='omniovcd' and row['dataset']=='SECOND': masks,metadata = fill_omni(adapter,row,masks,metadata)
        for spec,basis in zip(row['pilot_tasks'],metadata['task_basis']):
            basis.update(recipe_type=basis.get('recipe_type','dense_boolean'),
                         route='official_related' if spec['task_type']=='related' else 'task_adapted', frozen_method_policy=policies[method])
        paths = []; shape = pixels(Path(row['image_t1']),True).shape[:2]
        if len(masks)!=len(row['pilot_tasks']): raise ValueError('Missing task masks')
        with np.load(metadata['model_state'],allow_pickle=False) as state:
            for i,(spec,basis,mask) in enumerate(zip(row['pilot_tasks'],metadata['task_basis'],masks)):
                if mask.shape!=shape or not np.array_equal(mask,reconstruct(spec,basis,state)):
                    raise ValueError(f'Independent composition differs: {row["id"]}/{i}')
                paths.append(save(out/'predictions'/method/row['id']/f'task_{i:03}.png',mask))
            if method!='omniovcd' or row['dataset']!='SECOND':
                compact={key:state[key] for key in state.files}
        if method!='omniovcd' or row['dataset']!='SECOND':
            np.savez_compressed(metadata['model_state'],**compact)
            del compact
        record = dict(source_id=row['id'],dataset=row['dataset'],method=method,generated_in_this_run=True,
            agent_semantics=True,task_adapted=False,slurm_job_id=os.environ['SLURM_JOB_ID'],
            task_keys=[task_key(t) for t in row['pilot_tasks']],predictions=paths,seconds=time.monotonic()-started,
            metadata=metadata,ground_truth_inputs=False,input_images={k:row[k] for k in ('image_t1','image_t2')},
            mask_areas=[int(np.asarray(m,bool).sum()) for m in masks])
        write(receipt,record); print('COMPLETE',method,row['id'],round(record['seconds'],1),record['mask_areas'],flush=True)
    if smoke: write(work/'smoke_qc.json',dict(status='passed',sources=[r['id'] for r in inputs],independent_composition='passed',ground_truth_inputs=False,method_policy=policies[method]))


def generation_qc(out):
    import numpy as np
    from verifier_v1_official_generate import task_key,pixels
    inputs = read(out/'generator_inputs.json'); counts = {}
    for method in METHODS:
        count = 0
        for row in inputs:
            receipt = read(out/'generation'/method/f'{row["id"]}.json')
            if receipt['ground_truth_inputs'] or receipt['task_keys']!=[task_key(t) for t in row['pilot_tasks']]: raise ValueError('Invalid generation receipt')
            if len(receipt['predictions'])!=len(row['pilot_tasks']): raise ValueError('Incomplete receipt')
            with np.load(receipt['metadata']['model_state'],allow_pickle=False) as state:
                for spec,basis,path in zip(row['pilot_tasks'],receipt['metadata']['task_basis'],receipt['predictions']):
                    if not np.array_equal(reconstruct(spec,basis,state),pixels(Path(path))>0): raise ValueError(f'Model reconstruction differs: {method}/{row["id"]}')
                    count += 1
        counts[method] = dict(sources=len(inputs),formula_passed=count,provenance_passed=len(inputs))
    report = dict(status='passed',sources=len(inputs),agent_semantics=True,methods=counts,
        tasks_per_method=sum(len(r['pilot_tasks']) for r in inputs),ground_truth_read=False,
        normal_records=sum(v['formula_passed'] for v in counts.values()),failures=[])
    write(out/'generation_qc.json',report); print('GENERATION_QC',json.dumps(report),flush=True)


PATH_FIELDS = {'image_t1','image_t2','init_mask','gt_mask','raw_task_gt','model_state','receipt',
               'generator_receipt','original_change_gt','semantic_t1','semantic_t2','dataset_path'}


def absolute_record(record,base):
    def convert(value,key=''):
        if isinstance(value,dict): return {k:convert(v,k) for k,v in value.items()}
        if isinstance(value,list): return [convert(v,key) for v in value]
        if isinstance(value,str) and (key in PATH_FIELDS or key in ('fp','fn','ignore_mask')):
            p=Path(value); return str(p if p.is_absolute() else base/p)
        return value
    return convert(copy.deepcopy(record))


def export(out):
    from verifier_v1_official_generate import export as export_labels
    from verifier_v1_pilot_generate import semantic_truth,transition_filter
    cfg=read(out/'config.json'); annotations={}
    for row in read(out/'generator_inputs.json'):
        transitions=[(i,t) for i,t in enumerate(row['pilot_tasks']) if t['task_type']=='transition']
        if transitions:
            cmap,l1,l2=semantic_truth(row['source_image_id'])
            for i,t in transitions:
                p=t['task_params']; okay,info=transition_filter(cmap,l1,l2,p['source_class'],p['target_class'],cfg)
                if not okay: raise ValueError('Frozen transition fails completeness')
                annotations[f'{row["id"]}:{i}']=info
    write(out/'frozen_task_annotations.json',annotations)
    export_labels(out)
    generated=rows(out/'manifest.jsonl'); write_rows(out/'new_manifest.jsonl',generated)
    approved=[absolute_record(r,PILOT) for r in rows(PILOT/'manifest.jsonl')]; combined=approved+generated
    sources={(r['dataset'],r['source_image_id']) for r in combined}
    expected={(r['dataset'],r['source_image_id']) for r in read(out/'sources.json') if r['split']=='train'}
    if sources!=expected or len(sources)!=600: raise ValueError('Manifest does not cover exactly 600 agreed pairs')
    if len({r['id'] for r in combined})!=len(combined): raise ValueError('Duplicate training record ID')
    for r in combined:
        for key in ('image_t1','image_t2','init_mask','gt_mask','raw_task_gt'):
            if not Path(r[key]).is_file(): raise FileNotFoundError(r[key])
        for p in r['labels'].values():
            if not Path(p).is_file(): raise FileNotFoundError(p)
    write_rows(out/'manifest.jsonl',combined)
    report=dict(status='complete',independent_source_pairs=600,validation_source_pairs=60,test_source_pairs=60,
        diagnostic_records=len(combined),approved_pilot_records=len(approved),new_records=len(generated),
        normal_records=sum(not r['is_counterfactual'] for r in combined),counterfactual_records=sum(r['is_counterfactual'] for r in combined),
        review_status_counts=dict(Counter(r['review_status'] for r in combined)),source_pairs_by_dataset=dict(Counter(d for d,_ in sources)),
        methods=list(METHODS),split_qc='passed',generation_qc='passed',new_label_qc='passed',
        approved_labels_reused_unchanged=True,val_test_generation=False,manifest=str(out/'manifest.jsonl'),approved_pilot=str(PILOT))
    write(out/'dataset_report.json',report); write(out/'summary.json',report)
    print('DATASET_COMPLETE',json.dumps(report,ensure_ascii=False),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('stage',choices=('prepare','infer','qc','export','review-check'))
    p.add_argument('--output',type=Path,required=True); p.add_argument('--method',choices=METHODS); p.add_argument('--smoke',action='store_true')
    args=p.parse_args()
    if not os.environ.get('SLURM_JOB_ID'): raise RuntimeError('Submit processing through Slurm')
    out=args.output.resolve()
    if args.stage=='prepare': prepare(out)
    elif args.stage=='infer':
        if args.method is None: p.error('--method is required')
        infer(out,args.method,args.smoke)
    elif args.stage=='qc': generation_qc(out)
    elif args.stage=='review-check':
        import subprocess,sys
        approve_pilot()
        subprocess.run([sys.executable,'-B',str(REPO/'tools/check_verifier_v1_review.py'),str(PILOT/'audit.html'),
                        '--summary',str(PILOT/'summary.html'),'--final-package','--expect-sources','28',
                        '--report',str(PILOT/'html_qc.json')],check=True)
    else: export(out)


if __name__=='__main__': main()
