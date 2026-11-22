"""Joint FP/FN-only pilot with immutable predictions and explicit loss-ignore masks."""
from __future__ import annotations
import argparse
from collections import Counter
import json
import os
from pathlib import Path
import random
import re
import sys
import time
import numpy as np
from PIL import Image

REPO=Path(__file__).resolve().parents[3]
ROOT=Path(os.environ.get('CD_AGENT_ACCOUNT_ROOT','/guisongxia01/pangchao/wangyihan'))
DATASETS=Path(os.environ.get('CD_AGENT_DATASETS',REPO/'datasets'))
RS=DATASETS
DATA=RS/'SECOND'
LEVIR=DATASETS/'LEVIR-CD'
ACTIVE=Path(os.environ.get('AGENT_RUN',REPO/'.runtime/agent_run'))
SEED=20261118
CLASSES={1:'low vegetation',2:'non-vegetated ground surface',3:'tree',4:'water',5:'building',6:'sports field'}
NAMES={1:'low vegetation',2:'non-vegetated land',3:'trees',4:'water bodies',5:'buildings',6:'sports fields'}
ZH={1:'低矮植被',2:'非植被地表',3:'树木',4:'水体',5:'建筑',6:'运动场'}
CLEAR=(3,4,5,6)
FORBIDDEN=re.compile(r'sam3|segearth|ovcd',re.I)
sys.path[:0]=[str(REPO/'src/cd_agent/_compat'),str(REPO/'src')]


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')


def read(path): return json.loads(path.read_text())


def image(path,rgb=False):
    with Image.open(path) as im: return np.array(im.convert('RGB') if rgb else im)


def save(path,values):
    path.parent.mkdir(parents=True,exist_ok=True)
    Image.fromarray(np.asarray(values,np.uint8)*255).save(path)
    return str(path.resolve())


def task(kind,c,a=None,b=None,index=0):
    if kind!='transition' and c not in CLEAR: raise ValueError('Forbidden independent target')
    if kind=='transition' and ({a,b}=={1,2} or not ({a,b}&set(CLEAR))): raise ValueError('Forbidden conversion')
    n=NAMES[c]; z=ZH[c]
    if kind=='related':
        en=[f'Find all changes involving {n}.',f'Mark changed regions related to {n}.',f'Identify every area where {n} changed between the two images.']
        cn=[f'找出所有与{z}相关的变化',f'标出涉及{z}的变化区域',f'识别两期影像中{z}发生变化的地方']
    elif kind=='into':
        en=[f'Find areas that became {n}.',f'Identify places where {n} appeared in the later image.',f'Mark land that was converted into {n}.']
        cn=[f'找出转入{z}的区域',f'识别后期新增的{z}',f'标出由其他地物变为{z}的地方']
    elif kind=='outof':
        en=[f'Find areas where {n} disappeared.',f'Mark places where {n} changed into other land cover.',f'Identify regions formerly occupied by {n} that have been transformed.']
        cn=[f'找出{z}消失的区域',f'标出{z}转为其他地物的地方',f'识别原有{z}发生转出的区域']
    else:
        en=[f'Find areas that changed from {NAMES[a]} to {NAMES[b]}.',f'Identify places where {NAMES[b]} replaced {NAMES[a]}.',f'Mark regions converted from {NAMES[a]} into {NAMES[b]}.']
        cn=[f'找出由{ZH[a]}转为{ZH[b]}的区域',f'识别{ZH[b]}替代{ZH[a]}的地方',f'标出从{ZH[a]}变为{ZH[b]}的区域']
    i=index%3
    return dict(task_type=kind,task_params=dict(class_id=c) if kind!='transition' else dict(source_class=a,target_class=b),prompt_en=en[i],prompt_zh=cn[i],prompt_template_id=f'{kind}_{i}',prompt_templates_en=en,prompt_templates_zh=cn)


def compile_target(cmap,l1,l2,spec):
    p=spec['task_params']; k=spec['task_type']
    if k=='transition': return cmap&(l1==p['source_class'])&(l2==p['target_class'])
    c=p['class_id']
    if k=='related': return cmap&((l1==c)|(l2==c))
    if k=='into': return cmap&(l1!=c)&(l2==c)
    return cmap&(l1==c)&(l2!=c)


def semantic_truth(name):
    return image(DATA/'label_unch0chi1'/name)>0,image(DATA/'slabel1'/name),image(DATA/'slabel2'/name)


def transition_filter(cmap,l1,l2,a,b,cfg):
    from scipy import ndimage as ndi
    if {a,b}=={1,2}: return False,dict(reason='forbidden_ambiguous_conversion')
    if b in CLEAR: proxy=cmap&(l2==b); phase='t2'; cls=b
    elif a in CLEAR: proxy=cmap&(l1==a); phase='t1'; cls=a
    else: return False,dict(reason='no_clear_entity_proxy')
    raw=cmap&(l1==a)&(l2==b)
    if raw.sum()<cfg['min_area']: return False,dict(reason='transition_too_small')
    labels,_=ndi.label(proxy,np.ones((3,3),bool)); sizes=np.bincount(labels.ravel())
    ids,counts=np.unique(labels[raw],return_counts=True)
    cover=[]; fragments=[]
    for ident,pixels in zip(ids,counts):
        if not ident: continue
        cover.append(float(pixels/sizes[ident]))
        y,x=np.nonzero(labels==ident); box=(slice(y.min(),y.max()+1),slice(x.min(),x.max()+1))
        _,n=ndi.label(raw[box]&(labels[box]==ident),np.ones((3,3),bool)); fragments.append(int(n))
    minimum=min(cover) if cover else 0.; maximum=max(fragments) if fragments else 0
    info=dict(proxy_phase=phase,proxy_class=cls,matched_entities=len(cover),transition_completeness=minimum,min_completeness=cfg['transition_coverage'],max_fragments=maximum,allowed_fragments=cfg['max_transition_fragments'])
    if minimum<cfg['transition_coverage']: return False,dict(info,reason='partial_entity_conversion')
    if maximum>cfg['max_transition_fragments']: return False,dict(info,reason='fragmented_entity_conversion')
    return True,dict(info,reason='accepted')


def source_splits(pool,inspected,selected):
    uninspected=[n for n in pool if n not in (set(inspected)|set(selected))]
    n=round(len(pool)*.1); val=uninspected[:n]; evaluate=uninspected[n:2*n]; held=set(val)|set(evaluate)
    return dict(fit=[x for x in pool if x not in held],val=val,offline_eval=evaluate,label_inspected_fit_ids=inspected,selected=selected)


def choose_whu_entities(names,cfg,skip=()):
    chosen=[]; inspected=[]; details=[]; skipped=set(skip)
    for name in names:
        if name in skipped: continue
        raw=image(RS/'WHU-CD/label_unch0chi1'/name)>0
        inspected.append(name); area=int(raw.sum())
        entry=dict(source_image_id=name,raw_target_area=area)
        if area<cfg.get('whu_min_raw_area',256):
            entry['reason']='raw_gt_below_minimum'; details.append(entry); continue
        anchor=clean_diagnostic(raw,raw,cfg)
        entry['anchor_effective_foreground_union']=anchor['metrics']['effective_foreground_union']
        if not entry['anchor_effective_foreground_union']:
            entry['reason']='anchor_has_no_effective_foreground'; details.append(entry); continue
        entry['reason']='selected'; details.append(entry); chosen.append(name)
        if len(chosen)==6: break
    return chosen,inspected,details


def repair_whu(out):
    inputs=read(out/'generator_inputs.json'); cfg=read(out/'config.json')
    allocation=read(out/'source_splits.json'); whu=allocation['datasets']['WHU-CD']
    old=[r for r in inputs if r['dataset']=='WHU-CD']; old_names=[r['source_image_id'] for r in old]
    if len(old)!=6: raise ValueError('WHU repair expects exactly six current sources')
    cfg['whu_min_raw_area']=256
    chosen,inspected,details=choose_whu_entities(whu['fit'],cfg,old_names)
    report=dict(status='selected' if len(chosen)==6 else 'insufficient_positive_fit_sources',
        selection_domain='frozen_WHU_fit_only',minimum_raw_gt_area=256,
        anchor_effective_foreground_required=True,superseded_source_ids=old_names,
        inspected_fit_ids=inspected,scan=details,new_selected=chosen,
        other_source_pairs_unchanged=True)
    write(out/'whu_repair.json',report)
    if len(chosen)!=6: raise RuntimeError('WHU fit did not provide six nonempty entity sources; see whu_repair.json')
    replacements=iter(chosen); updated=[]
    for row in inputs:
        if row['dataset']!='WHU-CD': updated.append(row); continue
        name=next(replacements); row=dict(row)
        row.update(id='whu_'+Path(name).stem,source_image_id=name,source_group_id='WHU-CD/train/'+name,
            image_t1=str(RS/'WHU-CD/img1'/name),image_t2=str(RS/'WHU-CD/img2'/name))
        updated.append(row)
    assert [r for r in inputs if r['dataset']!='WHU-CD']==[r for r in updated if r['dataset']!='WHU-CD']
    whu['selected']=chosen
    looked=set(whu.get('label_inspected_fit_ids',[]))|set(old_names)|set(inspected)
    whu['label_inspected_fit_ids']=[name for name in whu['fit'] if name in looked]
    whu['superseded_source_ids']=old_names
    assert set(chosen)<=set(whu['fit']) and not (set(chosen)&(set(whu['val'])|set(whu['offline_eval'])))
    audit=read(out/'split_audit.json'); audit['datasets']['WHU-CD'].update(
        selected=6,selected_raw_gt_nonempty=6,label_inspected_fit=len(whu['label_inspected_fit_ids']),
        superseded_source_ids=old_names,repair=dict(minimum_raw_gt_area=256,
            anchor_effective_foreground_required=True,new_selected=chosen,inspected_fit_ids=inspected,
            rejection_counts=dict(Counter(r['reason'] for r in details if r['reason']!='selected'))))
    write(out/'generator_inputs.json',updated); write(out/'source_splits.json',allocation)
    write(out/'split_audit.json',audit); write(out/'config.json',cfg)
    print(json.dumps(report,ensure_ascii=False),flush=True)


def inference_rows(out,dataset):
    rows=read(out/'generator_inputs.json')
    return [row for row in rows if dataset is None or row['dataset']==dataset]


def retained_generation(out,filename,dataset):
    if dataset is None: return []
    prefix={'WHU-CD':'whu_','LEVIR-CD':'levir_','SECOND':'second_'}[dataset]
    path=out/filename
    return [row for row in read(path) if not row['id'].startswith(prefix)] if path.exists() else []


def prepare(out,args):
    if (out/'generator_inputs.json').exists(): raise FileExistsError('Choose a new run')
    cfg=dict(seed=SEED,min_area=args.min_area,ignore_radius=args.ignore_radius,transition_coverage=args.transition_coverage,max_transition_fragments=args.max_transition_fragments,transition_proxy='8-connected clear-class C-domain entity in its corresponding date',class_texts=CLASSES,sam3=dict(pixel_threshold=.5,score_threshold=.5,confidence_threshold=.1,tile=512,overlap_threshold=.25),segearth=dict(vocabulary=['background',*CLASSES.values()],foreground_score_threshold=.5,score_semantics='Fixed-vocabulary normalized uncalibrated semantic scores'),anchor_count=4,max_counterfactual=24,whu_min_raw_area=256)
    inputs=[]; splits={}; audits={}; selected_meta=[]; trans_stats=Counter(); trans_details=[]
    def add(dataset,name,tasks,counters=()):
        slug={'LEVIR-CD':'levir','WHU-CD':'whu','SECOND':'second'}[dataset]
        root=LEVIR/'train_original' if dataset=='LEVIR-CD' else RS/dataset
        dirs=('A','B') if dataset=='LEVIR-CD' else ('img1','img2')
        inputs.append(dict(id=f'{slug}_{Path(name).stem}',dataset=dataset,dataset_slug=slug,source_image_id=name,source_group_id=f'{dataset}/train/{name}',image_t1=str(root/dirs[0]/name),image_t2=str(root/dirs[1]/name),normal_tasks=tasks,counter_tasks=list(counters)))
    for dataset,total,names_folder,exclude_paths in (
        ('LEVIR-CD',445,LEVIR/'train_original/A',[ACTIVE/'dev2'/f'{p}.txt' for p in ('names','screen','screen30','confirm170')]),
        ('WHU-CD',1260,RS/'WHU-CD/img1',[ACTIVE/'whu/names_dev60.txt',ACTIVE/'whu/names_val128.txt']),
    ):
        names=sorted(p.name for p in names_folder.glob('*.png') if dataset=='LEVIR-CD' or p.name.startswith('train_'))
        if len(names)!=total: raise ValueError(f'{dataset} expected train {total}')
        excluded=set().union(*(p.read_text().split() for p in exclude_paths)); extra=set(); manifest_paths=[]
        if dataset=='WHU-CD':
            for folder,prefix in (('t54','T5.4'),('t54n','T5.4n')):
                for split in ('dev','val'):
                    p=ACTIVE/folder/'tasks'/f'{prefix}_{split}.jsonl'
                    if p.exists():
                        rows=[json.loads(line) for line in p.read_text().splitlines() if line.strip()]; ids={Path(row['image_t2']).name for row in rows}
                        manifest_paths.append(dict(path=str(p),task_count=len(rows),source_id_count=len(ids),additional_to_known_exclusion=len(ids-excluded))); extra.update(ids)
        extra_new=extra-excluded; excluded|=extra
        pool=sorted(set(names)-excluded); random.Random(SEED).shuffle(pool)
        if dataset=='WHU-CD':
            chosen,inspected,positive_scan=choose_whu_entities(pool,cfg)
            if len(chosen)!=6: raise ValueError('WHU train pool lacks six usable positive entities')
        else:
            chosen,inspected,positive_scan=pool[:6],[],[]
        splits[dataset]=source_splits(pool,inspected,chosen)
        audits[dataset]=dict(train_total=total,excluded_known=len(set(names)&excluded),additional_active_t5=len(extra_new&set(names)),eligible=len(pool),selected=6,exclusion_files=list(map(str,exclude_paths)),active_t5_manifests=manifest_paths,user_local_train_admission=dataset=='WHU-CD',label_inspected_fit=len(inspected),positive_entity_scan=positive_scan)
        for i,name in enumerate(chosen): add(dataset,name,[task('related',5,index=i)])
    exclusion=REPO/'doc/second_active_dev_ids.txt'; excluded=set(exclusion.read_text().split())
    names=sorted(p.name for p in (DATA/'img1').glob('train_*.png'))
    if len(names)!=2968: raise ValueError('Expected SECOND train2968')
    pool=sorted(set(names)-excluded); random.Random(SEED).shuffle(pool); inspected=pool[:384]; candidates=[]; anomaly=Counter()
    for name in inspected:
        cmap,l1,l2=semantic_truth(name)
        bad=int((~np.isin(l1,range(7))|~np.isin(l2,range(7))).sum()); one=int(((l1==0)^(l2==0)).sum())
        anomaly['illegal_class_pixels']+=bad; anomaly['one_sided_zero_pixels']+=one
        if bad or one: continue
        hist=np.bincount((l1[cmap].astype(int)*7+l2[cmap]).ravel(),minlength=49).reshape(7,7)
        options={}
        for c in CLEAR:
            related=int(hist[c,:].sum()+hist[:,c].sum()-hist[c,c]); into=int(hist[:,c].sum()-hist[c,c]); outof=int(hist[c,:].sum()-hist[c,c])
            if related<cfg['min_area'] or not (into or outof): continue
            accepted=[]
            for a,b in zip(*np.nonzero(hist>=cfg['min_area'])):
                a,b=int(a),int(b)
                if a==b or not a or not b or c not in (a,b): continue
                ok,info=transition_filter(cmap,l1,l2,a,b,cfg); trans_stats[info['reason']]+=1
                if ok: accepted.append(dict(a=a,b=b,quality=info,pixels=int(hist[a,b])))
            accepted.sort(key=lambda t:-t['pixels'])
            options[c]=dict(related=related,into=into,outof=outof,transitions=accepted)
        if options: candidates.append(dict(name=name,options=options))
    chosen=[]; used=set()
    for repeat in range(4):
        for c in CLEAR:
            available=[r for r in candidates if r['name'] not in used and c in r['options']]
            available.sort(key=lambda r:(not bool(r['options'][c]['transitions']),not bool(r['options'][c]['into'] and r['options'][c]['outof'])))
            if not available: raise ValueError(f'Insufficient clear SECOND class {c} samples')
            row=available[0]; used.add(row['name']); chosen.append((row,c))
    for i,(row,c) in enumerate(chosen):
        options=row['options'][c]; counters=[]
        for j,t in enumerate(options['transitions']):
            q=task('transition',c,t['a'],t['b'],i+j); counters.append(q)
            trans_details.append(dict(source=row['name'],source_class=t['a'],target_class=t['b'],**t['quality']))
        add('SECOND',row['name'],[task(k,c,index=i) for k in ('related','into','outof')],counters)
        selected_meta.append(dict(name=row['name'],class_id=c,related_pixels=options['related'],into_pixels=options['into'],outof_pixels=options['outof'],accepted_transition_tasks=len(counters)))
    splits['SECOND']=source_splits(pool,inspected,[r['name'] for r,c in chosen])
    audits['SECOND']=dict(train_total=2968,excluded_known=len(set(names)&excluded),eligible=len(pool),selected=16,exclusion_files=[str(exclusion)],label_inspected_fit=len(inspected),observed_label_anomalies=dict(anomaly))
    write(out/'config.json',cfg); write(out/'generator_inputs.json',inputs)
    write(out/'source_splits.json',dict(seed=SEED,datasets=splits))
    write(out/'sampling.json',dict(second_selected=selected_meta,transition_filter=dict(trans_stats),accepted_transition_details=trans_details))
    write(out/'split_audit.json',dict(status='pending_review',datasets=audits,independent_source_pairs=len(inputs),official_val_test_used=False,all_descendants_grouped=True,second_only_canonical_exclusion=True,historical_scans=False))


def generate_sam3(out,dataset=None):
    # Reuse installed SciPy without downloading or modifying the SAM3 environment.
    # Append after native packages so its NumPy/PyTorch remain authoritative.
    sys.path.append(str(ROOT/'conda_envs/qwen3vl-sft-env/lib/python3.12/site-packages'))
    sys.path.insert(0,str(ROOT/'Chang-Agent/third_party/OmniOVCD'))
    import torch
    import sam3_tools
    from cd_agent.workers.sam3_worker import Segmenter
    from cd_agent.tools import Toolbox
    from cd_agent.state import Mask
    from cd_agent import instances
    torch.manual_seed(SEED)
    worker=Segmenter(sam3_tools.build_processor(),sam3_tools,out/'fresh_sam3')
    records=retained_generation(out,'sam3_generation.json',dataset)
    for row in inference_rows(out,dataset):
        start=time.monotonic(); c=row['normal_tasks'][0]['task_params']['class_id']; calls=[]
        def segment(time,text,tile,score_thr):
            p=row['image_t1' if time=='t1' else 'image_t2']
            result=worker.infer(row['dataset_slug']+'_train',row['id']+'.png',time,text,tile,conf=.1,pixel_thr=.5,image_path=p)
            if result['cache_hit']: raise RuntimeError('Fresh SAM3 namespace unexpectedly hit cache')
            with np.load(result['path']) as data:
                h,w=map(int,data['shape']); keep=data['scores']>=score_thr
                dense=np.unpackbits(data['masks'][keep],axis=1,count=h*w).reshape(-1,h,w).astype(bool)
                values=instances.from_dense(dense,data['scores'][keep])
            calls.append(dict(time=time,**result)); return values,dict(backend='fresh_sam3',cache_hit=False)
        rgb=[image(row[k],True) for k in ('image_t1','image_t2')]
        box=Toolbox(rgb,out/'generator_tools'/row['id'],segment=segment)
        def execute(name,**args):
            result=box.execute(name,args)
            if not result.public['ok']:
                print(result.internal.get('traceback',''),file=sys.stderr); raise RuntimeError(result.text)
            return result.public
        t1=execute('segment_text',time='t1',text=CLASSES[c],tile=512,score_thr=.5)['inst']
        t2=execute('segment_text',time='t2',text=CLASSES[c],tile=512,score_thr=.5)['inst']
        matches=execute('match_instances',inst_t1=t1,inst_t2=t2,overlap_thr=.25)
        for slot,spec in enumerate(row['normal_tasks']):
            sets=[matches['appeared'],matches['disappeared']] if spec['task_type']=='related' else [matches['appeared' if spec['task_type']=='into' else 'disappeared']]
            key=execute('to_mask',sets=sets)['mask']; prediction=box.session.get(key,Mask).data
            save(out/'predictions/sam3_imc'/f'{row["id"]}_{slot}.png',prediction)
        records.append(dict(id=row['id'],dataset=row['dataset'],seconds=time.monotonic()-start,calls=calls,generated_fresh=True,class_text=CLASSES[c]))
        write(out/'sam3_generation.json',records); print('sam3',row['id'],f'{records[-1]["seconds"]:.1f}s',flush=True)


def generate_segearth(out,dataset=None):
    import torch
    from cd_agent.t13_worker import DynamicBackend
    cfg=read(REPO/'src/cd_agent/vision_t13.example.json')['dynamic']; cache=out/'fresh_segearth'; cache.mkdir(parents=True,exist_ok=True)
    torch.manual_seed(SEED); backend=DynamicBackend(cfg,cache); records=retained_generation(out,'segearth_generation.json',dataset)
    for row in inference_rows(out,dataset):
        start=time.monotonic(); scores=[]
        for ti,k in enumerate(('image_t1','image_t2'),1):
            torch.manual_seed(SEED)
            with torch.inference_mode(): values=backend.semantic(image(row[k],True),list(CLASSES.values()))
            np.save(cache/f'{row["id"]}_t{ti}_scores.npy',values); scores.append(values)
        c=row['normal_tasks'][0]['task_params']['class_id']; p1=scores[0][c]>.5; p2=scores[1][c]>.5
        for slot,spec in enumerate(row['normal_tasks']):
            prediction=p1^p2 if spec['task_type']=='related' else ~p1&p2 if spec['task_type']=='into' else p1&~p2
            save(out/'predictions/segearth_dense'/f'{row["id"]}_{slot}.png',prediction)
        records.append(dict(id=row['id'],dataset=row['dataset'],seconds=time.monotonic()-start,generated_fresh=True,generator_config=cfg))
        write(out/'segearth_generation.json',records); print('segearth',row['id'],f'{records[-1]["seconds"]:.1f}s',flush=True)



def tiny_components(mask,min_area):
    from scipy import ndimage as ndi
    labels,n=ndi.label(mask,np.ones((3,3),bool)); sizes=np.bincount(labels.ravel(),minlength=n+1)
    small=np.flatnonzero((sizes<min_area)&(np.arange(len(sizes))>0))
    return np.isin(labels,small),len(small)


def clean_diagnostic(pred,raw,cfg):
    from scipy import ndimage as ndi
    structure=np.ones((3,3),bool)
    small_gt,ng=tiny_components(raw,cfg['min_area']); clean=raw&~small_gt
    raw_fp=pred&~raw; raw_fn=~pred&raw
    small_fp,nfp=tiny_components(raw_fp,cfg['min_area']); small_fn,nfn=tiny_components(raw_fn,cfg['min_area'])
    tiny=small_gt|small_fp|small_fn
    pl,_=ndi.label(pred,structure); gl,_=ndi.label(clean,structure)
    ip=np.unique(pl[pred&clean]); ip=ip[ip>0]; ig=np.unique(gl[pred&clean]); ig=ig[ig>0]
    matched_p=pred&np.isin(pl,ip); matched_g=clean&np.isin(gl,ig)
    independent_p=pred&~matched_p; independent_g=clean&~matched_g
    protected=(independent_p|independent_g)&~tiny
    boundary=(matched_p^ndi.binary_erosion(matched_p,structure=structure,border_value=0))|(matched_g^ndi.binary_erosion(matched_g,structure=structure,border_value=0))
    tolerance=ndi.distance_transform_edt(~boundary)<=cfg['ignore_radius'] if boundary.any() and cfg['ignore_radius']>0 else np.zeros_like(raw)
    tolerance&=~protected
    ignore=tiny|tolerance; valid=~ignore
    fp=pred&~clean&valid; fn=~pred&clean&valid
    raw_union=int((pred|raw).sum()); effective_union=int(((pred|clean)&valid).sum())
    effective_iou=float((pred&clean&valid).sum()/effective_union) if effective_union else None
    stats=dict(min_area=cfg['min_area'],ignore_radius=cfg['ignore_radius'],removed_gt_components=ng,removed_gt_pixels=int(small_gt.sum()),tiny_raw_fp_components=nfp,tiny_raw_fn_components=nfn,
        raw_fp_area=int(raw_fp.sum()),raw_fn_area=int(raw_fn.sum()),tiny_ignored_fp_area=int((raw_fp&tiny).sum()),tiny_ignored_fn_area=int((raw_fn&tiny).sum()),boundary_ignored_fp_area=int((raw_fp&~tiny&tolerance).sum()),boundary_ignored_fn_area=int((raw_fn&~tiny&tolerance).sum()),
        matched_tolerance_pixels=int((tolerance&~tiny).sum()),ignored_pixels=int(ignore.sum()),ignore_fraction=float(ignore.mean()),ignored_residual_fraction=float(((raw_fp|raw_fn)&ignore).sum()/max(1,(raw_fp|raw_fn).sum())),
        independent_fp_preserved_pixels=int((independent_p&raw_fp&~tiny).sum()),independent_fn_preserved_pixels=int((independent_g&raw_fn&~tiny).sum()),matched_prediction_components=len(ip),matched_gt_components=len(ig))
    metrics=dict(raw_iou=float((pred&raw).sum()/raw_union) if raw_union else 1.,effective_iou=effective_iou,iou=effective_iou,raw_foreground_union=raw_union,effective_foreground_union=effective_union,
        fp_area=int(fp.sum()),fn_area=int(fn.sum()),mask_area=int(pred.sum()),target_area=int(clean.sum()),raw_target_area=int(raw.sum()),valid_area=int(valid.sum()),ignore_area=int(ignore.sum()),ignore_fraction=float(ignore.mean()),
        raw_fp_area=stats['raw_fp_area'],raw_fn_area=stats['raw_fn_area'],tiny_ignored_fp_area=stats['tiny_ignored_fp_area'],tiny_ignored_fn_area=stats['tiny_ignored_fn_area'],boundary_ignored_fp_area=stats['boundary_ignored_fp_area'],boundary_ignored_fn_area=stats['boundary_ignored_fn_area'],empty_effective_foreground=effective_union==0)
    return dict(clean=clean,fp=fp,fn=fn,ignore=ignore,protected=protected,metrics=metrics,cleaning=stats)


def raw_target(row,spec):
    name=row['source_image_id']
    if row['dataset']=='SECOND': return compile_target(*semantic_truth(name),spec)
    if row['dataset']=='LEVIR-CD': return image(LEVIR/'train_original/label'/name)>127
    return image(RS/'WHU-CD/label_unch0chi1'/name)>0


def quality(iou):
    if iou is None: return 'no_effective_foreground'
    return '[0,.3)' if iou<.3 else '[.3,.7)' if iou<.7 else '[.7,.9)' if iou<.9 else '[.9,.95)' if iou<.95 else '[.95,1]'


def export(out):
    cfg=read(out/'config.json'); inputs=read(out/'generator_inputs.json'); allocation=read(out/'source_splits.json')['datasets']; checks=Counter(); records=[]; base_records={}; cf_stats=Counter(); chosen_transitions=[]
    def record(row,gen,slot,spec,marker,base=None,prepared=None,transition_info=None):
        raw=raw_target(row,spec); path=out/'predictions'/gen/f'{row["id"]}_{slot}.png'
        if gen=='gt_anchor': save(path,raw)
        pred=image(path)>0; d=prepared or clean_diagnostic(pred,raw,cfg); valid=~d['ignore']; ident=f'{row["id"]}_{gen}_{spec["task_type"]}_{marker}'; folder=out/'cases'/ident
        labels={k:save(folder/f'{k}.png',d[field]) for k,field in [('fp','fp'),('fn','fn'),('ignore_mask','ignore')]}
        raw_path=save(folder/'raw_task_gt.png',raw); clean_path=save(folder/'gt_clean.png',d['clean'])
        cleaning=dict(d['cleaning'])
        if transition_info: cleaning.update(transition_info)
        metrics=d['metrics']; natural=gen!='gt_anchor' and metrics['effective_foreground_union']>0 and metrics['iou'] is not None and metrics['iou']>=.9
        tags=[]
        if metrics['fp_area']: tags.append('false_positive')
        if metrics['fn_area']: tags.append('false_negative')
        if cleaning['removed_gt_components']: tags.append('tiny_gt_ignored')
        if cleaning['independent_fp_preserved_pixels']: tags.append('independent_false_positive')
        if cleaning['independent_fn_preserved_pixels']: tags.append('independent_false_negative')
        if natural: tags.append('natural_high_score')
        if gen=='gt_anchor': tags.append('gt_anchor')
        if metrics['empty_effective_foreground']: tags.append('no_effective_foreground')
        item=dict(id=ident,dataset=row['dataset'],source_group_id=row['source_group_id'],source_image_id=row['source_image_id'],official_split='train',internal_split='fit',review_status='pending_review',image_t1=row['image_t1'],image_t2=row['image_t2'],**spec,
            generator=gen,init_mask=str(path.resolve()),gt_mask=clean_path,raw_task_gt=raw_path,labels=labels,metrics=metrics,cleaning=cleaning,quality_band=quality(metrics['iou']),error_tags=tags,is_counterfactual=base is not None,base_task_id=base['id'] if base else None,
            counterfactual_group_id=base['counterfactual_group_id'] if base else f'{row["id"]}_{gen}_{slot}',
            source=dict(dataset_path=str(LEVIR/'train_original' if row['dataset']=='LEVIR-CD' else RS/row['dataset']),original_change_gt=str(LEVIR/'train_original/label'/row['source_image_id']) if row['dataset']=='LEVIR-CD' else str(RS/row['dataset']/'label_unch0chi1'/row['source_image_id']),generator_config=cfg['sam3' if gen=='sam3_imc' else 'segearth'] if gen!='gt_anchor' else dict(source='unaltered raw task truth'),gt_lineage='Raw task truth retained; small task components removed only in separate cleaned target and covered by ignore'),
            isolation=dict(all_descendants_grouped=True,known_actual_exclusions_applied=True,source_unit='native_complete_pair'))
        invariant=dict(fp_inside_m=not (d['fp']&~pred).any(),fn_outside_m=not (d['fn']&pred).any(),fp_fn_disjoint=not (d['fp']&d['fn']).any(),ignore_excluded=not ((d['fp']|d['fn'])&d['ignore']).any(),
            fp_formula=np.array_equal(d['fp'],pred&~d['clean']&valid),fn_formula=np.array_equal(d['fn'],~pred&d['clean']&valid),valid_reconstruction=np.array_equal(((pred&~d['fp'])|d['fn'])[valid],d['clean'][valid]),
            raw_gt_preserved=np.array_equal(raw,image(raw_path)>0),candidate_unchanged=np.array_equal(pred,image(path)>0),removed_gt_ignored=not ((raw&~d['clean'])&valid).any(),independent_bodies_protected=not (d['protected']&d['ignore']).any(),
            fp_funnel=cleaning['raw_fp_area']-cleaning['tiny_ignored_fp_area']-cleaning['boundary_ignored_fp_area']==metrics['fp_area'],fn_funnel=cleaning['raw_fn_area']-cleaning['tiny_ignored_fn_area']-cleaning['boundary_ignored_fn_area']==metrics['fn_area'],
            clean_prompt=not any(FORBIDDEN.search(t) for t in [spec['prompt_en'],spec['prompt_zh'],*spec['prompt_templates_en'],*spec['prompt_templates_zh']]),three_templates=len(set(spec['prompt_templates_en']))>=3 and len(set(spec['prompt_templates_zh']))>=3,
            allowed_task=spec['task_type']=='transition' or spec['task_params']['class_id'] in CLEAR,
            gt_anchor_zero=gen!='gt_anchor' or not (d['fp']|d['fn']).any(),source_in_fit=row['source_image_id'] in allocation[row['dataset']]['fit'])
        if base:
            base_valid=~(image(base['labels']['ignore_mask'])>0); base_gt=image(base['gt_mask'])>0
            invariant.update(cf_nonempty_m=bool(pred.any()),cf_effective_difference=bool(((base_gt^d['clean'])&base_valid&valid).any()),cf_same_pixels=all(np.array_equal(image(base[k]),image(item[k])) for k in ('image_t1','image_t2','init_mask')))
        if spec['task_type']=='transition': invariant['complete_transition']=transition_info is not None and transition_info['transition_completeness']>=cfg['transition_coverage'] and transition_info['max_fragments']<=cfg['max_transition_fragments']
        for k,passed in invariant.items(): checks[k+('_passed' if passed else '_failed')]+=1
        records.append(item); return item
    for row in inputs:
        for gen in ('sam3_imc','segearth_dense'):
            for slot,spec in enumerate(row['normal_tasks']):
                item=record(row,gen,slot,spec,str(slot)); base_records[(row['id'],gen,slot)]=item
    second=[r for r in inputs if r['dataset']=='SECOND']; accepted=0
    for round_id in (0,1):
        for ix,row in enumerate(second):
            if accepted>=cfg['max_counterfactual']: break
            gen=('sam3_imc','segearth_dense')[(ix+round_id)%2]; base=base_records[(row['id'],gen,0)]; pred=image(base['init_mask'])>0
            if not pred.any(): cf_stats['empty_m_rejected']+=1; continue
            cmap,l1,l2=semantic_truth(row['source_image_id']); options=[*row['counter_tasks'],*row['normal_tasks'][1:]]
            for j,spec in enumerate(options):
                info=None
                if spec['task_type']=='transition':
                    a,b=spec['task_params']['source_class'],spec['task_params']['target_class']; ok,info=transition_filter(cmap,l1,l2,a,b,cfg)
                    if not ok: cf_stats['transition_rejected_at_export']+=1; continue
                raw=compile_target(cmap,l1,l2,spec); d=clean_diagnostic(pred,raw,cfg)
                difference=((image(base['gt_mask'])>0)^d['clean'])&~(image(base['labels']['ignore_mask'])>0)&~d['ignore']
                if not difference.any(): cf_stats['no_effective_gt_difference']+=1; continue
                record(row,gen,0,spec,f'cf{round_id}_{j}',base,d,info); accepted+=1; cf_stats['accepted']+=1
                if info: chosen_transitions.append(dict(source=row['source_image_id'],**spec['task_params'],**info))
                break
    anchors=[next(r for r in inputs if r['dataset']==d) for d in ('LEVIR-CD','WHU-CD')]+second[:2]
    for ix,row in enumerate(anchors): record(row,'gt_anchor',0,row['normal_tasks'][0],f'anchor{ix}')
    failures=sum(v for k,v in checks.items() if k.endswith('_failed'))
    (out/'manifest.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in records))
    qc=dict(status='passed' if not failures else 'failed',records=len(records),invariants=dict(checks),failures=failures,fp_fn_only=True,cleaning_config={k:cfg[k] for k in ('min_area','ignore_radius','transition_coverage','max_transition_fragments')},counterfactual=dict(cf_stats),transition_records=chosen_transitions,automatic_only=True)
    write(out/'label_qc.json',qc)
    funnel_keys=('raw_fp_area','raw_fn_area','tiny_ignored_fp_area','tiny_ignored_fn_area','boundary_ignored_fp_area','boundary_ignored_fn_area','fp_area','fn_area')
    summary=dict(status='pending_review',independent_source_pairs=len(inputs),diagnostic_records=len(records),datasets=dict(Counter(r['dataset'] for r in records)),sources_by_dataset=dict(Counter(r['dataset'] for r in inputs)),generators=dict(Counter(r['generator'] for r in records)),tasks=dict(Counter(r['task_type'] for r in records)),quality_bands=dict(Counter(r['quality_band'] for r in records)),coverage=dict(Counter(t for r in records for t in r['error_tags'])),qc_status=qc['status'],invariant_failures=failures,counterfactual=dict(cf_stats),natural_high_score_records=sum('natural_high_score' in r['error_tags'] for r in records),empty_effective_foreground_records=sum(r['metrics']['empty_effective_foreground'] for r in records),
        pixel_funnel={k:sum(r['metrics'][k] for r in records) for k in funnel_keys},mean_ignore_fraction=float(np.mean([r['metrics']['ignore_fraction'] for r in records])),
        transition_filter=read(out/'sampling.json')['transition_filter'],selected_transition_records=len(chosen_transitions),generation_seconds={g:sum(v['seconds'] for v in read(out/f'{g}_generation.json')) for g in ('sam3','segearth')})
    write(out/'summary.json',summary); print(json.dumps(summary,ensure_ascii=False),flush=True)
    if failures: raise RuntimeError('FP/FN label QC failed')


def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('stage',choices=['prepare','repair_whu','sam3','segearth','export']); p.add_argument('--output',type=Path,required=True); p.add_argument('--count',type=int,default=28)
    p.add_argument('--dataset',choices=['LEVIR-CD','WHU-CD','SECOND']); p.add_argument('--min-area',type=int,default=16); p.add_argument('--ignore-radius',type=int,default=2); p.add_argument('--transition-coverage',type=float,default=.95); p.add_argument('--max-transition-fragments',type=int,default=1)
    args=p.parse_args()
    if not os.environ.get('SLURM_JOB_ID'): raise RuntimeError('Slurm only')
    if args.count!=28: raise ValueError('Joint pilot is fixed at 6+6+16 source pairs')
    args.output.mkdir(parents=True,exist_ok=True)
    {'prepare':lambda:prepare(args.output,args),'repair_whu':lambda:repair_whu(args.output),'sam3':lambda:generate_sam3(args.output,args.dataset),'segearth':lambda:generate_segearth(args.output,args.dataset),'export':lambda:export(args.output)}[args.stage]()

if __name__=='__main__': main()
