"""Thin, offline adapters around official OVCD inference implementations.

GT is deliberately absent from this module's interface. Evaluation-framework
containers are replaced for OmniOVCD; its model and inference bodies are intact.
"""
from __future__ import annotations
import ast
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import numpy as np
from PIL import Image

import os
ROOT = Path(os.environ.get('CD_AGENT_ACCOUNT_ROOT', '/guisongxia01/pangchao/wangyihan'))
REPOSITORY = Path(__file__).resolve().parents[3]
REFERENCE = Path(os.environ.get('CD_AGENT_THIRD_PARTY', REPOSITORY / 'third_party'))
MODEL_ROOT = Path(os.environ.get('CD_AGENT_MODELS', REPOSITORY / 'models'))
CHECKPOINT = MODEL_ROOT / 'sam3/sam3.pt'
MCI_WEIGHTS = {
    'sam': MODEL_ROOT / 'SAM/sam_vit_h_4b8939.pth',
    'dinov2': MODEL_ROOT / 'dinov2-base',
    'clip': MODEL_ROOT / 'CLIP/ViT-B-16.pt',
    'simfeatup': MODEL_ROOT / 'SimFeatUp/xclip_jbu_one_million_aid.ckpt',
}
SCIENCE = ROOT / 'conda_envs/qwen3vl-sft-env/lib/python3.12/site-packages'
CANONICAL = {1:'low vegetation',2:'non-vegetated ground surface',3:'tree',4:'water',5:'building',6:'sports field'}


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def required_classes(tasks):
    result = set()
    for task in tasks:
        p = task['task_params']
        result.update([p['source_class'],p['target_class']] if task['task_type']=='transition' else [p['class_id']])
    return sorted(result)


def binary_task(spec, branches):
    """Use official per-class change candidates and its two semantic branches."""
    p=spec['task_params']; kind=spec['task_type']
    if kind=='transition':
        a,b=p['source_class'],p['target_class']
        return (branches[a]['change']|branches[b]['change'])&branches[a]['t1']&branches[b]['t2']
    c=p['class_id']; v=branches[c]
    if kind=='related': return v['change']
    if kind=='into': return v['change']&~v['t1']&v['t2']
    return v['change']&v['t1']&~v['t2']


class FrameworkSample:
    def __init__(self, **meta): self.metainfo=meta
    def set_metainfo(self, values): self.metainfo.update(values)
    def set_data(self, values): self.__dict__.update(values)


def load_official_omni(path):
    """Remove only evaluation-framework imports/decorator, keeping all bodies.

    BaseSegmentor is an nn.Module here, PixelData holds tensors, and skimage's
    binary 8-connectivity label call is supplied by equivalent SciPy CC labeling.
    No neural operations, fusion, matching or thresholds are replaced.
    """
    import torch
    from scipy import ndimage
    tree=ast.parse(path.read_text())
    keep=[]
    skipped=('mmseg','mmengine','scipy','skimage')
    for node in tree.body:
        if isinstance(node,ast.ImportFrom) and (node.module or '').split('.')[0] in skipped:
            continue
        if isinstance(node,ast.ClassDef): node.decorator_list=[]
        keep.append(node)
    tree.body=keep
    def label(mask, connectivity=2, background=0):
        assert connectivity==2 and background==0
        return ndimage.label(np.asarray(mask!=0,bool),structure=np.ones((3,3),bool))[0]
    scope=dict(__file__=str(path),__name__='official_omni_pilot',BaseSegmentor=torch.nn.Module,
               PixelData=lambda **kw:SimpleNamespace(**kw),measure=SimpleNamespace(label=label))
    exec(compile(tree,str(path),'exec'),scope)
    return scope


class OmniAdapter:
    def __init__(self, work):
        import torch
        self.work=work; self.repo=REFERENCE/'OmniOVCD'
        sys.path.insert(0,str(self.repo))
        sys.path.append(str(SCIENCE))
        scope=load_official_omni(self.repo/'OmniOVCD_seg.py')
        build=scope['build_sam3_image_model']
        def local_build(**kw):
            kw.update(checkpoint_path=str(CHECKPOINT),bpe_path=str(self.repo/'sam3/assets/bpe_simple_vocab_16e6.txt.gz'),load_from_HF=False)
            model=build(**kw); model._device=torch.device('cuda'); return model
        scope['build_sam3_image_model']=local_build
        cfg=self.config('LEVIR-CD',5)
        self.model=scope['OmniOVCDSeg'](**cfg).eval()

    def config(self,dataset,cls):
        files={1:'low-vegetation',2:'surface',3:'tree',4:'water',5:'building',6:'playground'}
        suffix='levir_cd' if dataset=='LEVIR-CD' else 'whu_cd' if dataset=='WHU-CD' else 'second_'+files[cls]
        source=self.repo/'configs'/f'cfg_{suffix}.py'
        scope={}; exec(compile(source.read_text(),str(source),'exec'),scope)
        cfg=scope['model'].copy(); cfg['classname_path']=str(self.repo/cfg['classname_path'])
        cfg['enable_vis']=False
        return cfg

    def infer(self,row):
        if row.get("agent_semantics"): return self.infer_agent(row)
        import torch
        branches={}; configurations={}; model=self.model
        shape=np.asarray(Image.open(row['image_t1'])).shape[:2]
        tile=256 if row['dataset'] in ('LEVIR-CD','WHU-CD') else max(shape)
        for c in required_classes(row['pilot_tasks']):
            cfg=self.config(row['dataset'],c); configurations[c]=cfg
            words=[]; indices=[]
            for index,line in enumerate(Path(cfg['classname_path']).read_text().splitlines()):
                for word in line.split(','): words.append(word.strip()); indices.append(index)
            for key,value in cfg.items():
                if key!='classname_path': setattr(model,key,value)
            model.processor.confidence_threshold=cfg['confidence_threshold']
            model.query_words=words; model.query_idx=torch.tensor(indices,dtype=torch.long,device='cuda')
            model.num_cls=max(indices)+1; model.num_queries=len(words)
            output={k:np.zeros(shape,bool) for k in ('t1','t2','change')}
            for y in range(0,shape[0],tile):
                for x in range(0,shape[1],tile):
                    y2=min(y+tile,shape[0]); x2=min(x+tile,shape[1]); paths=[]
                    for phase in (1,2):
                        path=self.work/'tiles'/f'{row["id"]}_{y}_{x}_t{phase}.png'
                        if not path.exists():
                            path.parent.mkdir(parents=True,exist_ok=True)
                            with Image.open(row[f'image_t{phase}']) as im: im.crop((x,y,x2,y2)).save(path)
                        paths.append(str(path))
                    sample=FrameworkSample(img1_path=paths[0],img2_path=paths[1],ori_shape=(y2-y,x2-x))
                    with torch.inference_mode(): result=model.predict(None,[sample])[0]
                    for phase in (1,2):
                        logits=getattr(result,f'seg{phase}_logits').data
                        pred=logits.argmax(0); pred[logits.max(0)[0]<model.prob_thd]=model.bg_idx
                        output[f't{phase}'][y:y2,x:x2]=(pred==1).cpu().numpy()
                    output['change'][y:y2,x:x2]=result.pred_sem_seg.data[0].cpu().numpy()>0
            branches[c]=output
        return [binary_task(t,branches) for t in row['pilot_tasks']],dict(
            official_repository='https://github.com/Erxucomeon/OmniOVCD',configs=configurations,
            input_adapter=dict(tile_size=tile,nonoverlapping_native_crops=True,stitch='original coordinates'),
            compatibility='Only mmseg evaluation containers/decorator and equivalent SciPy binary CC; official inference bodies unchanged',
            task_adapter='Official category-specific change AND directional semantic branches; transition uses union of endpoint changes and T1 source/T2 target')


class ReAAdapter:
    def __init__(self,work):
        self.work=work; self.repo=REFERENCE/'ReA-OVCD'
        sys.path.insert(0,str(self.repo)); sys.path.append(str(SCIENCE))
        self.segmentor=None; self.dataset=None

    def infer(self,row):
        if row.get("agent_semantics"): return self.infer_agent(row)
        import torch
        from ovcd_segmentor import OVCDSegmentor
        from ovcd_pipeline import OpenVocabularyChangeDetector
        from ovcd_postprocess import BoundaryDistanceChangeFilter
        from configs.ovcd_configs import get_config
        name={'LEVIR-CD':'levircd','WHU-CD':'whucd','SECOND':'second_raw'}[row['dataset']]
        cfg=get_config(name)
        if self.dataset!=name:
            if self.segmentor is not None:
                del self.segmentor; torch.cuda.empty_cache()
            self.segmentor=OVCDSegmentor(class_names=cfg['class_names'],device='cuda',
                confidence_threshold=cfg['confidence_threshold'],prob_thd=cfg['prob_thd'],
                use_sem_seg=True,use_presence_score=cfg.get('use_presence',True),
                sam3_checkpoint=str(CHECKPOINT),bpe_path=str(self.repo/'sam3/assets/bpe_simple_vocab_16e6.txt.gz')).eval()
            self.segmentor.processor.model._device=torch.device('cuda')
            self.dataset=name
        pp=cfg['postprocess'].copy()
        if row['dataset']=='SECOND':
            pp.update(boundary_distance_tau=cfg['pc_boundary_tau'],min_change_area=cfg['pc_min_change_area'],logit_margin=cfg.get('pc_logit_margin',0.))
        detector=OpenVocabularyChangeDetector(class_names=cfg['class_names'],device='cuda',
            change_threshold=cfg['change_threshold'],presence_alpha=cfg.get('presence_alpha',1.),
            use_presence_filtering=cfg.get('use_presence',True),instance_filter=BoundaryDistanceChangeFilter(**pp))
        results=[]
        with torch.inference_mode():
            for phase in (1,2):
                with Image.open(row[f'image_t{phase}']) as im:
                    rgb=im.convert('RGB')
                    if cfg['slide_stride']>0 and cfg['slide_crop']>0:
                        result=self.segmentor.slide_inference_single(rgb,stride=cfg['slide_stride'],crop_size=cfg['slide_crop'])
                    else: result=self.segmentor.forward_single(rgb)
                results.append(result)
            cd=detector.detect_change(*results)
            self.agent_cd=cd
        mapping={1:3,2:2,3:4,4:1,5:5,6:6} if row['dataset']=='SECOND' else {5:1}
        t1=cd.pred_t1.cpu().numpy(); t2=cd.pred_t2.cpu().numpy()
        changes=cd.per_class_change
        branches={c:dict(t1=t1==native,t2=t2==native,change=np.asarray(changes[native],bool)) for c,native in mapping.items()}
        native_to_canonical={v:k for k,v in mapping.items()}
        np.savez(self.work/f'{row["id"]}_semantics.npz',t1=t1.astype(np.uint8),t2=t2.astype(np.uint8))
        return [binary_task(t,branches) for t in row['pilot_tasks']],dict(
            official_repository='https://github.com/Funny0101/ReA-OVCD',config=cfg,postprocess=pp,
            input_source_paths=[row['image_t1'],row['image_t2']],official_dataset_path_unused=True,
            effective_mode='per_class_bcd' if row['dataset']=='SECOND' else 'bcd',
            native_to_canonical=native_to_canonical,task_adapter='Official per-class refined changes AND original T1/T2 semantic labels; transition uses union of endpoint changes',
            presence_note='Official detect_change currently leaves temporal presence filtering commented out; retained unchanged')


class DynamicAdapter:
    def __init__(self,work,configs):
        import torch
        self.work=work; self.configs=configs
        repo = REFERENCE / 'DynamicEarth'
        third_party = repo / 'third_party'
        runtime = REPOSITORY / '.runtime/dynamic-earth'
        sys.path[:0] = [
            str(repo), str(third_party / 'SegEarth_OV'), str(third_party / 'segment_anything'),
            str(third_party / 'SimFeatUp'), str(runtime),
        ]
        self.repo=repo
        from segment_anything import sam_model_registry
        from dynamic_earth.sam_ext import MaskProposal
        from transformers import AutoModel,AutoImageProcessor
        self.mp=MaskProposal()
        self.sam=sam_model_registry['vit_h'](checkpoint=str(MCI_WEIGHTS['sam'])).to('cuda').eval()
        dinov2 = str(MCI_WEIGHTS['dinov2'])
        self.dino=AutoModel.from_pretrained(dinov2,local_files_only=True).to('cuda').eval()
        self.processor=AutoImageProcessor.from_pretrained(dinov2,do_resize=False,do_center_crop=False,local_files_only=True)
        import open_clip_custom.openai as clip_loader
        def local_download(url,cache_dir=None):
            if not url.endswith('/ViT-B-16.pt'): raise ValueError('Unexpected checkpoint request')
            return str(MCI_WEIGHTS['clip'])
        clip_loader.download_pretrained_from_url=local_download
        self.identifier=None; self.vocab=None

    def identifier_for(self,config):
        import torch
        from torchvision import transforms
        from dynamic_earth.identifier.segearth_ov_ext import SegEarth_OV
        vocabulary=config['vocabulary']
        if self.vocab!=vocabulary:
            if self.identifier is not None:
                del self.identifier; torch.cuda.empty_cache()
            names=self.work/'classes.txt'; names.write_text('\n'.join(vocabulary)+'\n')
            self.identifier=SegEarth_OV(clip_type='CLIP',vit_type='ViT-B/16',model_type='SegEarth',
                name_path=str(names),device='cuda',ignore_residual=True,feature_up=True,
                feature_up_cfg=dict(model_name='jbu_one',model_path=str(MCI_WEIGHTS['simfeatup'])),cls_token_lambda=-.3).eval()
            self.vocab=vocabulary
        side=config['identifier_resize']
        processor=transforms.Compose([transforms.ToTensor(),transforms.Normalize([.48145466,.4578275,.40821073],[.26862954,.26130258,.27577711]),transforms.Resize((side,side))])
        return self.identifier,processor

    def infer(self,row):
        if row.get("agent_semantics"): return self.infer_agent(row)
        import torch
        from segment_anything.utils.amg import rle_to_mask
        from dynamic_earth.comparator.bi_match import bitemporal_match
        from dynamic_earth.identifier.utils import identify
        images=[np.asarray(Image.open(row[f'image_t{i}']).convert('RGB')) for i in (1,2)]
        branches={}; retained={}; details={}; proposal_cache={}; compared={}
        for c in required_classes(row['pilot_tasks']):
            key=row['dataset'] if row['dataset']!='SECOND' else str(c)
            cfg=self.configs[key]; points=cfg['points_per_side']; angle=cfg['change_confidence_threshold']
            if points not in proposal_cache:
                self.mp.make_mask_generator(model=self.sam,points_per_side=points,points_per_batch=64,
                    pred_iou_thresh=.5,stability_score_thresh=.95,stability_score_offset=.9,box_nms_thresh=.7,min_mask_region_area=0)
                self.mp.set_hyperparameters(match_hist=False,area_thresh=.8)
                data,n1=self.mp.forward(*images)
                masks=np.asarray([rle_to_mask(r).astype(bool) for r in data['rles']],bool).reshape(-1,*images[0].shape[:2])
                proposal_cache[points]=(masks,n1)
            if (points,angle) not in compared:
                masks,n1=proposal_cache[points]
                if len(masks):
                    masks,n1=bitemporal_match(*images,masks,self.dino,self.processor,n1,device='cuda',change_confidence_threshold=angle,
                        model_config=dict(model_type='DINOv2',feature_dim=768,patch_size=14))
                compared[points,angle]=(masks,n1)
            masks,n1=compared[points,angle]; model,processor=self.identifier_for(cfg)
            labels=[]
            class Capture:
                def predict(self,*args,**kwargs):
                    values=model.predict(*args,**kwargs); labels.append(np.asarray(values,int)); return values
            if len(masks):
                cmasks,_=identify(*images,masks,n1,Capture(),processor,device='cuda',is_instance_class=cfg['is_instance_class'])
                l1,l2=labels
            else:
                cmasks=masks; l1=l2=np.empty(0,int)
            shape=images[0].shape[:2]
            merge=lambda values:np.any(values,axis=0) if len(values) else np.zeros(shape,bool)
            change=merge(cmasks)
            branches[c]=dict(change=change,t1=merge(masks[l1==1]),t2=merge(masks[l2==1]))
            retained[c]=dict(masks=masks,t1=l1,t2=l2)
            details[c]=dict(cfg,proposals=len(proposal_cache[points][0]),changed_proposals=len(masks),category_changed_proposals=len(cmasks))
        predictions=[]
        for spec in row['pilot_tasks']:
            if spec['task_type']!='transition':
                c=spec['task_params']['class_id']; data=retained[c]; l1,l2=data['t1'],data['t2']; kind=spec['task_type']
                if kind=='related': pred=branches[c]['change']
                else:
                    keep=(l1!=l2)&((l2==1) if kind=='into' else (l1==1))
                    values=data['masks'][keep]; pred=np.any(values,0) if len(values) else np.zeros(images[0].shape[:2],bool)
            else: pred=binary_task(spec,branches)
            predictions.append(pred)
        return predictions,dict(official_repository='https://github.com/likyoo/DynamicEarth',configs=details,
            native_resolution=True,task_adapter='Official identify result for related; same proposal semantic assignments for into/outof; endpoint class branches for transition')

# V3 Agent composition: only RGB, structured task, and native model states enter here.
def planner_mask(task, B, Y1, Y2):
    p=task['task_params']; kind=task['task_type']
    if kind=='transition': return B & (Y1==p['source_class']) & (Y2==p['target_class'])
    c=p['class_id']
    if kind=='related': return B & ((Y1==c)|(Y2==c))
    if kind=='into': return B & (Y2==c) & (Y1!=c)
    if kind=='outof': return B & (Y1==c) & (Y2!=c)
    raise ValueError(kind)


def finish_agent(adapter,row,state,basis,metadata):
    path=adapter.work/f'{row["id"]}_model_state.npz'
    np.savez(path,**state)
    masks=[planner_mask(task,state[b['B_key']],state[b['Y1_key']],state[b['Y2_key']])
           for task,b in zip(row['pilot_tasks'],basis)]
    metadata.update(model_state=str(path.resolve()),task_basis=basis,
        input_source_paths=[row['image_t1'],row['image_t2']],gt_used=False,
        task_adapter='Agent Boolean composition of native model change domain and phase semantics; no GT input')
    return masks,metadata


def basis_keys(B,Y1,Y2,mode):
    return dict(B_key=B,Y1_key=Y1,Y2_key=Y2,semantic_mode=mode)


def rea_agent(self,row):
    # Run exactly the native pipeline once; retain complete phase semantics and B.
    legacy=dict(row,agent_semantics=False)
    _,metadata=self.infer(legacy)
    # infer stores native phase maps; the reliable native CD result is cached below.
    cd=self.agent_cd
    mapping=metadata['native_to_canonical']
    n1=cd.pred_t1.cpu().numpy(); n2=cd.pred_t2.cpu().numpy()
    y1=np.zeros(n1.shape,np.uint8); y2=np.zeros(n2.shape,np.uint8)
    for native,c in mapping.items(): y1[n1==native]=c; y2[n2==native]=c
    state=dict(Y1=y1,Y2=y2,B=cd.change_mask.cpu().numpy().astype(bool))
    for native,c in mapping.items(): state[f'related_{c}']=np.asarray(cd.per_class_change[native],bool)
    basis=[]
    for task in row['pilot_tasks']:
        p=task['task_params']; key=f'related_{p["class_id"]}' if task['task_type']=='related' else 'B'
        basis.append(basis_keys(key,'Y1','Y2','fullclassargmax'))
    metadata['weights']=str(CHECKPOINT)
    metadata['semantic_note']='Native complete class predictions and refined reliable B; related retains native per-class change output'
    return finish_agent(self,row,state,basis,metadata)


def omni_agent(self,row):
    import torch
    model=self.model
    images=[Image.open(row[f'image_t{i}']).convert('RGB') for i in (1,2)]
    shape=(images[0].height,images[0].width)
    tile=256 if row['dataset'] in ('LEVIR-CD','WHU-CD') else max(shape)
    state={}; configs={}; words_by_class={}
    def configure(cfg,words,indices):
        for key,value in cfg.items():
            if key!='classname_path': setattr(model,key,value)
        model.processor.confidence_threshold=cfg['confidence_threshold']
        model.query_words=words; model.query_idx=torch.tensor(indices,dtype=torch.long,device='cuda')
        model.num_cls=max(indices)+1; model.num_queries=len(words)
    def phase_predict(image):
        with torch.inference_mode():
            if model.slide_crop>0 and (model.slide_crop<image.width or model.slide_crop<image.height):
                scores=model.slide_inference(image,model.slide_stride,model.slide_crop)
            else: scores=model._inference_single_view(image)
            if model.num_cls!=model.num_queries:
                idx=torch.nn.functional.one_hot(model.query_idx).T.view(model.num_cls,model.num_queries,1,1)
                scores=(scores.unsqueeze(0)*idx).max(1)[0]
            pred=scores.argmax(0)
            pred[scores.max(0)[0]<model.prob_thd]=model.bg_idx
        return pred
    for c in required_classes(row['pilot_tasks']):
        cfg=self.config(row['dataset'],c); configs[c]=cfg
        lines=Path(cfg['classname_path']).read_text().splitlines()
        words_by_class[c]=lines[1]
        words=[]; indices=[]
        for index,line in enumerate(lines):
            for word in line.split(','): words.append(word.strip()); indices.append(index)
        configure(cfg,words,indices)
        for phase in (1,2):
            state[f'Y{phase}_{c}']=np.zeros(shape,np.uint8)
            state[f'unmatched{phase}_{c}']=np.zeros(shape,bool)
        for y in range(0,shape[0],tile):
            for x in range(0,shape[1],tile):
                box=(x,y,min(x+tile,shape[1]),min(y+tile,shape[0]))
                preds=[phase_predict(im.crop(box)) for im in images]
                instances=[model._extract_instances(pred==1) for pred in preds]
                for phase in (1,2):
                    state[f'Y{phase}_{c}'][y:box[3],x:box[2]]=(preds[phase-1].cpu().numpy()==1).astype(np.uint8)*c
                    # Preserve exact official asymmetric overlap matching, before its final OR.
                    output=np.zeros((box[3]-y,box[2]-x),bool)
                    for instance in instances[phase-1]:
                        area=instance.sum()
                        if area==0 or area<model.t12_min_instance_area: continue
                        if not any((instance*ref).sum()/area>=model.instance_iou_threshold for ref in instances[2-phase]): output|=instance.astype(bool)
                    state[f'unmatched{phase}_{c}'][y:box[3],x:box[2]]=output
        state[f'related_{c}']=state[f'unmatched1_{c}']|state[f'unmatched2_{c}']
    ambiguous=any(t['task_type']=='transition' and ({t['task_params']['source_class'],t['task_params']['target_class']}&{1,2}) for t in row['pilot_tasks'])
    if ambiguous:
        # Joint seven-class SFID pass, retaining native fusion and thresholds.
        words=['background']; indices=[0]
        for c in range(1,7):
            cfg=self.config(row['dataset'],c)
            foreground=Path(cfg['classname_path']).read_text().splitlines()[1]
            for word in foreground.split(','): words.append(word.strip()); indices.append(c)
        cfg=self.config(row['dataset'],5); configure(cfg,words,indices)
        configs['fullclassargmax']=dict(cfg,query_words=words,query_classes=indices)
        for phase,image in enumerate(images,1):
            state[f'Y{phase}_full']=np.zeros(shape,np.uint8)
            for y in range(0,shape[0],tile):
                for x in range(0,shape[1],tile):
                    box=(x,y,min(x+tile,shape[1]),min(y+tile,shape[0]))
                    state[f'Y{phase}_full'][y:box[3],x:box[2]]=phase_predict(image.crop(box)).cpu().numpy().astype(np.uint8)
    basis=[]
    for task in row['pilot_tasks']:
        p=task['task_params']; kind=task['task_type']
        if kind=='transition':
            a,b=p['source_class'],p['target_class']; key=f'transition_B_{a}_{b}'
            state[key]=state[f'related_{a}']|state[f'related_{b}']
            full=bool({a,b}&{1,2})
            basis.append(basis_keys(key,'Y1_full' if full else f'Y1_{a}','Y2_full' if full else f'Y2_{b}','fullclassargmax' if full else 'native_target_instances'))
        else:
            c=p['class_id']; key=f'related_{c}' if kind=='related' else f'unmatched{2 if kind=="into" else 1}_{c}'
            basis.append(basis_keys(key,f'Y1_{c}',f'Y2_{c}','native_target_instances'))
    return finish_agent(self,row,state,basis,dict(official_repository='https://github.com/Erxucomeon/OmniOVCD',configs=configs,
        weights=str(CHECKPOINT),input_adapter=dict(tile_size=tile,nonoverlapping_native_crops=True),
        semantic_note='Native SFID per-phase segmentation and original asymmetric instance overlap matching; phase-specific unmatched domains retained before OR; ambiguous endpoints use joint full-class argmax'))


def dynamic_agent(self,row):
    import torch
    from segment_anything.utils.amg import rle_to_mask
    from dynamic_earth.comparator.bi_match import bitemporal_match
    from dynamic_earth.identifier.utils import identify
    images=[np.asarray(Image.open(row[f'image_t{i}']).convert('RGB')) for i in (1,2)]
    shape=images[0].shape[:2]
    primary=next(t['task_params']['class_id'] for t in row['pilot_tasks'] if t['task_type']=='related')
    key=row['dataset'] if row['dataset']!='SECOND' else str(primary)
    cfg=dict(self.configs[key])
    if row['dataset']=='WHU-CD': cfg['points_per_side']=int(np.ceil(max(shape)/32))
    points=cfg['points_per_side']
    self.mp.make_mask_generator(model=self.sam,points_per_side=points,points_per_batch=64,
        pred_iou_thresh=.5,stability_score_thresh=.95,stability_score_offset=.9,box_nms_thresh=.7,min_mask_region_area=0)
    self.mp.set_hyperparameters(match_hist=False,area_thresh=.8)
    data,n1=self.mp.forward(*images)
    masks=np.asarray([rle_to_mask(r).astype(bool) for r in data['rles']],bool).reshape(-1,*shape)
    proposal_count=len(masks)
    if len(masks):
        masks,n1=bitemporal_match(*images,masks,self.dino,self.processor,n1,device='cuda',
            change_confidence_threshold=cfg['change_confidence_threshold'],model_config=dict(model_type='DINOv2',feature_dim=768,patch_size=14))
    merge=lambda a:np.any(a,0) if len(a) else np.zeros(shape,bool)
    state=dict(B=merge(masks))
    related_classes={t['task_params']['class_id'] for t in row['pilot_tasks'] if t['task_type']=='related'}
    details=dict(primary_class=primary,change_domain_config=cfg,proposal_count=proposal_count,changed_proposals=len(masks))
    for c in sorted(related_classes):
        rcfg=dict(self.configs[row['dataset'] if row['dataset']!='SECOND' else str(c)])
        # Frozen task lists have exactly one related class per source.
        if c!=primary: raise ValueError('Frozen source unexpectedly has multiple related classes')
        rcfg['points_per_side']=points
        model,processor=self.identifier_for(rcfg); labels=[]
        class Capture:
            def predict(self,*args,**kwargs):
                values=model.predict(*args,**kwargs); labels.append(np.asarray(values,int)); return values
        if len(masks):
            cmasks,_=identify(*images,masks,n1,Capture(),processor,device='cuda',is_instance_class=rcfg['is_instance_class'])
            l1,l2=labels
        else: cmasks=masks; l1=l2=np.empty(0,int)
        state[f'related_{c}']=merge(cmasks)
        state[f'Y1_related_{c}']=merge(masks[l1==1]).astype(np.uint8)*c
        state[f'Y2_related_{c}']=merge(masks[l2==1]).astype(np.uint8)*c
    if any(t['task_type']!='related' for t in row['pilot_tasks']):
        fullcfg=dict(cfg,vocabulary=['background','low vegetation','ground,bareland','tree,forest','water,river,pond,sea','building','sports field'])
        model,processor=self.identifier_for(fullcfg)
        with torch.inference_mode():
            for phase,image in enumerate(images,1):
                tensor=processor(Image.fromarray(image)).unsqueeze(0).to('cuda').half()
                scores=model.forward_slide(tensor,[dict(ori_shape=shape)],model.slide_stride,model.slide_crop)[0]
                scores=(scores*model.logit_scale).softmax(0)
                if model.num_classes!=model.num_queries:
                    idx=torch.nn.functional.one_hot(model.query_idx).T.view(model.num_classes,model.num_queries,1,1)
                    scores=(scores.unsqueeze(0)*idx).max(1)[0]
                pred=scores.argmax(0); pred[scores.max(0)[0]<model.prob_thd]=model.bg_idx
                state[f'Y{phase}']=pred.cpu().numpy().astype(np.uint8)
        details['fullclass_semantics_config']=fullcfg
    basis=[]
    for task in row['pilot_tasks']:
        if task['task_type']=='related':
            c=task['task_params']['class_id']; basis.append(basis_keys(f'related_{c}',f'Y1_related_{c}',f'Y2_related_{c}','native_proposal_labels'))
        else: basis.append(basis_keys('B','Y1','Y2','fullclassargmax'))
    return finish_agent(self,row,state,basis,dict(official_repository='https://github.com/likyoo/DynamicEarth',configs=details,
        weights={name: str(path) for name, path in MCI_WEIGHTS.items()},
        semantic_note='B is pre-identify DINO-retained proposal union; directions use dense native SegEarth full-class argmax; related retains official identify; WHU point spacing 32 pixels'))


ReAAdapter.infer_agent=rea_agent
OmniAdapter.infer_agent=omni_agent
DynamicAdapter.infer_agent=dynamic_agent
