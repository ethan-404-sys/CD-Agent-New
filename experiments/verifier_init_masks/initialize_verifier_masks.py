"""Canonical initial-mask CLI; retain frozen sources, methods, and successful masks."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import sys

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE/'initial_masks'))
import verifier_v1_dataset as dataset


def regression(out):
    import numpy as np
    from PIL import Image
    pilot=dataset.PILOT
    inputs=dataset.read(pilot/'generator_inputs.json'); checked={}
    for method in dataset.METHODS:
        count=0
        for row in inputs:
            receipt=dataset.read(pilot/'generation'/method/f'{row["id"]}.json')
            with np.load(pilot/receipt['metadata']['model_state'],allow_pickle=False) as state:
                for spec,basis,path in zip(row['pilot_tasks'],receipt['metadata']['task_basis'],receipt['predictions']):
                    expected=dataset.reconstruct(spec,basis,state)
                    with Image.open(pilot/path) as image: actual=np.asarray(image)>0
                    if not np.array_equal(expected,actual): raise ValueError(f'Frozen mask changed: {method}/{row["id"]}')
                    count+=1
        checked[method]=count
    if sum(checked.values())!=222: raise ValueError('Expected 222 approved normal masks')
    report=dict(status='passed',approved_source_pairs=len(inputs),normal_masks=checked,
                total_normal_masks=222,masks_changed=False,ground_truth_read=False)
    dataset.write(out/'logs/frozen_regression.json',report)
    print('FROZEN_REGRESSION',json.dumps(report),flush=True)


def check_failure(out):
    import numpy as np
    from verifier_v1_generation_qc import independent_compile
    path=out/'generation/dynamic_earth_mci/levir_train_69_model_state.npz'
    with np.load(path,allow_pickle=False) as archive: state={key:archive[key] for key in archive.files}
    masks=state['proposals']; grid=np.asarray([448,448],np.int64)
    ys=np.arange(448)*masks.shape[1]//448; xs=np.arange(448)*masks.shape[2]//448
    support=masks[:,ys[:,None],xs[None,:]].sum((1,2)).astype(np.int64)
    spec=next(row['pilot_tasks'][0] for row in dataset.read(out/'generator_inputs.json') if row['id']=='levir_train_69')
    basis=dict(recipe_type='proposal_labels',route='official_related',selection_mode='related',
               proposal_masks_key='proposals',proposal_gate_key='binary_5_related_gate',conf_threshold=0.,
               phase1_labels_key='binary_5_labels1',phase2_labels_key='binary_5_labels2',
               phase1_scores_key='binary_5_scores1',phase2_scores_key='binary_5_scores2')
    for phase in (1,2):
        state[f'checked_support{phase}']=support; state[f'checked_grid{phase}']=grid
        basis[f'phase{phase}_support_key']=f'checked_support{phase}'
        basis[f'phase{phase}_grid_key']=f'checked_grid{phase}'
    compiled=independent_compile(spec,basis,state,True)
    if not np.array_equal(compiled,state['related_5']): raise ValueError('Native mask changed on failure fixture')
    # A supported NaN must still fail: this is not a blanket nonfinite bypass.
    bad=int(np.flatnonzero(~np.isfinite(state['binary_5_scores1']).all(1))[0])
    state['checked_support1']=support.copy(); state['checked_support1'][bad]=1
    try: independent_compile(spec,basis,state,True)
    except ValueError: rejected=True
    else: rejected=False
    if not rejected: raise ValueError('Supported nonfinite score was accepted')
    report=dict(status='passed',source_id='levir_train_69',native_mask_unchanged=True,
                supported_nan_rejected=True,empty_sampled_proposals=int((support==0).sum()))
    dataset.write(out/'logs/failure_fixture_qc.json',report)
    print('FAILURE_FIXTURE',json.dumps(report),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage',choices=('infer','qc','export','regression','check-failure'))
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--method',choices=dataset.METHODS)
    p.add_argument('--source-id',action='append')
    p.add_argument('--smoke',action='store_true')
    p.add_argument('--pilot-package',type=Path)
    args=p.parse_args()
    if not os.environ.get('SLURM_JOB_ID'): raise RuntimeError('Submit through Slurm')
    out=args.output.resolve()
    if args.pilot_package: dataset.PILOT=args.pilot_package.resolve()
    if args.stage=='regression': regression(out)
    elif args.stage=='check-failure': check_failure(out)
    elif args.stage=='qc': dataset.generation_qc(out)
    elif args.stage=='export': dataset.export(out)
    else:
        if not args.method: p.error('--method required for infer')
        dataset.infer(out,args.method,args.smoke,args.source_id)


if __name__=='__main__': main()
