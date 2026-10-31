"""Reuse the existing aligned folding/sTASEP implementation for scalar tasks."""
import argparse
from concurrent.futures import ProcessPoolExecutor,as_completed
import hashlib
import json
from pathlib import Path
import sys
import time
import numpy as np

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,'/root/graph_polish_20261031')
from run_aligned_profile_ablation_20261030 import structure_job,simulation_job
SOURCE='/root/fusion_encoders_20261031/source'
PARAMETERS=Path('/root/graph_polish_20261031/data/physics_parameters_train_only.json')


def save(path,v):
    path=Path(path);tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(v,indent=2,allow_nan=False));tmp.replace(path)


def one(row,folder,parameters):
    path=Path(folder)/f'{row["uid"]}.npz'
    if path.exists():
        with np.load(path,allow_pickle=False) as z:
            assert str(z['dna_sha256'])==row['uid']
        return row['uid']
    f=structure_job((SOURCE,row['dna']))
    seed=int(hashlib.sha256(('scalar_features:'+row['uid']).encode()).hexdigest()[:8],16)
    f.update(simulation_job((SOURCE,row['dna'],f,parameters,seed)))
    f['dna_sha256']=row['uid'];n=len(f['cod'])
    # Scalar examples have no fabricated measured profile; these placeholders
    # exist only for the established collation helper and are never optimized.
    f['label']=np.zeros(n,dtype=np.float32);f['observed']=np.zeros(n,dtype=bool)
    assert f['physics'].shape==(n,2) and f['structure'].shape==(n,5)
    with path.with_suffix('.tmp').open('wb') as h:np.savez_compressed(h,**f)
    path.with_suffix('.tmp').replace(path)
    return row['uid']


def main(args):
    task=json.loads((ROOT/'data'/f'{args.task}.json').read_text());folder=ROOT/'features';folder.mkdir(exist_ok=True)
    parameters=json.loads(PARAMETERS.read_text())
    rows=list({r['uid']:r for r in task['records']}.values());started=time.monotonic();completed=0
    save(ROOT/f'{args.task}_features_status.json',dict(stage='folding_and_simulation',completed=0,total=len(rows),unix=time.time()))
    with ProcessPoolExecutor(max_workers=2) as pool:
        jobs=[pool.submit(one,r,folder,parameters) for r in rows]
        for f in as_completed(jobs):
            f.result();completed+=1
            if completed%5==0 or completed==len(rows):
                save(ROOT/f'{args.task}_features_status.json',dict(stage='completed' if completed==len(rows) else 'folding_and_simulation',
                    completed=completed,total=len(rows),seconds=time.monotonic()-started,unix=time.time()))
    save(ROOT/f'{args.task}_features_locked.json',dict(task=args.task,n=len(rows),
        task_sha256=hashlib.sha256((ROOT/'data'/f'{args.task}.json').read_bytes()).hexdigest(),
        physics_parameters_sha256=hashlib.sha256(PARAMETERS.read_bytes()).hexdigest(),
        physics_source='Frozen mouse train-only proxy from the transferred current profile model; no scalar test-label fitting',
        n_simulations=4,simulation_aggregation='mean and std, matching existing current-model inputs',
        truncated_sequences=0,files={r['uid']:hashlib.sha256((folder/f'{r["uid"]}.npz').read_bytes()).hexdigest() for r in rows}))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--task',choices=['protein','te','rpf'],required=True)
    args=p.parse_args()
    try:main(args)
    except BaseException as error:
        save(ROOT/f'{args.task}_features_failure.json',dict(error=repr(error),unix=time.time()));raise
