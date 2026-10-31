"""Bounded serial queue for frozen data, nine arms and held-out verification."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parent


def save(path,obj):
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(obj,indent=2));tmp.replace(path)


def main():
    assert json.loads((ROOT/'gpu_preflight.json').read_text())['status']=='passed'
    assert not (ROOT/'launch.json').exists()
    from train_fusion_encoders_20261031 import ARMS
    sources=['run_fusion_trial_20261031.py','train_fusion_encoders_20261031.py','evaluate_fusion_encoders_20261031.py',
        'prepare_fusion_data_20261031.py','run_aligned_profile_ablation_20261030.py','physics_scorer/fusion_encoder_model.py',
        'physics_scorer/profile_model.py','physics_scorer/diagnostic_model.py','physics_scorer/model.py',
        'source/evaluator/seq2ribo/geometry.py','source/evaluator/seq2ribo/simulation.py','source/evaluator/seq2ribo/utils.py']
    frozen=json.loads((ROOT/'data/features_locked.json').read_text())
    for r in frozen['features']:
        p=ROOT/'data/frozen_inputs/features'/f'{r["gene"]}.npz'
        if not p.exists():p=ROOT/'data/features'/f'{r["gene"]}.npz'
        assert hashlib.sha256(p.read_bytes()).hexdigest()==r['sha256']
    save(ROOT/'protocol_locked.json',dict(unix=time.time(),arms=ARMS,seeds=[30101,30102,30103],
        data_features_manifest_sha256=hashlib.sha256((ROOT/'data/features_locked.json').read_bytes()).hexdigest(),
        source_hashes={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in sources},
        initializations='random; no BERT/T5 pretrained weights',primary='fresh120 equal-transcript log1p profile PCC; unit-mass seed ensembles',
        selection='validation seed mean, before tests',bootstrap='2000 paired genes, exploratory unadjusted intervals'))
    save(ROOT/'launch.json',dict(pid=os.getpid(),unix=time.time()))
    jobs=[('train',a,s) for a in ARMS for s in [30101,30102,30103]]+[('evaluate',None,None)]
    for stage,arm,seed in jobs:
        save(ROOT/'status.json',dict(stage=stage,arm=arm,seed=seed,unix=time.time()))
        if stage=='train':cmd=[sys.executable,'-u',str(ROOT/'train_fusion_encoders_20261031.py'),'--data-root',str(ROOT/'data'),'--out',str(ROOT/'runs'),'--arm',arm,'--seed',str(seed)]
        else:cmd=[sys.executable,'-u',str(ROOT/'evaluate_fusion_encoders_20261031.py')]
        with (ROOT/f'{stage}_{arm or "all"}_{seed or 0}.log').open('xb') as log:r=subprocess.run(cmd,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,timeout=3600)
        if r.returncode:raise RuntimeError(f'{stage} {arm} {seed} failed: {r.returncode}')


if __name__=='__main__':
    try:main()
    except BaseException as e:save(ROOT/'status.json',dict(stage='failed',error=repr(e),unix=time.time()));raise
