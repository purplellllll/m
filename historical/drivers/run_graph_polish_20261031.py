"""Prepare, preflight, train 24 models, then lock and evaluate all results."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parent
os.environ.update(USE_TF='0',PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',CUDA_VISIBLE_DEVICES='0')


def save(path,obj):
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(obj,indent=2));tmp.replace(path)


def execute(stage,command,timeout=3600,**metadata):
    save(ROOT/'status.json',dict(stage=stage,unix=time.time(),**metadata))
    name='_'.join([stage,*map(str,metadata.values())])+'.log'
    with (ROOT/name).open('xb') as log:
        result=subprocess.run(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,timeout=timeout)
    if result.returncode:
        raise RuntimeError(f'{stage} failed with code {result.returncode}; see {name}')


def main():
    assert not (ROOT/'launch.json').exists()
    save(ROOT/'launch.json',dict(pid=os.getpid(),unix=time.time()))
    execute('cpu_checks',[sys.executable,'-m','physics_scorer.test_graph_polish_model'])
    execute('prepare',[sys.executable,'-u',str(ROOT/'prepare_graph_polish_20261031.py')])
    execute('preflight',[sys.executable,'-u',str(ROOT/'preflight_graph_polish_20261031.py')])
    from train_graph_polish_20261031 import sha, MODES, SEEDS, RECIPE
    assert json.loads((ROOT/'gpu_preflight.json').read_text())['status']=='passed'
    prior=ROOT.parent/'fusion_encoders_20261031';(ROOT/'references').mkdir()
    references=[]
    for arm,name in [('bilstm_baseline','ref_bilstm'),('transformer','ref_transformer'),('context_x2','ref_context_x2')]:
        for seed in SEEDS:
            locked=json.loads((prior/f'runs/{arm}_{seed}/locked.json').read_text())
            assert sha(locked['path'])==locked['sha256']
            dest=ROOT/f'references/{name}_{seed}.pt';shutil.copy2(locked['path'],dest)
            references.append(dict(locked,arm=name,path=str(dest),kind='fusion'))
    save(ROOT/'references_locked.json',dict(models=references))
    sources=['run_graph_polish_20261031.py','train_graph_polish_20261031.py','evaluate_graph_polish_20261031.py',
        'preflight_graph_polish_20261031.py','prepare_graph_polish_20261031.py','train_fusion_encoders_20261031.py',
        'run_aligned_profile_ablation_20261030.py','prepare_fusion_data_20261031.py',
        'physics_scorer/graph_polish_model.py','physics_scorer/test_graph_polish_model.py',
        'physics_scorer/fusion_encoder_model.py','physics_scorer/profile_model.py','physics_scorer/model.py',
        'physics_scorer/diagnostic_model.py','source/evaluator/seq2ribo/utils.py',
        'source/evaluator/seq2ribo/geometry.py','source/evaluator/seq2ribo/simulation.py']
    manifest=json.loads((ROOT/'data/features_locked.json').read_text())
    for r in manifest['features']:
        p=ROOT/'data/frozen_inputs/features'/f'{r["gene"]}.npz'
        if not p.exists():p=ROOT/'data/features'/f'{r["gene"]}.npz'
        assert sha(p)==r['sha256']
    save(ROOT/'protocol_locked.json',dict(unix=time.time(),arms=MODES,seeds=SEEDS,recipe=RECIPE,
        source_hashes={p:sha(ROOT/p) for p in sources},data_features_manifest_sha256=sha(ROOT/'data/features_locked.json'),
        selection='mean per-seed validation PCC among four non-ablated routes before tests',
        primary='fresh120 equal-transcript log1p position PCC; unit-mass prediction ensembles',
        reference_limit='old recipe and smaller parameter count; reference only',
        initializations='random; identical parameters across eight new arms',near_homology_clustered=False))
    with (ROOT/'runtime_freeze.txt').open('w') as f:subprocess.run([sys.executable,'-m','pip','freeze'],stdout=f,check=True)
    for arm in MODES:
        for seed in SEEDS:
            execute('train',[sys.executable,'-u',str(ROOT/'train_graph_polish_20261031.py'),
                '--data-root',str(ROOT/'data'),'--out',str(ROOT/'runs'),'--arm',arm,'--seed',str(seed)],arm=arm,seed=seed)
    execute('evaluate',[sys.executable,'-u',str(ROOT/'evaluate_graph_polish_20261031.py')])
    assert json.loads((ROOT/'verification.json').read_text())['status']=='passed'


if __name__=='__main__':
    try:main()
    except BaseException as error:
        save(ROOT/'status.json',dict(stage='failed',error=repr(error),unix=time.time()));raise
