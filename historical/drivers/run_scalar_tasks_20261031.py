"""Bounded scalar-task experiment queue; key exists only in process memory."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parent


def save(p,v):
    t=p.with_suffix('.tmp');t.write_text(json.dumps(v,indent=2,allow_nan=False));t.replace(p)


def execute(stage,command,timeout,**metadata):
    name='_'.join([stage,*map(str,metadata.values())]);save(ROOT/'status.json',dict(stage=stage,unix=time.time(),**metadata))
    with (ROOT/f'{name}.log').open('xb') as log:
        result=subprocess.run(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,timeout=timeout)
    if result.returncode:raise RuntimeError(f'{name} exited {result.returncode}')


def main():
    assert not (ROOT/'launch.json').exists()
    key=sys.stdin.readline().strip();assert key
    os.environ.update(SWANLAB_API_KEY=key,SWANLAB_PROBE_GIT='false',SWANLAB_DISABLE_GIT='True',
        SWANLAB_PROBE_RUNTIME='false',SWANLAB_PROBE_REQUIREMENTS='false',USE_TF='0',
        PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',CUDA_VISIBLE_DEVICES='0')
    del key
    save(ROOT/'launch.json',dict(pid=os.getpid(),unix=time.time()))
    sources=[p for p in ROOT.glob('*.py')]+list((ROOT/'physics_scorer').glob('*.py'))
    save(ROOT/'protocol_locked.json',dict(source_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
        seeds=[30101,30102,30103],arms=['frozen_probe','finetune'],max_steps_per_run=2000,validation_every=100,
        selection='Best scalar Pearson on validation only; select arm across3 seeds before any test',
        note='Aim to exceed baselines; this is not an assertion of SOTA. Native seq2ribo TE split remains unavailable.',
        te_single_target_runs=48,protein_runs=6,rpf_frozen_probe_runs=6,
        maximum_optimizer_steps=126000,task_order=['protein','te','rpf'],unix=time.time()))
    execute('scalar_checks',[sys.executable,str(ROOT/'test_scalar_downstream_20261031.py')],120)
    if not (ROOT/'protein_features_locked.json').exists():
        raise RuntimeError('Finish the already-running protein feature preparation before launching training; no duplicate workers')
    execute('scalar_gpu_preflight',[sys.executable,str(ROOT/'preflight_scalar_tasks_20261031.py')],180)
    for seed in [30101,30102,30103]:
        for arm in ['frozen_probe','finetune']:
            execute('train',[sys.executable,'-u',str(ROOT/'train_scalar_tasks_20261031.py'),'--task','protein',
                '--arm',arm,'--seed',str(seed),'--max-steps','2000'],8*3600,task='protein',arm=arm,seed=seed)
    save(ROOT/'protein_training_complete.json',dict(unix=time.time(),test_evaluated=False))
    # Baseline kernels are isolated from both old experiments' environments.
    runtime=ROOT/'venv/bin/python'
    deadline=time.monotonic()+3600
    while True:
        status=json.loads((ROOT/'runtime_status.json').read_text())
        if status['stage']=='installed_pending_numerical_parity':break
        if status['stage']=='failed':raise RuntimeError('Isolated baseline runtime failed; preserve completed scalar weights')
        if time.monotonic()>deadline:raise TimeoutError('Baseline runtime preparation exceeded1h')
        time.sleep(15)
    execute('baseline_backend_check',[str(runtime),str(ROOT/'check_scalar_mamba_backend_20261031.py')],300)
    # Protein baseline CPU simulations can overlap the longer human RNA folds.
    with (ROOT/'evaluate_protein.log').open('xb') as log:
        protein_evaluation=subprocess.Popen([str(runtime),'-u',str(ROOT/'evaluate_scalar_protein_20261031.py')],cwd=ROOT,
            stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
    save(ROOT/'protein_evaluation_launch.json',dict(pid=protein_evaluation.pid,unix=time.time()))
    execute('prepare',[sys.executable,'-u',str(ROOT/'prepare_scalar_features_20261031.py'),'--task','te'],72*3600,task='te')
    for use_utr in [False,True]:
        for target in range(4):
            for seed in [30101,30102,30103]:
                for kind in ['ours','seq2ribo']:
                    command=[str(runtime),'-u',str(ROOT/'train_scalar_tasks_20261031.py'),'--task','te',
                        '--arm','finetune','--seed',str(seed),'--max-steps','2000','--target-index',str(target),'--model',kind]
                    if use_utr:command.append('--use-utr')
                    execute('train',command,8*3600,task='te_utr' if use_utr else 'te',target=target,model=kind,seed=seed)
    save(ROOT/'te_training_complete.json',dict(unix=time.time(),test_evaluated=False))
    execute('evaluate_te',[str(runtime),'-u',str(ROOT/'evaluate_scalar_te_20261031.py')],4*3600)
    execute('prepare',[sys.executable,'-u',str(ROOT/'prepare_scalar_features_20261031.py'),'--task','rpf'],72*3600,task='rpf')
    for seed in [30101,30102,30103]:
        for arm in ['full_probe','rna_context_only']:
            execute('train',[str(runtime),'-u',str(ROOT/'train_scalar_rpf_20261031.py'),'--arm',arm,'--seed',str(seed)],12*3600,task='rpf',arm=arm,seed=seed)
    execute('evaluate_rpf',[str(runtime),'-u',str(ROOT/'evaluate_scalar_rpf_20261031.py')],6*3600)
    if protein_evaluation.wait(timeout=60)!=0:raise RuntimeError('Protein paired baseline evaluation failed; inspect its dedicated log')
    save(ROOT/'status.json',dict(stage='completed',unix=time.time(),strict_native_sota_certified=False,
        limitations='TE uses common refit, RPF length-filtered native subset; native TE split unavailable'))


if __name__=='__main__':
    try:main()
    except BaseException as error:
        save(ROOT/'status.json',dict(stage='failed',error_type=type(error).__name__,unix=time.time()));raise
