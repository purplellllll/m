"""Secret remains in process memory; execute only after an explicit source decision."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parent


def save(path,obj):
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(obj,indent=2));tmp.replace(path)


def execute(stage,command,timeout,**metadata):
    save(ROOT/'status.json',dict(stage=stage,unix=time.time(),**metadata))
    name='_'.join([stage,*map(str,metadata.values())])+'.log'
    with (ROOT/name).open('xb') as f:
        result=subprocess.run(command,cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,timeout=timeout)
    if result.returncode:raise RuntimeError(f'{stage} failed: {result.returncode}; see {name}')


def main():
    assert not (ROOT/'launch.json').exists()
    key=sys.stdin.readline().strip();assert key
    os.environ.update(SWANLAB_API_KEY=key,SWANLAB_PROBE_GIT='false',SWANLAB_DISABLE_GIT='True',
        SWANLAB_PROBE_RUNTIME='false',SWANLAB_PROBE_REQUIREMENTS='false',USE_TF='0',PYTHONDONTWRITEBYTECODE='1',
        OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',CUDA_VISIBLE_DEVICES='0')
    del key
    save(ROOT/'launch.json',dict(pid=os.getpid(),unix=time.time()))
    deadline=time.monotonic()+24*3600
    while not (ROOT/'data_choice.json').exists():
        save(ROOT/'status.json',dict(stage='waiting_for_data_choice',unix=time.time()))
        if time.monotonic()>deadline:raise TimeoutError('Data decision was not provided within 24 hours')
        time.sleep(15)
    approval=json.loads((ROOT/'data_choice.json').read_text())
    if approval['choice']!='related_liver_repartition':
        save(ROOT/'status.json',dict(stage='waiting_for_original_train_files',unix=time.time()));return
    execute('prepare',[sys.executable,'-u',str(ROOT/'prepare_expanded_riboseq_20261031.py')],24*3600)
    execute('model_checks',[sys.executable,'-m','physics_scorer.test_reliable_graph_context'],300)
    execute('preflight',[sys.executable,'-u',str(ROOT/'preflight_expanded_riboseq_20261031.py')],600)
    from train_expanded_riboseq_20261031 import ARMS,RECIPE,sha
    sources=['run_expanded_riboseq_20261031.py','prepare_expanded_riboseq_20261031.py','train_expanded_riboseq_20261031.py',
        'preflight_expanded_riboseq_20261031.py','evaluate_expanded_riboseq_20261031.py',
        'physics_scorer/reliable_graph_context.py','physics_scorer/test_reliable_graph_context.py',
        'physics_scorer/graph_polish_model.py','physics_scorer/fusion_encoder_model.py',
        'physics_scorer/profile_model.py','physics_scorer/model.py','physics_scorer/diagnostic_model.py',
        'run_aligned_profile_ablation_20261030.py','prepare_fusion_data_20261031.py',
        'source/evaluator/seq2ribo/geometry.py','source/evaluator/seq2ribo/simulation.py','source/evaluator/seq2ribo/utils.py']
    save(ROOT/'protocol_locked.json',dict(arms=ARMS,recipe=RECIPE,seeds=[30101,30102,30103],unix=time.time(),
        source_hashes={p:sha(ROOT/p) for p in sources},features_sha256=sha(ROOT/'data/features_locked.json'),
        evaluation_interval_optimizer_steps=100,selection='mean best validation PCC; before test evaluation',
        source='related-study LIVER repartition, NOT original RiboGL train split',original_split_reproduced=False))
    for seed in [30101,30102,30103]:
        for arm in ARMS:
            execute('train',[sys.executable,'-u',str(ROOT/'train_expanded_riboseq_20261031.py'),
                '--root',str(ROOT),'--arm',arm,'--seed',str(seed),'--max-steps','3000','--swanlab-mode','online'],6*3600,arm=arm,seed=seed)
    execute('evaluate',[sys.executable,'-u',str(ROOT/'evaluate_expanded_riboseq_20261031.py')],3600)


if __name__=='__main__':
    try:main()
    except BaseException as error:
        # Never persist environment or secret-bearing exception repr.
        save(ROOT/'status.json',dict(stage='failed',error_type=type(error).__name__,unix=time.time()));raise
