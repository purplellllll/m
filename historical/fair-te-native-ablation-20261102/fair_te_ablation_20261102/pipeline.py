"""Serial resource queue: existing continuation, controlled training, audit, GitHub publication."""
from pathlib import Path
import sys,json,subprocess,time,os
HERE=Path(__file__).resolve().parent;WORK=HERE.parent.parent;TE=WORK/'runs/te_stability_20261102'
CUDA=WORK/'.runtime/scorer_cuda/Scripts/python.exe'
ARMS=['full','no_physics','no_graph','no_context','seq_transformer','seq2ribo_TE_adapt','RiboGL_TE_adapt','RiboDecode_TE_adapt']
SEEDS=[30101,30102]
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def save(name,v):(HERE/name).write_text(json.dumps(v,indent=2,ensure_ascii=False),encoding='utf-8')
def execute(python,script,args,log_name):
    with (HERE/log_name).open('ab') as log:
        child=subprocess.Popen([str(python),'-u',str(HERE/script),*args],cwd=WORK,stdout=log,stderr=subprocess.STDOUT,creationflags=subprocess.CREATE_NO_WINDOW)
        save('active_process.json',dict(pid=child.pid,script=script,args=args,started_unix=time.time()))
        code=child.wait()
    assert code==0,(script,args,code,log_name)
def main():
    deadline=time.time()+48*3600
    deps=[TE/'native_compare/complete.json',TE/'te_100000/training_complete.json',HERE/'inputs_locked.json',HERE/'all_preflight_complete.json']
    while not all(p.exists() for p in deps):
        missing=[str(p.relative_to(WORK)) for p in deps if not p.exists()]
        save('pipeline_status.json',dict(stage='waiting_for_serial_resources',dependencies=missing,unix=time.time()))
        for failure in [TE/'te_100000/training_failure.json',TE/'native_compare/comparison_failure.json',HERE/'preflight_failure.json']:
            if failure.exists():raise RuntimeError(str(failure)+': '+str(read(failure)))
        if time.time()>deadline:raise TimeoutError('Prerequisites not complete within48hours')
        time.sleep(10)
    if not (HERE/'full_length_100000/complete.json').exists():
        save('pipeline_status.json',dict(stage='reused903_test_for_locked100000_models',unix=time.time()))
        execute(CUDA,'evaluate_100000.py',[],'evaluate_100000.log')
    for arm in ARMS:
        assert read(HERE/'preflight'/f'{arm}.json')['status']=='passed'
        for seed in SEEDS:
            marker=HERE/'models'/f'{arm}_{seed}'/'complete.json'
            if marker.exists():continue
            save('pipeline_status.json',dict(stage='controlled_training',arm=arm,seed=seed,total_models=16,unix=time.time()))
            execute(CUDA,'experiment.py',['train','--arm',arm,'--seed',str(seed)],f'train_{arm}_{seed}.log')
    save('training_complete.json',dict(status='completed',models=16,steps_per_model=3000,from_scratch=True))
    if not (HERE/'all_checkpoints_locked.json').exists():execute(CUDA,'evaluate.py',['lock'],'lock.log')
    for arm in ARMS:
        for seed in SEEDS:
            if (HERE/'test_predictions'/f'{arm}_{seed}.json').exists():continue
            save('pipeline_status.json',dict(stage='locked_test_inference',arm=arm,seed=seed,unix=time.time()))
            execute(CUDA,'evaluate.py',['predict','--arm',arm,'--seed',str(seed)],f'predict_{arm}_{seed}.log')
    if not (HERE/'complete.json').exists():execute(CUDA,'evaluate.py',['summarize'],'summarize.log')
    assert read(HERE/'independent_audit.json')['status']=='passed'
    save('pipeline_status.json',dict(stage='publishing_completed_audited_results',unix=time.time()))
    if not (HERE/'publication_complete.json').exists():
        for attempt in range(3):
            try:execute(CUDA,'publish.py',[],'publish.log');break
            except AssertionError:
                if attempt==2:raise
                time.sleep(10)
    save('pipeline_status.json',dict(stage='completed_and_published',publication=read(HERE/'publication_complete.json'),unix=time.time()))
    print('Controlled benchmark, audits, and GitHub publication complete',flush=True)
if __name__=='__main__':
    try:main()
    except BaseException as e:save('pipeline_failure.json',dict(error=repr(e),unix=time.time()));raise
