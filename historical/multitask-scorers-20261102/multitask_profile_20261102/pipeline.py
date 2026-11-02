"""Persistently run the bounded4-model queue without altering the older pipeline."""
from pathlib import Path
import subprocess,sys,time,os,json
HERE=Path(__file__).resolve().parent;WORK=HERE.parents[1];PY=WORK/'.runtime/scorer_cuda/Scripts/python.exe'
def save(p,v):
    t=p.with_suffix('.tmp');t.write_text(json.dumps(v,indent=2,ensure_ascii=False),encoding='utf8');t.replace(p)
def run(script,args):
    name=script+'_'+'_'.join(args);stdout=(HERE/(name+'.stdout.log')).open('a',encoding='utf8');stderr=(HERE/(name+'.stderr.log')).open('a',encoding='utf8')
    p=subprocess.Popen([str(PY),'-u',str(HERE/script),*args],cwd=str(WORK),stdout=stdout,stderr=stderr,creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)
    save(HERE/'active_process.json',dict(pid=p.pid,script=script,args=args,started_unix=time.time()));start=time.monotonic()
    while p.poll() is None:
        save(HERE/'heartbeat.json',dict(pid=p.pid,alive=True,elapsed=time.monotonic()-start,unix=time.time()));time.sleep(5)
    stdout.close();stderr.close()
    if p.returncode:raise RuntimeError(f'{script} {args} exited {p.returncode}; see named stderr log')
def main():
    save(HERE/'pipeline_status.json',dict(stage='waiting_for_neutral_stage',pid=os.getpid()))
    while not (WORK/'runs/multitask_transformer_20261102/evaluation/complete.json').exists():time.sleep(5)
    save(HERE/'pipeline_status.json',dict(stage='waiting_for_features',pid=os.getpid()))
    while not (HERE/'features_complete.json').exists():time.sleep(5)
    run('train.py',['preflight'])
    for arm in ['sequence','dynamics']:
        for seed in [30101,30102]:
            save(HERE/'pipeline_status.json',dict(stage='training',arm=arm,seed=seed,pid=os.getpid(),unix=time.time()))
            run('train.py',['train','--arm',arm,'--seed',str(seed)])
    save(HERE/'pipeline_status.json',dict(stage='locked_test_evaluation',pid=os.getpid()))
    run('evaluate.py',[])
    save(HERE/'pipeline_status.json',dict(stage='completed',pid=os.getpid(),unix=time.time()))
if __name__=='__main__':
    try:main()
    except BaseException as e:save(HERE/'pipeline_failure.json',dict(error=repr(e),unix=time.time()));raise
