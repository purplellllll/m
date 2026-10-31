"""Return compact scalar training progress without credentials or test labels."""
import json
import os
from pathlib import Path
import time
ROOT=Path(__file__).resolve().parent


def read(path):return json.loads(path.read_text()) if path.exists() else None


def main():
    state=read(ROOT/'status.json') or {};launch=read(ROOT/'launch.json') or {};pid=launch.get('pid')
    alive=False
    if pid:
        try:os.kill(pid,0);alive=True
        except ProcessLookupError:pass
    summary=dict(state=state,controller_alive=alive,pid=pid,complete_scalar_weights=0,test_evaluations=[])
    progress=[]
    for out in sorted((ROOT/'runs').glob('*')) if (ROOT/'runs').exists() else []:
        status=read(out/'status.json') or {};lock=read(out/'locked.json')
        if lock:summary['complete_scalar_weights']+=1
        if status:
            progress.append(dict(run=out.name,stage=status.get('stage'),step=status.get('step',status.get('steps')),
                latest_validation_pcc=status.get('validation_pcc'),best_validation_pcc=(lock or {}).get('best_validation_pcc')))
    summary['runs']=progress
    summary['completed_runs']=sum(r['stage']=='completed' for r in progress)
    for task in ['protein','te','rpf']:
        status=read(ROOT/f'{task}_features_status.json')
        if status:summary[f'{task}_features']=status
        if (ROOT/f'{task}_evaluation_complete.json').exists():summary['test_evaluations'].append(task)
    summary['now']=time.time();print(json.dumps(summary,indent=2,allow_nan=False))


if __name__=='__main__':main()
