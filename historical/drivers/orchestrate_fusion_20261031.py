"""Advance through setup, frozen data, CUDA preflight, training and evaluation."""
import json
from pathlib import Path
import subprocess
import time

ROOT=Path(__file__).resolve().parent


def progress(stage,**kw):
    p=ROOT/'orchestration_status.json';tmp=p.with_suffix('.tmp')
    tmp.write_text(json.dumps(dict(stage=stage,unix=time.time(),**kw),indent=2));tmp.replace(p)


def main():
    begun=time.monotonic();progress('waiting_for_setup')
    while True:
        b=json.loads((ROOT/'bootstrap_status.json').read_text())
        d=json.loads((ROOT/'related_liver/status.json').read_text())
        if b['stage']=='failed' or d['stage']=='failed':raise RuntimeError(f'Setup failed: {b}; {d}')
        if b['stage']==d['stage']=='completed':break
        if time.monotonic()-begun>3600:raise TimeoutError('Setup wait exceeded3600s')
        time.sleep(10)
    py=ROOT/'venv/bin/python'
    for stage,file,timeout in [('prepare','prepare_fusion_data_20261031.py',3600),('preflight','preflight_fusion_gpu_20261031.py',600),('experiment','run_fusion_trial_20261031.py',108000)]:
        progress(stage)
        with (ROOT/f'{stage}.log').open('xb') as log:
            r=subprocess.run([str(py),'-u',str(ROOT/file)],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,timeout=timeout)
        if r.returncode:raise RuntimeError(f'{stage} failed: {r.returncode}; see {stage}.log')
    progress('completed')


if __name__=='__main__':
    try:main()
    except BaseException as e:progress('failed',error=repr(e));raise
