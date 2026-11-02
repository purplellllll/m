"""Resume a finite prepare/train/test pipeline; stop on errors, no silent retries."""
from pathlib import Path
import subprocess
import sys
import time
from common import ROOT,WORK,read,save,prepare,ensure_features,verify_sources

def run():
    gpu=WORK/'.runtime/scorer_cuda/Scripts/python.exe'
    prepare();verify_sources()
    def child(script,args,log):
        with (ROOT/log).open('a',encoding='utf-8') as handle:subprocess.run([str(gpu),str(ROOT/script),*args],cwd=WORK,stdout=handle,stderr=subprocess.STDOUT,check=True)
    if not (ROOT/'gpu_preflight.json').exists():child('train.py',['--preflight'],'gpu_preflight.log')
    save(ROOT/'pipeline_status.json',dict(stage='development_features',started_unix=time.time()))
    ensure_features(read(ROOT/'train_validation.json'),'development_features')
    if not (ROOT/'training_complete.json').exists():
        save(ROOT/'pipeline_status.json',dict(stage='training',unix=time.time()));child('train.py',[],'training.log')
    assert (ROOT/'selection_locked.json').exists()
    save(ROOT/'pipeline_status.json',dict(stage='heldout_features',selection_locked=True,unix=time.time()))
    ensure_features(read(ROOT/'test_inputs.json'),'test_features')
    if not (ROOT/'evaluation_complete.json').exists():
        save(ROOT/'pipeline_status.json',dict(stage='heldout_local_evaluation',selection_locked=True,unix=time.time()));child('evaluate.py',[],'evaluation.log')
    save(ROOT/'pipeline_status.json',dict(stage='completed',unix=time.time(),test_n=903,paper_coefficients_used=False))

if __name__=='__main__':
    try:run()
    except BaseException as error:save(ROOT/'pipeline_status.json',dict(stage='failed',error=repr(error),unix=time.time(),automatic_retry=False));raise
