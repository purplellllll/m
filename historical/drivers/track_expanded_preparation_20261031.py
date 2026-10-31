"""Publish actual preparation progress while the independent training queue runs."""
import json
import os
from pathlib import Path
import sys
import time
os.environ.update(SWANLAB_PROBE_GIT='false',SWANLAB_PROBE_RUNTIME='false',SWANLAB_PROBE_REQUIREMENTS='false',SWANLAB_DISABLE_GIT='True')
import swanlab

ROOT=Path(__file__).resolve().parent


def read(name):
    path=ROOT/name
    return json.loads(path.read_text()) if path.exists() else {}


def main():
    key=sys.stdin.readline().strip();assert key
    swanlab.login(api_key=key,save=False);del key
    run=swanlab.init(project='RiboSeq-GNN-Transformer-20261031',experiment_name='full-data-preparation',
        mode='online',public=False,logdir=str(ROOT/'swanlog'),config=dict(n_train=5564,n_validation=696,n_test=696,
        source='related-study public LIVER repartition; not original RiboGL split'))
    (ROOT/'preparation_swanlab_run.json').write_text(json.dumps(dict(id=run.id,url=getattr(run,'url',None)),indent=2))
    step=0;deadline=time.monotonic()+24*3600
    try:
        while time.monotonic()<deadline:
            status=read('status.json');prepared=read('prepare_status.json');panel=read('data/panel.json')
            if prepared.get('stage')=='completed':
                swanlab.log({'preparation/completed':1,'preparation/elapsed_seconds':prepared['seconds']},step=step);return
            if status.get('stage')=='failed':
                swanlab.log({'preparation/failed':1},step=step);return
            stage=prepared.get('stage','initializing')
            swanlab.log({f'preparation/{stage}_completed':prepared.get('completed',0),
                'preparation/total':prepared.get('total',len(panel.get('records',[]))),
                'preparation/elapsed_seconds':prepared.get('seconds',0)},step=step)
            step+=1;time.sleep(60)
    finally:swanlab.finish()


if __name__=='__main__':main()
