"""Compact process and output progress from the declared experiment directory."""
import json
import os
from pathlib import Path
import time

ROOT=Path(__file__).resolve().parent


def read(name):
    path=ROOT/name
    return json.loads(path.read_text()) if path.exists() else {}


state=read('status.json');launch=read('launch.json');alive=False
if launch:
    try:os.kill(launch['pid'],0);alive=True
    except ProcessLookupError:pass
latest={}
if state.get('stage')=='train':latest=read(f'runs/{state["arm"]}_{state["seed"]}/status.json')
if state.get('stage')=='prepare':latest=read('prepare_status.json')
print(json.dumps(dict(process_alive=alive,stage=state.get('stage'),arm=state.get('arm'),seed=state.get('seed'),
    completed_weights=len(list((ROOT/'runs').glob('*/locked.json'))),epoch=latest.get('epoch'),
    validation_pcc=latest.get('validation_pcc'),features_completed=latest.get('completed'),
    last_update_seconds=round(time.time()-latest.get('unix',state.get('unix',time.time())),1),error=state.get('error'))))
