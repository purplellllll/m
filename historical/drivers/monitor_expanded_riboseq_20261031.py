"""Compact status of the source-gated expanded-data training queue."""
import json
import os
from pathlib import Path
import time
ROOT=Path(__file__).resolve().parent


def read(path):return json.loads(path.read_text()) if path.exists() else {}


state=read(ROOT/'status.json');launch=read(ROOT/'launch.json');alive=False
if launch:
    try:os.kill(launch['pid'],0);alive=True
    except ProcessLookupError:pass
details=read(ROOT/'prepare_status.json') if state.get('stage')=='prepare' else {}
if state.get('stage')=='train':details=read(ROOT/f'runs/{state["arm"]}_{state["seed"]}/status.json')
print(json.dumps(dict(stage=state.get('stage'),process_alive=alive,arm=state.get('arm'),seed=state.get('seed'),
    step=details.get('step'),best_validation_pcc=details.get('best_pcc'),preparation_stage=details.get('stage'),
    prepared=details.get('completed'),total=details.get('total'),completed_weights=len(list((ROOT/'runs').glob('*/locked.json'))),
    seconds_since_update=round(time.time()-details.get('unix',state.get('unix',time.time())),1))))
