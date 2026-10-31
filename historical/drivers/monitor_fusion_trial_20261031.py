"""Compact read-only status for the isolated server experiment."""
import json
from pathlib import Path
import time

ROOT=Path(__file__).resolve().parent
def read(name):
    p=ROOT/name
    return json.loads(p.read_text()) if p.exists() else {}
o=read('orchestration_status.json');s=read('status.json');b=read('bootstrap_status.json');p=read('prepare_status.json')
a=read(f'runs/{s.get("arm")}_{s.get("seed")}/status.json')
print(json.dumps(dict(orchestration=o.get('stage'),bootstrap=b.get('stage'),preparation=p.get('stage'),
    stage=s.get('stage'),arm=s.get('arm'),seed=s.get('seed'),completed_weights=len(list((ROOT/'runs').glob('*/locked.json'))),
    epoch=a.get('epoch'),validation_pcc=a.get('validation_pcc'),last_update_seconds=round(time.time()-a.get('unix',s.get('unix',o.get('unix',time.time()))),1),
    error=s.get('error') or o.get('error') or b.get('error'))))
