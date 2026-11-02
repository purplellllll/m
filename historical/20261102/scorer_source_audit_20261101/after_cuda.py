"""Bounded follow-through for an installer already running before the length probe existed."""
import json
from pathlib import Path
import subprocess
import time

ROOT=Path(__file__).resolve().parent
deadline=time.monotonic()+7200
while time.monotonic()<deadline:
    try:state=json.loads((ROOT/'cuda_status.json').read_text(encoding='utf-8'))
    except (FileNotFoundError,json.JSONDecodeError):state={}
    if state.get('stage')=='failed':raise RuntimeError(state)
    if state.get('stage')=='completed':
        if not (ROOT/'length_probe_cuda.json').exists():
            subprocess.run([state['python'],str(ROOT/'length_probe.py'),'--device','cuda'],check=True)
        break
    time.sleep(5)
else:raise TimeoutError('CUDA installation did not finish within two hours')
