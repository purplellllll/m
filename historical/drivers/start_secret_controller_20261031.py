"""Detach a controller and pass authentication via a pipe, without storing credentials."""
import json
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parent
key=sys.stdin.readline().strip();assert key
with (ROOT/'controller.log').open('xb') as log:
    process=subprocess.Popen([sys.executable,'-u',str(ROOT/'run_expanded_riboseq_20261031.py')],
        cwd=ROOT,stdin=subprocess.PIPE,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,text=True)
    process.stdin.write(key+'\n');process.stdin.flush();process.stdin.close()
del key
print(json.dumps(dict(status='controller_started',pid=process.pid)))
