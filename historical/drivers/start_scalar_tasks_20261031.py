"""Start detached scalar experiments, pass SwanLab key over an anonymous pipe."""
import json
from pathlib import Path
import subprocess
import sys
ROOT=Path(__file__).resolve().parent
key=sys.stdin.readline().strip();assert key
with (ROOT/'controller.log').open('xb') as log:
    p=subprocess.Popen([sys.executable,'-u',str(ROOT/'run_scalar_tasks_20261031.py')],cwd=ROOT,
        stdin=subprocess.PIPE,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,text=True)
    p.stdin.write(key+'\n');p.stdin.flush();p.stdin.close()
del key
print(json.dumps(dict(status='scalar_controller_started',pid=p.pid)))
