"""Install an isolated GPU runtime; leave the active CPU interpreter untouched."""
from pathlib import Path
import json
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parent
WORK=ROOT.parent.parent
ENV=WORK/'.runtime/scorer_cuda'

def status(stage,**fields):
    (ROOT/'cuda_status.json').write_text(json.dumps(dict(stage=stage,unix=time.time(),**fields),indent=2),encoding='utf-8')

try:
    status('creating_environment')
    subprocess.run([sys.executable,'-m','venv','--system-site-packages',str(ENV)],check=True)
    python=ENV/'Scripts/python.exe'
    status('installing_cuda_torch',version='2.11.0',index='https://download.pytorch.org/whl/cu126')
    subprocess.run([str(python),'-m','pip','install','--no-cache-dir','--timeout','120','--ignore-installed','--no-deps','https://download.pytorch.org/whl/cu126/torch-2.11.0%2Bcu126-cp312-cp312-win_amd64.whl#sha256=1b7d728ddcf84b12f64ca10ec5f6ce91dd9edcc5e62ed5f0a7260ca00f029e8b','--disable-pip-version-check'],check=True)
    status('checking_gpu')
    subprocess.run([str(python),str(ROOT/'runtime_probe.py'),'--device','cuda'],check=True,cwd=WORK)
    subprocess.run([str(python),str(ROOT/'length_probe.py'),'--device','cuda'],check=True,cwd=WORK)
    status('completed',python=str(python))
except BaseException as error:
    status('failed',error=repr(error))
    raise
