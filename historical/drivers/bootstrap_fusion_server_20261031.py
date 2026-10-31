"""Create an isolated runtime on the newly authorized A800 server."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import urllib.request

ROOT=Path('/root/fusion_encoders_20261031')


def status(stage,**kw):
    p=ROOT/'bootstrap_status.json';tmp=p.with_suffix('.tmp')
    tmp.write_text(json.dumps(dict(stage=stage,unix=time.time(),**kw),indent=2));tmp.replace(p)


def main():
    ROOT.mkdir(exist_ok=True);venv=ROOT/'venv'
    assert not venv.exists(),'Keep existing environments unchanged'
    status('creating_venv');subprocess.check_call([sys.executable,'-m','venv','--without-pip',str(venv)])
    py=venv/'bin/python';installer=ROOT/'get-pip.py'
    with urllib.request.urlopen('https://bootstrap.pypa.io/get-pip.py',timeout=90) as r:installer.write_bytes(r.read())
    status('installing_pip');subprocess.check_call([str(py),str(installer)])
    status('installing_torch')
    subprocess.check_call([str(py),'-m','pip','install','torch==2.7.1+cu118','--index-url','https://download.pytorch.org/whl/cu118'])
    status('installing_dependencies')
    subprocess.check_call([str(py),'-m','pip','install','transformers==5.7.0','numpy==1.26.4','pandas==2.2.3','scipy==1.15.3','ViennaRNA==2.7.0'])
    code="import torch,transformers,RNA; assert torch.cuda.is_available(); print(torch.__version__,transformers.__version__,RNA.__version__,torch.cuda.get_device_name())"
    subprocess.check_call([str(py),'-u','-c',code],env=dict(os.environ,USE_TF='0'))
    status('completed',python=str(py))


if __name__=='__main__':
    try:main()
    except BaseException as e:status('failed',error=repr(e));raise
