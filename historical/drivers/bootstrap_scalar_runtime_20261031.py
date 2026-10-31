"""Install optional baseline CUDA kernels into a new, isolated environment."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
import urllib.request

ROOT=Path(__file__).resolve().parent
VENV=ROOT/'venv'
BASE='/root/riboseq_graph_v2_20261031/venv/lib/python3.10/site-packages'


def save(v):
    p=ROOT/'runtime_status.json';t=p.with_suffix('.tmp');t.write_text(json.dumps(v,indent=2));t.replace(p)


def main():
    subprocess.run([sys.executable,'-m','venv','--without-pip',str(VENV)],check=True)
    site=VENV/'lib/python3.10/site-packages'
    (site/'prior_runtime.pth').write_text('import site; site.addsitedir('+repr(BASE)+')\n')
    py=str(VENV/'bin/python')
    subprocess.run([py,'-m','pip','install','--disable-pip-version-check','--no-deps','einops==0.8.0'],check=True)
    # Mamba's optional language-generation imports require the4.x API. Pin
    # locally instead of changing the active profile experiment's5.x runtime.
    subprocess.run([py,'-m','pip','install','--disable-pip-version-check','--no-deps','--ignore-installed',
        'transformers==4.44.2','tokenizers==0.19.1','huggingface-hub==0.36.0'],check=True)
    filename='mamba_ssm-2.2.6.post3+cu11torch2.7cxx11abiTRUE-cp310-cp310-linux_x86_64.whl'
    url='https://github.com/state-spaces/mamba/releases/download/v2.2.6.post3/'+filename
    wheel=ROOT/filename;save(dict(stage='downloading_baseline_kernel',source=url,unix=time.time()))
    if not wheel.exists():
        with urllib.request.urlopen(url,timeout=60) as response,wheel.with_suffix('.partial').open('wb') as out:
            while True:
                chunk=response.read(1024*1024)
                if not chunk:break
                out.write(chunk)
        wheel.with_suffix('.partial').replace(wheel)
    assert wheel.stat().st_size==391436861
    digest=hashlib.sha256(wheel.read_bytes()).hexdigest()
    subprocess.run([py,'-m','pip','install','--disable-pip-version-check','--no-deps',str(wheel)],check=True)
    subprocess.run([py,'-c','import torch,mamba_ssm,einops; assert torch.__version__=="2.7.1+cu118"; print(mamba_ssm.__version__,einops.__version__)'],check=True)
    save(dict(stage='installed_pending_numerical_parity',wheel_sha256=digest,source=url,unix=time.time()))


if __name__=='__main__':
    try:main()
    except BaseException as error:
        save(dict(stage='failed',error_type=type(error).__name__,unix=time.time()));raise
