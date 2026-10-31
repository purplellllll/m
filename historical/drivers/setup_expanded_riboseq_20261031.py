"""Prepare an isolated experiment and tracking environment without changing prior runs."""
import json
from pathlib import Path
import shutil
import subprocess
import sys

ROOT=Path('/root/riboseq_graph_v2_20261031')
PRIOR=Path('/root/graph_polish_20261031')
ROOT.mkdir(exist_ok=False)
shutil.copytree(PRIOR/'physics_scorer',ROOT/'physics_scorer',ignore=shutil.ignore_patterns('__pycache__'))
shutil.copytree(PRIOR/'source',ROOT/'source')
for name in ['run_aligned_profile_ablation_20261030.py','prepare_fusion_data_20261031.py']:
    shutil.copy2(PRIOR/name,ROOT/name)
subprocess.run([sys.executable,'-m','venv',str(ROOT/'venv')],check=True)
site=next((ROOT/'venv/lib').glob('python*/site-packages'))
prior_site=next(Path('/root/fusion_encoders_20261031/venv/lib').glob('python*/site-packages'))
(site/'prior_runtime.pth').write_text(str(prior_site)+'\n')
with (ROOT/'tracking_install.log').open('w') as log:
    subprocess.run([str(ROOT/'venv/bin/python'),'-m','pip','install','swanlab'],stdout=log,stderr=subprocess.STDOUT,check=True)
print(json.dumps(dict(status='ready',root=str(ROOT))))
