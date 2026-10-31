"""Create a separate experiment directory from immutable prior helper sources."""
from pathlib import Path
import shutil

ROOT=Path('/root/graph_polish_20261031')
PRIOR=Path('/root/fusion_encoders_20261031')
ROOT.mkdir(exist_ok=False)
(ROOT/'physics_scorer').mkdir()
for p in (PRIOR/'physics_scorer').glob('*.py'):
    shutil.copy2(p,ROOT/'physics_scorer'/p.name)
for name in ['train_fusion_encoders_20261031.py','run_aligned_profile_ablation_20261030.py','prepare_fusion_data_20261031.py']:
    shutil.copy2(PRIOR/name,ROOT/name)
for name in ['utils.py','geometry.py','simulation.py']:
    source=PRIOR/'source/evaluator/seq2ribo'/name
    dest=ROOT/'source/evaluator/seq2ribo'/name;dest.parent.mkdir(parents=True,exist_ok=True)
    shutil.copy2(source,dest)
print(str(ROOT))
