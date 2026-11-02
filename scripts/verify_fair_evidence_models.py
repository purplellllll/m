"""Verify the new release's hashes, strict scalar loading and synthetic input contract."""
from pathlib import Path
import sys,json,argparse
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from physics_scorer.checkpoint_io import load_exported_checkpoint
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--models-root',type=Path,required=True);parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    torch.set_num_threads(2);torch.backends.mha.set_fastpath_enabled(False);torch.manual_seed(31)
    records=json.loads((args.models_root/'manifest.json').read_text(encoding='utf-8'))['models'];assert len(records)==14
    valid=torch.tensor([[True]*12,[True]*8+[False]*4]);cod=torch.randint(0,64,(2,12));cod[~valid]=64
    inputs=(cod,valid,torch.rand(2,12,2),torch.rand(2,12,5),torch.tensor([[0,12],[8,18]]));results=[]
    for row in records:
        model,meta=load_exported_checkpoint(args.models_root/row['path'],row['sha256'])
        assert meta['spec']==row['metadata']['spec']
        with torch.no_grad():p=model(inputs)
        assert p.shape==(2,) and torch.isfinite(p).all();results.append(dict(path=row['path'],strict_load=True,hash_checked=True,finite_scalar_output=True))
    args.output.write_text(json.dumps(dict(status='passed',models=len(results),torch=str(torch.__version__),device='cpu',results=results,biological_metrics_recomputed=False),indent=2),encoding='utf-8')
if __name__=='__main__':main()
