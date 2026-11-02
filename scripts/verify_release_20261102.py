"""Check all published own-model hashes, strict state loading and synthetic inference."""
import argparse
import json
from pathlib import Path
import sys
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from physics_scorer.checkpoint_io import load_exported_checkpoint

def main():
    p=argparse.ArgumentParser();p.add_argument('--models-root',type=Path,required=True);p.add_argument('--output',type=Path);args=p.parse_args()
    torch.set_num_threads(2);torch.manual_seed(31);torch.backends.mha.set_fastpath_enabled(False)
    manifest=json.loads((args.models_root/'manifest.json').read_text(encoding='utf-8'));assert len(manifest['models'])==60
    valid=torch.tensor([[True]*12,[True]*8+[False]*4]);cod=torch.randint(0,64,(2,12));cod[~valid]=64
    inputs=(cod,valid,torch.rand(2,12,2),torch.rand(2,12,5),torch.tensor([[0,12],[8,18]]));counts={}
    for record in manifest['models']:
        model,metadata=load_exported_checkpoint(args.models_root/record['path'],record['sha256']);assert metadata['spec']==record['spec']
        kind=record['spec']['kind'];counts[kind]=counts.get(kind,0)+1
        with torch.no_grad():
            if kind=='HumanTEScalar':
                props=torch.rand(2,3);prediction=model(inputs,props);assert prediction.shape==(2,)
                if not model.use_properties:torch.testing.assert_close(prediction,model(inputs,props+100),atol=0,rtol=0)
                state=model.state_dict();broken={k:v for k,v in state.items() if k!='head.net.0.weight'}
                try:model.load_state_dict(broken,strict=True);raise AssertionError('Missing trained scalar head was accepted')
                except RuntimeError:pass
                model.load_state_dict(state,strict=True)
            elif kind=='BranchProbe':prediction=model(torch.rand(2,3,384),torch.rand(2,6));assert prediction.shape==(2,)
            elif kind=='CodonVMAE':prediction=model(cod,cod,sample=False)[0];assert prediction.shape==(2,12,64)
            else:prediction=model(*inputs)[0];assert prediction.shape==valid.shape
        assert torch.isfinite(prediction).all();del model
    result=dict(status='passed',models=60,kinds=counts,hashes_verified=True,strict_state_load=True,
        missing_human_head_rejected=True,no_stats_input_invariance=True,synthetic_inference=True,torch=torch.__version__,not_biological_validation=True)
    if args.output:args.output.write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result))

if __name__=='__main__':main()
