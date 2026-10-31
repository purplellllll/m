"""Verify published hashes and synthetic forward passes; no biological evaluation."""
import argparse
import json
from pathlib import Path
import sys
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from physics_scorer.checkpoint_io import load_exported_checkpoint


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--models-root',type=Path,default=ROOT/'models')
    parser.add_argument('--output',type=Path);args=parser.parse_args()
    torch.set_num_threads(1);torch.manual_seed(31)
    manifest=json.loads((args.models_root/'manifest.json').read_text());results=[]
    valid=torch.tensor([[True]*12,[True]*8+[False]*4]);cod=torch.randint(0,64,(2,12));cod[~valid]=64
    inputs=(cod,valid,torch.rand(2,12,2),torch.rand(2,12,5),torch.tensor([[0,12],[8,18]]))
    for record in manifest['models']:
        model,metadata=load_exported_checkpoint(args.models_root/record['path'],record['sha256'])
        assert metadata['spec']==record['spec']
        with torch.no_grad():
            if record['spec']['kind']=='TaskScalarModel':
                prediction=model(inputs,torch.rand(2,6));assert prediction.shape==(2,1)
            else:
                prediction,_=model(*inputs);assert prediction.shape==valid.shape and (prediction[~valid]==0).all()
        assert torch.isfinite(prediction).all()
        results.append(dict(path=record['path'],sha256_verified=True,strict_load_passed=True,
            synthetic_shape=list(prediction.shape),experiment_complete=record['experiment_complete']))
        del model
    output=dict(status='passed',n_models=len(results),torch=torch.__version__,biological_metrics_computed=False,
        training_started=False,results=results)
    if args.output:args.output.write_text(json.dumps(output,indent=2),encoding='utf-8')
    print(json.dumps(dict(status='passed',n_models=len(results),biological_metrics_computed=False,training_started=False)))


if __name__=='__main__':main()
