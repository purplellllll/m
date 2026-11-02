"""Predict from an aligned single-CDS NPZ; never generate surrogate zero physics."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from physics_scorer.checkpoint_io import load_exported_checkpoint

def main():
    p=argparse.ArgumentParser();p.add_argument('--models-root',type=Path,required=True);p.add_argument('--model',action='append')
    p.add_argument('--features',type=Path,required=True);p.add_argument('--properties',type=float,nargs=3,metavar=('GC','MFE_NT','CAI'))
    p.add_argument('--device',default='cpu');args=p.parse_args();torch.set_num_threads(2);torch.backends.mha.set_fastpath_enabled(False)
    manifest=json.loads((args.models_root/'manifest.json').read_text(encoding='utf-8'))
    selected=args.model or manifest['selected_paths'];registry={r['path']:r for r in manifest['models']}
    with np.load(args.features,allow_pickle=False) as z:features={k:z[k].copy() for k in ['cod','physics','structure','edges']}
    cod=torch.tensor(features['cod'],dtype=torch.long,device=args.device)[None];valid=torch.ones_like(cod,dtype=torch.bool)
    inputs=(cod,valid,torch.tensor(features['physics'],device=args.device)[None],torch.tensor(features['structure'],device=args.device)[None],
            torch.tensor(features['edges'],dtype=torch.long,device=args.device))
    props=None if args.properties is None else torch.tensor([args.properties],device=args.device)
    values=[]
    for name in selected:
        record=registry[name];model,metadata=load_exported_checkpoint(args.models_root/name,record['sha256'],args.device)
        if metadata['spec']['kind']!='HumanTEScalar':raise ValueError('Expected a human TE export')
        with torch.no_grad():score=float(model(inputs,props)[0])
        values.append(dict(score=score,seed=metadata['seed'],arm=metadata['arm'],model=name));del model
    print(json.dumps(dict(score=sum(v['score'] for v in values)/len(values),task='human_HEK293_CDS_only_TE',models=values)))

if __name__=='__main__':main()
