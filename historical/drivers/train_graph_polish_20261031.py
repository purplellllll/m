"""One controlled architecture arm, trained without opening held-out labels."""
import argparse
import math
import os
from pathlib import Path
import random
import sys
import time
os.environ.setdefault('USE_TF','0')
import numpy as np
import torch
PACKAGE_ROOT=Path(__file__).resolve().parent
if not (PACKAGE_ROOT/'physics_scorer').exists():PACKAGE_ROOT=PACKAGE_ROOT.parent
sys.path.insert(0,str(PACKAGE_ROOT))
from physics_scorer.graph_polish_model import GraphPolishModel, MODES
from train_fusion_encoders_20261031 import data, collate, native_profile_loss, sha, save

SEEDS=[30101,30102,30103]
RECIPE=dict(lr=.0003,minimum_lr=.00003,warmup_epochs=2,max_epochs=30,patience=7,
            batch_size=8,weight_decay=.01,clip=1.,auxiliary=.1)


def main(args):
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    assert torch.cuda.is_available()
    torch.cuda.set_per_process_memory_fraction(.12)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    seed=args.seed;torch.manual_seed(seed);np.random.seed(seed);random.seed(seed)
    out=Path(args.out)/f'{args.arm}_{seed}';out.mkdir(parents=True,exist_ok=False)
    setting=dict(mode=args.arm);model=GraphPolishModel(**setting).cuda()
    rows,fs=data(Path(args.data_root));tr=[i for i,r in enumerate(rows) if r['split']=='train'];va=[i for i,r in enumerate(rows) if r['split']=='validation']
    optimizer=torch.optim.AdamW(model.parameters(),lr=RECIPE['lr'],weight_decay=RECIPE['weight_decay'])
    save(out/'inputs_locked.json',dict(arm=args.arm,seed=seed,setting=setting,recipe=RECIPE,
        driver_sha256=sha(__file__),model_sha256=sha(PACKAGE_ROOT/'physics_scorer/graph_polish_model.py'),
        panel_sha256=sha(Path(args.data_root)/'panel.json'),parameters=sum(v.numel() for v in model.parameters()),
        torch_version=torch.__version__,n_train=400,n_validation=80,n_test_evaluated=0))
    best=float('inf');stale=0;started=time.monotonic()
    for epoch in range(RECIPE['max_epochs']):
        order=np.random.default_rng(seed+epoch).permutation(tr);model.train();losses=[]
        for start in range(0,len(order),8):
            progress=epoch+start/len(order)
            factor=min(1.,(progress+.02)/2) if progress<2 else .1+.9*.5*(1+math.cos(math.pi*(progress-2)/28))
            for group in optimizer.param_groups:group['lr']=RECIPE['lr']*factor
            x,y,mask=collate([fs[i] for i in order[start:start+8]])
            x=tuple(v.cuda() for v in x);y=y.cuda();mask=mask.cuda()
            optimizer.zero_grad(set_to_none=True);q,aux=model(*x)
            loss=native_profile_loss(q,y,mask)+.1*torch.stack([native_profile_loss(v,y,mask) for v in aux.values()]).mean()
            assert torch.isfinite(loss);loss.backward()
            assert torch.isfinite(torch.nn.utils.clip_grad_norm_(model.parameters(),1.))
            optimizer.step();losses.append(float(loss.detach()))
        model.eval();vloss=[];vpcc=[]
        with torch.no_grad():
            for start in range(0,len(va),8):
                ix=va[start:start+8];x,y,mask=collate([fs[i] for i in ix]);x=tuple(v.cuda() for v in x);y=y.cuda();mask=mask.cuda()
                q,_=model(*x)
                for j in range(len(ix)):
                    vloss.append(float(native_profile_loss(q[j:j+1],y[j:j+1],mask[j:j+1])))
                    a=y[j][mask[j]].cpu().numpy().astype(float);b=q[j][mask[j]].cpu().numpy().astype(float)
                    assert np.ptp(a)>0 and np.ptp(b)>0;vpcc.append(float(np.corrcoef(a,b)[0,1]))
        value=float(np.mean(vloss));pcc=float(np.mean(vpcc))
        event=dict(epoch=epoch+1,arm=args.arm,seed=seed,train_loss=float(np.mean(losses)),
            validation_native_loss=value,validation_pcc=pcc,input_gate=float(model.input_logit.sigmoid().detach()),
            lr=optimizer.param_groups[0]['lr'],seconds=time.monotonic()-started)
        with (out/'history.jsonl').open('a') as f:f.write(__import__('json').dumps(event)+'\n')
        save(out/'status.json',dict(stage='training',unix=time.time(),**event));print(__import__('json').dumps(event),flush=True)
        if value<best:
            best=value;best_pcc=pcc;best_epoch=epoch+1;best_gate=event['input_gate'];stale=0
            state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        else:stale+=1
        if stale>=RECIPE['patience']:break
    path=out/'best.pt';torch.save(dict(state_dict=state,setting=setting,arm=args.arm,seed=seed),path)
    save(out/'locked.json',dict(path=str(path),sha256=sha(path),arm=args.arm,seed=seed,setting=setting,
        kind='polish',validation_loss=best,validation_pcc=best_pcc,best_epoch=best_epoch,input_gate=best_gate,
        parameters=sum(v.numel() for v in model.parameters()),n_test_evaluated=0))
    save(out/'status.json',dict(stage='completed',unix=time.time()))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--data-root',required=True);parser.add_argument('--out',required=True)
    parser.add_argument('--arm',choices=list(MODES),required=True);parser.add_argument('--seed',type=int,required=True)
    main(parser.parse_args())
