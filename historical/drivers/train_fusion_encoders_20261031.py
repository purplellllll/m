"""Train one frozen fusion/encoder arm; held-out evaluation is a separate step."""
import argparse
import hashlib
import json
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
from physics_scorer.fusion_encoder_model import FusionEncoderModel
from physics_scorer.diagnostic_model import native_profile_loss
from run_aligned_profile_ablation_20261030 import collate

ARMS={
    'bilstm_baseline':dict(encoder='bilstm'),
    'physics_x2':dict(encoder='bilstm',gains=[2.,1.,1.]),
    'gnn_x2':dict(encoder='bilstm',gains=[1.,2.,1.]),
    'context_x2':dict(encoder='bilstm',gains=[1.,1.,2.]),
    'learned_fusion':dict(encoder='bilstm',learned_gains=True),
    'bilstm_ln':dict(encoder='bilstm_ln'),
    'transformer':dict(encoder='transformer'),
    'bert':dict(encoder='bert'),
    't5':dict(encoder='t5'),
}


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path,obj):
    path=Path(path);temp=path.with_suffix('.tmp');temp.write_text(json.dumps(obj,indent=2),encoding='utf-8');temp.replace(path)


def data(root):
    panel=json.loads((root/'panel.json').read_text());rows=[r for r in panel['records'] if r['split'] in ['train','validation']];fs=[]
    assert len([r for r in rows if r['split']=='train'])==400 and len([r for r in rows if r['split']=='validation'])==80
    for r in rows:
        path=root/'frozen_inputs/features'/f'{r["gene"]}.npz'
        if not path.exists():path=root/'features'/f'{r["gene"]}.npz'
        with np.load(path,allow_pickle=False) as z:fs.append({k:z[k].copy() for k in z.files})
    return rows,fs


def main(args):
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    device=torch.device(args.device)
    if device.type=='cuda':
        if not torch.cuda.is_available():raise RuntimeError('CUDA is unavailable; no implicit CPU training fallback')
        torch.cuda.set_per_process_memory_fraction(.12)
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    seed=args.seed;torch.manual_seed(seed);np.random.seed(seed);random.seed(seed)
    out=Path(args.out)/f'{args.arm}_{seed}';out.mkdir(parents=True,exist_ok=False)
    setting=ARMS[args.arm];model=FusionEncoderModel(**setting).to(device)
    rows,fs=data(Path(args.data_root));tr=[i for i,r in enumerate(rows) if r['split']=='train'];va=[i for i,r in enumerate(rows) if r['split']=='validation']
    optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.01)
    best=float('inf');best_state=None;stale=0;begun=time.monotonic()
    manifest=dict(driver_sha256=sha(__file__),model_sha256=sha(PACKAGE_ROOT/'physics_scorer/fusion_encoder_model.py'),
        data_panel_sha256=sha(Path(args.data_root)/'panel.json'),arm=args.arm,seed=seed,setting=setting,
        pretrained=False,encoder_ff=model.context_ff,parameters=sum(p.numel() for p in model.parameters()),
        torch_version=torch.__version__,device=str(device),n_train=400,n_validation=80,n_test_evaluated=0)
    if model.family in ['bert','t5']:
        import transformers
        manifest['transformers_version']=transformers.__version__
    save(out/'inputs_locked.json',manifest)
    for epoch in range(18):
        model.train();losses=[];order=np.random.default_rng(seed+epoch).permutation(tr)
        for start in range(0,len(order),8):
            ix=order[start:start+8];x,y,mask=collate([fs[i] for i in ix]);x=tuple(v.to(device) for v in x);y=y.to(device);mask=mask.to(device)
            optimizer.zero_grad(set_to_none=True);q,a=model(*x)
            loss=native_profile_loss(q,y,mask)+.1*torch.stack([native_profile_loss(v,y,mask) for v in a.values()]).mean()
            assert torch.isfinite(loss);loss.backward();assert torch.isfinite(torch.nn.utils.clip_grad_norm_(model.parameters(),1.))
            optimizer.step();losses.append(float(loss.detach()))
        model.eval();vloss=[];vpcc=[]
        with torch.no_grad():
            for start in range(0,len(va),8):
                ix=va[start:start+8];x,y,mask=collate([fs[i] for i in ix]);x=tuple(v.to(device) for v in x);y=y.to(device);mask=mask.to(device)
                q,_=model(*x)
                for j in range(len(ix)):
                    vloss.append(float(native_profile_loss(q[j:j+1],y[j:j+1],mask[j:j+1])))
                    a=y[j][mask[j]].cpu().numpy().astype(float);b=q[j][mask[j]].cpu().numpy().astype(float)
                    assert np.ptp(a)>0 and np.ptp(b)>0;vpcc.append(float(np.corrcoef(a,b)[0,1]))
        value=float(np.mean(vloss));pcc=float(np.mean(vpcc));g=model.effective_gains().detach().cpu().tolist()
        event=dict(epoch=epoch+1,arm=args.arm,seed=seed,train_loss=float(np.mean(losses)),validation_native_loss=value,
            validation_pcc=pcc,gains=g,seconds=time.monotonic()-begun)
        with (out/'history.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(event)+'\n')
        save(out/'status.json',dict(stage='training',unix=time.time(),**event));print(json.dumps(event),flush=True)
        if value<best:
            best=value;best_pcc=pcc;best_epoch=epoch+1;best_gains=g;stale=0
            best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        else:stale+=1
        if stale>=4:break
    path=out/'best.pt';torch.save(dict(state_dict=best_state,setting=setting,arm=args.arm,seed=seed),path)
    save(out/'locked.json',dict(path=str(path),sha256=sha(path),arm=args.arm,seed=seed,setting=setting,
        validation_loss=best,validation_pcc=best_pcc,best_epoch=best_epoch,gains=best_gains,pretrained=False,
        parameters=manifest['parameters'],n_test_evaluated=0))
    save(out/'status.json',dict(stage='completed',unix=time.time()))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--data-root',required=True);p.add_argument('--out',required=True)
    p.add_argument('--arm',choices=list(ARMS),required=True);p.add_argument('--seed',type=int,required=True)
    p.add_argument('--device',choices=['cuda','cpu'],default='cuda');args=p.parse_args()
    try:main(args)
    except BaseException as error:
        root=Path(args.out);root.mkdir(parents=True,exist_ok=True)
        save(root/f'failure_{args.arm}_{args.seed}.json',dict(error=repr(error),unix=time.time()));raise
