"""Validation-only frozen-weight scan of one fusion branch at a time."""
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F

WORK=Path(__file__).resolve().parents[1];sys.path.insert(0,str(WORK))
from physics_scorer.fusion_encoder_model import FusionEncoderModel
from run_aligned_profile_ablation_20261030 import collate

OUT=WORK/'runs/fusion_encoders_20261031'
SNAP=WORK/'runs/bilstm_diagnostics_20261030/completed_snapshot/bilstm_diagnostics_20261030'
SEEDS=[30101,30102,30103];BRANCHES=['physics','gnn','context'];GRID=[0.,.5,1.,2.,4.,8.]


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def score(y,q):
    return float(np.corrcoef(y.astype(float),q.astype(float))[0,1]) if len(y)>1 and np.ptp(y)>0 and np.ptp(q)>0 else np.nan


def main():
    torch.set_num_threads(1);torch.set_num_interop_threads(1);begun=time.monotonic()
    OUT.mkdir(exist_ok=True);assert not (OUT/'gain_selection.json').exists(),'Preserve the existing scan'
    panel=json.loads((SNAP/'panel.json').read_text())
    rows=[r for r in panel['records'] if r['split']=='validation'];assert len(rows)==80
    fs=[];input_hashes=[]
    for r in rows:
        path=SNAP/'frozen_inputs/features'/f'{r["gene"]}.npz'
        with np.load(path,allow_pickle=False) as z:fs.append({k:z[k].copy() for k in z.files})
        input_hashes.append(dict(gene=r['gene'],sha256=sha(path)))
    candidates={'baseline':[1.,1.,1.]}
    for j,b in enumerate(BRANCHES):
        for gain in GRID:
            if gain==1:continue
            gains=[1.,1.,1.];gains[j]=gain;candidates[f'{b}_{gain:g}']=gains
    protocol=dict(status='validation_only',seeds=SEEDS,gain_grid=GRID,candidates=candidates,
        n_validation=80,n_test_evaluated=0,selection='Mean per-seed validation position PCC; independent single-branch optima; tie within1e-6 prefers unit gain',
        interpretation='Frozen-network sensitivity, not identifiable biological importance; fixed gains can be absorbed by the first fusion linear layer after retraining',
        model='Original width424/native-loss/aux0.1 full three-way late fusion, not latest no-late model',
        backend='local CPU; test for algebraic cached-projection equivalence; no test predictions',
        prior_panel_sha256=sha(SNAP/'panel.json'),validation_inputs=input_hashes,source_sha256=sha(__file__))
    (OUT/'gain_scan_protocol.json').write_text(json.dumps(protocol,indent=2),encoding='utf-8')
    per=[];rms=[];gradients=[];checkpoints=[];max_error=0.
    for seed in SEEDS:
        path=SNAP/f'w424_native_aux1_{seed}/best.pt';digest=sha(path)
        lock=json.loads((path.parent/'locked.json').read_text());assert digest==lock['sha256']
        model=FusionEncoderModel().eval();state=torch.load(path,map_location='cpu',weights_only=True)['state_dict']
        missing=model.load_state_dict(state,strict=False)
        assert set(missing.missing_keys)=={'fixed_gains','gain_logits'} and not missing.unexpected_keys
        model.requires_grad_(False);captured=[]
        hook=model.fusion[0].register_forward_pre_hook(lambda module,args:captured.append(args[0].detach()))
        cache=[]
        with torch.no_grad():
            for start in range(0,len(fs),8):
                batch=fs[start:start+8];x,_,_=collate(batch);q,_=model(*x)
                h=captured.pop();assert not captured
                for j,f in enumerate(batch):
                    n=len(f['cod']);parts=h[j,:n].split(424,-1)
                    projected=[F.linear(part,model.fusion[0].weight[:,k*424:(k+1)*424]) for k,part in enumerate(parts)]
                    rebuilt=F.softplus(F.linear(F.gelu(sum(projected)+model.fusion[0].bias),model.fusion[2].weight,model.fusion[2].bias)).squeeze(-1)
                    error=float((rebuilt-q[j,:n]).abs().max());max_error=max(max_error,error);assert error<1e-4
                    cache.append((projected,f))
                    for k,b in enumerate(BRANCHES):rms.append(dict(seed=seed,gene=rows[start+j]['gene'],branch=b,
                        embedding_rms=float(parts[k].square().mean().sqrt()),projected_rms=float(projected[k].square().mean().sqrt()),
                        fusion_weight_norm=float(model.fusion[0].weight[:,k*424:(k+1)*424].norm())))
            for name,gains in candidates.items():
                for i,(projected,f) in enumerate(cache):
                    h=sum(g*p for g,p in zip(gains,projected))+model.fusion[0].bias
                    q=F.softplus(F.linear(F.gelu(h),model.fusion[2].weight,model.fusion[2].bias)).squeeze(-1).numpy()
                    mask=f['observed'];r=score(f['label'][mask],q[mask]);assert np.isfinite(r)
                    per.append(dict(seed=seed,candidate=name,gene=rows[i]['gene'],pcc=r,gain_physics=gains[0],gain_gnn=gains[1],gain_context=gains[2]))
        hook.remove()
        gains=torch.ones(3,requires_grad=True);losses=[]
        for projected,f in cache:
            h=sum(g*p for g,p in zip(gains,projected))+model.fusion[0].bias
            q=F.softplus(F.linear(F.gelu(h),model.fusion[2].weight,model.fusion[2].bias)).squeeze(-1)
            mask=torch.from_numpy(f['observed']);y=torch.from_numpy(f['label'])[mask];q=q[mask]
            losses.append(-F.cosine_similarity(q-q.mean(),y-y.mean(),dim=0,eps=1e-8))
        torch.stack(losses).mean().backward()
        for b,g in zip(BRANCHES,gains.grad.tolist()):gradients.append(dict(seed=seed,branch=b,derivative_negative_validation_pcc=g))
        checkpoints.append(dict(seed=seed,path=str(path),sha256=digest));assert sha(path)==digest
        print(json.dumps(dict(seed=seed,status='scanned',seconds=time.monotonic()-begun)),flush=True)
    per=pd.DataFrame(per);per.to_csv(OUT/'gain_scan_per_transcript.csv',index=False)
    baseline=per[per.candidate=='baseline'].set_index(['seed','gene']).pcc;summary=[]
    for name,g in per.groupby('candidate',sort=False):
        delta=(g.set_index(['seed','gene']).pcc-baseline).reset_index();by_gene=delta.groupby('gene').pcc.mean().to_numpy()
        boot=np.random.default_rng(301031).integers(0,len(by_gene),(2000,len(by_gene)));lo,hi=np.percentile(by_gene[boot].mean(1),[2.5,97.5])
        seed_delta=delta.groupby('seed').pcc.mean()
        summary.append(dict(candidate=name,mean_validation_pcc=float(g.pcc.mean()),delta_from_unit=float(by_gene.mean()),
            ci_low=float(lo),ci_high=float(hi),positive_seeds=int((seed_delta>0).sum()),
            **{f'seed_{s}_delta':float(seed_delta.loc[s]) for s in SEEDS}))
    summary=pd.DataFrame(summary);summary.to_csv(OUT/'gain_scan_summary.csv',index=False)
    chosen={};branch_best={};base=float(summary[summary.candidate=='baseline'].mean_validation_pcc.iloc[0])
    for j,b in enumerate(BRANCHES):
        group=summary[summary.candidate.eq('baseline')|summary.candidate.str.startswith(b+'_')]
        best=group.sort_values('mean_validation_pcc',ascending=False).iloc[0]
        name='baseline' if best.mean_validation_pcc<=base+1e-6 else best.candidate
        chosen[b]=candidates[name][j]
        branch_best[b]=dict(candidate=name,gain=chosen[b],mean_validation_pcc=float(group[group.candidate==name].mean_validation_pcc.iloc[0]))
    result=dict(status='validation_scan_completed',selection_unix=time.time(),preferred_gains=chosen,branch_best=branch_best,
        baseline_mean_validation_pcc=base,checkpoints=checkpoints,n_test_evaluated=0,cached_projection_max_abs_error=max_error,
        torch_version=torch.__version__,cpu_only=True,seconds=time.monotonic()-begun)
    (OUT/'gain_selection.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    pd.DataFrame(rms).to_csv(OUT/'branch_embedding_rms.csv',index=False);pd.DataFrame(gradients).to_csv(OUT/'gain_local_derivatives.csv',index=False)
    print(json.dumps(result),flush=True)


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--fine',action='store_true');args=parser.parse_args()
    if args.fine:
        OUT=OUT/'gain_refinement';OUT.mkdir(exist_ok=True)
        GRID=[.75,.9,1.,1.1,1.25,1.5]
    main()
