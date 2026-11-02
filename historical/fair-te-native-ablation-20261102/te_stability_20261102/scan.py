"""Matched V3 latent-search development branches, finite lambda grid."""
from pathlib import Path
import sys,hashlib,time,math
import numpy as np
import torch
ROOT=Path(__file__).resolve().parent;WORK=ROOT.parent.parent
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(WORK))
from locked_scorer import LockedScorer
from stability import StabilityScorer,read,save,sha
from run_pilot import module
from latent_decoder.model import CodonVMAE,collate_records,mutable_positions,synonymous_logits,MASK,AA_IDS
from latent_decoder.tokenizer import CODONS
from latent_decoder.prepare_data import GENETIC_CODE
OUT=ROOT/'lambda_scan';OUT.mkdir(exist_ok=True)
LAMBDAS=[0.,.25,.5,.75,1.]
ARMS=['own_multi','ribodecode_multi','ribodecode_mfe']
STEPS=12

@torch.no_grad()
def main():
    torch.set_num_threads(2);torch.backends.mha.set_fastpath_enabled(False);torch.backends.cuda.matmul.allow_tf32=False
    assert read(ROOT/'scorer_locked.json')['status']=='passed'
    stability_report=read(ROOT/'stability/summary.json');assert stability_report['usable_for_exploratory_scan']
    candidates=[r for r in read(ROOT/'stability/development_records.json') if r['split']=='val']
    rows=[]
    for lo,hi in [(80,199),(200,349),(350,500)]:
        group=[r for r in candidates if lo<=r['n_codons']<=hi]
        rows.append(min(group,key=lambda r:hashlib.sha256(('te_stability_scan:'+r['uid']).encode()).hexdigest()))
    evaluator=module('stability_external',WORK/'runs/vmae_own_full30_cpu_20261031/evaluate_cpu.py');evaluator.OUT=OUT/'external';evaluator.OUT.mkdir(exist_ok=True)
    # Keep training-compatible RNAfold2.7.0 loaded in this process. Final properties are recomputed separately with2.7.2.
    external_protocol=evaluator.source_protocol();hard,condition=evaluator.ribodecode_model(torch)
    scorer=LockedScorer('cuda');stability=StabilityScorer()
    generator=WORK/'runs/vmae_tiny_size_only_20261025/checkpoints/fold0_uniform.pt'
    ck=torch.load(generator,map_location='cpu',weights_only=False)
    model=CodonVMAE(**ck['config']['model']).cuda().eval().requires_grad_(False);model.load_state_dict(ck['model'],strict=True)
    model_seed_scale=float(scorer.models[0][2]['target_scale'])
    train=[r['half_life_log_target'] for r in read(ROOT/'stability/development_records.json') if r['split']=='train']
    stability_scale=float(np.std(train))
    bank_path=OUT/'score_bank.json';bank=read(bank_path) if bank_path.exists() else {}
    def score(dna,row):
        uid=hashlib.sha256(dna.encode()).hexdigest()
        if uid not in bank:
            bank[uid]=dict(**scorer(dna),ribodecode_score=evaluator.hard_score(torch,hard,condition,dna))
            save(bank_path,bank)
        return dict(**bank[uid],stability_log_score=stability(dna,row['utr5'],row['utr3']))
    originals=[score(r['dna'],r) for r in rows]
    ribo_scale=max(.1,float(np.std([r['ribodecode_score'] for r in originals])))
    protocol=dict(status='locked_before_candidates',cases=[dict(uid=r['uid'],gene=r['gene'],n_codons=r['n_codons']) for r in rows],
        lambdas=LAMBDAS,arms=ARMS,proposals_per_cell=STEPS,total_proposals=3*len(ARMS)*len(LAMBDAS)*STEPS,
        generator_sha256=sha(generator),scorer_lock_sha256=sha(ROOT/'scorer_locked.json'),stability_model_sha256=sha(ROOT/'stability/model.npz'),
        inputs_sha256=sha(ROOT/'stability/development_records.json'),external_protocol=external_protocol,
        objective='lambda*T+(1-lambda)*P. T=translation score delta/scaler. Own/ribodecode_multi P=mean(GC_delta/5percent,CAI_delta/.05,-MFE_per_nt_delta/.05,half_life_log_delta/train_label_std). ribodecode_mfe P=-MFE_per_nt_delta/.05.',
        scaling=dict(own=model_seed_scale,ribodecode=ribo_scale,stability=stability_scale),
        limits=['Three validation CDS, 12 proposals per grid cell; exploratory, not final test',
                'All generated metrics are predictions; no measured candidate TE or half-life',
                'RiboDecode-score-guided V3 branches, NOT native RiboDecode optimizer with protein-specific MFE surrogate',
                'GC reward preserved from historical search; no evidence higher GC always beneficial',
                'Auxiliary half-life ridge is separate from our three-branch TE network',
                'Fixed UTR context for each case; seq2ribo uses CDS only',
                'New lambda endpoints differ from historical additive-own-score scans'],source_sha256=sha(__file__))
    path=OUT/'protocol_locked.json'
    if path.exists():assert read(path)==protocol
    else:save(path,protocol)
    save(OUT/'inputs_locked.json',rows)
    def reward(v,original,arm,lam):
        own=(v['own_score']-original['own_score'])/model_seed_scale
        ribo=(v['ribodecode_score']-original['ribodecode_score'])/ribo_scale
        mfe=(original['mfe_per_nt']-v['mfe_per_nt'])/.05
        props=np.mean([(v['gc_percent']-original['gc_percent'])/5.,(v['cai']-original['cai'])/.05,mfe,
            (v['stability_log_score']-original['stability_log_score'])/stability_scale])
        return float(lam*(own if arm=='own_multi' else ribo)+(1-lam)*(mfe if arm=='ribodecode_mfe' else props))
    results=read(OUT/'results.json') if (OUT/'results.json').exists() else []
    for ci,(row,original) in enumerate(zip(rows,originals)):
        dna=row['dna'];tokens=collate_records([dict(cds_dna=dna)]).cuda();active=mutable_positions(tokens)
        mu,lv=model.encode(tokens.masked_fill(active,MASK),tokens);sigma=(lv*.5).exp()
        for arm in ARMS:
            for lam in LAMBDAS:
                if any(z['case_uid']==row['uid'] and z['arm']==arm and z['lambda']==lam for z in results):continue
                rng=torch.Generator(device='cuda').manual_seed(20261102+ci)
                offset=torch.zeros_like(mu);latent_best=-float('inf');best=original;best_reward=0.;trace=[];begun=time.monotonic()
                for step in range(STEPS):
                    trial=offset if step==0 else offset+torch.randn(mu.shape,generator=rng,device='cuda')*active[...,None]*max(.25,.95**(step//10))
                    trial=trial*active[...,None];rms=trial[active].square().mean().sqrt();trial*=min(1.,8/float(rms.clamp_min(1e-12)))
                    ids=synonymous_logits(model.decode(mu+sigma*trial,tokens),tokens).argmax(-1);ids[~active]=tokens[~active]
                    assert torch.equal(AA_IDS.to(ids.device)[ids],AA_IDS.to(ids.device)[tokens])
                    candidate=''.join(CODONS[int(i)] for i in ids[0])
                    assert candidate[:3]==dna[:3] and candidate[-3:]==dna[-3:] and len(candidate)==len(dna)
                    assert all(GENETIC_CODE[candidate[j:j+3]]==GENETIC_CODE[dna[j:j+3]] for j in range(0,len(dna),3))
                    value=score(candidate,row);r=reward(value,original,arm,lam);trace.append(dict(step=step,reward=r,**value))
                    if r>latent_best:latent_best=r;offset=trial.clone()
                    if r>best_reward:best=value;best_reward=r
                    save(OUT/'status.json',dict(stage='search',case=ci+1,arm=arm,lambda_value=lam,step=step+1,steps=STEPS,completed_cells=len(results),total_cells=45,unique_scored_sequences=len(bank)))
                results.append(dict(case_uid=row['uid'],n_codons=row['n_codons'],arm=arm,**{'lambda':lam},original=original,best=best,best_reward=best_reward,
                    trace=trace,seconds=time.monotonic()-begun,unique_proposals=len({r['sequence_sha256'] for r in trace})))
                save(OUT/'results.json',results);print('scan',len(results),45,arm,lam,'reward',best_reward,flush=True)
    save(OUT/'search_complete.json',dict(status='completed',cells=len(results),attempted_proposals=540,unique_scored_sequences=len(bank),test_read=False))
if __name__=='__main__':
    try:main()
    except BaseException as error:
        save(OUT/'failure.json',dict(error=repr(error),unix=time.time()));raise
