"""Matched proposal-budget synonymous hill climb versus VMAE, human TE development only."""
from pathlib import Path
import hashlib
import importlib.util
import json
import sys
import time
import numpy as np
import torch

ROOT=Path(__file__).resolve().parent
WORK=ROOT.parent.parent
PILOT=WORK/'runs/human_te_pilot_20261101'
sys.path.insert(0,str(PILOT));sys.path.insert(0,str(WORK))
from human_score import HumanTEScorer
from run_pilot import read,save,sha
from latent_decoder.model import CodonVMAE,collate_records,mutable_positions,synonymous_logits,MASK,AA_IDS
from latent_decoder.tokenizer import CODONS

def genetic_code():
    from latent_decoder.prepare_data import GENETIC_CODE
    return GENETIC_CODE

def select():
    rows=[r for r in read(PILOT/'selected_development_rows.json') if r['split']=='val' and r['dna'].startswith('ATG')]
    selected=[]
    for low,high in [(80,199),(200,349),(350,500)]:
        available=[r for r in rows if low<=len(r['dna'])//3<=high];assert available
        selected.append(min(available,key=lambda r:hashlib.sha256(('direct_vs_vmae:'+r['uid']).encode()).hexdigest()))
    return selected

def selfcheck():
    code=genetic_code();groups={}
    for codon,aa in code.items():groups.setdefault(aa,[]).append(codon)
    dna='ATGGCTGAATTCTAA';rng=np.random.default_rng(1);codons=[dna[j:j+3] for j in range(0,len(dna),3)]
    for j in [1,2,3]:
        alternatives=[c for c in groups[code[codons[j]]] if c!=codons[j]]
        codons[j]=str(rng.choice(alternatives))
    edited=''.join(codons)
    assert edited!=dna and edited[:3]==dna[:3] and edited[-3:]==dna[-3:]
    assert [code[dna[j:j+3]] for j in range(0,len(dna),3)]==[code[edited[j:j+3]] for j in range(0,len(dna),3)]
    save(ROOT/'selfcheck.json',dict(status='passed',synonymous_changes=True,start_stop_preserved=True))

@torch.no_grad()
def main():
    selfcheck();torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.mha.set_fastpath_enabled(False)
    assert torch.cuda.is_available();scorer=HumanTEScorer('cuda');rows=select();code=genetic_code();groups={}
    for codon,aa in code.items():groups.setdefault(aa,[]).append(codon)
    generator=WORK/'runs/vmae_tiny_size_only_20261025/checkpoints/fold0_uniform.pt'
    protocol=dict(task='Human HEK293 CDS-only TE, experimental pilot head',scorer_arm=scorer.arm,
        scorer_checkpoints=[z[3] for z in scorer.models],generator_sha256=sha(generator),cases=[r['uid'] for r in rows],
        proposals_per_method=20,objective='Own human TE scalar only, lambda1; properties reported separately',
        direct='Greedy synonymous hill climb; edit1/2/4 mutable codons with probabilities0.5/0.3/0.2; always a changed codon',
        vmae='Original V3 frozen; initial masked decoder plus19 Gaussian latent proposals, same radius8 and noise schedule',
        seed=20261101,original_fallback=True,amino_acids_and_terminals_fixed=True,
        limits=['3 development cases, not903 heldout test','20 attempted proposals; unique queries and wall times can differ',
                'Pilot head256train96validation; mouse-proxy four-run physics; not expanded final head',
                'All generated-sequence results are predictions, no measured candidate expression'],source_sha256=sha(__file__))
    save(ROOT/'protocol_locked.json',protocol);save(ROOT/'inputs_locked.json',rows)
    payload=torch.load(generator,map_location='cpu',weights_only=False)
    model=CodonVMAE(**payload['config']['model']).cuda().eval().requires_grad_(False);model.load_state_dict(payload['model'],strict=True)
    results=read(ROOT/'search_results.json') if (ROOT/'search_results.json').exists() else []
    for cached in results:
        for value in cached['trace']+[cached['original']]:scorer.cache[value['canonical_rna_sha256']]=value
    for case,row in enumerate(rows):
        dna=row['dna'];original=scorer(dna);tokens=collate_records([dict(cds_dna=dna)]).cuda();active=mutable_positions(tokens)
        mutable=active[0].nonzero().flatten().cpu().numpy();assert len(mutable)>0
        def valid(candidate):
            assert len(candidate)==len(dna) and candidate[:3]==dna[:3] and candidate[-3:]==dna[-3:]
            assert all(code[candidate[j:j+3]]==code[dna[j:j+3]] for j in range(0,len(dna),3))
        for method in ['direct_synonymous','vmae_latent']:
            if any(r['case_uid']==row['uid'] and r['method']==method for r in results):continue
            begun=time.monotonic();best=original;trace=[]
            seed=20261101+int(hashlib.sha256(dna.encode()).hexdigest()[:8],16)
            rng=np.random.default_rng(seed);latent_rng=torch.Generator(device='cuda').manual_seed(seed)
            if method=='vmae_latent':
                mu,lv=model.encode(tokens.masked_fill(active,MASK),tokens);sigma=(.5*lv).exp();offset=torch.zeros_like(mu);latent_best=-float('inf')
            for step in range(20):
                if method=='direct_synonymous':
                    codons=[best['dna'][j:j+3] for j in range(0,len(dna),3)]
                    count=min(int(rng.choice([1,2,4],p=[.5,.3,.2])),len(mutable))
                    for j in rng.choice(mutable,count,replace=False):
                        choices=[c for c in groups[code[codons[j]]] if c!=codons[j]];assert choices
                        codons[j]=str(rng.choice(choices))
                    candidate=''.join(codons)
                else:
                    trial=offset if step==0 else offset+torch.randn(mu.shape,generator=latent_rng,device='cuda')*active[...,None]*max(.25,.95**(step//10))
                    trial=trial*active[...,None];rms=trial[active].square().mean().sqrt();trial*=min(1.,8/float(rms.clamp_min(1e-12)))
                    ids=synonymous_logits(model.decode(mu+sigma*trial,tokens),tokens).argmax(-1);ids[~active]=tokens[~active]
                    amino_ids=AA_IDS.to(ids.device)
                    assert torch.equal(amino_ids[ids],amino_ids[tokens]);candidate=''.join(CODONS[int(i)] for i in ids[0])
                valid(candidate);value=scorer(candidate);trace.append(dict(step=step,**value))
                if method=='vmae_latent' and value['own_score']>latent_best:latent_best=value['own_score'];offset=trial
                if value['own_score']>best['own_score']:best=value
                save(ROOT/'status.json',dict(stage='search',case=case+1,total_cases=3,method=method,step=step+1,steps=20,best_own_score=best['own_score']))
            result=dict(case_uid=row['uid'],n_codons=len(dna)//3,method=method,original=original,best=best,trace=trace,
                attempts=20,unique_proposals=len({z['sequence_sha256'] for z in trace}),seconds=time.monotonic()-begun,
                changed_codons=sum(best['dna'][j:j+3]!=dna[j:j+3] for j in range(0,len(dna),3)))
            results.append(result);save(ROOT/'search_results.json',results)
    save(ROOT/'search_complete.json',dict(status='completed',cases=3,methods=2,total_proposals=120))

if __name__=='__main__':
    try:main()
    except BaseException as error:save(ROOT/'failure.json',dict(error=repr(error),unix=time.time()));raise
