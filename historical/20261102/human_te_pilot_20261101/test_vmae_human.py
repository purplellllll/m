"""Small end-to-end VMAE -> experimental human TE scorer check on development DNA."""
from pathlib import Path
import hashlib
import json
import sys
import torch
sys.path.insert(0,str(Path(__file__).resolve().parent))
from human_score import HumanTEScorer
from run_pilot import ROOT,WORK,save,sha
from latent_decoder.model import CodonVMAE,collate_records,mutable_positions,synonymous_logits,MASK,AA_IDS
from latent_decoder.tokenizer import CODONS

torch.set_num_threads(2);torch.manual_seed(20261101);torch.backends.mha.set_fastpath_enabled(False)
scorer=HumanTEScorer('cpu')
rows=json.loads((ROOT/'selected_development_rows.json').read_text(encoding='utf-8'))
row=min([r for r in rows if r['split']=='val' and r['dna'].startswith('ATG')],key=lambda r:len(r['dna']))
path=WORK/'runs/vmae_tiny_size_only_20261025/checkpoints/fold0_uniform.pt';payload=torch.load(path,map_location='cpu',weights_only=False)
model=CodonVMAE(**payload['config']['model']).eval().requires_grad_(False);model.load_state_dict(payload['model'],strict=True)
original=scorer(row['dna']);tokens=collate_records([dict(cds_dna=row['dna'])]);active=mutable_positions(tokens);candidates=[]
with torch.no_grad():
    mu,lv=model.encode(tokens.masked_fill(active,MASK),tokens)
    for step in range(3):
        offset=torch.zeros_like(mu) if step==0 else torch.randn_like(mu)*active[...,None]*.5
        ids=synonymous_logits(model.decode(mu+(.5*lv).exp()*offset,tokens),tokens).argmax(-1);ids[~active]=tokens[~active]
        assert torch.equal(AA_IDS[ids],AA_IDS[tokens])
        dna=''.join(CODONS[int(i)] for i in ids[0]);assert dna[:3]==row['dna'][:3] and dna[-3:]==row['dna'][-3:]
        candidates.append(dict(step=step,**scorer(dna)))
best=max([original]+candidates,key=lambda r:r['own_score'])
save(ROOT/'human_vmae_integration_check.json',dict(status='passed',task='human_HEK293_TE',generator_sha256=sha(path),
    original=original,candidates=candidates,best=best,amino_acid_and_terminal_constraints_verified=True,
    scope='One shortest development CDS,3 proposals, lambda1; interface verification, not biological efficacy or full search'))
print('PASS: original V3 -> synonymous DNA -> trained human TE head; 3 proposals',original['own_score'],best['own_score'])
