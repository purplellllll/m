"""Compare conditional posterior samples with standard-normal prior decoding."""
import argparse,json,time
from pathlib import Path
import numpy as np
import torch

from .diagnose_mask_plateau import choose
from .latent_optimization import design_probabilities
from .model import AA_IDS,CODONS,PAD,SYNONYMS,CodonVMAE,collate_records,mutable_positions
from .train import codon_frequency,load_data,sha256


@torch.no_grad()
def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',required=True)
    parser.add_argument('--data',required=True)
    parser.add_argument('--output',required=True)
    args=parser.parse_args()
    out=Path(args.output)
    out.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(2)
    start=time.monotonic()
    digest=sha256(args.checkpoint)
    checkpoint=torch.load(args.checkpoint,map_location='cpu',weights_only=False)
    model=CodonVMAE(**checkpoint['config']['model']).eval().requires_grad_(False)
    model.load_state_dict(checkpoint['model'])
    splits,_=load_data(args.data,allow_development=True)
    records=choose(splits['val'],16,short=True)
    counts=codon_frequency(splits['train'])
    rng=torch.Generator().manual_seed(56173)
    rows=[]
    for begin in range(0,len(records),4):
        group=records[begin:begin+4]
        tokens=collate_records(group)
        active=mutable_positions(tokens)
        valid=tokens<PAD
        mu,logvar=model.encode(tokens,tokens)
        prior=counts[None,None,:]*SYNONYMS[tokens]
        prior/=prior.sum(-1,keepdim=True)
        for mode in ['original','posterior','standard_normal']:
            for draw in range(1 if mode=='original' else 8):
                if time.monotonic()-start>1200:
                    raise TimeoutError('Sampling audit exceeded 20-minute limit')
                if mode=='original':
                    predicted=tokens.clamp_max(63)
                else:
                    noise=torch.randn(mu.shape,generator=rng)
                    z=mu+torch.exp(.5*logvar)*noise if mode=='posterior' else noise
                    z=z*valid[...,None]
                    probabilities=design_probabilities(model.decode(z,tokens),tokens)
                    predicted=probabilities.argmax(-1)
                    assert torch.equal(AA_IDS[predicted][valid],AA_IDS[tokens][valid])
                    assert torch.equal(predicted[valid&~active],tokens[valid&~active])
                nll=-prior.gather(-1,predicted[...,None]).squeeze(-1).clamp_min(1e-12).log()
                for index,record in enumerate(group):
                    sequence=''.join(CODONS[x] for x in predicted[index,valid[index]].tolist())
                    row=dict(sequence_id=record['sequence_id'],mode=mode,draw=draw,cds_dna=sequence,
                        gc_fraction=(sequence.count('G')+sequence.count('C'))/len(sequence),
                        frequency_nll=float(nll[index,active[index]].mean()),
                        mutable_hamming=float((predicted[index,active[index]]!=tokens[index,active[index]]).float().mean()),
                        protein_preserved=True)
                    rows.append(row)
        print(json.dumps(dict(completed=begin+len(group))),flush=True)
    (out/'samples.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows),encoding='utf-8')
    summary={}
    for mode in ['original','posterior','standard_normal']:
        data=[r for r in rows if r['mode']==mode]
        unique=[len({r['cds_dna'] for r in data if r['sequence_id']==record['sequence_id']}) for record in records]
        summary[mode]=dict(samples=len(data),median_unique=float(np.median(unique)),unique_per_cds=unique,
            mean_gc=float(np.mean([r['gc_fraction'] for r in data])),
            gc_range=[min(r['gc_fraction'] for r in data),max(r['gc_fraction'] for r in data)],
            mean_frequency_nll=float(np.mean([r['frequency_nll'] for r in data])),
            mean_mutable_hamming=float(np.mean([r['mutable_hamming'] for r in data])))
    assert sha256(args.checkpoint)==digest
    result=dict(summary=summary,checkpoint_sha256=digest,source_unchanged=True,
                seed=56173,records=16,samples_per_distribution=8,device='cpu')
    (out/'summary.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result),flush=True)


if __name__=='__main__':
    main()
