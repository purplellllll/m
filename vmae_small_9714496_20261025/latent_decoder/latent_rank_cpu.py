"""Covariance geometry supplement: AU alone cannot locate loss of independent directions."""
import argparse
import gc
import gzip
import json
from pathlib import Path

import torch

from .budget_js_extension import save_json
from .latent_collapse_cpu import Pool, emit, masked
from .length_diagnostics import length_group
from .model import CodonVMAE, collate_records
from .train import sha256


def within_geometry(pool):
    within = pool.cross/pool.n-(pool.aa_total.T/pool.aa_n.clamp_min(1))@pool.aa_total/pool.n
    var = within.diag().clamp_min(0)
    scale = var.sqrt().clamp_min(1e-12)
    corr = within/scale[:, None]/scale[None, :]
    corr[var<=1e-12, :] = 0
    corr[:, var<=1e-12] = 0
    eigen = torch.linalg.eigvalsh(within).clamp_min(0).flip(0)
    return dict(covariance_participation=float(var.sum().square()/within.square().sum().clamp_min(1e-30)),
        correlation_participation=float(corr.diag().sum().square()/corr.square().sum().clamp_min(1e-30)),
        dimensions_95pct=int((eigen.cumsum(0)<.95*eigen.sum()).sum())+int(eigen.sum()>0))


@torch.no_grad()
def measure_rank(model, records):
    pools, captures, handles = {}, {}, []
    def capture(name):
        def hook(module, args, result):
            captures[name] = result.detach()
        return hook
    def before(module, args):
        captures['encoder_input'] = args[0].detach()
    handles.append(model.encoder.register_forward_pre_hook(before))
    for i, layer in enumerate(model.encoder.layers):
        handles.append(layer.register_forward_hook(capture(f'encoder_block_{i+1}')))
    for name, layer in [('encoder_norm', model.encoder.norm), ('mu', model.mu), ('decoder_projection', model.from_latent)]:
        handles.append(layer.register_forward_hook(capture(name)))
    try:
        for i, record in enumerate(records):
            tokens = collate_records([record])
            for mode, inputs in [('clean',tokens), ('mask173',masked(tokens,record['sequence_id'],173)[0])]:
                captures.clear()
                model(inputs,tokens,sample=False)
                for stage, values in captures.items():
                    key='/'.join((record['panel'],length_group(record['n_codons']-1),mode,stage))
                    if key not in pools:
                        pools[key]=Pool(values.shape[-1],covariance=True)
                    pools[key].add(values,tokens)
            emit('rank_sequence',index=i+1,total=len(records))
    finally:
        for handle in handles:
            handle.remove()
    return {key:dict(**pool.summary(),within_aa_geometry=within_geometry(pool)) for key,pool in pools.items()}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--checkpoint',action='append',required=True)
    args=parser.parse_args()
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    plan=json.loads((args.output/'plan.json').read_text())
    chosen={r['sequence_id']:r for r in plan['selected']}
    records=[]
    with gzip.open(args.data/'corpus.jsonl.gz','rt') as f:
        for line in f:
            r=json.loads(line)
            if r['sequence_id'] in chosen:
                r.update(chosen[r['sequence_id']]); records.append(r)
    assert len(records)==24
    for entry in args.checkpoint:
        name,path=entry.split('=',1)
        assert sha256(path)==plan['checkpoints'][name]
        target=args.output/name/'layer_rank.json'
        if target.exists():
            continue
        state=torch.load(path,map_location='cpu',weights_only=False,mmap=True)
        model=CodonVMAE(**state['config']['model']).eval().requires_grad_(False)
        model.load_state_dict(state['model'])
        del state
        result=measure_rank(model,records)
        save_json(target,dict(source_sha256=sha256(Path(__file__)),device='cpu',layers=result))
        del model,result
        gc.collect()
    emit('rank_completed')


if __name__=='__main__':
    main()
