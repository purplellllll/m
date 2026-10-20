"""Paired context ablations on fixed masked targets; never updates model weights."""
import argparse
import json
import time
from pathlib import Path

import torch
from torch.nn import functional as F

from .model import AA_IDS, MASK, CodonVMAE, corrupt, mutable_positions, synonymous_logits
from .train import batches, codon_frequency, load_data, sha256


def context_inputs(tokens, inputs, selected, counts, generator):
    visible = mutable_positions(tokens) & ~selected
    permuted = inputs.clone()
    aa = AA_IDS.to(tokens.device)[tokens]
    for row in range(len(tokens)):
        for amino_acid in range(20):
            at = visible[row] & (aa[row] == amino_acid)
            values = inputs[row, at]
            if len(values) > 1:
                permuted[row, at] = values[torch.randperm(len(values), generator=generator)]
    modes = synonymous_logits(counts.log().expand(*tokens.shape, 64), tokens).argmax(-1)
    return dict(normal=inputs, shuffled_visible=permuted,
                frequency_mode_visible=torch.where(visible, modes, inputs),
                no_mutable_context=tokens.masked_fill(mutable_positions(tokens), MASK))


def select_stratified(records):
    ordered = sorted(records, key=lambda r: (r['n_codons'], r['sequence_id']))
    selected = []
    for quartile in range(4):
        bucket = ordered[len(ordered)*quartile//4:len(ordered)*(quartile+1)//4]
        selected.extend(sorted(bucket, key=lambda r:r['sequence_id'])[:32])
    return selected


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--data', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    start = time.monotonic()
    source_hash = sha256(args.checkpoint)
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    model = CodonVMAE(**checkpoint['config']['model']).eval().requires_grad_(False)
    model.load_state_dict(checkpoint['model'])
    splits, _ = load_data(args.data, allow_development=True)
    records = select_stratified(splits['val'])
    by_cds = {tuple(r['codon_ids']):r['sequence_id'] for r in records}
    counts = codon_frequency(splits['train'])
    mask_rng = torch.Generator().manual_seed(20261019)
    shuffle_rng = torch.Generator().manual_seed(38173)
    results = []
    for batch_index, tokens in enumerate(batches(records, 4)):
        if time.monotonic()-start > 1200:
            raise TimeoutError('Context audit exceeded 20-minute limit')
        inputs, selected = corrupt(tokens, .15, mask_rng)
        conditions = context_inputs(tokens, inputs, selected, counts, shuffle_rng)
        conditions['frequency_baseline'] = None
        rows = [dict(sequence_id=by_cds[tuple(t[t<64].tolist())], n_codons=int((t<64).sum()),
                     masked_sites=int(at.sum()), conditions={}) for t, at in zip(tokens,selected)]
        for name, corrupted in conditions.items():
            raw = counts.log().expand(*tokens.shape, 64) if corrupted is None else model(corrupted,tokens,sample=False)[0]
            logits = synonymous_logits(raw,tokens)
            ce = F.cross_entropy(logits.transpose(1,2),tokens.clamp_max(63),reduction='none')
            for row, at in enumerate(selected):
                rows[row]['conditions'][name] = dict(ce_sum=ce[row,at].sum().item(),
                    correct=int((logits[row].argmax(-1)[at]==tokens[row,at]).sum()))
        results.extend(rows)
        with (out/'records.jsonl').open('a',encoding='utf-8') as handle:
            for row in rows:
                handle.write(json.dumps(row)+'\n')
        print(json.dumps(dict(batch=batch_index+1, records=len(results))),flush=True)
    sites = sum(r['masked_sites'] for r in results)
    summary = {}
    for name in results[0]['conditions']:
        summary[name] = dict(ce=sum(r['conditions'][name]['ce_sum'] for r in results)/sites,
                            accuracy=sum(r['conditions'][name]['correct'] for r in results)/sites)
    assert source_hash == sha256(args.checkpoint)
    payload = dict(metrics=summary, sequences=len(results), masked_sites=sites,
                   checkpoint_sha256=source_hash, source_unchanged=True, mask_seed=20261019,
                   shuffle_seed=38173, length_stratified=True, device='cpu', torch_version=torch.__version__)
    (out/'summary.json').write_text(json.dumps(payload,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(payload),flush=True)


if __name__ == '__main__':
    main()
