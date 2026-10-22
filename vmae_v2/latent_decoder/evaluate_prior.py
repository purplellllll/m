"""Paired masked reconstruction, frequency baseline and latent permutation audit."""
import argparse
import json
from pathlib import Path

import torch
from torch.nn import functional as F

from .diagnose_mask_plateau import choose
from .model import AA_IDS, CodonVMAE, corrupt, mutable_positions, synonymous_logits
from .train import batches, codon_frequency, load_data, sha256


def shuffle_latents(mu, tokens, selected, generator):
    # Keep AA and input visibility fixed so hidden targets never receive visible-target latents.
    shuffled = mu.clone()
    mutable = mutable_positions(tokens)
    aa = AA_IDS.to(tokens.device)[tokens]
    for amino_acid in range(20):
        for masked in (False, True):
            positions = mutable & (aa == amino_acid) & (selected == masked)
            values = mu[positions]
            if len(values) > 1:
                shuffled[positions] = values[torch.randperm(len(values), device=mu.device, generator=generator)]
    return shuffled


@torch.no_grad()
def evaluate_prior(model, records, counts, device, repeats=3):
    if repeats < 1:
        raise ValueError('At least one permutation is required')
    model.eval()
    mask_rng = torch.Generator(device=device).manual_seed(173)
    shuffle_rng = torch.Generator(device=device).manual_seed(9473)
    sums = dict(masked_mutable_ce=0., frequency_masked_ce=0.)
    shuffled_sums = [0.] * repeats
    correct = prior_correct = sites = 0
    counts = counts.to(device)
    for batch_index, tokens in enumerate(batches(records, 4)):
        tokens = tokens.to(device)
        inputs, selected = corrupt(tokens, .15, mask_rng)
        logits, mu, _ = model(inputs, tokens, sample=False)
        logits = synonymous_logits(logits, tokens)
        prior = synonymous_logits(counts.log().expand(*tokens.shape, 64), tokens)
        targets = tokens.clamp_max(63)
        for name, scores in [('masked_mutable_ce', logits), ('frequency_masked_ce', prior)]:
            ce = F.cross_entropy(scores.transpose(1, 2), targets, reduction='none')
            sums[name] += ce[selected].sum().item()
        correct += int((logits.argmax(-1)[selected] == tokens[selected]).sum())
        prior_correct += int((prior.argmax(-1)[selected] == tokens[selected]).sum())
        sites += int(selected.sum())
        for repeat in range(repeats):
            shuffled = shuffle_latents(mu, tokens, selected, shuffle_rng)
            scores = synonymous_logits(model.decode(shuffled, tokens), tokens)
            ce = F.cross_entropy(scores.transpose(1, 2), targets, reduction='none')
            shuffled_sums[repeat] += ce[selected].sum().item()
        if (batch_index + 1) % 8 == 0:
            print(json.dumps(dict(batches=batch_index + 1, masked_sites=sites)), flush=True)
    if not sites:
        raise ValueError('No masked mutable positions')
    result = {name: total / sites for name, total in sums.items()}
    result.update(masked_sites=sites, sequences=len(records), masked_mutable_accuracy=correct/sites,
                  frequency_masked_accuracy=prior_correct/sites,
                  shuffled_masked_ce_repeats=[total/sites for total in shuffled_sums])
    result['shuffled_masked_ce'] = sum(result['shuffled_masked_ce_repeats']) / repeats
    result['masked_ce_relative_gain'] = 1 - result['masked_mutable_ce']/result['frequency_masked_ce']
    result['shuffle_masked_ce_increase'] = result['shuffled_masked_ce'] - result['masked_mutable_ce']
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--data', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(output)
    torch.set_num_threads(4)
    checkpoint_hash = sha256(args.checkpoint)
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    model = CodonVMAE(**checkpoint['config']['model'])
    model.load_state_dict(checkpoint['model'])
    splits, _ = load_data(args.data, allow_development=True)
    records = choose(splits['val'], 256)
    print(json.dumps(dict(event='start', validation_records=len(records), device='cpu')), flush=True)
    result = evaluate_prior(model, records, codon_frequency(splits['train']), torch.device('cpu'))
    assert sha256(args.checkpoint) == checkpoint_hash
    result.update(checkpoint_sha256=checkpoint_hash, source_unchanged=True, device='cpu',
                  torch_version=torch.__version__, sample_ids=[r['sequence_id'] for r in records],
                  mask_seed=173, shuffle_seed=9473, mask_probability=.15, batch_size=4,
                  shuffle_scope='Within batch and AA, separately for masked and visible mutable sites',
                  evaluation_scope='256 validation CDS, all length ranges; three permutations, one mask')
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k!='sample_ids'}), flush=True)


if __name__ == '__main__':
    main()
