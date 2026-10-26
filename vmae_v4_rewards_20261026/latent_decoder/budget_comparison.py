"""Paired exploratory B5 / RiboDecode-1.3.0 budget adaptations."""
import argparse
import ast
import hashlib
import json
import math
import time
from pathlib import Path

import torch
from torch.nn import functional as F

from .dual_space_ablation import CountedScorer, masks, project_joint, regularizers
from .latent_optimization import TranslationObjective, design_probabilities, state_hash, write_rows
from .model import AA_IDS, CODONS, CodonVMAE, collate_records
from .train import sha256

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'external/ribodecode_1_3_0/RiboDecode/train_optim_sim.py'


def project_budget(p, tokens, limit):
    """Keep the largest log-probability advantages over the original codon."""
    active, _ = masks(tokens)
    active[:, 0] = False
    proposed = p.argmax(-1)
    advantage = p.max(-1).values.clamp_min(1e-30).log() - p.gather(-1, tokens[..., None]).squeeze(-1).clamp_min(1e-30).log()
    eligible = active & (proposed != tokens)
    ranked = advantage.masked_fill(~eligible, -torch.inf).argsort(dim=-1, descending=True, stable=True)
    keep = torch.zeros_like(active)
    keep.scatter_(1, ranked[:, :limit], True)
    selected = torch.where(keep & eligible, proposed, tokens)
    assert bool(((selected != tokens).sum(-1) <= limit).all())
    assert torch.equal(AA_IDS.to(tokens.device)[selected], AA_IDS.to(tokens.device)[tokens])
    assert torch.equal(selected[:, [0, -1]], tokens[:, [0, -1]])
    return F.one_hot(selected, 64).float()


def native_generator(n, device):
    # Extract only the shipped class: importing its module copies files and starts optimization.
    tree = ast.parse(SOURCE.read_text(encoding='utf-8'))
    node = next(x for x in tree.body if isinstance(x, ast.ClassDef) and x.name == 'seq_codon_gen')
    namespace = dict(torch=torch, nn=torch.nn, F=F)
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(SOURCE), 'exec'), namespace)
    gen = namespace['seq_codon_gen'](dict(optim_batchsize=1, max_len=n * 3)).to(device)
    dataset = ast.parse(SOURCE.with_name('dataset.py').read_text(encoding='utf-8'))
    vocab = next(ast.literal_eval(x.value) for x in dataset.body if isinstance(x, ast.Assign)
                 and any(isinstance(t, ast.Name) and t.id == 'my_vocab' for t in x.targets))
    return gen, torch.tensor([vocab[c] for c in CODONS], device=device)


def summarize(output):
    rows = [json.loads(s) for s in (output / 'results.jsonl').read_text().splitlines()]
    summary = []
    for budget in (5, 10, 20):
        paired = {}
        for arm in ('B5', 'RiboDecode'):
            eligible = [r for r in rows if r['arm'] == arm and r['budget_percent'] <= budget]
            best = {}
            for r in eligible:
                key = (r['sequence_id'], r['seed'])
                if key not in best or r['best']['hard_score'] > best[key]['best']['hard_score']:
                    best[key] = r
            paired[arm] = best
        keys = sorted(set(paired['B5']) & set(paired['RiboDecode']))
        if keys:
            summary.append(dict(budget_percent=budget, paired_runs=len(keys),
                **{arm: sum(paired[arm][k]['best']['hard_score'] for k in keys) / len(keys) for arm in paired},
                mean_paired_difference=sum(paired['B5'][k]['best']['hard_score'] - paired['RiboDecode'][k]['best']['hard_score'] for k in keys) / len(keys)))
    (output / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    args = parser.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    old = json.loads((ROOT / 'results_eight_cds_comparison_20261020/plan.json').read_text())
    config = json.loads((ROOT / 'deliverables/b0_b5_frozen_20261020/reference/protocol.json').read_text())
    checkpoint_path = ROOT / 'deliverables/vmae_50ep_last_epoch50_20261019.pt'
    assert sha256(checkpoint_path) == config['checkpoint_sha256']
    assert sha256(ROOT / 'score_model/best_model.p') == config['scorer_sha256']
    assert sha256(ROOT / 'dataset/manifest.json') == config['dataset_manifest_sha256']
    assert sha256(ROOT / 'dataset/tokenizer.json') == config['tokenizer_sha256']
    ids = {r['sequence_id'] for r in old['records']}
    originals = {}
    for line in (ROOT / 'results_eight_cds_comparison_20261020/candidates.jsonl').read_text().splitlines():
        r = json.loads(line)
        if r['sequence_id'] in ids:
            originals[r['sequence_id']] = r['initial']['cds_dna']
    assert set(originals) == ids
    plan = dict(version='budget-comparison-exploratory-20261021', records=old['records'], seeds=old['seeds'],
        budgets_percent=[5, 10, 20], denominator='all CDS codons including fixed start and stop',
        limit='floor(percent*n_codons/100)', steps=80, arms=['B5', 'RiboDecode'],
        constraints='synonymous; fixed first/last codon; no GC or JS hard filter',
        projection='rank proposed mutations by log(p_new/p_original); retain top k; straight-through gradient',
        scoring='common frozen HEK293T scorer, mRNA_count=4.5, terminal stop omitted',
        queries_per_run=160, backwards_per_run=80, setup_queries_per_cds=1,
        b5='frozen joint latent/direct scales, learning rates and three regularizers; Gaussian latent RMS .05',
        ribodecode='shipped 65-channel generator, Gaussian initialization, AdamW lr .1 wd 1e-4, native target=100 loss and scheduler; common scorer; budget projection',
        archive='best legal candidate including original; high-budget summary also admits lower-budget winners within same CDS/seed/arm',
        limitations=['8 previously used development CDS; exploratory only', 'budget-adapted methods, not unchanged historical implementations',
                     'matched scorer calls/backwards, not FLOPs or wall time', 'best found under fixed search budget, not global optimum'],
        device=args.device, checkpoint_sha256=sha256(checkpoint_path), scorer_sha256=config['scorer_sha256'],
        native_source_sha256=sha256(SOURCE), source_sha256=sha256(Path(__file__)), frozen_b5_config=config)
    (output / 'plan.json').write_text(json.dumps(plan, indent=2), encoding='utf-8')
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    model = CodonVMAE(**checkpoint['config']['model']).eval().requires_grad_(False)
    model.load_state_dict(checkpoint['model'])
    model.to(args.device)
    scorer = CountedScorer(TranslationObjective(ROOT / 'score_model')).to(args.device).eval().requires_grad_(False)
    hashes = state_hash(model), state_hash(scorer)
    start = time.monotonic()
    for record in old['records']:
        identity = record['sequence_id']
        tokens = collate_records([dict(cds_dna=originals[identity])]).to(args.device)
        n = tokens.shape[1]
        active, support = masks(tokens)
        with torch.no_grad():
            mu, logvar = model.encode(tokens, tokens)
            sigma = (.5 * logvar).exp()
            baseline = float(scorer(F.one_hot(tokens, 64).float(), tokens))
        for seed in old['seeds']:
            for percent in (5, 10, 20):
                limit = n * percent // 100
                for arm in ('B5', 'RiboDecode'):
                    torch.manual_seed(seed)
                    run_start = time.monotonic()
                    before = scorer.forward_sequences
                    best = dict(step=0, hard_score=baseline, cds_dna=originals[identity], changed_codons=0)
                    if arm == 'B5':
                        key = int.from_bytes(hashlib.sha256(f'{seed}:{identity}'.encode()).digest()[:8], 'big') % (2**63 - 1)
                        noise = torch.randn(mu.shape[1:], generator=torch.Generator().manual_seed(key)).to(args.device)[None] * active[..., None]
                        latent = (noise / (noise.square().sum() / (active.sum() * noise.shape[-1])).sqrt() * .05).requires_grad_()
                        direct = torch.zeros((*tokens.shape, 64), device=args.device, requires_grad=True)
                        lr = config['lr'] / math.sqrt(2) * config['arm_lr_multipliers']['B5']
                        optimizer = torch.optim.Adam([dict(params=[latent], lr=lr), dict(params=[direct], lr=lr * config['direct_lr_factor'])])
                        def probabilities():
                            decoder = model.decode(mu + sigma * config['latent_scale'] * latent, tokens)
                            p = design_probabilities(decoder + config['direct_scale'] * direct * support, tokens)
                            terms = regularizers(p, design_probabilities(decoder, tokens), latent, direct, active, support)
                            return p, sum(config['regularization'][k] * v.mean() for k, v in terms.items())
                    else:
                        generator, order = native_generator(n, args.device)
                        native_mask = torch.zeros((1, n, 65), device=args.device)
                        native_mask[:, :, order] = torch.where(active[..., None], support, F.one_hot(tokens, 64).bool()).float()
                        native_mask[:, 0] = 0
                        native_mask[:, 0, order[tokens[0, 0]]] = 1
                        optimizer = torch.optim.AdamW(generator.parameters(), lr=.1, weight_decay=1e-4)
                        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[40, 60, 70, 85])
                        def probabilities():
                            return generator.generate(native_mask).transpose(1, 2)[:, :, order], 0.
                    print(json.dumps(dict(event='run_start', sequence_id=identity, seed=seed, budget_percent=percent, arm=arm, limit=limit)), flush=True)
                    for step in range(1, 81):
                        optimizer.zero_grad(set_to_none=True)
                        p, penalty = probabilities()
                        hard = project_budget(p, tokens, limit)
                        score = scorer(hard + p - p.detach(), tokens).mean()
                        loss = -score + penalty if arm == 'B5' else (100 - .2 * score.expm1()).abs() / 100
                        assert bool(torch.isfinite(loss))
                        loss.backward()
                        if arm == 'B5':
                            torch.nn.utils.clip_grad_norm_([latent, direct], 1., error_if_nonfinite=True)
                        optimizer.step()
                        if arm == 'B5':
                            project_joint(latent, direct, active, support, component_radius=16.)
                        else:
                            scheduler.step()
                        with torch.no_grad():
                            p, _ = probabilities()
                            hard = project_budget(p, tokens, limit)
                            value = float(scorer(hard, tokens))
                            selected = hard.argmax(-1)
                            candidate = dict(step=step, hard_score=value, cds_dna=''.join(CODONS[t] for t in selected[0].tolist()), changed_codons=int((selected != tokens).sum()))
                        write_rows(output / 'candidates.jsonl', [dict(sequence_id=identity, seed=seed, budget_percent=percent, arm=arm, **candidate)])
                        if value > best['hard_score']:
                            best = candidate
                        if step % 20 == 0:
                            print(json.dumps(dict(event='progress', arm=arm, step=step, best_gain=best['hard_score'] - baseline, seconds=time.monotonic()-run_start)), flush=True)
                    assert scorer.forward_sequences - before == 160
                    write_rows(output / 'results.jsonl', [dict(sequence_id=identity, seed=seed, budget_percent=percent, arm=arm,
                        baseline=baseline, limit=limit, best=best, gain=best['hard_score']-baseline, scorer_queries=160, backwards=80, seconds=time.monotonic()-run_start)])
                    summarize(output)
    assert hashes == (state_hash(model), state_hash(scorer))
    assert all(p.grad is None for p in list(model.parameters()) + list(scorer.parameters()))
    (output / 'completion_receipt.json').write_text(json.dumps(dict(completed=True, seconds=time.monotonic()-start,
        scorer_queries=scorer.forward_sequences, weights_unchanged=True)), encoding='utf-8')


if __name__ == '__main__':
    main()
