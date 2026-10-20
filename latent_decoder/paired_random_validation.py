"""Frozen B4/B5/B6 comparison with sequence-keyed, genuinely random starts."""
import argparse
import hashlib
import json
import platform
import time
from pathlib import Path

import torch

from .b6_reproducibility import frozen_config, verify_reference
from .dual_space_ablation import CountedScorer, optimize_batch
from .latent_optimization import TranslationObjective, state_hash
from .model import CodonVMAE
from .train import codon_frequency, load_data, sha256


SEEDS = [20261023, 20261024, 20261025]
BINS = [(30, 128, 13), (129, 256, 13), (257, 512, 13),
        (513, 1024, 13), (1025, 1500, 12)]


def select_records(records):
    selected = []
    for lo, hi, count in BINS:
        candidates = [r for r in records if lo <= r['n_codons'] <= hi]
        candidates.sort(key=lambda r: hashlib.sha256(
            ('paired-validation-20261020:' + r['sequence_id']).encode()).hexdigest())
        if len(candidates) < count: raise ValueError('Insufficient validation records')
        selected.extend(candidates[:count])
    assert len({r['sequence_id'] for r in selected}) == 64
    return selected


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--reference', default='deliverables/b0_b5_frozen_20261020')
    p.add_argument('--data', default='dataset')
    p.add_argument('--plan', required=True)
    p.add_argument('--prepare', action='store_true')
    p.add_argument('--checkpoint')
    p.add_argument('--scorer', default='score_model')
    p.add_argument('--output')
    p.add_argument('--evaluator-smoke-receipt')
    p.add_argument('--seed', type=int)
    p.add_argument('--arms', nargs='+', choices=['B7', 'B8'])
    p.add_argument('--handoff-config', help='Required B7/B8 development handoff configuration JSON')
    p.add_argument('--device', default='cuda', choices=['cpu', 'cuda'])
    args = p.parse_args()
    reference = Path(args.reference)
    manifest = verify_reference(reference)
    original, config = frozen_config(reference, ['B4', 'B5', 'B6'])
    if args.arms:
        assert len(set(args.arms)) == len(args.arms)
        original, config = frozen_config(reference, args.arms)
        if not args.handoff_config: p.error('--arms B7/B8 requires --handoff-config')
        config['handoff'] = json.loads(Path(args.handoff_config).read_text())
        h = config['handoff']
        assert h['method'] in ('kl', 'reencode') and h['temperature'] == 1.
        assert isinstance(h['steps'], int) and h['steps'] > 0 and h['lr'] > 0 and h['eta'] >= 0
    splits, _ = load_data(args.data, allow_development=True)
    plan_path = Path(args.plan)
    if args.prepare:
        selected = select_records(splits['val'])
        plan = dict(version='paired-random-validation-20261020', main_candidate='B5',
            controls=['B4', 'B6'], seeds=SEEDS, random_start_rms=.05,
            initialization='shared sequence-keyed Gaussian latent direction, exact RMS .05; direct zero',
            selection='64 hash-ranked val records in five predeclared length strata; no score filtering',
            length_strata=BINS, evaluation_ids=[r['sequence_id'] for r in selected],
            records=[{k:r[k] for k in ('sequence_id','n_codons','cluster_id')} for r in selected],
            dataset_manifest_sha256=sha256(Path(args.data)/'manifest.json'),
            reference_manifest_sha256=sha256(reference/'freeze_manifest.json'),
            checkpoint_sha256=manifest['checkpoint_sha256'], scorer_sha256=manifest['scorer_sha256'],
            batch_size=1, steps=80, scorer_forwards_per_arm_cds=240, backwards_per_arm_cds=80,
            common_setup_scorer_forwards_per_cds=6,
            batch_note='All new arms use batch 1 to cover full supported length range; old runs remain frozen',
            primary_contrasts=['B5_minus_B4','B5_minus_B6'],
            analysis='average each arm over all seeds within CDS, then paired CDS/cluster bootstrap; never select best seed',
            predictor_evaluation='RiboDecode, as requested; same optimizer scorer, not independent validation')
        with plan_path.open('x', encoding='utf-8') as f: json.dump(plan,f,indent=2)
        print(json.dumps(dict(plan=str(plan_path), records=len(selected), seeds=SEEDS)))
        return
    if not args.checkpoint or not args.output or args.seed is None:
        p.error('--checkpoint, --output and --seed are required for execution')
    plan = json.loads(plan_path.read_text())
    assert args.seed in plan['seeds']
    formal=plan['version']=='b456-formal-validation-20261020'
    if formal and args.arms:
        raise ValueError('Choose and freeze handoff on development data before creating a new formal B7/B8 plan')
    if not formal:assert plan['seeds']==SEEDS
    if formal:
        if not args.evaluator_smoke_receipt:p.error('Formal runs require a successful independent-evaluator smoke receipt')
        smoke=json.loads(Path(args.evaluator_smoke_receipt).read_text())
        assert smoke['completed'] and smoke['smoke'] and smoke['weights_unchanged']
        assert smoke['evaluator_lock_sha256']==plan['independent_evaluator_lock_sha256']
    assert plan['reference_manifest_sha256'] == sha256(reference/'freeze_manifest.json')
    assert plan['dataset_manifest_sha256'] == sha256(Path(args.data)/'manifest.json')
    assert plan['checkpoint_sha256'] == sha256(args.checkpoint)
    assert plan['scorer_sha256'] == sha256(Path(args.scorer)/'best_model.p')
    assert original['tokenizer_sha256'] == sha256(Path(args.data)/'tokenizer.json')
    config.update(batch_size=plan['batch_size'], random_start_seed=args.seed,
        random_start_rms=plan['random_start_rms'], timeout_seconds=24*3600,
        evaluation_ids=plan['evaluation_ids'])
    output = Path(args.output); output.mkdir(parents=True, exist_ok=False)
    (output/'protocol.json').write_text(json.dumps(config,indent=2))
    (output/'plan.json').write_text(json.dumps(plan,indent=2))
    torch.set_num_threads(4); torch.manual_seed(args.seed)
    if args.device == 'cuda':
        torch.cuda.manual_seed_all(args.seed)
        torch.cuda.reset_peak_memory_stats()
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    model = CodonVMAE(**checkpoint['config']['model']).eval().requires_grad_(False)
    model.load_state_dict(checkpoint['model']); model = model.to(args.device)
    scorer = CountedScorer(TranslationObjective(args.scorer)).eval().requires_grad_(False).to(args.device)
    hashes = (state_hash(model),state_hash(scorer))
    by_id = {r['sequence_id']:r for r in splits[plan.get('split','val')]}
    selected = [by_id[i] for i in plan['evaluation_ids']]
    if formal:
        from .formal_validation_plan import select_formal,SEEDS as FORMAL_SEEDS
        assert plan['seeds']==FORMAL_SEEDS
        expected=select_formal(splits['test'],plan['excluded_clusters'])
    else:expected=select_records(splits['val'])
    assert [r['sequence_id'] for r in expected] == plan['evaluation_ids']
    counts = codon_frequency(splits['train']).to(args.device)
    if args.device == 'cuda': torch.cuda.synchronize()
    start = time.monotonic(); results=[]
    for index, record in enumerate(selected):
        results.extend(optimize_batch(model,scorer,[record],counts,config,output,start))
        print(json.dumps(dict(event='cds_complete',seed=args.seed,completed=index+1,total=len(selected),
            elapsed_seconds=time.monotonic()-start)),flush=True)
    if args.device == 'cuda': torch.cuda.synchronize()
    assert hashes == (state_hash(model),state_hash(scorer))
    assert all(p.grad is None for p in list(model.parameters())+list(scorer.parameters()))
    extra = 2 if args.arms else 0
    assert scorer.forward_sequences == len(selected)*(6+len(config['run_arms'])*(240+extra))
    verify_reference(reference)
    receipt = dict(completed=True,seed=args.seed,records=len(selected),
        elapsed_seconds=time.monotonic()-start,scorer_forward_sequences=scorer.forward_sequences,
        common_setup_scorer_forwards=6*len(selected),
        peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated() if args.device=='cuda' else None,
        python=platform.python_version(),torch=torch.__version__,cuda=torch.version.cuda,
        gpu=torch.cuda.get_device_name(0) if args.device=='cuda' else None,
        plan_sha256=sha256(plan_path),source_sha256=sha256(Path(__file__)),
        optimizer_source_sha256=sha256(Path(__file__).with_name('dual_space_ablation.py')),
        handoff_config_sha256=sha256(args.handoff_config) if args.handoff_config else None,
        model_weights_unchanged=True,scorer_weights_unchanged=True,
        arm_metrics={arm:dict(records=len([r for r in results if r['arm']==arm]),
            mean_gain=sum(r['best_hard_gain'] for r in results if r['arm']==arm)/len(selected),
            search_seconds=sum(r['elapsed_seconds'] for r in results if r['arm']==arm),
            scorer_forwards=sum(r['scorer_forward_sequences'] for r in results if r['arm']==arm),
            backwards=sum(r['backward_sequence_evaluations'] for r in results if r['arm']==arm),
            decoder_evaluations=sum(r['decoder_sequence_evaluations'] for r in results if r['arm']==arm))
            for arm in config['run_arms']})
    (output/'completion_receipt.json').write_text(json.dumps(receipt,indent=2))
    print(json.dumps(receipt),flush=True)


if __name__ == '__main__': main()
