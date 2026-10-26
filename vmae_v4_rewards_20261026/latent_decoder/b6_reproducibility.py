"""Add B6 using a hash-frozen B0-B5 protocol; never recalibrate or retune."""
import argparse
import copy
import json
import platform
import shutil
import time
from pathlib import Path

import numpy as np
import torch

from .dual_space_ablation import ARMS, CountedScorer, optimize_batch
from .latent_optimization import TranslationObjective, state_hash
from .model import CodonVMAE
from .train import codon_frequency, load_data, sha256


def frozen_config(reference, selected_arms):
    original = json.loads((reference / 'reference/protocol.json').read_text())
    if not original['output_space_budget'] or original['steps'] != 80 or original['batch_size'] != 4:
        raise ValueError('Expected the frozen 80-step, batch-4 v2 protocol')
    config = copy.deepcopy(original)
    config['arm_lr_multipliers']['B6'] = original['arm_lr_multipliers']['B4']
    assert config['arm_lr_multipliers']['B6'] == original['arm_lr_multipliers']['B5']
    config['run_arms'] = selected_arms
    for arm, label in [('B7', 'Direct -> Latent'), ('B8', 'Direct -> Latent + Regularization')]:
        if arm in selected_arms:
            config['arm_lr_multipliers'][arm] = original['arm_lr_multipliers']['B4']
            config['arms'][arm] = label
    config['arms']['B6'] = 'Latent -> Direct + Regularization'
    for key, value in original.items():
        if key not in ('arms', 'arm_lr_multipliers'):
            assert config[key] == value
    config['b6_extension'] = dict(schedule='B4: 40 latent, freeze z, 40 direct',
        regularization='B5 coefficients in both phases; direct L2/decoder KL are zero in latent phase',
        retuning=False, original_protocol_preserved=True)
    return original, config


def verify_reference(reference):
    manifest = json.loads((reference / 'freeze_manifest.json').read_text())
    for name, expected in manifest['files_sha256'].items():
        assert sha256(reference / name) == expected, name
    for name in ['model.py', 'train.py', 'tokenizer.py', 'latent_optimization.py',
                 'diagnose_mask_plateau.py', 'replay_single.py']:
        expected = manifest['files_sha256']['source/latent_decoder/' + name]
        assert sha256(Path(__file__).parent / name) == expected, name
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--data', required=True)
    parser.add_argument('--scorer', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--mode', choices=['b6', 'repeat'], default='b6')
    parser.add_argument('--seed', type=int, default=20261020)
    parser.add_argument('--device', choices=['cuda', 'cpu'], default='cuda')
    args = parser.parse_args()
    reference = Path(args.reference)
    manifest = verify_reference(reference)
    selected_arms = ['B6'] if args.mode == 'b6' else list(ARMS)[1:] + ['B6']
    original, config = frozen_config(reference, selected_arms)
    assert config['checkpoint_sha256'] == sha256(args.checkpoint) == manifest['checkpoint_sha256']
    assert config['scorer_sha256'] == sha256(Path(args.scorer) / 'best_model.p') == manifest['scorer_sha256']
    assert config['dataset_manifest_sha256'] == sha256(Path(args.data) / 'manifest.json')
    assert config['tokenizer_sha256'] == sha256(Path(args.data) / 'tokenizer.json')
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(reference / 'reference/protocol.json', output / 'reference_protocol.json')
    (output / 'protocol.json').write_text(json.dumps(config, indent=2) + '\n')
    torch.set_num_threads(4)
    torch.manual_seed(args.seed)
    if args.device == 'cuda': torch.cuda.manual_seed_all(args.seed)
    start = time.monotonic()
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    model = CodonVMAE(**checkpoint['config']['model']).eval().requires_grad_(False)
    model.load_state_dict(checkpoint['model'])
    model = model.to(args.device)
    scorer = CountedScorer(TranslationObjective(args.scorer)).eval().requires_grad_(False).to(args.device)
    frozen_hashes = (state_hash(model), state_hash(scorer))
    splits, _ = load_data(args.data, allow_development=True)
    by_id = {r['sequence_id']: r for r in splits['val']}
    records = [by_id[identity] for identity in original['evaluation_ids']]
    counts = codon_frequency(splits['train']).to(args.device)
    print(json.dumps(dict(event='start', mode=args.mode, seed=args.seed, records=len(records),
                          run_arms=selected_arms, steps=config['steps'], retuning=False)), flush=True)
    results = []
    for begin in range(0, len(records), config['batch_size']):
        results += optimize_batch(model, scorer, records[begin:begin + config['batch_size']], counts, config, output, start)
    assert frozen_hashes == (state_hash(model), state_hash(scorer))
    assert all(p.grad is None for p in list(model.parameters()) + list(scorer.parameters()))
    assert config['checkpoint_sha256'] == sha256(args.checkpoint)
    assert config['scorer_sha256'] == sha256(Path(args.scorer) / 'best_model.p')
    verify_reference(reference)
    metrics = {}
    for arm in ['B0'] + selected_arms:
        rows = [r for r in results if r['arm'] == arm]
        metrics[arm] = dict(records=len(rows), mean_best_gain=float(np.mean([r['best_hard_gain'] for r in rows])),
            improved=sum(r['best_hard_gain'] > 1e-5 for r in rows), final_feasible=sum(r['final']['feasible'] for r in rows),
            mean_hamming=float(np.mean([r['best']['mutable_hamming'] for r in rows])),
            mean_decoder_kl=float(np.mean([r['best'].get('decoder_kl', 0.) for r in rows])),
            mean_direct_l2=float(np.mean([r['best'].get('direct_l2', 0.) for r in rows])),
            scorer_forward_sequences=sum(r['scorer_forward_sequences'] for r in rows),
            backward_sequence_evaluations=sum(r['backward_sequence_evaluations'] for r in rows))
    assert all(metrics[a]['scorer_forward_sequences'] == 3840 and metrics[a]['backward_sequence_evaluations'] == 1280 for a in selected_arms)
    receipt = dict(mode=args.mode, seed=args.seed, interpretation='technical repeat; no random initialization of design variables',
        arms=metrics, records=len(records), steps=config['steps'], elapsed_seconds=time.monotonic()-start,
        model_and_scorer_unchanged=True, reference_unchanged=True,
        freeze_manifest_sha256=sha256(reference / 'freeze_manifest.json'),
        python=platform.python_version(), torch=torch.__version__, cuda=torch.version.cuda,
        gpu=torch.cuda.get_device_name(0) if args.device == 'cuda' else None,
        cudnn=torch.backends.cudnn.version(), cudnn_deterministic=torch.backends.cudnn.deterministic,
        cudnn_benchmark=torch.backends.cudnn.benchmark,
        matmul_tf32=torch.backends.cuda.matmul.allow_tf32,
        cudnn_tf32=torch.backends.cudnn.allow_tf32,
        source_sha256={p.name:sha256(p) for p in [Path(__file__), Path(__file__).with_name('dual_space_ablation.py')]})
    (output / 'summary.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps(receipt), flush=True)


if __name__ == '__main__':
    main()
