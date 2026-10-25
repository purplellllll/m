"""Matched translation-only searches; no training or checkpoint changes."""
import csv
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
from torch.nn import functional as F
from latent_decoder.analyze_random_validation import sequence_metrics, fold_candidate
from latent_decoder.budget_js_extension import save_json
from latent_decoder.dual_space_ablation import CountedScorer
from latent_decoder.latent_collapse_cpu import Pool
from latent_decoder.latent_information_probe import load_model
from latent_decoder.latent_optimization import TranslationObjective, state_hash
from latent_decoder.latent_search_probe_cpu import search
from latent_decoder.model import MASK, PAD, collate_records, mutable_positions
from latent_decoder.train import sha256
from latent_decoder.v2_train import corpus


def main():
    import fcntl
    out = ROOT/'small_translation_20261025'
    out.mkdir(exist_ok=False)
    with (out/'queue.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        status = dict(state='initializing', completed=0, total=18, goal_complete=False)
        save_json(out/'status.json', status)
        begun = time.monotonic()
        try:
            torch.set_num_threads(4)
            torch.set_num_interop_threads(1)
            torch.manual_seed(20261024)
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
            assert torch.cuda.is_available()
            original = ROOT.parent/'vmae_adjustment_20261024'
            small = ROOT.parent/'vmae_small_size_only_20261025'
            completion = json.loads((small/'fixed_split_complete.json').read_text())
            cases = json.loads((ROOT/'cases.json').read_text())
            prior = json.loads((ROOT/'results/results.json').read_text())
            full_prior = json.loads((ROOT/'full_data_search/results.json').read_text())
            full = json.loads((original/'full_data_followup/control/complete.json').read_text())
            checkpoints = dict(frozen=original/'priors/fold0_uniform_best.pt',
                               full_control=Path(full['selected']),
                               small=small/'results/fold_0/uniform/best.pt')
            assert sha256(checkpoints['small']) == completion['best_sha256']
            assert sha256(checkpoints['full_control']) == full['selected_sha256']
            assert len(cases['cases']) == 6
            dependencies = json.loads((ROOT/'source_manifest.json').read_text())
            for name, digest in dependencies.items():
                assert sha256(ROOT/name) == digest, name
            data = corpus(original/'data', 0)
            assert {c['record']['sequence_id'] for c in cases['cases']} <= {r['sequence_id'] for r in data['val']}
            calibration = [r for r in data['train'] if r['sequence_id'] in cases['calibration_ids']]
            assert len(calibration) == 12
            weights = json.loads((ROOT.parent/'vmae_transfer_assessment_20261024/runs/latent_research_20261024/search_probe_6cds/cai_weights.json').read_text())
            scorer = CountedScorer(TranslationObjective(ROOT/'score_model')).cuda().eval().requires_grad_(False)
            subset = json.loads((ROOT/'scorer_subset_receipt.json').read_text())
            assert sha256(ROOT/'canonical_condition.pt') == subset['canonical_condition_sha256']
            with torch.no_grad():
                scorer.scorer.condition.copy_(torch.load(ROOT/'canonical_condition.pt', map_location='cpu', weights_only=True).cuda())
            assert state_hash(scorer.scorer) == subset['model_state_hash']
            scorer_hash = state_hash(scorer)
            save_json(out/'plan.json', dict(
                checkpoint_sha256={a: sha256(p) for a, p in checkpoints.items()},
                source_sha256=sha256(Path(__file__)), dependencies=dependencies,
                cases_sha256=sha256(ROOT/'cases.json'), steps=80, learning_rate=.05,
                radius=8., temperature=1., seed=20261024, encoder='all mutable codons masked',
                edit_budget=None, js_penalty=0., js_hard_limit=None, direct_l2=0.,
                native_reference='Saved actual native return with 80 updates; not full defaults',
                current_gpu_rerun=list(checkpoints), outer_test_used=False,
                scope='Six reused validation CDS, exploratory translation-score comparison; no retraining'))
            results, comparison, fasta = [], [], []
            for arm, checkpoint in checkpoints.items():
                model = load_model(checkpoint, 'cuda')
                model_hash = state_hash(model)
                pool = Pool(model.latent_dim, covariance=True)
                with torch.no_grad():
                    for record in calibration:
                        tokens = collate_records([record]).cuda()
                        mu, _ = model.encode(tokens, tokens)
                        pool.add(mu.cpu(), tokens.masked_fill(~mutable_positions(tokens), PAD).cpu())
                weak = torch.tensor(pool.summary()['within_aa']['variance_by_dim']) <= .01
                save_json(out/(arm+'_train_geometry.json'), pool.summary())
                for case in cases['cases']:
                    record = case['record']
                    identity = record['sequence_id']
                    status.update(state='searching', arm=arm, sequence_id=identity)
                    save_json(out/'status.json', status)
                    destination = out/arm/identity
                    destination.mkdir(parents=True, exist_ok=False)
                    tokens = collate_records([record])
                    with torch.no_grad():
                        native = collate_records([dict(cds_dna=case['native_reference']['cds_dna'])]).cuda()
                        native_score = float(scorer(F.one_hot(native, 64).float(), native))
                    assert abs(native_score-case['native_reference']['hard_score']) < 2e-5
                    row = search(model, scorer, record, 80, weak, destination,
                                 arms=('latent_radius8',),
                                 encoder_tokens=tokens.masked_fill(mutable_positions(tokens), MASK))[0]
                    row.update(model_arm=arm, native_reference=case['native_reference'])
                    results.append(row)
                    previous = next((r for r in prior+full_prior if r['model_arm'] == arm and r['sequence_id'] == identity), None)
                    properties = dict(**sequence_metrics(row['best']['cds_dna'], weights), **fold_candidate(row['best']['cds_dna']))
                    summary = dict(model_arm=arm, sequence_id=identity, n_aa=row['n_aa'],
                        original_score=row['original_score'], initial_decoder_score=row['initial_decoder_hard_score'],
                        best_score=row['best']['hard_score'], native_score=native_score,
                        gain_from_original=row['gain'], delta_from_native=row['best']['hard_score']-native_score,
                        old_same_model_score=previous['best']['hard_score'] if previous else None,
                        search_seconds=row['seconds'], search_queries=row['search_scorer_queries'],
                        changed_codons=sum(x != y for x, y in zip(collate_records([dict(cds_dna=row['best']['cds_dna'])])[0].tolist(), tokens[0].tolist())),
                        **properties)
                    comparison.append(summary)
                    fasta.append(f">{arm}|aa={row['n_aa']}|hard={row['best']['hard_score']:.8f}\n{row['best']['cds_dna']}\n")
                    save_json(out/'results.json', results)
                    save_json(out/'comparison.json', comparison)
                    status['completed'] += 1
                    save_json(out/'status.json', status)
                    print(json.dumps(dict(event='case_complete', **summary)), flush=True)
                assert state_hash(model) == model_hash and all(p.grad is None for p in model.parameters())
                del model
                torch.cuda.empty_cache()
            assert state_hash(scorer) == scorer_hash and all(p.grad is None for p in scorer.parameters())
            assert len(comparison) == status['total'] == 18
            with (out/'comparison.csv').open('w', newline='') as handle:
                writer = csv.DictWriter(handle, fieldnames=list(comparison[0]))
                writer.writeheader()
                writer.writerows(comparison)
            (out/'candidates.fasta').write_text(''.join(fasta))
            status.update(state='completed', source_weights_unchanged=True, seconds=time.monotonic()-begun,
                          seq2ribo_assessed=False, outer_test_used=False)
        except BaseException as error:
            status.update(state='failed', error=repr(error))
            raise
        finally:
            save_json(out/'status.json', status)


if __name__ == '__main__':
    main()
