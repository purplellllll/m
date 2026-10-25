"""Frozen inner-fold search diagnosis; raw metrics do not establish five-objective success."""
import argparse
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

import torch
from torch.nn import functional as F

from .analyze_random_validation import sequence_metrics, fold_candidate
from .budget_js_extension import save_json
from .dual_space_ablation import CountedScorer, center_direct, masks
from .latent_collapse_cpu import Pool
from .latent_optimization import TranslationObjective, design_probabilities, state_hash
from .length_diagnostics import gradient_summary
from .model import AA, AA_IDS, CODONS, MASK, PAD, CodonVMAE, collate_records
from .train import codon_frequency, sha256
from .v2_data import pick
from .v2_train import corpus


ARMS = ('latent_radius025', 'latent_radius8', 'latent_expand', 'decoder_hidden', 'direct_logits')
ROOT = Path(__file__).resolve().parents[1]


def latent_delta(value, basis=None):
    return value if basis is None else value@basis.T


def project(value, active, radius, basis=None):
    with torch.no_grad():
        value.mul_(active[..., None])
        rms = latent_delta(value,basis)[active].square().mean().sqrt()
        value.mul_((radius/rms.clamp_min(1e-12)).clamp_max(1.))


def expanded_radius(radius, stalled, soft_gain):
    # Test the hard switching threshold when soft progress has not produced hard progress.
    return min(8.,2*radius) if stalled >= 5 and soft_gain > 1e-6 else radius


def shifted_logits(model, tokens, mu, sigma, base_logits, base_hidden, variable, arm, basis=None):
    if arm.startswith('latent_'):
        return model.decode(mu+sigma*latent_delta(variable,basis), tokens)
    if arm == 'decoder_hidden':
        return model.output(model.decoder(base_hidden+variable, tokens == PAD))
    assert arm == 'direct_logits'
    return base_logits+variable


def candidate(probabilities, tokens):
    ids = probabilities.argmax(-1)
    valid = tokens < PAD
    aa_ids = AA_IDS.to(tokens.device)
    assert torch.equal(aa_ids[ids[valid]], aa_ids[tokens[valid]])
    assert torch.equal(ids[:, 0], tokens[:, 0]) and torch.equal(ids[:, -1], tokens[:, -1])
    return ''.join(CODONS[i] for i in ids[0, valid[0]].tolist())


def validate_native(original, native):
    before=collate_records([dict(cds_dna=original)])
    after=collate_records([dict(cds_dna=native)])
    assert torch.equal(AA_IDS[before],AA_IDS[after])
    assert original[:3]==native[:3]
    return dict(protein_preserved=True,start_preserved=True,stop_preserved=original[-3:]==native[-3:])


def search(model, scorer, record, steps, weak, destination, resume=False, *, arms=None, latent_basis=None, temperature=1., encoder_tokens=None):
    assert math.isfinite(temperature) and temperature>0
    arms=ARMS if arms is None else arms
    assert arms and set(arms)<=set(ARMS)
    if latent_basis is not None:
        assert latent_basis.ndim==2 and latent_basis.shape[0]==model.latent_dim
        assert 0<latent_basis.shape[1]<=model.latent_dim and torch.isfinite(latent_basis).all()
        assert all(arm.startswith('latent_') for arm in arms)
    device = next(model.parameters()).device
    tokens = collate_records([record]).to(device)
    weak = weak.to(device)
    latent_basis = latent_basis.to(device) if latent_basis is not None else None
    gain = getattr(model, 'recipe', {}).get('gain', 1.)
    active, support = masks(tokens)
    assert active.any(), 'Probe requires a CDS with at least one synonymous choice'
    encoder_tokens=tokens if encoder_tokens is None else encoder_tokens.to(device)
    assert encoder_tokens.shape==tokens.shape
    assert ((encoder_tokens==tokens)|((encoder_tokens==MASK)&active)).all()
    with torch.no_grad():
        mu, lv = model.encode(encoder_tokens, tokens)
        sigma = (.5*lv).exp()
        base_logits = model.decode(mu, tokens)
        base_hidden = model.from_latent(mu*gain)+model.aa_embedding(model.aa_ids[tokens])
        natural_p = design_probabilities(base_logits, tokens)
        initial_p = design_probabilities(base_logits/temperature, tokens)
        original_score = float(scorer(F.one_hot(tokens, 64).float(), tokens))
        initial_score = float(scorer(initial_p, tokens))
        initial_hard = float(scorer(F.one_hot(initial_p.argmax(-1), 64).float(), tokens))
    baseline = dict(cds_dna=record['cds_dna'], hard_score=original_score, step=-1)
    recon = candidate(initial_p, tokens)
    results = []
    for arm in arms:
        if resume and (destination/f'{arm}.json').exists():
            saved=json.loads((destination/f'{arm}.json').read_text())
            assert saved['sequence_id']==record['sequence_id'] and saved['steps']==steps
            results.append(saved)
            continue
        begun = time.monotonic()
        width = model.latent_dim if arm.startswith('latent_') else (
            model.mu.in_features if arm == 'decoder_hidden' else 64)
        if latent_basis is not None:width=latent_basis.shape[1]
        variable = torch.zeros((*tokens.shape, width), device=device, requires_grad=True)
        optimizer = torch.optim.Adam([variable], lr=.05)
        best = dict(cds_dna=recon, hard_score=initial_hard, step=0)
        archive={record['cds_dna']:dict(baseline)}
        archive.setdefault(recon,dict(best))
        best_variable = variable.detach().clone()
        unique, first_change, last_gradient = {recon}, None, None
        radius = .25 if arm in ('latent_radius025','latent_expand') else 8.
        stalled, last_adjustment_soft = 0, initial_score
        before = scorer.forward_sequences
        trace = []
        for step in range(1, steps+1):
            optimizer.zero_grad(set_to_none=True)
            logits = shifted_logits(model, tokens, mu, sigma, base_logits, base_hidden, variable, arm, latent_basis)
            probabilities = design_probabilities(logits/temperature, tokens)
            hard = F.one_hot(probabilities.argmax(-1), 64).float()
            score = scorer(hard+probabilities-probabilities.detach(), tokens).mean()
            gradient, = torch.autograd.grad(-score, variable)
            assert torch.isfinite(gradient).all()
            variable.grad = gradient
            last_gradient = gradient_summary(gradient[active])
            torch.nn.utils.clip_grad_norm_([variable], 1., error_if_nonfinite=True)
            optimizer.step()
            if arm == 'direct_logits':
                with torch.no_grad():
                    variable.copy_(center_direct(variable, support))
            step_radius = radius
            project(variable, active, step_radius, latent_basis)
            with torch.no_grad():
                logits = shifted_logits(model, tokens, mu, sigma, base_logits, base_hidden, variable, arm, latent_basis)
                probabilities = design_probabilities(logits/temperature, tokens)
                dna = candidate(probabilities, tokens)
                hard_score = float(scorer(F.one_hot(probabilities.argmax(-1), 64).float(), tokens))
                soft_score = float(scorer(probabilities, tokens))
                assert math.isfinite(hard_score) and math.isfinite(soft_score)
                archive.setdefault(dna,dict(cds_dna=dna,hard_score=hard_score,step=step))
                unique.add(dna)
                selected = probabilities.argmax(-1)
                edits = int((selected != tokens).sum())
                if first_change is None and dna != recon:
                    first_change = step
                if hard_score > best['hard_score']:
                    best = dict(cds_dna=dna, hard_score=hard_score, step=step)
                    best_variable = variable.detach().clone()
                    stalled = 0
                else:
                    stalled += 1
                if arm == 'latent_expand':
                    updated = expanded_radius(radius,stalled,soft_score-last_adjustment_soft)
                    if updated != radius:
                        stalled, last_adjustment_soft = 0, soft_score
                    radius = updated
                mapped = latent_delta(variable,latent_basis)
                dz = sigma*mapped if arm.startswith('latent_') else None
                response = F.linear(dz, model.from_latent.weight)*gain if dz is not None else (
                    variable if arm == 'decoder_hidden' else None)
                effective_logits = center_direct(logits-base_logits, support)
                row = dict(step=step, hard_score=hard_score, soft_score=soft_score,
                    project_radius=step_radius, next_radius=radius,
                    soft_hard_gap=soft_score-hard_score, changed_codons=edits,
                    parameter_rms=float(mapped[active].square().mean().sqrt()),
                    decoder_input_displacement_rms=float(response[active].square().mean().sqrt()) if response is not None else None,
                    centered_supported_logit_displacement_rms=float(effective_logits[support].square().mean().sqrt()),
                    gradient_dims99=last_gradient['dims_for_99pct_gradient_energy'])
                row['gradient_rms']=float(gradient[active].square().mean().sqrt())
                trace.append(row)
            if step == 1 or step % 10 == 0:
                print(json.dumps(dict(event='search', sequence_id=record['sequence_id'], arm=arm,
                    **row, seconds=time.monotonic()-begun)), flush=True)
        queries = scorer.forward_sequences-before
        assert queries == 3*steps
        interventions = {}
        if arm.startswith('latent_'):
            with torch.no_grad():
                for label, channels in (('reset_weak', weak), ('reset_strong', ~weak)):
                    altered = latent_delta(best_variable,latent_basis).clone(); altered[..., channels] = 0.
                    p = design_probabilities(model.decode(mu+sigma*altered, tokens)/temperature, tokens)
                    value = float(scorer(F.one_hot(p.argmax(-1), 64).float(), tokens))
                    interventions[label] = dict(dimensions=int(channels.sum()), hard_score=value,
                        delta_from_best_search=value-best['hard_score'])
        winner = best if best['hard_score'] > original_score else baseline
        result = dict(sequence_id=record['sequence_id'], n_aa=record['n_codons']-1, arm=arm,
            steps=steps, search_scorer_queries=queries, intervention_queries=len(interventions),
            original_score=original_score, initial_decoder_hard_score=initial_hard,
            initial_decoder_soft_score=initial_score, initial_decoder_changed_codons=sum(
                a!=b for a,b in zip(collate_records([dict(cds_dna=recon)]).to(device)[0],tokens[0])),
            initial_decoder_cds=recon, best_search=best, best=winner,
            gain=winner['hard_score']-original_score, unique_candidates=len(unique),
            first_decoder_candidate_change_step=first_change, trace=trace,
            final_gradient=last_gradient, best_displacement=gradient_summary(latent_delta(best_variable,latent_basis)[active]),
            search_coordinate_dimensions=width, output_latent_dimensions=model.latent_dim,
            encoder_masked_positions=int((encoder_tokens==MASK).sum()),
            hard_candidates=list(archive.values()),
            temperature=temperature,
            initial_max_probability_t1=float(natural_p[active].max(-1).values.mean()),
            initial_max_probability_search=float(initial_p[active].max(-1).values.mean()),
            initial_entropy_t1=float(-(natural_p[active]*natural_p[active].clamp_min(1e-30).log()).sum(-1).mean()),
            initial_entropy_search=float(-(initial_p[active]*initial_p[active].clamp_min(1e-30).log()).sum(-1).mean()),
            channel_interventions=interventions, seconds=time.monotonic()-begun)
        result['initial_decoder_changed_codons'] = int(result['initial_decoder_changed_codons'])
        save_json(destination/f'{arm}.json', result)
        results.append(result)
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--steps', type=int, default=80)
    parser.add_argument('--per-length', type=int, default=2)
    parser.add_argument('--native-python', type=Path)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    assert 1 <= args.steps <= 80 and 1 <= args.per_length <= 2
    torch.set_num_threads(4); torch.manual_seed(20261024)
    args.output.mkdir(parents=True, exist_ok=args.resume)
    data = corpus(args.data, 0)
    records = pick(data['val'], args.per_length, 'latent-search-inner-probe-20261024')
    calibration = pick(data['train'], 4, 'latent-search-train-calibration-20261024')
    assert not {r['v2_group'] for r in records} & {r['v2_group'] for r in calibration}
    saved = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    model = CodonVMAE(**saved['config']['model']).eval().requires_grad_(False)
    model.load_state_dict(saved['model'], strict=True)
    scorer = CountedScorer(TranslationObjective(ROOT/'score_model')).eval().requires_grad_(False)
    frozen = state_hash(model), state_hash(scorer)
    counts = codon_frequency(data['train'])
    weights = {c:float(counts[i]/max(counts[j] for j,a in enumerate(AA) if a==AA[i])) for i,c in enumerate(CODONS)}
    pool = Pool(model.latent_dim, covariance=True)
    with torch.no_grad():
        for record in calibration:
            tokens = collate_records([record]); mu, _ = model.encode(tokens,tokens)
            active, _ = masks(tokens); pool.add(mu,tokens.masked_fill(~active,PAD))
    within = pool.summary()['within_aa']['variance_by_dim']
    weak = torch.tensor(within) <= .01
    plan = dict(source_sha256=sha256(Path(__file__)), checkpoint_sha256=sha256(args.checkpoint),
        dependencies_sha256={n:sha256(Path(__file__).with_name(n)) for n in (
            'model.py','latent_optimization.py','dual_space_ablation.py','analyze_random_validation.py','v2_data.py','v2_train.py')},
        scorer_sha256=sha256(ROOT/'score_model/best_model.p'), folds_sha256=sha256(args.data/'folds.json'),
        records=records, calibration_ids=[r['sequence_id'] for r in calibration],
        train_calibration_geometry=pool.summary(), weak_channel_mask=weak.tolist(),
        fold=0, split='inner_val', steps=args.steps, arms=ARMS, seed=20261024,
        learning_rate=.05, edit_budget=None, js_penalty=0., js_hard_limit=None,
        native_protocol='If requested: shipped return pipeline, translation-only,80updates/batch1/oneouterround; '
            '160native target+negative queries plus80common candidate scores. At steps80, probe arms also '
            'use240search common calls, plus shared initialization/interventions; not equal FLOPs or walltime.',
        boundary='Probe fixes original start/stop. Native keeps its shipped return, including synonymous stop changes; raw native reference is not normalized.',
        objective='RiboDecode common HEK293T hard straight-through score only; no regularizer',
        interpretation='Radius/coordinate diagnostic, not multi-objective optimization or outer acceptance. '
            'Hidden and logit coordinates have different geometry: record actual decoder/logit displacement.',
        adaptive_rule='Own threshold diagnostic inspired by local search, not a LOL-BO reproduction: '
            'double .25..8 after5steps with no new best hard score but positive soft progress.',
        seq2ribo='Deferred official CUDA evaluation; no replacement score',
        gc_mfe_target='Per-CDS native RiboDecode returned sequence as user reference; raw deltas until tolerances specified')
    if args.resume:
        old=json.loads((args.output/'plan.json').read_text())
        for key in ('checkpoint_sha256','scorer_sha256','folds_sha256','records','calibration_ids','steps'):
            assert old[key]==plan[key], key
        save_json(args.output/'resume_revision.json',dict(previous_source_sha256=old['source_sha256'],
            current_source_sha256=plan['source_sha256'],
            reason='Native return may change synonymous terminal stop; retain raw output instead of requiring probe-only stop constraint. Reuse completed searches unchanged.'))
        save_json(args.output/'plan_current.json',plan)
    else:
        save_json(args.output/'plan.json', plan)
    save_json(args.output/'cai_weights.json', weights)
    cache, all_results = {}, []
    def properties(dna):
        if dna not in cache:
            cache[dna] = dict(**sequence_metrics(dna,weights), **fold_candidate(dna))
            assert cache[dna]['viennarna_version'] == '2.7.2'
        return cache[dna]
    status = dict(state='running', completed=0, total=len(records)*len(ARMS))
    try:
        for record in records:
            destination = args.output/record['sequence_id']; destination.mkdir(exist_ok=args.resume)
            save_json(args.output/'status.json', dict(status, sequence_id=record['sequence_id']))
            rows = search(model,scorer,record,args.steps,weak,destination,args.resume)
            native = None
            if args.native_python:
                native_dir = destination/'native'
                native_started=time.monotonic()
                command=[str(args.native_python),'-u',str(ROOT/'tools/run_three_cds_native.py'),
                    '--package',str(ROOT/'external/ribodecode_1_3_0/RiboDecode'),
                    '--output',str(native_dir),'--cds',record['cds_dna'],'--seed','20261024']
                if not (native_dir/'completion_receipt.json').exists():
                    with (destination/'native.log').open('wb') as log:
                        subprocess.run(command,cwd=ROOT,env=dict(os.environ,PYTHONPATH=str(ROOT)),
                            stdout=log,stderr=subprocess.STDOUT,check=True)
                receipt=json.loads((native_dir/'completion_receipt.json').read_text())
                native_dna=receipt['native_returned_cds']
                t=collate_records([dict(cds_dna=native_dna)])
                original_tokens=collate_records([record])
                native_validity=validate_native(record['cds_dna'],native_dna)
                candidates_path=native_dir/'results_natural/HEK293T_optim_mfe_dist-optim/0/samples/optim_results.txt'
                candidate_dnas=[line.split('\t')[0] for line in candidates_path.read_text().splitlines()]
                assert len(candidate_dnas)==80
                archive=[]
                with torch.no_grad():
                    for dna in candidate_dnas:
                        c=collate_records([dict(cds_dna=dna)])
                        assert torch.equal(AA_IDS[c],AA_IDS[original_tokens])
                        archive.append(dict(cds_dna=dna,hard_score=float(scorer(F.one_hot(c,64).float(),original_tokens))))
                native_score=next(x['hard_score'] for x in archive if x['cds_dna']==native_dna)
                save_json(destination/'native_common_archive.json',archive)
                native=dict(cds_dna=native_dna,hard_score=native_score,
                    native_target_and_negative_queries=receipt['native_target_and_negative_scorer_queries'],
                    common_rescore_queries=80,best_common_archive=max(archive,key=lambda x:x['hard_score']),
                    native_generation_seconds=receipt['elapsed_seconds'],**native_validity,
                    cold_generation_and_common_rescore_seconds=time.monotonic()-native_started,**properties(native_dna))
                save_json(destination/'native_reference.json',native)
            for row in rows:
                row.update(properties=properties(row['best']['cds_dna']), original_properties=properties(record['cds_dna']),
                    native_reference=native, seq2ribo_te=None)
                if native:
                    row['relative_to_native']=dict(hard_score=row['best']['hard_score']-native['hard_score'],
                        gc_fraction=row['properties']['gc_fraction']-native['gc_fraction'],
                        cai=row['properties']['train_reference_cai']-native['train_reference_cai'],
                        mfe=row['properties']['mfe']-native['mfe'])
                save_json(destination/f"{row['arm']}.json",row)
                all_results.append(row); status['completed']+=1
            save_json(args.output/'results.json',all_results)
            print(json.dumps(dict(event='record_complete',sequence_id=record['sequence_id'],
                n_aa=record['n_codons']-1,completed=status['completed'],total=status['total'])),flush=True)
        assert frozen == (state_hash(model),state_hash(scorer))
        assert all(p.grad is None for p in list(model.parameters())+list(scorer.parameters()))
        status.update(state='completed', weights_unchanged=True, outer_test_used=False,
            seq2ribo_evaluated=False, goal_complete=False)
    except BaseException as error:
        status.update(state='failed',error=repr(error)); raise
    finally:
        save_json(args.output/'status.json',status)


if __name__ == '__main__':
    main()
