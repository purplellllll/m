"""Read-only matched evaluation of compact latents retaining full projected covariance."""
import argparse
import json
from pathlib import Path
import statistics

import torch

from .budget_js_extension import save_json
from .latent_collapse_cpu import GROUPS, ce_metrics
from .latent_correlated_projection import load_projected, project_state, projected_kl
from .latent_information_probe import load_model
from .latent_posterior_audit import DRAWS, audit, posterior_noise
from .latent_recovery_gpu import batch_mask, evaluate
from .length_diagnostics import length_group
from .model import PAD
from .paper_guided_latent_cpu import load_checkpoint
from .train import sha256
from .v2_data import read_corpus
from .v2_train import atomic_torch


def metrics(model, logits, tokens, mu, source_logvar, selected):
    zero = torch.zeros_like(mu)
    result = ce_metrics(logits, tokens, zero, zero, selected)
    kl = projected_kl(mu, source_logvar, model.noise_projection)[tokens < PAD].mean()
    result['kl'] = float(kl)
    result['fixed_loss'] = float(torch.tensor(result['full_ce']) + torch.tensor(result['ce']) + .001 * kl.cpu())
    return result


def summarize(rows, posterior=False):
    modes = ('mean', 'posterior') if posterior else (None,)
    by_length = {}
    for group in GROUPS:
        by_length[group] = {}
        for mode in modes:
            chosen = [row for row in rows if row['group'] == group and (mode is None or row['mode'] == mode)]
            assert chosen
            by_length[group][mode] = {key: statistics.mean(row[key] for row in chosen)
                for key in ('ce', 'full_ce', 'kl', 'fixed_loss')}
    macro = {mode: {key: statistics.mean(by_length[group][mode][key] for group in GROUPS)
        for key in ('ce', 'full_ce', 'kl', 'fixed_loss')} for mode in modes}
    if not posterior:
        return dict(rows=rows, by_length={g: by_length[g][None] for g in GROUPS}, macro=macro[None])
    return dict(rows=rows, by_length=by_length, macro=macro)


@torch.no_grad()
def evaluate_projected(model, records, seeds=(175, 176), posterior=False):
    rows = []
    for record in records:
        for seed in seeds:
            tokens, inputs, selected = batch_mask([record], seed, 'cuda')
            mu, source_lv = model.distribution(inputs, tokens)
            modes = [('mean', None)] + ([('posterior', draw) for draw in DRAWS] if posterior else [])
            for mode, draw in modes:
                z = mu if draw is None else model.sample(mu, source_lv,
                    posterior_noise(record['sequence_id'], draw, source_lv.shape).to(mu.device))
                result = metrics(model, model.decode(z * (tokens < PAD)[..., None], tokens),
                    tokens, mu, source_lv, selected)
                assert all(torch.isfinite(torch.tensor(value)) for value in result.values())
                rows.append(dict(sequence_id=record['sequence_id'], group=length_group(record['n_codons'] - 1),
                    mask_seed=seed, mode=mode, draw_seed=draw, **result))
    return summarize(rows, posterior)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    root = args.root
    assert json.loads((root / 'objective_confirmation/status.json').read_text())['state'] == 'completed'
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    assert torch.cuda.is_available()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    output = root / 'correlated_projection'
    output.mkdir(exist_ok=True)
    reference = json.loads((root / 'v1_safe_inner_all/V1_epoch50.json').read_text())
    ids = set(reference['plan']['records'])
    probe_ids = set(json.loads((root / 'v1_inner_reference/V1_epoch50.json').read_text())['plan']['records'])
    assert probe_ids <= ids
    records = [r for r in read_corpus(root / 'source/data/corpus.jsonl.gz') if r['sequence_id'] in ids]
    probes = [r for r in records if r['sequence_id'] in probe_ids]
    assert len(records) == 154 and len(probes) == 12
    previous = json.loads((root / 'pca_recovery/plan.json').read_text())
    teacher_path = Path(previous['teacher_path'])
    assert sha256(teacher_path) == previous['teacher_sha256']
    teacher = load_checkpoint(teacher_path)
    plan = dict(source_sha256=sha256(Path(__file__)),
        helper_sha256=sha256(Path(__file__).with_name('latent_correlated_projection.py')),
        teacher_sha256=sha256(teacher_path), prior='Standard normal in uncentered projected coordinates',
        covariance='Exact U.T@diag(exp(source_lv))@U, with Cholesky logdet KL and shared256-D source noise',
        dimensions=[16, 32, 64], records=sorted(ids), posterior_records=sorted(probe_ids),
        posterior_draws=list(DRAWS), evaluation_beta=.001, outer_test_used=False, training_performed=False)
    if (output / 'plan.json').exists():
        assert json.loads((output / 'plan.json').read_text()) == plan
    else:
        save_json(output / 'plan.json', plan)
    status = dict(state='evaluating', completed=0)
    try:
        for name, path in [('V1', root / 'V1_epoch50_model_only.pt'), ('teacher', teacher_path)]:
            status.update(model=name)
            save_json(output / 'status.json', status)
            if not (output / (name + '.json')).exists():
                model = load_model(path, 'cuda')
                result = dict(mean=evaluate(model, records), posterior=audit(model, probes), checkpoint_sha256=sha256(path))
                save_json(output / (name + '.json'), result)
                del model
                torch.cuda.empty_cache()
        for dimension in plan['dimensions']:
            name = f'dim_{dimension}'
            status.update(model=name)
            save_json(output / 'status.json', status)
            if not (output / (name + '.json')).exists():
                report = json.loads((root / f'pca_recovery/inputs/dim{dimension}_check.json').read_text())['report']
                state = project_state(teacher, report)
                path = output / (name + '.pt')
                atomic_torch(path, state)
                model = load_projected(state, 'cuda')
                result = dict(mean=evaluate_projected(model, records),
                    posterior=evaluate_projected(model, probes, seeds=(175,), posterior=True),
                    checkpoint_sha256=sha256(path), note='Mean decoder identical subspace to diagonal-PCA initialization; full covariance and prior handled explicitly')
                save_json(output / (name + '.json'), result)
                del model, state
                torch.cuda.empty_cache()
            status['completed'] += 1
            print(json.dumps(dict(event='correlated_projection_evaluated', dimension=dimension)), flush=True)
        status.update(state='completed', completed=3)
    except BaseException as error:
        status.update(state='failed', error=repr(error))
        raise
    finally:
        save_json(output / 'status.json', status)


if __name__ == '__main__':
    main()
