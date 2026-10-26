"""One locked outer evaluation after all fifteen independent full-data fits finish."""
import argparse
import json
from pathlib import Path

import torch

from .budget_js_extension import save_json
from .latent_correlated_projection_gpu import evaluate_projected
from .latent_cv_evaluation import aggregate_scores, digest, outer_cohorts
from .latent_fullfold_recovery_gpu import FullfoldProjected, fold_data
from .latent_functional_validation import functional_validation
from .latent_information_probe import load_model
from .latent_posterior_audit import audit
from .latent_recovery_gpu import evaluate
from .length_diagnostics import length_group
from .paper_guided_latent_cpu import load_checkpoint
from .train import codon_frequency
from .v2_data import pick


PROTOCOL_SHA = '9321592dad92d23d2873089b20b8c7c454884416ccdd18c9e6b1650702b26899'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    root = args.root
    parent = root/'fullfold_recovery'
    status = json.loads((parent/'status.json').read_text())
    assert status['state'] == 'completed' and status['completed'] == 15
    assert digest(parent/'FINAL_EVALUATION_PROTOCOL.json') == PROTOCOL_SHA
    frozen = json.loads((parent/'all_source_sha256.json').read_text())
    for relative, expected in frozen.items():
        assert digest(root/'source'/relative) == expected, relative
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    assert torch.cuda.is_available()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    data = root/'source/data'
    cohorts = outer_cohorts(data)
    assert {int(f):len(rows) for f,rows in cohorts.items()} == {0:373,1:324,2:347,3:307,4:412}
    output = parent/'final_validation'
    output.mkdir(exist_ok=True)
    metadata = {r['sequence_id']:dict(fold=int(fold), homology_group=r['v2_group'],
        length_group=length_group(r['n_codons']-1)) for fold, rows in cohorts.items() for r in rows}
    assert len(metadata) == 1763
    targets = []
    for fold, records in cohorts.items():
        targets.append(dict(fold=int(fold), name='V1', checkpoint=str(root/'V1_epoch50_model_only.pt')))
        for name in ('control','candidate','compact'):
            complete = json.loads((parent/f'fold_{fold}'/name/'complete.json').read_text())
            assert complete['fold'] == int(fold) and complete['arm'] == name
            assert complete['used_cds'] == complete['full_training_cds'] > 30000
            path = Path(complete['selected_path'])
            assert path.resolve().parent == (parent/f'fold_{fold}'/name).resolve()
            assert digest(path) == complete['selected_sha256']
            targets.append(dict(fold=int(fold), name=name, checkpoint=str(path)))
    for target in targets:
        target['checkpoint_sha256'] = digest(Path(target['checkpoint']))
    plan = dict(source_sha256=digest(Path(__file__)), protocol_sha256=PROTOCOL_SHA, targets=targets,
        dependencies={name:digest(Path(__file__).with_name(name)) for name in (
            'latent_cv_evaluation.py','latent_functional_validation.py','latent_signal_geometry.py')},
        metadata=metadata, masks=[175,176], probes_per_length=4, probe_salt='locked-final-probe-20261023',
        recoding_repeats=2, selection='All models selected on their inner validation only; no outer-driven changes',
        outer_test_used=True, goal_complete=False)
    if (output/'plan.json').exists():
        assert json.loads((output/'plan.json').read_text()) == plan
    else:
        save_json(output/'plan.json', plan)
    progress = dict(state='evaluating', completed=0, total=20)
    try:
        combined = {name:[] for name in ('V1','control','candidate','compact')}
        for fold, records in cohorts.items():
            probes = pick(records, 4, plan['probe_salt'])
            assert len(probes) == 12
            train = fold_data(data, int(fold))['train']
            counts = codon_frequency(train)
            del train
            for target in [t for t in targets if t['fold'] == int(fold)]:
                name = target['name']
                progress.update(fold=int(fold), name=name)
                save_json(output/'status.json', progress)
                path = output/f'fold_{fold}_{name}.json'
                if not path.exists():
                    if name == 'compact':
                        state = load_checkpoint(Path(target['checkpoint']))
                        model = FullfoldProjected(noise_dim=state['noise_dim'], **state['config']['model']).cuda()
                        model.load_state_dict(state['model'], strict=True)
                        model.recipe = state['recipe']
                        model.eval().requires_grad_(False)
                        del state
                        mean = evaluate_projected(model, records)
                        posterior = evaluate_projected(model, probes, seeds=(175,), posterior=True)
                    else:
                        model = load_model(Path(target['checkpoint']), 'cuda')
                        mean = evaluate(model, records)
                        posterior = audit(model, probes)
                    functional = functional_validation(model, probes, counts, repeats=2)
                    result = dict(mean=mean, posterior=posterior, functional=functional,
                        plan=dict(**target, records=[r['sequence_id'] for r in records],
                            probe_records=[r['sequence_id'] for r in probes], outer_test_used=True))
                    save_json(path, result)
                    del model
                    torch.cuda.empty_cache()
                result = json.loads(path.read_text())
                assert result['plan']['checkpoint_sha256'] == target['checkpoint_sha256']
                assert set(result['plan']['records']) == {r['sequence_id'] for r in records}
                assert len(result['mean']['rows']) == 2*len(records)
                for row in result['mean']['rows']:
                    combined[name].append(dict(row, fold=int(fold)))
                progress['completed'] += 1
                print(json.dumps(dict(event='final_outer_target',fold=fold,name=name,metrics=result['mean']['macro'])),flush=True)
        summary = aggregate_scores(combined, metadata)
        summary.update(outer_test_used=True, protocol_sha256=PROTOCOL_SHA, goal_complete=False,
            interpretation='Prespecified V1-pretrained fivefold adaptation, excluding V1-seen and historical development homology groups. Inspect functional/posterior results before judging latent recovery.')
        save_json(output/'score_summary.json', summary)
        progress.update(state='completed', completed=20)
    except BaseException as error:
        progress.update(state='failed', error=repr(error))
        raise
    finally:
        save_json(output/'status.json', progress)


if __name__ == '__main__':
    main()
