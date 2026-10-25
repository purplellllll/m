"""Score locked previous-small-model candidates against native returns with seq2ribo."""
import csv
import hashlib
import json
from pathlib import Path
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
PACKAGES = ROOT/'.runtime/mamba_cuda_v2_2_4/packages'
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(PACKAGES))

import torch
from latent_decoder.budget_js_extension import save_json
from latent_decoder.latent_optimization import state_hash
from latent_decoder.tokenizer import AA, encode_cds


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    import fcntl
    out = ROOT/'small_seq2ribo_20261025'
    out.mkdir(exist_ok=False)
    with (out/'queue.lock').open('w') as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        status = dict(state='initializing', completed=0, total=54)
        save_json(out/'status.json', status)
        start = time.monotonic()
        try:
            source = ROOT.parent/'vmae_transfer_20261024'
            inputs = source/'small_translation_20261025/results.json'
            cases_path = source/'cases.json'
            searches = json.loads(inputs.read_text())
            cases = json.loads(cases_path.read_text())['cases']
            candidates = []
            for case in sorted(cases, key=lambda c:c['record']['n_codons']):
                record = case['record']
                result = next(r for r in searches if r['model_arm']=='small' and r['sequence_id']==record['sequence_id'])
                sequences = dict(original=record['cds_dna'], native=case['native_reference']['cds_dna'], small=result['best']['cds_dna'])
                protein = [AA[i] for i in encode_cds(record['cds_dna'])]
                assert all([AA[i] for i in encode_cds(dna)]==protein for dna in sequences.values())
                candidates.append(dict(sequence_id=record['sequence_id'], n_aa=result['n_aa'], sequences=sequences,
                    small_ribo_hard=result['best']['hard_score'], native_ribo_hard=case['native_reference']['hard_score']))
            assert len(candidates)==6
            lock_path = ROOT/'docs/seq2ribo_evaluator_lock_20261020.json'
            lock = json.loads(lock_path.read_text())
            seeds = [lock['seed'], 20261021, 20261022]
            assert len(set(seeds))==3
            parity = json.loads((ROOT/'seq2ribo_cuda_parity/completion_receipt.json').read_text())
            assert parity['cuda_parity_verified'] and parity['teacher_weights_unchanged']
            repository = ROOT/'tmp/seq2ribo_source_20261020/seq2ribo-main'
            for name, digest in {**lock['source_sha256'], **lock['weights_sha256']}.items():
                assert sha(repository/name)==digest, name
            for name, digest in parity['compiled_extensions'].items():
                assert sha(name)==digest
            save_json(out/'plan.json', dict(candidates=candidates, seeds=seeds,
                checkpoint='Previous 9714496-parameter small model, selected epoch35; not currently training 2498240-parameter model',
                search_results_sha256=sha(inputs), cases_sha256=sha(cases_path), evaluator_lock_sha256=sha(lock_path),
                source_sha256=sha(Path(__file__)), options=lock, selection='Best RiboDecode hard-score candidate frozen before seq2ribo evaluation; no seq2ribo reselection',
                native_reference='Same six actual saved native80-update returns, not full defaults',
                scope='Paired three-seed assessment on six reused validation CDS, not independent outer test or biological validation',
                current_training_unchanged=True, goal_complete=False))
            import RNA
            import mamba_ssm
            import selective_scan_cuda
            assert RNA.__version__==lock['viennarna_version']=='2.7.2'
            assert torch.cuda.is_available() and mamba_ssm.__version__=='2.2.4'
            assert Path(selective_scan_cuda.__file__).resolve().is_relative_to(PACKAGES.resolve())
            torch.set_num_threads(1)
            torch.set_num_interop_threads(1)
            torch.backends.cuda.matmul.allow_tf32=False
            torch.backends.cudnn.allow_tf32=False
            sys.path.insert(0, str(repository))
            from seq2ribo import Seq2Ribo
            predictor = Seq2Ribo(cell_line=lock['cell_line'], device='cuda', weights_dir=str(repository/'weights'), cache_dir=str(out/'geometry'))
            teacher = predictor._load_model('te', use_utr=False).eval().requires_grad_(False)
            teacher_hash = state_hash(teacher)
            assert teacher_hash==parity['teacher_state_hash']
            previous = {r['cds_dna']:r['cpu_reference_te'] for r in json.loads((ROOT/'seq2ribo_cuda_parity_inputs.json').read_text())}
            measurements, comparisons = [], []
            status['state']='scoring'
            for case in candidates:
                paired=[]
                for seed in seeds:
                    values={}
                    for label,dna in case['sequences'].items():
                        value=float(predictor.predict(dna.replace('T','U'), task='te', n_stasep_runs=lock['n_stasep_runs'],
                            init_p=lock['init_p'], seed=seed, return_scaled_te=lock['return_scaled_te'], batch_size=8)[0])
                        assert torch.isfinite(torch.tensor(value))
                        values[label]=value
                        if label=='native' and seed==lock['seed']:
                            assert dna in previous and abs(value-previous[dna])<1e-5*(1+abs(previous[dna]))
                        measurements.append(dict(sequence_id=case['sequence_id'], n_aa=case['n_aa'], label=label,
                            seed=seed, seq2ribo_te=value, sequence_sha256=hashlib.sha256(dna.encode()).hexdigest()))
                        save_json(out/'measurements.json', measurements)
                        status.update(completed=len(measurements), n_aa=case['n_aa'], seed=seed, label=label, seconds=time.monotonic()-start)
                        save_json(out/'status.json', status)
                    paired.append(dict(seed=seed, **values, delta_native=values['small']-values['native'],
                                       delta_original=values['small']-values['original']))
                row=dict(sequence_id=case['sequence_id'], n_aa=case['n_aa'],
                    original_te=statistics.mean(r['original'] for r in paired),
                    native_te=statistics.mean(r['native'] for r in paired), small_te=statistics.mean(r['small'] for r in paired),
                    mean_delta_native=statistics.mean(r['delta_native'] for r in paired),
                    mean_delta_original=statistics.mean(r['delta_original'] for r in paired),
                    positive_all_seeds=all(r['delta_native']>1e-5*(1+abs(r['native'])) for r in paired),
                    positive_seed_count=sum(r['delta_native']>1e-5*(1+abs(r['native'])) for r in paired),
                    small_ribo_hard=case['small_ribo_hard'], native_ribo_hard=case['native_ribo_hard'],
                    paired=paired)
                comparisons.append(row)
                save_json(out/'results.json', comparisons)
                print(json.dumps(dict(event='case_complete', **{k:v for k,v in row.items() if k!='paired'})), flush=True)
            assert len(measurements)==54 and len(comparisons)==6 and state_hash(teacher)==teacher_hash
            assert len({(r['sequence_id'],r['seed'],r['label']) for r in measurements})==54
            with (out/'comparison.csv').open('w', newline='') as stream:
                writer=csv.DictWriter(stream,fieldnames=[k for k in comparisons[0] if k!='paired'])
                writer.writeheader()
                writer.writerows({k:v for k,v in row.items() if k!='paired'} for row in comparisons)
            save_json(out/'completion_receipt.json',dict(state='completed', completed_calls=54, cases=6,
                seeds=seeds, positive_all_seed_cases=sum(r['positive_all_seeds'] for r in comparisons),
                mean_delta_positive_cases=sum(r['mean_delta_native']>0 for r in comparisons),
                teacher_state_hash=teacher_hash, teacher_weights_unchanged=True,
                seconds=time.monotonic()-start, goal_complete=False, outer_test_used=False))
            status.update(state='completed', seconds=time.monotonic()-start)
        except BaseException as error:
            status.update(state='failed', error=repr(error))
            raise
        finally:
            save_json(out/'status.json', status)


if __name__=='__main__':
    main()
