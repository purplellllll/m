"""Verify official inference and common scoring on the six paired CDS cases."""
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
from torch.nn import functional as F
from latent_decoder.budget_js_extension import save_json
from latent_decoder.latent_optimization import TranslationObjective, predictor_input, state_hash
from latent_decoder.model import collate_records
from latent_decoder.tokenizer import AA, encode_cds
from latent_decoder.train import sha256
from score_model.inference import InferenceModel, InferenceModel_conditon_spec


def main():
    out = ROOT/'ribodecode_verification_20261025'
    out.mkdir(exist_ok=False)
    status = dict(state='running', completed=0, total=18)
    save_json(out/'status.json', status)
    try:
        torch.set_num_threads(4)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        assert torch.cuda.is_available()
        source = ROOT.parent/'vmae_transfer_20261024'
        cases = json.loads((source/'cases.json').read_text())
        searches = json.loads((source/'small_translation_20261025/results.json').read_text())
        subset = json.loads((ROOT/'scorer_subset_receipt.json').read_text())
        scorer = TranslationObjective(ROOT/'score_model').cuda().eval().requires_grad_(False)
        native = InferenceModel_conditon_spec(str(ROOT/'score_model/model_config.json'), str(ROOT/'score_model/best_model.p'))
        native.prepare(batch_size=1, condition='HEK293T', conditons_dir=str(ROOT/'score_model/conditions')+'/')
        native.predictorModel.eval().requires_grad_(False)
        assert state_hash(native.predictorModel) == state_hash(scorer.model) == subset['inference_state_hash']
        canonical = torch.load(source/'canonical_condition.pt', map_location='cpu', weights_only=True).cuda()
        context_error = float((native.dict_rna_condition['HEK293T']-canonical).abs().max())
        assert context_error < 2e-6
        with torch.no_grad():
            scorer.condition.copy_(canonical)
        assert state_hash(scorer) == subset['model_state_hash']
        assert native.mRNA_count_batch.shape == (1, 1) and float(native.mRNA_count_batch[0, 0]) == 4.5
        initial = state_hash(scorer), state_hash(native.predictorModel)
        rows = []
        for case in cases['cases']:
            record = case['record']
            optimized = next(r for r in searches if r['model_arm'] == 'small' and r['sequence_id'] == record['sequence_id'])
            candidates = dict(original=record['cds_dna'], native_returned=case['native_reference']['cds_dna'], small_optimized=optimized['best']['cds_dna'])
            original_protein = [AA[i] for i in encode_cds(record['cds_dna'])]
            for label, dna in candidates.items():
                assert [AA[i] for i in encode_cds(dna)] == original_protein
                assert dna[:3] == record['cds_dna'][:3]
                # Official training CDS omit the terminal stop; common scoring does the same.
                sense = dna[:-3]
                assert len(sense) <= 4500
                padded = sense+'N'*(4500-len(sense))
                encoded = InferenceModel.my_process_line_for_cds(native, padded)
                official_input = torch.from_numpy(encoded).float().transpose(1, 2)[:, :4].cuda()
                tokens = collate_records([dict(cds_dna=dna)]).cuda()
                onehot = F.one_hot(tokens, 64).float()
                assert torch.equal(official_input, predictor_input(onehot, tokens))
                with torch.no_grad():
                    official = float(native.predict_seq(official_input, condition='HEK293T')[0, 0])
                    common = float(scorer(onehot, tokens)[0])
                expected = dict(original=optimized['original_score'], native_returned=case['native_reference']['hard_score'], small_optimized=optimized['best']['hard_score'])[label]
                assert abs(official-common) < 2e-5 and abs(common-expected) < 2e-5
                rows.append(dict(sequence_id=record['sequence_id'], n_aa=record['n_codons']-1,
                    label=label, official_score=official, common_score=common, previous_score=expected,
                    absolute_error=abs(official-common), previous_error=abs(common-expected),
                    protein_preserved=True, start_preserved=True,
                    stop_preserved=dna[-3:]==record['cds_dna'][-3:]))
                save_json(out/'results.json', rows)
                status['completed'] = len(rows)
                save_json(out/'status.json', status)
        assert (state_hash(scorer), state_hash(native.predictorModel)) == initial
        assert len(rows) == 18
        save_json(out/'completion_receipt.json', dict(
            state='completed', passed=True, rows=len(rows), max_absolute_error=max(r['absolute_error'] for r in rows),
            max_previous_score_error=max(r['previous_error'] for r in rows), condition_max_error=context_error,
            abundance=4.5, condition='HEK293T', source_sha256=sha256(Path(__file__)),
            official_inference_sha256=sha256(ROOT/'score_model/inference.py'),
            weight_file_sha256=sha256(ROOT/'score_model/best_model.p'),
            official_model_state_hash=initial[1], source_weights_unchanged=True,
            native_reference='Saved actual native 80-update returns, not a fresh optimizer run or full defaults',
            stop_convention='All methods rescored without terminal stop, matching official training CDS',
            validation='Official predict_seq interface and independent official nucleotide preprocessing match common score',
            biological_translation_validated=False, seq2ribo_assessed=False))
        status.update(state='completed', passed=True)
    except BaseException as error:
        status.update(state='failed', error=repr(error))
        raise
    finally:
        save_json(out/'status.json', status)


if __name__ == '__main__':
    main()
