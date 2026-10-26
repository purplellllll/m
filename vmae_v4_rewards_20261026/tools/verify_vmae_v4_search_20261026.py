"""GPU smoke test for V4 reward-based latent search on one short TRAIN CDS."""
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))


def main():
    import torch
    from latent_decoder.v2_train import corpus
    from latent_decoder.latent_information_probe import load_model
    from latent_decoder.latent_optimization import TranslationObjective, state_hash
    from latent_decoder.dual_space_ablation import CountedScorer
    from tools.train_vmae_v4_20261026 import BASE, OUT, CHECKPOINT, CHECKPOINT_SHA, digest, save
    from tools.vmae_v4_rewards_20261026 import search_v4

    torch.set_num_threads(2)
    assert digest(CHECKPOINT)==CHECKPOINT_SHA
    data=corpus(BASE.parent/'vmae_adjustment_20261024/data',0)
    eligible=[r for r in data['train'] if 12<=r['n_codons']<=40]
    assert eligible
    record=min(eligible,key=lambda r:hashlib.sha256(('V4-search-preflight:'+r['sequence_id']).encode()).hexdigest())
    model=load_model(CHECKPOINT,'cuda').eval().requires_grad_(False)
    scorer=CountedScorer(TranslationObjective(BASE/'score_model')).cuda().eval().requires_grad_(False)
    canonical=BASE.parent/'vmae_transfer_20261024/canonical_condition.pt'
    subset=json.loads((BASE/'scorer_subset_receipt.json').read_text())
    assert digest(canonical)==subset['canonical_condition_sha256']
    with torch.no_grad():
        scorer.scorer.condition.copy_(torch.load(canonical,map_location='cpu',weights_only=True).cuda())
    assert state_hash(scorer.scorer)==subset['model_state_hash']
    weights=json.loads((OUT/'cai_weights.json').read_text())
    result=search_v4(model,scorer,record,weights,steps=10)
    assert len(result['trace'])==10 and result['trace'][-1]['mfe_feedback'] is not None
    assert result['model_weights_unchanged'] and digest(CHECKPOINT)==CHECKPOINT_SHA
    save(OUT/'search_preflight.json',dict(passed=True,split='TRAIN',sequence_id=record['sequence_id'],
        n_codons=record['n_codons'],steps=10,real_rnafold_feedback_verified=True,
        exact_fold_unique=result['exact_fold_unique'],best=result['best'],
        model_weights_unchanged=True,selection=result['selection']))
    print(json.dumps(dict(passed=True,steps=10,exact_fold_unique=result['exact_fold_unique'])),flush=True)


if __name__=='__main__':
    main()
