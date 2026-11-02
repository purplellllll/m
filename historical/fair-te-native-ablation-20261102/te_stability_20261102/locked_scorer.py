"""Strictly load continued backbone AND scalar head; reuse frozen feature pipeline."""
from pathlib import Path
import sys,hashlib
import numpy as np
import torch
ROOT=Path(__file__).resolve().parent
WORK=ROOT.parent.parent
sys.path.insert(0,str(WORK/'runs/human_te_pilot_20261101'))
import human_score as legacy
from run_pilot import Head,ReliableGraphContext,TaskScalarModel,module,read,sha,save

class LockedScorer(legacy.HumanTEScorer):
    def __init__(self,device='cuda'):
        self.device=device;self.arm='finetune2_no_stats';self.models=[];self.cache={}
        legacy.ROOT=ROOT/'te_continuation'
        self.source=WORK/'runs/graph_polish_20261031/gpu_result/completed_snapshot/graph_polish_20261031/source'
        self.parameters=read(self.source.parent/'data/physics_parameters_train_only.json')
        self.helper=module('locked_human_features',WORK/'repos/riboseq_scorer_20261031/historical/drivers/run_aligned_profile_ablation_20261030.py')
        self.probe=module('locked_human_batch',WORK/'runs/mrfp_branch_probe_20261031/run_probe.py')
        self.cai=read(WORK/'runs/vmae_tiny_30cds_20261025/cai_weights.json')
        selection=read(ROOT/'te_continuation/selection_locked.json')
        assert selection['protocol_sha256']==sha(ROOT/'te_continuation/protocol_locked.json')
        for record in selection['selected_models']:
            path=Path(record['best_path']);assert sha(path)==record['checkpoint_sha256']
            ck=torch.load(path,map_location='cpu',weights_only=True)
            base=ReliableGraphContext(**ck['config']);base.load_state_dict(ck['backbone'],strict=True)
            head=Head();head.load_state_dict(ck['head'],strict=True)
            self.models.append((TaskScalarModel(base,'te').to(device).eval().requires_grad_(False),head.to(device).eval().requires_grad_(False),ck,sha(path)))
    def __call__(self,dna):
        value=super().__call__(dna)
        value.update(arm='continued_human_no_stats',outside_pilot_length_domain=not 80<=len(dna)//3<=1500,
            validation_scope='7373train922validation, 18000steps per seed; mouse-proxy physics',task='human_HEK293_CDS_only_TE')
        return value

def check():
    torch.set_num_threads(2);torch.backends.mha.set_fastpath_enabled(False);torch.backends.cuda.matmul.allow_tf32=False
    scorer=LockedScorer()
    rows=[r for r in read(ROOT/'te_continuation/train_validation.json') if r['split']=='val']
    errors=[]
    for index in [0,11,49]:
        value=scorer(rows[index]['dna'])
        expected=[m[2]['validation_predictions'][index] for m in scorer.models]
        errors.extend(abs(a-b) for a,b in zip(value['seed_scores'],expected))
    assert max(errors)<2e-5,errors
    save(ROOT/'scorer_locked.json',dict(status='passed',checkpoint_sha256=[m[3] for m in scorer.models],
        source_sha256=sha(__file__),regression_samples=3,max_prediction_error=max(errors),backbone_and_head_strict=True,
        old_test_read=False,stats_input_zero=True))
    print('locked scorer parity',max(errors),flush=True)
if __name__=='__main__':check()
