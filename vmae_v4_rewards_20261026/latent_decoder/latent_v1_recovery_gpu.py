"""Capacity-matched recovery from V1; evaluate only V1-unseen safe inner groups."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

import torch

from .budget_js_extension import save_json
from .latent_capacity_gpu import compress_state, initial_check, rank_dimensions
from .latent_recovery_gpu import BASE, initialize, panels
from .length_diagnostics import length_group
from .paper_guided_latent_cpu import load_checkpoint
from .train import sha256
from .v2_data import pick
from .v2_train import atomic_torch


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    args=parser.parse_args();root=args.root
    assert json.loads((root/'gene_style_probe/complete.json').read_text())['goal_complete'] is False
    assert json.loads((root/'batch_recovery/length_profiles/status.json').read_text())['state']=='completed'
    torch.set_num_threads(4);torch.set_num_interop_threads(1)
    assert torch.cuda.is_available()
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    output=root/'v1_recovery';output.mkdir(exist_ok=True)
    initial=root/'V1_epoch50_model_only.pt'
    assert sha256(initial)=='b00c9eb229cb80a1ee935cda49a7af7d635b90a39399a671a54f7d5cfec7e32b'
    data=root/'source/data';training=panels(data,0,1024,64)['train']
    records=pick(training,8,'capacity-train-ranking-20261023')
    records=[dict(r,length_group=length_group(r['n_codons']-1)) for r in records]
    plan=dict(source_sha256=sha256(Path(__file__)),initial_sha256=sha256(initial),
        core_sha256=sha256(Path(__file__).with_name('latent_recovery_gpu.py')),
        worker_sha256=sha256(Path(__file__).with_name('latent_batch_recovery_gpu.py')),
        capacity_sha256=sha256(Path(__file__).with_name('latent_capacity_gpu.py')),
        dimensions=[256,128,64,32],seed=20261031,accumulation=16,lr=1e-5,profile='long_fast',
        microsteps=1152,cds_exposures=2304,ranking_ids=[r['sequence_id'] for r in records],
        evaluation='All154 old-test-safe INNER validation; same masks175/176; original V1 is a frozen reference',
        initialization='External V1 pretraining; reported validation homology groups exclude all original V1 train/val exposure',
        outer_test_used=False,goal_complete=False)
    if (output/'plan.json').exists():assert json.loads((output/'plan.json').read_text())==plan
    else:save_json(output/'plan.json',plan)
    inputs=output/'inputs';inputs.mkdir(exist_ok=True)
    if not (inputs/'complete.json').exists():
        teacher,_=initialize(initial,BASE);teacher.eval().requires_grad_(False)
        scores,mean=rank_dimensions(teacher,records)
        original=load_checkpoint(initial)
        save_json(inputs/'ranking.json',dict(scores=scores.tolist(),mean_mu=mean.tolist(),training_ids=plan['ranking_ids']))
        for dimension in plan['dimensions']:
            compressed,indices=compress_state(original,dimension,scores,mean)
            target=inputs/f'initial_dim{dimension}.pt';atomic_torch(target,compressed)
            model,_=initialize(target,BASE);model.eval().requires_grad_(False)
            checks=initial_check(model,teacher,records)
            if dimension==256:
                assert max(abs(r['teacher']['ce']-r['compressed']['ce']) for r in checks)<1e-6
            save_json(inputs/f'dim{dimension}_check.json',dict(retained_indices=indices.tolist(),rows=checks,
                checkpoint_sha256=sha256(target),compression_is_approximate=dimension<256))
            del model,compressed
            torch.cuda.empty_cache()
        del teacher,original
        torch.cuda.empty_cache()
        save_json(inputs/'complete.json',dict(dimensions=plan['dimensions'],teacher_unchanged=sha256(initial)==plan['initial_sha256']))
    status=dict(state='running',completed=0)
    try:
        for index,dimension in enumerate(plan['dimensions']):
            status.update(state='training',completed=index,dimension=dimension);save_json(output/'status.json',status)
            command=[sys.executable,'-u','-m','latent_decoder.latent_batch_recovery_gpu',
                '--data',str(data),'--initial',str(inputs/f'initial_dim{dimension}.pt'),
                '--validation-reference',str(root/'v1_safe_inner_all/V1_epoch50.json'),
                '--output',str(output/'results'/f'dim_{dimension}'),'--lr','1e-5',
                '--accumulation','16','--seed','20261031','--profile','long_fast']
            print(json.dumps(dict(event='v1_recovery_trial',dimension=dimension,command=command)),flush=True)
            subprocess.run(command,cwd=root/'source',check=True)
        status.update(state='completed',completed=4)
    except BaseException as error:
        status.update(state='failed',error=repr(error));raise
    finally:save_json(output/'status.json',status)


if __name__=='__main__':main()
