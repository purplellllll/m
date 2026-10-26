"""Train-only PCA initialization after joint inner selection; untouched outer tests."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

import torch

from .budget_js_extension import save_json
from .latent_capacity_gpu import initial_check
from .latent_information_probe import load_model
from .latent_pca_initialization import compress_pca
from .latent_posterior_audit import audit
from .latent_recovery_gpu import batch_mask, panels
from .length_diagnostics import length_group
from .model import PAD
from .paper_guided_latent_cpu import load_checkpoint
from .train import sha256
from .v2_data import pick, read_corpus
from .v2_train import atomic_torch


@torch.no_grad()
def calibration(model, records):
    captured={};h,means,variances,weights=[],[],[],[]
    def before(module,args):captured['hidden']=args[0]
    hook=model.mu.register_forward_pre_hook(before)
    try:
        for record in records:
            tokens,masked,_=batch_mask([record],173,'cuda');valid=tokens<PAD
            for inputs in (tokens,masked):
                mu,lv=model.encode(inputs,tokens)
                n=int(valid.sum())
                h.append(captured['hidden'][valid].cpu().double())
                means.append(mu[valid].cpu().double());variances.append(lv[valid].cpu().double())
                weights.append(torch.full((n,),1/(len(records)*2*n),dtype=torch.float64))
    finally:hook.remove()
    return tuple(torch.cat(x) for x in (h,means,variances,weights))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    args=parser.parse_args();root=args.root
    assert json.loads((root/'objective_recovery/status.json').read_text())['state']=='completed'
    torch.set_num_threads(4);torch.set_num_interop_threads(1)
    assert torch.cuda.is_available()
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    reference=json.loads((root/'v1_safe_inner_all/V1_epoch50.json').read_text())['macro']
    candidates=[]
    for path in (root/'objective_recovery/results').glob('*/complete.json'):
        complete=json.loads(path.read_text());plan=json.loads((path.parent/'plan.json').read_text())
        for row in complete['history']:
            if row['microstep'] and row['ce']<reference['ce'] and row['fixed_loss']<reference['fixed_loss']:
                checkpoint=path.parent/f"inner_micro{row['microstep']:05d}.pt"
                assert checkpoint.exists()
                candidates.append((row['fixed_loss'],row['ce'],checkpoint,plan['recipe'],row))
    assert candidates,'No joint inner candidate; do not silently relax selection'
    _,_,checkpoint,recipe,selection=min(candidates,key=lambda x:(x[0],x[1]))
    assert recipe['sigma'] is None and recipe['skip_layer']==0 and recipe['gain']==1.
    data=root/'source/data';train=panels(data,0,1024,64)['train']
    records=[dict(r,length_group=length_group(r['n_codons']-1)) for r in pick(train,8,'capacity-train-ranking-20261023')]
    output=root/'pca_recovery';output.mkdir(exist_ok=True)
    inputs=output/'inputs';inputs.mkdir(exist_ok=True)
    plan=dict(source_sha256=sha256(Path(__file__)),helper_sha256=sha256(Path(__file__).with_name('latent_pca_initialization.py')),
        objective_worker_sha256=sha256(Path(__file__).with_name('latent_objective_recovery_gpu.py')),
        teacher_sha256=sha256(checkpoint),teacher_path=str(checkpoint),selection=selection,recipe=recipe,
        dimensions=[256,64,32,16],ranking_ids=[r['sequence_id'] for r in records],
        calibration='24 training CDS, clean/masked equal views, equal CDS weight; full mean covariance PCA; ridge1e-4 relative trace',
        posterior='Diagonal Gaussian approximation to projected marginals, not an exact rotation of full covariance',
        budget='Each dimension including256 receives same additional2304CDS exposures with same selected beta/mean objective',
        outer_test_used=False,goal_complete=False)
    if (output/'plan.json').exists():assert json.loads((output/'plan.json').read_text())==plan
    else:save_json(output/'plan.json',plan)
    if not (inputs/'complete.json').exists():
        teacher=load_model(checkpoint,'cuda');values=calibration(teacher,records);state=load_checkpoint(checkpoint)
        for dimension in plan['dimensions']:
            compressed,report=compress_pca(state,*values,dimension)
            target=inputs/f'initial_dim{dimension}.pt';atomic_torch(target,compressed)
            model=load_model(target,'cuda');checks=initial_check(model,teacher,records)
            save_json(inputs/f'dim{dimension}_check.json',dict(report=report,training_checks=checks,checkpoint_sha256=sha256(target)))
            del model,compressed;torch.cuda.empty_cache()
        del teacher,state,values;torch.cuda.empty_cache()
        save_json(inputs/'complete.json',dict(dimensions=plan['dimensions'],teacher_unchanged=sha256(checkpoint)==plan['teacher_sha256']))
    status=dict(state='training',completed=0)
    try:
        for index,dimension in enumerate(plan['dimensions']):
            status.update(state='training',completed=index,dimension=dimension);save_json(output/'status.json',status)
            command=[sys.executable,'-u','-m','latent_decoder.latent_objective_recovery_gpu',
                '--data',str(data),'--initial',str(inputs/f'initial_dim{dimension}.pt'),
                '--validation-reference',str(root/'v1_safe_inner_all/V1_epoch50.json'),
                '--output',str(output/'results'/f'dim_{dimension}'),'--lr','1e-5','--accumulation','16',
                '--profile','long_fast','--seed','20261031','--beta',str(recipe['training_beta']),
                '--mean-weight',str(recipe['mean_weight'])]
            print(json.dumps(dict(event='pca_recovery_trial',dimension=dimension,command=command)),flush=True)
            subprocess.run(command,cwd=root/'source',check=True)
        status.update(state='posterior_checks',completed=4);save_json(output/'status.json',status)
        identities=set(json.loads((root/'v1_inner_reference/V1_epoch50.json').read_text())['plan']['records'])
        probes=[r for r in read_corpus(data/'corpus.jsonl.gz') if r['sequence_id'] in identities]
        assert len(probes)==12
        targets=[('V1',root/'V1_epoch50_model_only.pt'),('teacher',checkpoint)]
        targets += [(f'dim_{d}',output/'results'/f'dim_{d}'/'best.pt') for d in plan['dimensions']]
        (output/'posterior').mkdir(exist_ok=True)
        for name,path in targets:
            model=load_model(path,'cuda');result=audit(model,probes)
            result['plan']=dict(checkpoint_sha256=sha256(path),records=sorted(identities),mask_seed=175,
                noise='Four valid diagonal posterior draws; noise coordinates are NOT paired across PCA capacities',outer_test_used=False)
            save_json(output/'posterior'/f'{name}.json',result)
            del model;torch.cuda.empty_cache()
        status.update(state='completed',completed=4)
    except BaseException as error:
        status.update(state='failed',error=repr(error));raise
    finally:save_json(output/'status.json',status)


if __name__=='__main__':main()
