"""Check round-boundary recovery against a completed uninterrupted mixed run."""
import json
from pathlib import Path
import shutil
import subprocess
import sys


def main():
    base=Path(__file__).resolve().parents[1]
    root=base.parent/'vmae_30cds_20261025'
    gold=root/'preflight/mixed_tail'
    out=root/'recovery_validation'
    out.mkdir(exist_ok=False)
    resumed=out/'resumed_mixed'
    def copy_asset(source,destination):
        if Path(source).name=='best_model.p':
            __import__('os').link(source,destination);return destination
        return shutil.copy2(source,destination)
    shutil.copytree(gold,resumed,copy_function=copy_asset)
    (resumed/'completion_receipt.json').rename(resumed/'original_complete_before_validation.json')
    protocol=json.loads((gold/'runtime_protocol.json').read_text())
    task=out/'task.json';task.write_text(json.dumps(protocol['task']))
    with (out/'resume.log').open('wb') as log:
        subprocess.run([sys.executable,'-u',str(base/'tools/native_full_gpu_resume_20261025.py'),
            '--package',str(root/'external/RiboDecode'),'--task',str(task),
            '--output',str(resumed),'--mode','preflight','--resume-round','1'],
            stdout=log,stderr=subprocess.STDOUT,check=True)
    import torch
    import numpy as np
    torch.set_num_threads(1)
    x,y=[torch.load(p/'round_02_state.pt',map_location='cpu',weights_only=False) for p in (gold,resumed)]
    maxerror=0.
    for module in ('z','mfe'):
        for name,value in x[module].items():
            error=float((value.double()-y[module][name].double()).abs().max())
            maxerror=max(maxerror,error)
            assert torch.allclose(value,y[module][name],atol=1e-6,rtol=1e-6),name
    assert x['per_round_selected']==y['per_round_selected']
    assert x['python_rng']==y['python_rng']
    assert np.array_equal(x['numpy_rng'][1],y['numpy_rng'][1]) and x['numpy_rng'][2:]==y['numpy_rng'][2:]
    assert torch.equal(x['torch_rng'],y['torch_rng'])
    assert all(torch.equal(a,b) for a,b in zip(x['cuda_rng'],y['cuda_rng']))
    assert np.array_equal(x['previous_distribution'],y['previous_distribution'])
    suffix='results_natural/HEK293T_optim_mfe_dist-optim/1/samples/optim_results.txt'
    left,right=[(p/suffix).read_text().splitlines() for p in (gold,resumed)]
    assert len(left)==len(right)==80
    for a,b in zip(left,right):
        a,b=a.split('\t'),b.split('\t');assert a[0]==b[0]
        assert all(abs(float(u)-float(v))<1e-6*(1+abs(float(u))) for u,v in zip(a[1:],b[1:]))
    receipt=json.loads((resumed/'completion_receipt.json').read_text())
    assert receipt['native_target_negative_queries']==320 and receipt['resumed_from_completed_round']==1
    result=dict(passed=True,full_round_resumed=2,candidate_dna_equal=True,candidate_scores_within_1e_6=True,
        max_state_tensor_error=maxerror,python_numpy_torch_cpu_cuda_rng_equal=True,
        selected_sequences_equal=True,next_round_distribution_equal=True,nominal_queries=320)
    (out/'completion_receipt.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result),flush=True)


if __name__=='__main__':main()
