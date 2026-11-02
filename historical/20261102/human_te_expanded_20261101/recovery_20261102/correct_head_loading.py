"""Repair the missing scalar-head load without changing locked training or selection."""
from pathlib import Path
import importlib.util
import json
import shutil
import sys
import time

ROOT=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(ROOT))
from common import WORK,PILOT,read,save,sha,verify_sources,module

def main():
    verify_sources();selection=read(ROOT/'selection_locked.json');assert selection['arm']=='no_stats'
    archive=Path(__file__).resolve().parent/'invalid_random_head_outputs';archive.mkdir(exist_ok=True)
    for name in ['pipeline_status.json','test_summary.json','test_predictions.json','test_predictions.csv','own_test_predictions_locked.json','evaluation_complete.json','test_open_receipt.json','独立测试比较.md']:
        if (ROOT/name).exists() and not (archive/name).exists():shutil.copy2(ROOT/name,archive/name)
    source=ROOT/'evaluate.py';spec=importlib.util.spec_from_file_location('locked_test_evaluation',source)
    evaluator=importlib.util.module_from_spec(spec);spec.loader.exec_module(evaluator)
    import numpy as np
    import torch
    from train import setup_gpu,batches
    setup_gpu();head_class=evaluator.Head
    val_rows=[r for r in read(ROOT/'train_validation.json') if r['split']=='val']
    indices=batches(val_rows)[0];group=[val_rows[j] for j in indices];features={}
    for row in group:
        with np.load(ROOT/'features'/f'{row["uid"]}.npz',allow_pickle=False) as z:features[row['uid']]={k:z[k].copy() for k in z.files}
    probe=module('recovery_validation_batch',WORK/'runs/mrfp_branch_probe_20261031/run_probe.py')
    inputs,_=probe.batch(group,features,'cuda');checks=[]
    for record in selection['selected_models']:
        assert sha(record['best_path'])==record['checkpoint_sha256']
        checkpoint=torch.load(record['best_path'],map_location='cpu',weights_only=True)
        base=evaluator.ReliableGraphContext(**checkpoint['config']);base.load_state_dict(checkpoint['backbone'],strict=True)
        model=evaluator.TaskScalarModel(base,'te').cuda().eval().requires_grad_(False)
        head=head_class();head.load_state_dict(checkpoint['head'],strict=True)
        assert all(torch.equal(head.state_dict()[k],v) for k,v in checkpoint['head'].items())
        head=head.cuda().eval().requires_grad_(False)
        with torch.no_grad():
            pooled,_=model.representations(inputs)
            predicted=(head(pooled[:,2],torch.zeros(len(group),3,device='cuda'))*checkpoint['target_scale']+checkpoint['target_mean']).cpu().numpy()
        expected=np.asarray(checkpoint['validation_predictions'])[indices]
        error=float(np.max(np.abs(predicted-expected)));assert error<5e-5,(record['seed'],error)
        checks.append(dict(seed=record['seed'],validation_batch=len(group),max_abs_error=error,scalar_head_state_exact=True))
        del model,head;torch.cuda.empty_cache()
    save(Path(__file__).resolve().parent/'validation_reload_regression.json',dict(status='passed',checks=checks))
    records=iter(selection['selected_models']);loaded=[]
    def trained_head():
        record=next(records);checkpoint=torch.load(record['best_path'],map_location='cpu',weights_only=True)
        assert sha(record['best_path'])==record['checkpoint_sha256']
        head=head_class();head.load_state_dict(checkpoint['head'],strict=True)
        assert all(torch.equal(head.state_dict()[k],v) for k,v in checkpoint['head'].items())
        loaded.append(record['checkpoint_sha256']);return head
    evaluator.Head=trained_head
    save(Path(__file__).resolve().parent/'correction_protocol.json',dict(issue='Original evaluation omitted head.load_state_dict; old own-score column invalid',
        correction='Inject the same selected checkpoints scalar-head state before test inference; locked evaluator otherwise unchanged',
        source_sha256=sha(__file__),original_evaluator_sha256=sha(source),selection_sha256=sha(ROOT/'selection_locked.json'),
        model_training_or_selection_changed=False,test_based_tuning=False,validation_reload_checks=checks))
    evaluator.main();assert len(loaded)==2
    save(Path(__file__).resolve().parent/'correction_complete.json',dict(status='completed',selected_checkpoint_hashes=loaded,unix=time.time(),
        test_metrics_sha256=sha(ROOT/'test_predictions.json'),test_summary_sha256=sha(ROOT/'test_summary.json')))

if __name__=='__main__':
    try:main()
    except BaseException as error:save(Path(__file__).resolve().parent/'failure.json',dict(error=repr(error),unix=time.time()));raise
