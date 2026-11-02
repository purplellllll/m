"""Resume both TE seeds at the prior terminal learning rate, development only."""
from pathlib import Path
import sys,os,shutil,time
HERE=Path(__file__).resolve().parent;WORK=HERE.parent.parent
ORIGINAL=WORK/'runs/human_te_expanded_20261101'
OLD=HERE/'te_30000';ROOT=HERE/'te_100000'
sys.path.insert(0,str(ORIGINAL))
import common

def prepare():
    ROOT.mkdir(exist_ok=True)
    source=ROOT/'trainer_low_lr.py'
    if not source.exists():
        text=(ORIGINAL/'train.py').read_text(encoding='utf-8')
        lines=[s for s in text.splitlines() if s.strip().startswith('factor=min(')];assert len(lines)==1
        text=text.replace(lines[0],'                factor=.1  # Preserve the previous terminal learning rate.')
        assert text.count('if stale>=8:break')==1
        text=text.replace('                    if stale>=8:break','                    # No early stop: user requested100000 total steps.')
        source.write_text(text,encoding='utf-8')
    if not (ROOT/'protocol_locked.json').exists():
        assert common.read(OLD/'training_complete.json')['status']=='completed'
        shutil.copy2(OLD/'train_validation.json',ROOT/'train_validation.json')
        (ROOT/'features').mkdir(exist_ok=True)
        for row in common.read(ROOT/'train_validation.json'):
            src=OLD/'features'/f'{row["uid"]}.npz';dst=ROOT/'features'/src.name
            if not dst.exists():os.link(src,dst)
        initial={str(OLD/'models'/f'no_stats_{seed}'/name):common.sha(OLD/'models'/f'no_stats_{seed}'/name) for seed in common.SEEDS for name in ['best.pt','latest.pt']}
        protocol=dict(task='HEK293_TE_low_lr_continuation',max_total_steps=100000,initial_steps=30000,seeds=common.SEEDS,
            arms=['no_stats'],encoder_lr=1e-6,head_lr=3e-5,validation_every=500,early_stop_checks=None,
            train_validation_sha256=common.sha(ROOT/'train_validation.json'),initial_checkpoint_sha256=initial,
            test_policy='Previously opened903 samples never read; no new TE test confirmation claimed',
            source_sha256={str(Path(__file__).resolve()):common.sha(__file__),str(source):common.sha(source),str(ORIGINAL/'common.py'):common.sha(ORIGINAL/'common.py')})
        common.save(ROOT/'protocol_locked.json',protocol)
        import torch
        for seed in common.SEEDS:
            out=ROOT/'models'/f'no_stats_{seed}';out.mkdir(parents=True,exist_ok=True)
            for name in ['best.pt','latest.pt']:
                c=torch.load(OLD/'models'/f'no_stats_{seed}'/name,map_location='cpu',weights_only=False)
                assert (c['step']==30000 if name=='latest.pt' else c['step']<=30000);c['protocol_sha256']=common.sha(ROOT/'protocol_locked.json');torch.save(c,out/name)
    return source
def verify():
    p=common.read(ROOT/'protocol_locked.json')
    for name,digest in {**p['source_sha256'],**p['initial_checkpoint_sha256']}.items():assert common.sha(name)==digest,name
    return p
def save(path,data):
    if isinstance(data,dict):
        if 'total_models' in data:data['total_models']=2
        if Path(path).name=='training_complete.json':data['models']=2
    common.save(path,data)
def wait_for_resources():
    ROOT.mkdir(exist_ok=True)
    dependency=HERE/'native_compare/complete.json'
    started=time.time()
    while not dependency.exists():
        common.save(ROOT/'queue_status.json',dict(stage='queued_after_native_comparison',dependency=str(dependency),requested_total_steps=100000,seeds=common.SEEDS,queued_unix=started))
        for failure in [HERE/'native_compare/queue_failure.json',HERE/'native_compare/comparison_failure.json',HERE/'native_compare/own_search/failure.json']:
            if failure.exists():raise RuntimeError('Dependency failed: '+str(failure))
        if time.time()-started>86400:raise TimeoutError('Comparison dependency not complete within24hours; inspect queue')
        time.sleep(10)
    common.save(ROOT/'queue_status.json',dict(stage='starting_training',requested_total_steps=100000,seeds=common.SEEDS))

if __name__=='__main__':
    try:
        wait_for_resources()
        source=prepare();trainer=common.module('te_low_lr_100000',source)
        trainer.ROOT=ROOT;trainer.MAX_STEPS=100000;trainer.EVERY=500;trainer.ARMS=['no_stats'];trainer.verify_sources=verify;trainer.save=save
        trainer.train()
    except BaseException as e:common.save(ROOT/'training_failure.json',dict(error=repr(e),unix=time.time()));raise
