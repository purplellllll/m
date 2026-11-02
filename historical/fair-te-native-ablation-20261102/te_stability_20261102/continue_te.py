"""Bounded development-only continuation. Never opens the previously used test set."""
from pathlib import Path
import sys, shutil, os, json, hashlib, time

HERE = Path(__file__).resolve().parent
WORK = HERE.parent.parent
OLD = WORK / 'runs/human_te_expanded_20261101'
ROOT = HERE / 'te_continuation'
sys.path.insert(0, str(OLD))
import common
import train as trainer

def prepare():
    ROOT.mkdir(exist_ok=True)
    if (ROOT/'protocol_locked.json').exists():
        return
    common.verify_sources()
    shutil.copy2(OLD/'train_validation.json', ROOT/'train_validation.json')
    rows = common.read(ROOT/'train_validation.json')
    (ROOT/'features').mkdir(exist_ok=True)
    for row in rows:
        src = OLD/'features'/f'{row["uid"]}.npz'
        dst = ROOT/'features'/src.name
        if not dst.exists():
            os.link(src, dst)
    initial = {}
    for seed in common.SEEDS:
        for name in ['latest.pt', 'best.pt']:
            p = OLD/'models'/f'no_stats_{seed}'/name
            initial[str(p)] = common.sha(p)
    protocol = dict(task='human_HEK293_TE_development_continuation',
        train_validation_sha256=common.sha(ROOT/'train_validation.json'),
        seeds=common.SEEDS, arms=['no_stats'], initial_checkpoint_sha256=initial,
        max_total_steps=18000, validation_every=500, early_stop_checks=8,
        schedule='Original cosine formula extended to 18000 total steps; learning rate rises at resume. Not a steps-only causal experiment.',
        test_policy='Previously opened 903 samples are not read or used for selection.',
        source_sha256={str(Path(__file__).resolve()):common.sha(__file__),str(OLD/'train.py'):common.sha(OLD/'train.py'),str(OLD/'common.py'):common.sha(OLD/'common.py')})
    common.save(ROOT/'protocol_locked.json', protocol)
    import torch
    for seed in common.SEEDS:
        out = ROOT/'models'/f'no_stats_{seed}'
        out.mkdir(parents=True, exist_ok=True)
        for name in ['latest.pt','best.pt']:
            ck = torch.load(OLD/'models'/f'no_stats_{seed}'/name,map_location='cpu',weights_only=False)
            ck['protocol_sha256'] = common.sha(ROOT/'protocol_locked.json')
            torch.save(ck, out/name)

def verify():
    p = common.read(ROOT/'protocol_locked.json')
    for path, digest in p['source_sha256'].items():
        assert common.sha(path)==digest, path
    for path, digest in p['initial_checkpoint_sha256'].items():
        assert common.sha(path)==digest, path
    return p

def save(path, data):
    if isinstance(data,dict):
        if 'total_models' in data: data['total_models']=2
        if Path(path).name=='training_complete.json': data['models']=2
    common.save(path,data)

if __name__=='__main__':
    try:
        prepare()
        trainer.ROOT=ROOT
        trainer.MAX_STEPS=18000
        trainer.EVERY=500
        trainer.ARMS=['no_stats']
        trainer.verify_sources=verify
        trainer.save=save
        trainer.train()
    except BaseException as error:
        common.save(ROOT/'training_failure.json',dict(error=repr(error),unix=time.time()))
        raise
