"""Frozen epoch6 full latent optimization and independent evaluation."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import shutil
import statistics
import subprocess
import sys
import time

BASE=Path('/root/gpufree-data/vmae_transfer_assessment_20261024')
REFERENCE=BASE.parent/'vmae_v4_full_optimization_20261026'
ROOT=BASE.parent/'vmae_v4_epoch6_current_20261027'
PARENT=BASE.parent/'vmae_v4_rewards_20261026'
EXT=ROOT/'extension'
PYTHON='/root/gpufree-data/vmae_gpu_env/bin/python'
sys.path.insert(0,str(BASE))
EXPECTED={6:'8e95e3d3e5c62917c01a78a3add4ab45ea1afa7a1282748ce8d637f3fbea02e3'}
KEYS=('ribodecode_score','seq2ribo_score','cai','gc_percent','mfe_kcal_mol','mfe_per_nt')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
    tmp.replace(path)


def choose_checkpoint(scores,expected_ids):
    """Require every VAL identity; seq2ribo is deliberately absent from the decision API."""
    identities=set(expected_ids)
    if not identities or len(identities)!=len(expected_ids) or set(scores)!={0,1,2,3}:
        raise ValueError('Expected four checkpoints and unique validation identities')
    if any(set(rows)!=identities for rows in scores.values()):
        raise ValueError('Checkpoint comparison is incomplete or uses different identities')
    if any(not math.isfinite(float(x)) for rows in scores.values() for x in rows.values()):
        raise ValueError('Nonfinite combined reward')
    means={epoch:statistics.mean(rows.values()) for epoch,rows in scores.items()}
    maximum=max(means.values())
    winner=max(epoch for epoch,value in means.items() if maximum-value<=1e-8)
    return dict(best_epoch=winner,mean_optimized_reward=means,additional_epochs=5 if winner==3 else 0,
        selection='Mean optimized Hard+GC_reward+CAI_reward+MFE_reward on fixed 24 VAL CDS',
        tie_policy='Within 1e-8 choose latest epoch',seq2ribo_used_for_selection=False,test_used_for_selection=False)


def prepare():
    import torch
    ROOT.mkdir(exist_ok=True)
    checkpoint=ROOT/'checkpoints/epoch6.pt'
    assert sha(checkpoint)==EXPECTED[6]
    state=torch.load(checkpoint,map_location='cpu',weights_only=False)
    assert state['epoch']==6
    original=json.loads((REFERENCE/'plan.json').read_text())
    validation=json.loads((REFERENCE/'validation_panel.json').read_text())
    test=json.loads((REFERENCE/'test_panel.json').read_text())
    assert len(validation)==24 and len(test)==30
    assert not set(original['validation_ids']) & set(original['test_ids'])
    plan=dict(original,checkpoints={'6':EXPECTED[6]},selection='User froze epoch6, original training VAL objective; no seq2ribo selection',
        phase_order=['Fixed epoch6 full search VAL24','Fixed epoch6 full search TEST30'],
        source_sha256={n:sha(BASE/'tools'/n) for n in [Path(__file__).name,'vmae_v4_rewards_20261026.py',
            'train_vmae_v4_extended_20261026.py','vmae_multiobjective_core_20261026.py','run_vmae_30cds_20261025.py']})
    if (ROOT/'plan.json').exists():
        assert json.loads((ROOT/'plan.json').read_text())==plan
    else:
        save(ROOT/'plan.json',plan)
        save(ROOT/'validation_panel.json',validation);save(ROOT/'test_panel.json',test)
        shutil.copy2(REFERENCE/'cai_weights.json',ROOT/'cai_weights.json')
    return plan,validation,test


def refresh():
    if not (ROOT/'plan.json').exists():
        return
    plan=json.loads((ROOT/'plan.json').read_text())
    split_by_id={i:'validation' for i in plan['validation_ids']}
    split_by_id.update({i:'test' for i in plan['test_ids']})
    rows=[]
    for path in sorted((ROOT/'metrics').glob('*/*.json')):
        row=json.loads(path.read_text())
        rows.append(dict(row,panel_split=split_by_id[row['sequence_id']]))
    lookup={(r['sequence_id'],r['method']):r for r in rows}
    assert len(lookup)==len(rows)
    summaries={}
    for split in ('validation','test'):
        summaries[split]={}
        for method in sorted({r['method'] for r in rows}):
            selected=[r for r in rows if r['panel_split']==split and r['method']==method]
            if selected:
                summaries[split][method]=dict(n=len(selected),means={k:statistics.mean(r[k] for r in selected) for k in KEYS})
    save(ROOT/'summary.json',dict(panels=summaries,independent_seq2ribo=True,selection_uses_seq2ribo=False,
        generated_unix=time.time(),complete_validation_expected=48,initial_test_expected_at_least=60))
    if rows:
        with (ROOT/'metrics.csv.tmp').open('w',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(rows[0]))
            writer.writeheader();writer.writerows(rows)
        (ROOT/'metrics.csv.tmp').replace(ROOT/'metrics.csv')
    text='# V4 full optimization and independent seq2ribo\n\nPartial N must not be compared as equal panels.\n'
    for split,methods in summaries.items():
        text+='\n## '+split+'\n\n| Method | N | RiboDecode | seq2ribo | CAI | GC % | MFE | MFE/nt |\n|---|---:|---:|---:|---:|---:|---:|---:|\n'
        for method,v in methods.items():
            text+='| '+method+' | '+str(v['n'])+' | '+' | '.join(f"{v['means'][k]:.5f}" for k in KEYS)+' |\n'
    if (ROOT/'decision.json').exists():
        text+='\nDecision (VAL optimized reward only):\n\n```json\n'+(ROOT/'decision.json').read_text()+'```\n'
    text+='\nThese are predictor outputs, not measured expression. TEST panel was previously explored.\n'
    (ROOT/'REPORT.md').write_text(text)


def run():
    import fcntl
    import torch
    from latent_decoder.latent_information_probe import load_model
    from latent_decoder.latent_optimization import state_hash
    from tools import run_vmae_30cds_20261025 as evaluation
    from tools import vmae_v4_rewards_20261026 as core
    plan,validation,test=prepare()
    with (ROOT/'workflow.lock').open('a') as handle:
        fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (ROOT/'complete.json').exists():
            return
        evaluation.OUT=ROOT
        evaluation.refresh=refresh
        scorer,predictor,teacher,lock=evaluation.environment()
        reference_hashes=[state_hash(m) for m in (scorer,teacher)]
        weights=json.loads((ROOT/'cai_weights.json').read_text())
        fold_cache=json.loads(((ROOT/'fold_cache.json') if (ROOT/'fold_cache.json').exists() else (REFERENCE/'fold_cache.json')).read_text())
        original_fold=core.exact_fold
        current={}
        started=time.time()

        def status(stage,**extra):
            counts={s:{m:v['n'] for m,v in methods.items()} for s,methods in
                json.loads((ROOT/'summary.json').read_text())['panels'].items()} if (ROOT/'summary.json').exists() else {}
            save(ROOT/'status.json',dict(state='running',stage=stage,heartbeat_unix=time.time(),
                elapsed_seconds=time.time()-started,counts=counts,**extra))

        def cached_fold(dna,local_cache):
            key=hashlib.sha256(dna.encode()).hexdigest()
            fresh=key not in fold_cache
            result=original_fold(dna,fold_cache)
            local_cache[key]=dict(result)
            if fresh:
                save(ROOT/'fold_cache.json',fold_cache)
            status('exact_rnafold',**current,shared_unique_folds=len(fold_cache))
            return result

        core.exact_fold=cached_fold
        models={}
        paths={epoch:ROOT/f'checkpoints/epoch{epoch}.pt' for epoch in EXPECTED}
        expected=dict(EXPECTED)

        def get_model(epoch):
            if epoch not in models:
                assert sha(paths[epoch])==expected[epoch]
                models[epoch]=load_model(paths[epoch],'cuda').eval().requires_grad_(False)
            return models[epoch]

        def measure(record,method,dna,seconds=0.,queries=0):
            destination=ROOT/'metrics'/record['sequence_id']/(method+'.json')
            if destination.exists():
                row=json.loads(destination.read_text())
                assert row['cds_dna']==dna and row['sequence_sha256']==hashlib.sha256(dna.encode()).hexdigest()
                return
            status('independent_seq2ribo',**current)
            evaluation.measure(record,method,dna,scorer,predictor,lock,seconds,queries)
            print('SCORED',method,record['sequence_id'],flush=True)

        def compare(records,epochs,split):
            for record in sorted(records,key=lambda r:(r['n_codons'],r['sequence_id'])):
                current.clear();current.update(panel_split=split,sequence_id=record['sequence_id'],n_aa=record['n_codons']-1)
                measure(record,'original',record['cds_dna'])
                for epoch in epochs:
                    method='v3' if epoch==0 else 'v4_epoch'+str(epoch)
                    current['method']=method
                    destination=ROOT/'search'/method/record['sequence_id']/'result.json'
                    if destination.exists():
                        result=json.loads(destination.read_text())
                        assert result['checkpoint_sha256']==expected[epoch] and result['steps']==80
                        assert result['source_sha256']==plan['source_sha256']
                    else:
                        status('latent_search_80_steps',**current)
                        model=get_model(epoch)
                        before=scorer.forward_sequences
                        folds_before=len(fold_cache)
                        t=time.monotonic()
                        seed=int(hashlib.sha256(('v4-full-20261026:'+record['sequence_id']).encode()).hexdigest()[:8],16)
                        result=core.search_v4(model,scorer,record,weights,steps=80,seed=seed)
                        result.update(checkpoint_sha256=expected[epoch],source_sha256=plan['source_sha256'],seed=seed,
                            sequence_id=record['sequence_id'],epoch=epoch,seconds=time.monotonic()-t,
                            hard_scorer_queries=scorer.forward_sequences-before,
                            physical_new_folds=len(fold_cache)-folds_before,
                            candidate_locked_before_seq2ribo=True)
                        save(destination,result)
                        print('OPTIMIZED',method,record['sequence_id'],result['best']['total_reward'],flush=True)
                    measure(record,method,result['best']['cds_dna'],result['seconds'],result['hard_scorer_queries'])
                    assert [state_hash(m) for m in (scorer,teacher)]==reference_hashes

        try:
            compare(validation,[6],'validation')
            compare(test,[6],'test')
            refresh()
            assert [state_hash(m) for m in (scorer,teacher)]==reference_hashes
            assert sha(paths[6])==EXPECTED[6]
            save(ROOT/'complete.json',dict(completed=True,validation_cases=24,test_cases=30,
                current_epoch=6,checkpoint_sha256=EXPECTED[6],scorer_weights_unchanged=True))
            save(ROOT/'status.json',dict(state='completed',heartbeat_unix=time.time(),current_epoch=6))
        except BaseException as exc:
            save(ROOT/'status.json',dict(state='failed',heartbeat_unix=time.time(),error=repr(exc),
                current_case=current,completed_cases_preserved=True))
            raise


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepare',action='store_true')
    args=parser.parse_args()
    if args.prepare:
        prepare()
    else:
        run()
