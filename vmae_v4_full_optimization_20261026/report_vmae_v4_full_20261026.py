"""Report tuning ablation separately from the preserved original V3 hard-only baseline."""
import argparse
import csv
import json
from pathlib import Path
import statistics
import time

ROOT=Path('/root/gpufree-data/vmae_v4_full_optimization_20261026')
OLD=ROOT.parent/'vmae_tiny_30cds_20261025'
KEYS=('ribodecode_score','seq2ribo_score','cai','gc_percent','mfe_kcal_mol','mfe_per_nt')


def report():
    plan=json.loads((ROOT/'plan.json').read_text())
    previous=json.loads((OLD/'plan.json').read_text())
    assert previous['checkpoint_sha256']==plan['checkpoints']['0']
    assert previous['search']['steps']==plan['steps']==80
    assert previous['search']['lr']==plan['latent_lr'] and previous['search']['radius']==plan['latent_radius']
    assert json.loads((ROOT/'cai_weights.json').read_text())==json.loads((OLD/'cai_weights.json').read_text())
    reference={p.parent.name:json.loads(p.read_text()) for p in (OLD/'metrics').glob('*/small.json')}
    assert set(reference)==set(plan['test_ids'])
    text='# V4 完整优化阶段报告\n\n'
    text+='V3同奖励搜索用于区分微调的作用；原V3仅Hard搜索作为历史方法基线另列。不同N不得直接排名。\n\n'
    summary=json.loads((ROOT/'summary.json').read_text())
    for split,methods in summary['panels'].items():
        text+='## '+split+'\n\n| 方法 | N | RiboDecode | seq2ribo | CAI | GC% | MFE | MFE/nt |\n|---|---:|---:|---:|---:|---:|---:|---:|\n'
        if split=='test':
            text+='| V3原方法：仅Hard奖励 | 30 | '+' | '.join(f'{statistics.mean(r[k] for r in reference.values()):.5f}' for k in KEYS)+' |\n'
        for method,value in methods.items():
            label=method+'（同奖励搜索）' if method=='v3' else method
            text+='| '+label+' | '+str(value['n'])+' | '+' | '.join(f"{value['means'][k]:.5f}" for k in KEYS)+' |\n'
    deltas=[]
    for p in (ROOT/'metrics').glob('*/*.json'):
        row=json.loads(p.read_text())
        if row['sequence_id'] not in reference or row['method']=='original':
            continue
        old=reference[row['sequence_id']]
        assert row['n_aa']==old['n_aa']
        assert all('seq2ribo_seed_'+str(s) in row and 'seq2ribo_seed_'+str(s) in old for s in (20261020,20261021,20261022))
        deltas.append(dict(sequence_id=row['sequence_id'],method=row['method'],
            comparator='V3_original_hard_only',**{k:row[k]-old[k] for k in KEYS}))
    text+='\n## 与原V3方法的同身份配对差值\n\n| 方法 | 配对N | ΔHard | Δseq2ribo | ΔCAI | ΔGC百分点 | ΔMFE |\n|---|---:|---:|---:|---:|---:|---:|\n'
    for method in sorted({r['method'] for r in deltas}):
        selected=[r for r in deltas if r['method']==method]
        text+='| '+method+' | '+str(len(selected))+' | '+' | '.join(f'{statistics.mean(r[k] for r in selected):+.5f}' for k in KEYS[:-1])+' |\n'
    if deltas:
        with (ROOT/'paired_vs_v3_original.csv.tmp').open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(deltas[0]));w.writeheader();w.writerows(deltas)
        (ROOT/'paired_vs_v3_original.csv.tmp').replace(ROOT/'paired_vs_v3_original.csv')
    if (ROOT/'decision.json').exists():
        text+='\n正式条件判定：\n\n```json\n'+(ROOT/'decision.json').read_text()+'```\n'
    else:
        text+='\n完整比较尚未结束，追加五轮尚未判定；不按部分结果提前触发。\n'
    text+='\nseq2ribo只作独立评价。结果是预测评分，不代表实测表达。\n'
    tmp=ROOT/'COMPARISON_ZH.md.tmp'
    tmp.write_text(text,encoding='utf-8');tmp.replace(ROOT/'COMPARISON_ZH.md')


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--watch',action='store_true')
    args=parser.parse_args()
    while True:
        report()
        if not args.watch or json.loads((ROOT/'status.json').read_text())['state'] in ('completed','failed'):
            break
        time.sleep(60)
