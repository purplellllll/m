"""Compare frozen epoch6 with existing methods on the identical 30 CDS identities."""
import argparse
import csv
import json
import math
from pathlib import Path
import statistics
import time

BASE = Path('/root/gpufree-data')
ROOT = BASE/'vmae_v4_epoch6_current_20261027'
KEYS = ('ribodecode_score', 'seq2ribo_score', 'gc_percent', 'cai', 'mfe_kcal_mol', 'mfe_per_nt')


def save(name, value):
    target = ROOT/name
    temporary = target.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    temporary.replace(target)


def report():
    panel = json.loads((ROOT/'test_panel.json').read_text())
    ids = {r['sequence_id'] for r in panel}
    rows = []
    sources = [
        ('vmae_30cds_20261025', {'small': 'VMAE_9.7M_Hard', 'native': 'RiboDecode_native', 'original': 'original'}),
        ('vmae_tiny_30cds_20261025', {'small': 'V3_2.5M_Hard'}),
        ('vmae_v4_full_optimization_20261026', {'v3': 'V3_same_reward', 'v4_epoch2': 'V4_epoch2', 'v4_epoch3': 'V4_epoch3'}),
        ('rna_external_comparison_20261026', None),
        (ROOT.name, {'v4_epoch6': 'V4_epoch6_current'}),
    ]
    for directory, mapping in sources:
        for path in sorted((BASE/directory/'metrics').glob('*/*.json')):
            row = json.loads(path.read_text())
            if row['sequence_id'] not in ids or (mapping is not None and row['method'] not in mapping):
                continue
            rows.append(dict(row, method=mapping[row['method']] if mapping is not None else row['method'], source_file=str(path)))
    baseline_csv = BASE/'codon_baselines_20261025/metrics.csv'
    if baseline_csv.exists():
        with baseline_csv.open(encoding='utf-8-sig', newline='') as stream:
            for row in csv.DictReader(stream):
                if row['sequence_id'] in ids and row['method'] in ('codontransformer', 'fpp_codonbert'):
                    row.update({key: float(row[key]) for key in KEYS})
                    rows.append(dict(row, source_file=str(baseline_csv)))
    lookup = {(r['sequence_id'], r['method']): r for r in rows}
    assert len(lookup) == len(rows), 'Duplicate method/identity; inspect provenance'
    assert all(math.isfinite(float(r[k])) for r in rows for k in KEYS)
    methods = sorted({r['method'] for r in rows})
    means = {m: dict(n=sum(r['method'] == m for r in rows), means={
        k: statistics.mean(float(r[k]) for r in rows if r['method'] == m) for k in KEYS}) for m in methods}
    paired = {}
    for method in methods:
        if method == 'V4_epoch6_current':
            continue
        common = sorted(i for i in ids if (i, method) in lookup and (i, 'V4_epoch6_current') in lookup)
        if common:
            paired[method] = dict(n=len(common), sequence_ids=common, mean_delta={k: statistics.mean(
                float(lookup[i, 'V4_epoch6_current'][k])-float(lookup[i, method][k]) for i in common) for k in KEYS})
    save('all_model_summary.json', dict(methods=means, paired_epoch6_minus_baseline=paired,
        expected_test_cases=30, seq2ribo_used_for_selection=False, generated_unix=time.time()))
    fields = ['sequence_id', 'method', *KEYS, 'source_file']
    with (ROOT/'all_model_metrics.csv.tmp').open('w', newline='', encoding='utf-8-sig') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction='ignore')
        writer.writeheader(); writer.writerows(rows)
    (ROOT/'all_model_metrics.csv.tmp').replace(ROOT/'all_model_metrics.csv')
    text = '# 第6轮当前版本：完整隐空间优化与独立评估\n\n'
    text += '固定24条验证序列及30条测试序列；当前版本不按seq2ribo重新选序列。以下表格仅列测试集。\n\n'
    text += '| 方法 | N/30 | RiboDecode↑ | seq2ribo↑ | GC%↑ | CAI↑ | MFE↓ | MFE/nt↓ |\n|---|---:|---:|---:|---:|---:|---:|---:|\n'
    for method, value in means.items():
        text += '| '+method+' | '+str(value['n'])+' | '+' | '.join(f"{value['means'][k]:.6f}" for k in KEYS)+' |\n'
    text += '\n不同N的全表均值不可直接排名；下表在各方法与第6轮已完成的相同序列上配对。\n\n'
    text += '| 对照方法 | 配对N | ΔRiboDecode | Δseq2ribo | ΔGC百分点 | ΔCAI | ΔMFE | ΔMFE/nt |\n|---|---:|---:|---:|---:|---:|---:|---:|\n'
    for method, value in paired.items():
        text += '| '+method+' | '+str(value['n'])+' | '+' | '.join(f"{value['mean_delta'][k]:+.6f}" for k in KEYS)+' |\n'
    text += '\nV3_same_reward用于控制搜索奖励变化；V3_2.5M_Hard保留原Hard单目标方法。不同方法的搜索预算不同。\n'
    text += '\nRNop：官方权重不可获取。LinearDesign和完整RiboDecode以实际已完成N为准，不补造结果。\n'
    text += '\nseq2ribo：HEK293、CDS-only TE、三个固定随机种子，各8次sTASEP；评分为预测值，不代表实测表达倍数。\n'
    text += '\nGC按碱基比例、CAI按固定训练参考、MFE按原统一ViennaRNA设置；模型专属评分器仅报告已实际计算的结果。\n'
    (ROOT/'COMPARISON_ZH.md.tmp').write_text(text, encoding='utf-8')
    (ROOT/'COMPARISON_ZH.md.tmp').replace(ROOT/'COMPARISON_ZH.md')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--watch', action='store_true')
    args = parser.parse_args()
    while True:
        report()
        state = json.loads((ROOT/'status.json').read_text())['state'] if (ROOT/'status.json').exists() else 'prepared'
        if not args.watch or state in ('completed', 'failed'):
            break
        time.sleep(60)
