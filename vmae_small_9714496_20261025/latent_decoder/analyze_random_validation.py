"""Post-hoc metrics and CDS-paired statistics; no optimizer feedback."""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import csv
import gzip
import hashlib
import json
import math
from pathlib import Path
import numpy as np
from .tokenizer import CODONS, AA, encode_cds


def read_rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def cai_weights(data):
    counts=Counter({c:1 for c in CODONS})
    for path in sorted((data/'train').glob('*.gz')):
        with gzip.open(path,'rt') as handle:
            for line in handle:
                dna=json.loads(line)['cds_dna']
                counts.update(dna[i:i+3] for i in range(0,len(dna),3))
    maxima={aa:max(counts[c] for c,a in zip(CODONS,AA) if a==aa) for aa in set(AA)}
    return {c:counts[c]/maxima[a] for c,a in zip(CODONS,AA)}


def sequence_metrics(dna, weights):
    ids=encode_cds(dna)
    sense=[CODONS[i] for i in ids if AA[i]!='*']
    return dict(gc_fraction=(dna.count('G')+dna.count('C'))/len(dna),
        train_reference_cai=math.exp(sum(math.log(weights[c]) for c in sense)/len(sense)))


def synonym_changes(original, candidate):
    before=Counter(encode_cds(original));after=Counter(encode_cds(candidate))
    assert [AA[i] for i in encode_cds(original)]==[AA[i] for i in encode_cds(candidate)]
    result={}; weighted_tv=0.; total=sum(v for i,v in before.items() if AA[i]!='*')
    for aa in sorted(set(AA)-{'*'}):
        group=[i for i,a in enumerate(AA) if a==aa]; n=sum(before[i] for i in group)
        if not n:continue
        delta={CODONS[i]:(after[i]-before[i])/n for i in group}
        result[aa]=delta; weighted_tv+=sum(abs(d) for d in delta.values())*.5*n/total
    return result,weighted_tv


def paired_interval(values, groups, seed=20261020):
    rng=np.random.default_rng(seed)
    clusters=sorted(set(groups));indices=[[i for i,g in enumerate(groups) if g==c] for c in clusters]
    draws=[]
    for _ in range(5000):
        chosen=rng.integers(0,len(clusters),len(clusters))
        sample=[values[i] for c in chosen for i in indices[c]]
        draws.append(float(np.mean(sample)))
    return dict(mean=float(np.mean(values)),median=float(np.median(values)),
        ci95=np.quantile(draws,[.025,.975]).tolist(),ci97_5=np.quantile(draws,[.0125,.9875]).tolist(),
        wins=sum(v>1e-6 for v in values),ties=sum(abs(v)<=1e-6 for v in values),losses=sum(v< -1e-6 for v in values))


def write_csv(path, rows):
    with path.open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def fold_candidate(dna):
    import RNA
    structure,mfe=RNA.fold(dna.replace('T','U'));assert len(structure)==len(dna)
    return dict(sequence_sha256=hashlib.sha256(dna.encode()).hexdigest(),mfe=float(mfe),
        mfe_per_nt=float(mfe)/len(dna),viennarna_version=RNA.__version__,
        temperature_celsius=float(RNA.cvar.temperature),dangles=int(RNA.cvar.dangles))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--results',required=True);p.add_argument('--data',default='dataset')
    p.add_argument('--plan',default='docs/b456_random_validation_plan_20261020.json')
    p.add_argument('--output',required=True);p.add_argument('--fold',action='store_true')
    args=p.parse_args();root=Path(args.results);data=Path(args.data);out=Path(args.output)
    plan_path=Path(args.plan);plan=json.loads(plan_path.read_text());identities=plan['evaluation_ids']
    seeds=plan['seeds'];arms=['B0','B4','B5','B6'];all_rows=[];costs=[];steps=[];initials={}
    direct_scale=None
    for seed in seeds:
        run=root/'runs'/('seed_'+str(seed))
        receipt=json.loads((run/'completion_receipt.json').read_text())
        protocol=json.loads((run/'protocol.json').read_text())
        if direct_scale is None:direct_scale=protocol['direct_scale']
        assert direct_scale==protocol['direct_scale']
        assert receipt['completed'] and receipt['plan_sha256']==hashlib.sha256(plan_path.read_bytes()).hexdigest()
        assert receipt['model_weights_unchanged'] and receipt['scorer_weights_unchanged']
        rows=read_rows(run/'results.jsonl')
        assert len(rows)==len(identities)*len(arms)
        assert {(r['sequence_id'],r['arm']) for r in rows}=={(i,a) for i in identities for a in arms}
        for r in rows:
            assert r['best']['feasible'] and r['best']['protein_preserved']
            if r['arm']!='B0':assert r['scorer_forward_sequences']==240 and r['backward_sequence_evaluations']==80
            all_rows.append(dict(seed=seed,**r))
        initial=read_rows(run/'random_initial.jsonl');assert len(initial)==len(identities)
        initials[seed]={r['sequence_id']:r for r in initial}
        run_steps=read_rows(run/'steps.jsonl');assert len(run_steps)==len(identities)*3*80
        for identity in identities:
            for arm in ['B4','B6']:
                z=[r['R_z'] for r in run_steps if r['sequence_id']==identity and r['arm']==arm and r['step']>=40]
                assert len(z)==41 and max(z)-min(z)<1e-8
        steps.extend(dict(seed=seed,**r) for r in run_steps)
        for arm,values in receipt['arm_metrics'].items(): costs.append(dict(seed=seed,arm=arm,**values,
            process_peak_cuda_bytes=receipt['peak_cuda_allocated_bytes'],common_setup_scorer_forwards=receipt['common_setup_scorer_forwards']))
    for identity in identities:
        assert len({initials[s][identity]['latent_sha256'] for s in seeds})==len(seeds)
    out.mkdir(parents=True,exist_ok=True)
    cache_path=out/'rnafold_cache.jsonl'
    cache={r['sequence_sha256']:r for r in read_rows(cache_path)} if cache_path.exists() else {}
    sequences=sorted({r['best']['cds_dna'] for r in all_rows})
    if args.fold:
        missing=[dna for dna in sequences if hashlib.sha256(dna.encode()).hexdigest() not in cache]
        with cache_path.open('a') as handle, ProcessPoolExecutor(max_workers=4) as pool:
            for index,row in enumerate(pool.map(fold_candidate,missing)):
                key=row['sequence_sha256'];cache[key]=row
                handle.write(json.dumps(row)+'\n');handle.flush()
                if (index+1)%20==0:print('RNAfold',index+1,'/',len(sequences),flush=True)
    weights=cai_weights(data);metrics=[];proportions=[]
    for r in all_rows:
        dna=r['best']['cds_dna'];baseline=r['initial']['cds_dna']
        change,tv=synonym_changes(baseline,dna);key=hashlib.sha256(dna.encode()).hexdigest()
        before=encode_cds(baseline);after=encode_cds(dna)
        substitutions=Counter(CODONS[a]+'>'+CODONS[b] for a,b in zip(before,after) if a!=b)
        replaced_by_aa=Counter(AA[a] for a,b in zip(before,after) if a!=b)
        original_by_aa=Counter(AA[a] for a in before)
        folded=cache.get(key,{})
        metrics.append(dict(seed=r['seed'],sequence_id=r['sequence_id'],arm=r['arm'],
            n_codons=len(dna)//3,ribodecode_score=r['best']['hard_score'],ribodecode_gain=r['best_hard_gain'],
            best_step=r['best']['step'],final_feasible=r['final']['feasible'],
            mutable_hamming=r['best']['mutable_hamming'],decoder_kl=r['best'].get('decoder_kl',0.),
            R_z=r['best'].get('latent_l2',0.),R_delta=r['best'].get('direct_l2',0.),
            actual_residual_rms=direct_scale*math.sqrt(r['best'].get('direct_l2',0.)),
            synonym_weighted_tv=tv,changed_codons=sum(substitutions.values()),
            all_codons_replacement_fraction=sum(substitutions.values())/len(before),**sequence_metrics(dna,weights),
            rnafold_mfe=folded.get('mfe'),rnafold_mfe_per_nt=folded.get('mfe_per_nt'),sequence_sha256=key))
        proportions.append(dict(seed=r['seed'],sequence_id=r['sequence_id'],arm=r['arm'],
            synonymous_fraction_change=change,substitution_counts=dict(substitutions),
            per_aa_replacement_fraction={aa:replaced_by_aa[aa]/n for aa,n in original_by_aa.items() if aa!='*'}))
    lookup={(r['seed'],r['sequence_id'],r['arm']):r for r in metrics}
    paired=[]
    metadata={r['sequence_id']:r for r in plan['records']}
    secondary=['gc_fraction','train_reference_cai','rnafold_mfe','rnafold_mfe_per_nt',
               'synonym_weighted_tv','mutable_hamming','R_z','R_delta','actual_residual_rms','decoder_kl']
    for identity in identities:
        row=dict(sequence_id=identity,n_codons=metadata[identity]['n_codons'],cluster_id=metadata[identity]['cluster_id'])
        for arm in arms:
            values=[lookup[s,identity,arm]['ribodecode_gain'] for s in seeds]
            row[arm+'_mean_gain']=float(np.mean(values));row[arm+'_seed_sd']=float(np.std(values,ddof=1))
            row[arm+'_unique_best_sequences']=len({lookup[s,identity,arm]['sequence_sha256'] for s in seeds})
            for key in secondary:
                values=[lookup[s,identity,arm][key] for s in seeds]
                row[arm+'_mean_'+key]=float(np.mean(values)) if all(v is not None for v in values) else None
        for control in ['B4','B6']:
            row['B5_minus_'+control]=row['B5_mean_gain']-row[control+'_mean_gain']
            row['B5_wins_seeds_vs_'+control]=sum(lookup[s,identity,'B5']['ribodecode_gain']>lookup[s,identity,control]['ribodecode_gain']+1e-6 for s in seeds)
            for key in secondary:
                a,b=row['B5_mean_'+key],row[control+'_mean_'+key]
                row['B5_minus_'+control+'_'+key]=a-b if a is not None and b is not None else None
        for arm in ['B4','B5','B6']:
            for key in ['gc_fraction','train_reference_cai','rnafold_mfe','rnafold_mfe_per_nt']:
                a,b=row[arm+'_mean_'+key],row['B0_mean_'+key]
                row[arm+'_minus_B0_'+key]=a-b if a is not None and b is not None else None
        paired.append(row)
    groups=[r['cluster_id'] for r in plan['records']]
    summary=dict(cds=len(identities),seeds=seeds,independent_unit='CDS; seeds averaged within CDS',
        bootstrap_components=len(set(groups)),bootstrap_draws=5000,bootstrap_seed=20261020,
        predictor='RiboDecode: same optimizer scorer, not independent',folded_unique=len(cache),required_unique=len(sequences),
        true_random_initializations_verified=True,
        contrasts={name:paired_interval([r[name] for r in paired],groups) for name in ['B5_minus_B4','B5_minus_B6']},
        arms={arm:dict(mean_gain=float(np.mean([r['ribodecode_gain'] for r in metrics if r['arm']==arm])),
            improved_cds_mean_gain=sum(r[arm+'_mean_gain']>1e-5 for r in paired),
            per_seed_mean_gain={str(s):float(np.mean([r['ribodecode_gain'] for r in metrics if r['arm']==arm and r['seed']==s])) for s in seeds},
            mean_within_cds_seed_sd=float(np.mean([r[arm+'_seed_sd'] for r in paired])),
            mean_hamming=float(np.mean([r['mutable_hamming'] for r in metrics if r['arm']==arm])),
            mean_cai=float(np.mean([r['train_reference_cai'] for r in metrics if r['arm']==arm]))) for arm in arms})
    summary['length_strata']={str(lo)+'-'+str(hi):dict(n=sum(lo<=r['n_codons']<=hi for r in paired),
        **{name:float(np.mean([r[name] for r in paired if lo<=r['n_codons']<=hi]))
           for name in ['B5_minus_B4','B5_minus_B6']}) for lo,hi,_ in plan['length_strata']}
    summary['B5_wins_all_three_seeds']={control:sum(r['B5_wins_seeds_vs_'+control]==len(seeds) for r in paired) for control in ['B4','B6']}
    summary['posthoc_deltas']={arm:{key:float(np.mean([lookup[s,i,arm][key]-lookup[s,i,'B0'][key]
        for s in seeds for i in identities])) for key in ['gc_fraction','train_reference_cai','rnafold_mfe','rnafold_mfe_per_nt']
        if all(r[key] is not None for r in metrics)} for arm in arms}
    summary['search_cost']={arm:dict(total_search_seconds=sum(r['search_seconds'] for r in costs if r['arm']==arm),
        total_scorer_forwards=sum(r['scorer_forwards'] for r in costs if r['arm']==arm),
        total_backwards=sum(r['backwards'] for r in costs if r['arm']==arm),
        total_decoder_evaluations=sum(r['decoder_evaluations'] for r in costs if r['arm']==arm)) for arm in ['B4','B5','B6']}
    summary['trajectory_means']={arm:{str(step):{key:float(np.mean([r[key] for r in steps if r['arm']==arm and r['step']==step]))
        for key in ['R_z','R_delta','actual_residual_rms','decoder_kl']} for step in [40,80]} for arm in ['B4','B5','B6']}
    write_csv(out/'candidate_metrics.csv',metrics);write_csv(out/'paired_cds.csv',paired);write_csv(out/'compute_cost.csv',costs)
    with (out/'synonymous_fraction_changes.jsonl').open('w') as f:
        for r in proportions:f.write(json.dumps(r)+'\n')
    with (out/'regularization_trajectories.jsonl').open('w') as f:
        for r in steps:f.write(json.dumps({k:r[k] for k in ['seed','sequence_id','arm','step','phase','R_z','R_delta','actual_residual_rms','decoder_kl','hard_score','feasible']})+'\n')
    with (out/'best_candidates.fasta').open('w') as f:
        for r in all_rows:f.write('>{}_{}_{}\n{}\n'.format(r['seed'],r['sequence_id'],r['arm'],r['best']['cds_dna']))
    (out/'analysis.json').write_text(json.dumps(summary,indent=2))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,3,figsize=(13,4))
    for arm in ['B4','B5','B6']:
        for ax,key in zip(axes,['R_z','R_delta','actual_residual_rms']):
            matrix=np.array([[r[key] for r in steps if r['arm']==arm and r['step']==step] for step in range(1,81)])
            median=np.median(matrix,axis=1);lo,hi=np.quantile(matrix,[.25,.75],axis=1)
            ax.plot(range(1,81),median,label=arm);ax.fill_between(range(1,81),lo,hi,alpha=.12)
            ax.set(title=key,xlabel='Optimization step');ax.axvline(40.5,color='grey',linestyle='--',linewidth=.8)
    axes[0].legend();fig.suptitle('Median and IQR across CDS x seeds (descriptive, not confidence intervals)')
    fig.tight_layout();fig.savefig(out/'regularization_trajectories.png',dpi=160);plt.close(fig)
    table='\n'.join('| {} | {:.6f} | {:.2%} | {:.4f} |'.format(a,v['mean_gain'],v['mean_hamming'],v['mean_cai']) for a,v in summary['arms'].items())
    contrasts='\n'.join('| {} | {:+.6f} | [{:+.6f}, {:+.6f}] | {}/{}/{} |'.format(n,v['mean'],*v['ci95'],v['wins'],v['ties'],v['losses']) for n,v in summary['contrasts'].items())
    cost_table='\n'.join('| {} | {:.2f} | {} | {} | {} |'.format(a,v['total_search_seconds'],v['total_scorer_forwards'],v['total_backwards'],v['total_decoder_evaluations']) for a,v in summary['search_cost'].items())
    posthoc_table='\n'.join('| {} | {:+.4f} | {:+.6f} | {} | {} |'.format(a,v['gc_fraction']*100,v['train_reference_cai'],
        '{:+.4f}'.format(v['rnafold_mfe']) if 'rnafold_mfe' in v else '未计算',
        '{:+.6f}'.format(v['rnafold_mfe_per_nt']) if 'rnafold_mfe_per_nt' in v else '未计算') for a,v in summary['posthoc_deltas'].items())
    report='''# B4/B5/B6 随机起点配对验证

Origin Skill：academic-research-suite / experiment-agent；日期：2026-10-20。完整输入与权重哈希见各 seed 的 protocol/完成收据及原冻结清单。

状态：ANALYZED。固定 VMAE 和 RiboDecode；'''+str(len(identities))+''' 条预定分层 CDS，'''+str(len(seeds))+''' 个真正改变潜在初值的 seed。B5 为预先指定主方法候选，B4/B6 为对照。不同 seed 的初值哈希已验证不同；同 CDS/seed 三方法共享同一初值。所有完成收据、预算、同义性、best 可行性、分阶段冻结检查通过。

| 方法 | seed 均值后平均增量 | 平均可变位改动 | train-reference CAI |
|---|---:|---:|---:|
'''+table+'''

| 主比较 | 平均配对差 | 探索性 95% 区间 | 胜/平/负（CDS） |
|---|---:|---|---|
'''+contrasts+'''

该表是均值优势，不能解释为每条序列都更好：B5 对 B4/B6 在全部三个 seed 均胜出的 CDS 数分别为 '''+str(summary['B5_wins_all_three_seeds']['B4'])+'/'+str(len(identities))+'、'+str(summary['B5_wins_all_three_seeds']['B6'])+'/'+str(len(identities))+'''。长度层差值见 analysis.json，不能将总体均值外推至每个长度层。

以 CDS 为分析单位，先平均三个 seed，再按原 cluster_id 组件做 5,000 次 bootstrap；97.5% 多重比较口径区间见 analysis.json。胜平负容差为 1e-6。seed×CDS 不构成相同数量的独立样本；长度分层均值不是原数据集自然分布加权均值。区间包含零时既不能断言优越，也不能断言等价/非劣。

[逐 CDS 配对表](paired_cds.csv)、[全部候选指标](candidate_metrics.csv)、[计算成本](compute_cost.csv)、[同义密码子比例变化](synonymous_fraction_changes.jsonl)、[逐步正则与真实残差](regularization_trajectories.jsonl)、[轨迹图](regularization_trajectories.png)、[离散候选](best_candidates.fasta)。轨迹阴影是 CDS×seed 的四分位范围，不是置信区间；actual_residual_rms 为加到 decoder logits 的实际 Δ RMS，R_z/R_Δ 为标准化参数平方正则量。

每方法每 CDS 每 seed 80 次反向、240 次评分器前向；共同设置另计 6 次评分器前向。compute_cost.csv 分别报告各方法耗时、decoder 调用和反向/评分调用；显存是整个 seed 进程峰值，不能解释为各方法独立峰值。

| 方法 | 三个 seed 搜索秒数合计 | 评分器序列前向 | 序列反向 | decoder 序列调用 |
|---|---:|---:|---:|---:|
'''+cost_table+'''

以下均为每 CDS 内相对 B0 的差，先对 seed 平均；GC 单位为百分点，MFE 为 kcal/mol。更负的 MFE 只是更稳定的计算结构，不直接代表更高表达。

| 方法 | GC 差 | train-reference CAI 差 | MFE 差 | MFE/nt 差 |
|---|---:|---:|---:|---:|
'''+posthoc_table+'''

用户指定继续使用 RiboDecode，它也参与优化，因此该分数不是独立预测器验证。CAI 是训练集计数加 1 平滑后的相对同义频率参考，不能冒充特定细胞高表达基因参考。GC、CAI、MFE 与同义比例不参与候选选择。RNAfold 完整性：'''+str(len(cache))+'/'+str(len(sequences))+''' 个独特 best 候选；未计算的 MFE 保持为空。

B6 的 R_z 在 Direct 阶段为常数，对该阶段正在更新的 Direct 变量没有梯度；R_Δ 与 decoder KL 在 Latent 阶段为零。B5 的三个正则在联合阶段可同时影响更新。第 40/80 步的均值见 analysis.json 的 trajectory_means。轨迹描述正则与调度的相互作用，不能单凭轨迹隔离每项正则对收益的因果贡献。

统计解释核对（11/11 checked）：长度层方向见 length_strata，保留逐 CDS 结果避免生态推断；长度分层选择限制外推，不按评分/可行性筛选或条件化；诊断基率不适用；未按极端基线选择，保留配对对照；所有 CDS/seed 完整才运行主分析，初次部署失败另记；两个主比较同时报告且给出 97.5% 区间，其余指标描述性；执行前固定清单、seed、初始化幅度和预算，仍属于探索性研究；不从内部预测相关性推断表达因果或反向生物学机制。

局限：这是固定子集和固定模型下的随机初值稳定性分析；未验证独立评分器、训练集同源泄漏排除或真实表达提升。各 seed 与逐序列结果完整保留，不按最高 seed 选优，也不删去回落样本。

部署记录：第一次上传漏带 baseline_codon_order.json，尚未产生优化结果即退出；补齐后三个 seed 全部正常完成。原失败日志保留在原始证据包 logs/deployment_missing_tokenizer_reference*，未因该错误筛除或更换样本。
'''
    if plan['version']=='b456-formal-validation-20261020':
        report=report.split('\n部署记录：')[0]
    (out/'report.md').write_text(report,encoding='utf-8')
    print(json.dumps(summary),flush=True)


if __name__=='__main__':main()
