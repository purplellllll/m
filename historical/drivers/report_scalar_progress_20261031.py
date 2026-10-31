"""Independently verify the downloaded scalar snapshot and describe real results."""
import hashlib
import json
from pathlib import Path
import zipfile
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]/'runs/scalar_downstream_20261031'


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    receipt=json.loads((ROOT/'progress_archive_receipt.json').read_text());archive=ROOT/'progress_snapshot.zip'
    assert archive.stat().st_size==receipt['bytes'] and sha(archive)==receipt['sha256']
    folder=ROOT/'progress_snapshot'
    with zipfile.ZipFile(archive) as z:
        for name in z.namelist():assert not Path(name).is_absolute() and '..' not in Path(name).parts
        z.extractall(folder)
    manifest=json.loads((folder/'progress_files_manifest.json').read_text())
    for name,digest in manifest['files'].items():assert sha(folder/name)==digest
    result=json.loads((folder/'protein_ours_locked_test_audit.json').read_text())
    # CSV preserves the shortest round-trip decimal of float32 predictions;
    # restore the source dtype before independently recomputing float64 PCC.
    pred=pd.read_csv(folder/'protein_ours_locked_test_predictions.csv',dtype={k:'float32' for k in ['ours','ours_30101','ours_30102','ours_30103']})
    errors=[]
    for m in result['metrics']:
        group=pred if m['subset']=='native219' else pred[pred.clean_test]
        pcc=float(np.corrcoef(group.observed.to_numpy(dtype=float),group[m['model']].to_numpy(dtype=float))[0,1])
        errors.append(abs(pcc-m['pearson']));assert errors[-1]<1e-12
    audit=dict(status='passed',snapshot_archive_sha256=receipt['sha256'],n_manifest_files_verified=len(manifest['files']),
        n_independent_Pearson_rechecks=len(errors),max_error=max(errors),baseline_comparison_complete=False)
    (ROOT/'local_progress_audit.json').write_text(json.dumps(audit,indent=2))
    state=json.loads((folder/'progress_snapshot.json').read_text())
    report=['# 标量下游任务首轮实测','',
        '标量输出已接入真实标签，蛋白任务六组初轮训练已完成；目前尚未证明在RiboDecode、seq2ribo全部下游任务上超过原模型，不能宣布SOTA。','',
        '## 任务对齐','',
        '| 对照 | 标量输出 | Pearson颗粒度 |','|---|---|---|',
        '| RiboDecode | 实际RNA表达和10552基因环境条件下的log1p(5×RPF_RPKM) | 同环境跨基因，再跨环境平均 |',
        '| seq2ribo TE | Mean、HEK293、LCL、RPE-1分别单任务；CDS-only/含UTR分别比较 | 相同划分的跨转录本相关 |',
        '| seq2ribo蛋白表达 | mRFP实测表达 | 同义序列变体间相关 |','',
        '三路中间表征分别做屏蔽均值/最大池化、归一化及可学习增益，再经非线性MLP输出标量。RPF另接入RNA和环境；UTR版另接入完整5′/3′UTR。归一化位置分布不会求和冒充TE。','',
        '## 蛋白初轮','',
        '每组2000步，以三个种子的验证均值选中端到端微调。权重和选择在打开测试之前锁定。编码器来自此前已完成的width424 GraphPolishModel；当时新一轮扩大LIVER训练尚未完成，不能称其为本次初始化。','',
        '| 配方 | 种子 | 最佳验证Pearson |','|---|---:|---:|']
    for r in state['runs']:
        name,seed=r['run'].rsplit('_',1)
        report.append(f'| {name} | {seed} | {r["best_validation_pcc"]:.6f} |')
    report += ['', '| 锁定后的测试集 | N | 三种子集成Pearson | MSE |','|---|---:|---:|---:|']
    for m in result['metrics']:
        if m['model']=='ours':report.append(f'| {m["subset"]} | {m["n"]} | {m["pearson"]:.6f} | {m["mse"]:.6f} |')
    report += ['',
        'mRFP原划分1021/219/219；统一大小写和T/U后存在2条训练—测试重复CDS。本模型从训练中剔除对应2行，实际1019训练、219验证，同时报告原219测试及去重217测试。完整数据来自同组CodonMoE；seq2ribo公开6条测试序列和标签全部精确匹配，但完整原始划分身份尚未完全核证。','',
        '测试阶段仅使用验证选定的配方及三种子集成；单种子成绩仅按预设报告，不据测试重新选种子。冻结的seq2ribo四种细胞蛋白权重正按官方API、每条100次sTASEP计算同集结果，完成后生成配对差值和区间。测试已打开，不能再用这219/217标签回调参数。','',
        '## 后台队列','',
        'TE有10348条CDS；当前为完整结构和四次模拟准备阶段。后续四标签×两输入×两模型×三种子，共48个单任务训练。双方使用同一RiboNN基因fold及同一缓存物理输入，是共同输入重训练，不能称为seq2ribo原始分层划分或原生仿真配方复现。对照只从其profile权重初始化，TE头重新训练，不使用可能见过测试标签的公开TE头。','',
        'RPF有3245241条有效基因—环境记录、36484条不同CDS，200训练环境、120测试环境。首阶段冻结当前编码器、训练条件标量头，并设置RNA/环境对照，共6个训练。样本为90–4500nt规范CDS共同子集，不能冒称原文全体样本SOTA。作者训练基因内留作我方验证的基因，在新环境评测中另外报告，不误标为本模型见过的训练基因。','',
        '队列最多126000优化器步：蛋白12000、TE96000、RPF18000；每100步完整验证、10次未提升早停。服务器独立项目/root/scalar_downstream_20261031，控制器PID10781，后台仍运行。','',
        '[SwanLab项目](https://swanlab.cn/@purplellllll/RiboSeq-GNN-Transformer-20261031)','',
        '## 核验','',
        '真实蛋白batch的标量形状、三路梯度、权重保存/重载通过。独立环境Mamba2.2.6.post3编译版与未修改2.2.4参考定义的四层模型最大差2.384e-7；四个profile预训练编码器上的CDS-only/UTR任务头8条件均通过前反向。既有编码器四个源码文件与其原训练项目逐个哈希相同。','',
        f'本地归档及{len(manifest["files"])}个文件SHA256核验通过；从保存预测独立重算全部8项测试Pearson，最大误差{max(errors):.3g}。','',
        '原始数据、划分与后续限制详见protocol.json、local_data_audit.json；训练及选模证据在progress_snapshot/runs/和protein_evaluation_locked.json。']
    (ROOT/'标量任务首轮实测与后续队列.md').write_text('\n'.join(report)+'\n',encoding='utf-8')
    print(json.dumps(audit));print(json.dumps(dict(completed_runs=state['completed_runs'],controller_alive=state['controller_alive'])))


if __name__=='__main__':main()
