from pathlib import Path
import json,hashlib
import numpy as np
import pandas as pd
import torch
HERE=Path(__file__).resolve().parent
def read(p):return json.loads(Path(p).read_text(encoding='utf8'))
def main():
    a=read(HERE/'evaluation/summary.json');d=pd.read_csv(HERE/'evaluation/predictions.csv');assert len(d)==1188
    errors=[]
    for m in a['metrics']:
        s=d[d.task==m['task']];errors.append(abs(float(np.corrcoef(s.observed,s[m['model']])[0,1])-m['pearson']))
    assert max(errors)<1e-10;gates=[]
    for seed in [30101,30102]:
        p=HERE/'models'/f'dynamics_{seed}'/'best.pt';assert hashlib.sha256(p.read_bytes()).hexdigest()==read(p.parent/'complete.json')['best_sha256']
        c=torch.load(p,map_location='cpu',weights_only=True);gates.append(dict(seed=seed,best_step=c['step'],effective_residual_gate=float(.1*torch.tanh(c['model']['diffusion_gate']))))
    (HERE/'verification.json').write_text(json.dumps(dict(status='passed',n=len(d),arithmetic_max_error=max(errors),strict_reload=a['reload_checks'],gates=gates),indent=2),encoding='utf8')
    lines=['# 条件潜表示Diffusion对照实验','',
        '原目标：尝试用Diffusion替代无额外收益的图通路。最新多任务模型已经无GNN，因此本轮是在它上面新增条件去噪扩散分支，与无Diffusion的同预算profile-dynamics基线比较；不能称直接GNN/DDPM等价替换。','',
        '## 配方','',
        '相同数据/划分/输入、192维两层Transformer、两种子、每种子6000更新、batch8和验证集三主要端点macro Pearson选模。公共骨干初始权重逐张量相等，采样次序相同。',
        '扩散16步线性beta从0.01至0.25，目标是detach后的context潜表示，条件是序列/AA/宿主/物理输入token；辅助epsilon MSE固定权重0.05，不使用表型标签作扩散目标。推理固定种子噪声和4步确定性DDIM(15,10,5,0)。',
        '以0.1*tanh门控残差融合回原context，门控初始0；记录最终门控。参数量、FLOPs和推理时间增加，因此不声称完全计算预算匹配。没有生成新RNA，也没有把表达/HL预测当真实标签。','',
        '## 实测端点结果','', '|任务|模型|N|Pearson|95%条件区间|','|---|---|---:|---:|---|']
    for r in a['metrics']:
        if r['model'] not in ['dynamics_ensemble','no_diffusion_ensemble']:continue
        label='Diffusion集成' if r['model']=='dynamics_ensemble' else '无Diffusion基线'
        lines.append(f"|{r['task']}|{label}|{r['n']}|{r['pearson']:.6f}|[{r['ci95'][0]:.6f}, {r['ci95'][1]:.6f}]|")
    lines+=['','|任务|Diffusion减无Diffusion Δr|95%配对区间|','|---|---:|---|']
    for r in a['paired']:
        if r['reference']=='no_diffusion_ensemble':lines.append(f"|{r['task']}|{r['delta']:.6f}|[{r['ci95'][0]:.6f}, {r['ci95'][1]:.6f}]|")
    lines+=['','## 解释边界','',
        '功能表达为217条Ecoli mRFP荧光；HL为601人源基因实测log10(HL+1)；可溶性为370条PURE蛋白测定。三者不是同批构建体，不能逐行拼接联合标签。测试历史已多次打开，本轮为探索，不是独立确认；区间未包括训练随机性、同源性或多重搜索。',
        '门控非零或分支有梯度只证明函数依赖，不证明有稳定额外收益。仍需同时看三项和配对区间，不按测试选择有利种子或时间步。','',
        '训练协议comparison字段沿用旧模板，未运行sequence组；实际对照为上轮启用物理但无Diffusion的dynamics模型，evaluation通过锁定逐样本预测复用该基线，no_diffusion_ensemble标识为准。原定功能表达+细胞mRNA半衰期+可溶性验收尚未通过；Diffusion测试不能替代缺失的同样本实测配对数据。',
        '原已发布多任务证据：[GitHub Release](https://github.com/purplellllll/m/releases/tag/multitask-scorers-20261102)。本轮Diffusion结果独立补充。']
    (HERE/'Diffusion测试报告.md').write_text('\n'.join(lines)+'\n',encoding='utf8');print(json.dumps(dict(gates=gates,paired=[r for r in a['paired'] if r['reference']=='no_diffusion_ensemble']),indent=2))
if __name__=='__main__':main()
