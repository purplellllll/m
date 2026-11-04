# Transformer+sTASEP v1：后续加减实验固定基线

用户于2026-11-05指定保留的旧版本：表达Pearson约0.87，宿主RPF参数版Transformer+sTASEP；主模型为profile/dynamics，两种子30101/30102集成。纯序列profile/sequence作为关闭sTASEP的配对对照一并保留。没有将后续LPT/Gated/CDS-only改版混入。

|版本|表达E217 Pearson|mRNA半衰期H601 Pearson|PURE可溶性S370 Pearson|
|---|---:|---:|---:|
|Transformer+sTASEP（主基线）|0.867530|0.574914|0.707576|
|Transformer（配对对照）|0.871980|0.573455|0.708881|

上述三列来自不同测定/宿主；H不是蛋白半衰期，S不是蛋白降解率。历史测试已多次查看；该基线不是SOTA或联合实测验收通过的声明。

## 下载与复验

固定tag：`baseline-transformer-stasep-v1-20261105`；源码基底commit：`7ac032ad09439e27f9eb32f2f3b495b5c0656b32`。权重仅包含自有两臂×两种子的4份安全导出，另含训练集拟合scaler、标签坐标和宿主占据参数。

```bash
git checkout baseline-transformer-stasep-v1-20261105
pip install -r multitask_scorer/requirements.txt
gh release download baseline-transformer-stasep-v1-20261105 --repo purplellllll/m --pattern transformer-stasep-v1-models.tar.gz
tar -xzf transformer-stasep-v1-models.tar.gz
python scripts/verify_transformer_stasep_baseline.py --models-root transformer_stasep_v1
```

```python
from multitask_scorer import Scorer
baseline = Scorer("transformer_stasep_v1", stage="profile", arm="dynamics", device="cpu")
control = Scorer("transformer_stasep_v1", stage="profile", arm="sequence", device="cpu")
# baseline.predict(dna=complete_CDS, host="human" or "ecoli", utr5=..., utr3=...)
```

保持原UTR/宿主/序列输入契约；只给CDS是另一输入条件，不能沿用native-context的H601成绩。ViennaRNA固定2.7.2。验证脚本核对4权重SHA、6项完整历史Pearson和3条真实验证序列×两臂×4输出的CPU重载；训练快照在historical/下，完整重训仍需公开原数据和历史路径准备，并非无数据一键训练包。

## 后续实验规则

从本tag/基线分支创建独立实验分支；原权重、scaler、参数、测试集和本卡保持冻结。每轮记录改动模块/输入/监督、种子、训练预算及父基线commit。用相同划分、输入权限和标签变换进行配对比较，先看验证集，再执行预定测试。需要改变输入或任务时明确新协议；不覆盖/冒称旧基线。

代码在multitask_scorer/；训练/特征快照在historical/multitask-scorers-20261102/multitask_profile_20261102/；协议及逐样本证据在evidence/multitask-scorers-20261102/multitask_profile_20261102/。BASELINE.json固定来源、所有模型和代码SHA。
