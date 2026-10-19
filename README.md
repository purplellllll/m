# DS-RiboDecode

研究潜在空间 z 与直接密码子空间 Δ 的联合优化。当前已完成条件 Codon β-VMAE 的 50 轮训练，以及梯度回放、masked loss 平台诊断和生成先验的初步联合评估。

## 当前状态（2026-10-19）

- 模型：Encoder/Decoder 各 6 层，hidden 512，8 heads，FFN 2048，每个密码子 latent 256，RoPE；输出使用同义密码子硬约束。
- 冻结数据：[dataset](dataset/README.md)，GENCODE v28 与 RiboDecode 去重 CDS，train/val/test 为 **35,710 / 3,949 / 4,342**。当前语料未纳入百万条 GEMORNA 数据，也未认证近同源隔离。
- 完整验证集：masked CE **0.976372**，训练集频率基线 **1.144230**，相对改善 **14.67%**。
- 新增 256 条验证 CDS 的配对评估：模型 masked CE **0.966981**，频率基线 **1.140876**，打乱 masked-input 潜变量后 CE **1.189174**。支持将模型作为候选生成先验进入下游对照试验；尚未验证 z/Δ 优化收益或表达量提升。

## 代码与结果

1. [架构、损失与训练说明](latent_decoder/README.md)
2. [50 轮完成记录](docs/VGPU_west_LSCO_lr_50ep_completion_2026-10-19.md)
3. [单条 CDS 回放与参数更新检查](docs/VGPU_single_record_replay_2026-10-19.md)
4. [完整数据 masked loss 平台诊断](docs/VGPU_mask_plateau_diagnosis_2026-10-19.md)
5. [生成先验联合评估](docs/VMAE_prior_joint_assessment_2026-10-19.md)
6. [权重校验清单](deliverables/vmae_50ep_preserved_manifest_20261019.json)

研究阶段不同，配置也不同。已完成的 50 轮 run 对应 `config_lsco_lr_20ep.json` 起训及 `config_lsco_lr_50ep.json` 延长；`config.json` 是架构默认配置，不等于本次运行配置。请按对应报告复现，不覆盖已有 run。

```bash
python -m unittest discover -s latent_decoder -t . -p "test_*.py" -v
```

训练和诊断依赖 PyTorch、NumPy；绘图需要 matplotlib。原 GPU 实验环境为 Python 3.8 / PyTorch 2.0.0+cu118；配对 CPU 评估为 PyTorch 2.14.0+cpu。不同设备的相同 seed 不保证相同掩码。

权重 `.pt` 和部署归档不提交到 Git：第 37 轮 best 与第 50 轮 last 已另存于本地及服务器并校验，仓库保存代码、指标和校验清单。仓库中的历史报告按其日期保留；历史 [2026-10-16 Release](https://github.com/purplellllll/m/releases/tag/vmae-20ep-20261016) 是旧架构，不能用于恢复本次模型。

## 研究方案

- [整体方案](docs/DS_RiboDecode_plan_2026-10-16.md)
- [双空间数学与实验设计](docs/ds_ribodecode_math_review.md)
- [数据审阅](docs/ds_ribodecode_data_review.md)

下一阶段需要在相同起始序列和计算预算下比较 z-only、Δ-only 与 z+Δ，并用独立评价检查提升。重建指标及潜变量依赖均不等同于 mRNA 表达量改善。
