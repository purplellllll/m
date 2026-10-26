# V4：从 V3 加入三个持续奖励并微调

V4 从 V3 已发布的 **2,498,240 参数、epoch42** 冻结权重开始，不加载此前多目标实验的微调权重。起点 SHA256：`6592a07293da25b18f9ed71584e48dc4da23fa121b4f735f10d26ee7da26c6af`。

“CG”沿用此前约定，表示 **GC 含量**。三个奖励都不设达标阈值，不截断收益；以各自亲本序列为零点只用于中心化。

```
R_GC  = (GC_candidate − GC_parent) / 0.05
R_CAI = (CAI_candidate − CAI_parent) / 0.05
R_MFE = (MFE_parent/nt − MFE_candidate/nt) / 0.05

R_total = RiboDecode_Hard + R_GC + R_CAI + R_MFE
```

GC 使用 0–1 分数，分母 0.05 对应 5 个百分点。CAI 使用固定 TRAIN 密码子频率参考，并排除 stop。MFE 为 ViennaRNA 2.7.2、37°C、dangles2 的真实全长 RNAfold 能量；除序列长度后参与跨序列训练。在同一蛋白、同长度的候选间，更负的 MFE/nt 与更负的全长 MFE 排序一致。

## 微调

保留 V3 架构和重建项：

```
L_reconstruction = CE_mask + CE_full + 0.001 × KL
L_train = L_reconstruction
          − 0.1 × (Hard + R_GC + R_CAI)
          + 0.1 × 16 × L_SCST_MFE
L_SCST_MFE = −stopgrad(R_MFE_sample − R_MFE_greedy) × Σ log p(sampled_codon)
```

Hard、GC 和 CAI 使用同义密码子约束下的离散前向及 straight-through 梯度。MFE 不假装可直接反传：每 16 个 microbatch 选择批内一条序列，真实折叠采样候选与贪心候选，通过 SCST 策略梯度训练；16 是周期采样补偿。SCST 代理损失可以很大或为负，不能按数值把它解释成 MFE 或模型改善程度。

- 固定同源分组 split0：TRAIN 23,812 / VAL 2,592 / TEST 6,392；不轮换五折。
- TRAIN 全量微调 3 轮，AdamW，LR `1e-5`，weight decay0.01，microbatch4，有效batch4，grad clip1。
- BF16 模型前向；FP32 参数、损失与评分；验证使用 FP32。
- 保留 seed20261026、MASK0.15、重建CE与KL；评分器冻结，保留既定 HEK293T 条件、丰度4.5。
- 在固定24条VAL（短中长各8）上，以真实离散候选的同一组合目标选择权重。保留epoch0基线；微调若未优于基线，最优文件可以仍为V3。
- TEST面板不参与训练或权重选择。此前30条对照已经被探索性分析使用，不能将其重新称为全新最终测试。

两个模型/优化目标分别记录：更新 VMAE 参数属于微调；固定 VMAE 参数、只更新输入潜变量属于隐空间搜索。

## V4 隐空间搜索

`tools/vmae_v4_rewards_20261026.py::search_v4` 使用同一组奖励：默认80步、AdamLR0.05、标准化半径8，MFE每10步用真实RNAfold反馈一次。搜索后对全部唯一归档候选做精确折叠，按完整 `R_total` 选择，原序列也保留在归档中。

搜索期间 VMAE 与评分器均冻结，返回前核查状态哈希不变。序列始终保持相同蛋白、起始和终止密码子。seq2ribo不参与奖励或候选挑选。

## 已执行验证

- 14项本地及服务器检查通过，覆盖三个奖励方向、越过原序列后继续奖励、SCST正负梯度、PAD、固定stop和旧公共函数回归。
- GPU长序列预检通过：4×1500 codons，真实RNAfold反馈，有限损失和梯度；峰值CUDA分配1,167,349,248 bytes。此预检不执行optimizer.step。
- 真实TRAIN短序列的10步GPU隐空间搜索通过，覆盖MFE反馈，6个唯一精确折叠；V3权重与评分器状态未变。
- 实际微调是否已更新参数以 `first_update.json` 为准，进度以 `status.json` / `progress.json` 为准，训练完成以 `complete.json` 为准。

## 运行位置与复现

服务器工作目录：`/root/gpufree-data/vmae_v4_rewards_20261026`。执行源：`/root/gpufree-data/vmae_transfer_assessment_20261024`，Python：`/root/gpufree-data/vmae_gpu_env/bin/python`。

```bash
cd /root/gpufree-data/vmae_transfer_assessment_20261024
python -m unittest tools.test_vmae_v4_rewards_20261026 tools.test_vmae_multiobjective_core_20261026 -q
python tools/train_vmae_v4_20261026.py --preflight
python tools/train_vmae_v4_20261026.py
```

既有训练已启动时不要重复运行。程序使用进程锁；恢复时校验原plan。`last.pt` 保存完整epoch边界的优化器/RNG状态，epoch内部中断会从上一完整epoch恢复；没有宣称逐microbatch断点恢复。

这是原服务器路径的运行快照，完整复现需要 `plan.json` 指明的固定数据、V3权重、评分器资产与条件文件；不要直接在缺少这些资产的GitHub克隆目录中启动。

V4是本次单调奖励试验，不代表GC、CAI、MFE和翻译评分一定同时提高。完成后须分别报告四项奖励、两个翻译评分器、长度分组及运行成本；不能只依据训练代理loss宣称效果更好。
