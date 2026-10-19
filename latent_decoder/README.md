# Protein-Conditioned Codon-level β-VMAE

按 `DS_RiboDecode_VMAE_Architecture_Acceptance.docx`（v1.0，2026-10-17）实现的架构。当前默认是正式研究版；“正式”指配置规模，不代表通过全部科研表现验收。本实现是 DS-RiboDecode 的研究设计，不是 LSCO 原架构的严格复现。

2026-10-19 状态：已完成 50 轮 GPU 训练、独立参数更新回放和 masked loss 平台诊断。参见[完成记录](../docs/VGPU_west_LSCO_lr_50ep_completion_2026-10-19.md)及[生成先验联合评估](../docs/VMAE_prior_joint_assessment_2026-10-19.md)。`evaluate_prior.py` 额外在相同掩码位置比较模型、频率基线和潜变量置换；`shuffled_masked_ce` 与历史 clean-input 的 `shuffled_latent_ce` 含义不同。下方 2026-10-17 验收表保留当日检查范围，后续进展以这些报告为准。

学习率选择实验使用独立配置：0.01、0.1、0.001 各 5 epochs，见 [启动与曲线说明](LR_SWEEP.md)。

## 架构与配置

| 项目 | 正式版 `config.json` | 缩小版 `config_mvp.json` |
|---|---:|---:|
| Encoder / Decoder 层数 | 6 / 6 | 4 / 4 |
| hidden / heads / FFN | 512 / 8 / 2048 | 384 / 6 / 1536 |
| 每密码子 latent 维度 | 256 | 128 |
| 参数量 | 38,302,784 | 14,403,776 |
| 位置机制 | RoPE | RoPE |
| 注意力 | 双向、非自回归 | 双向、非自回归 |
| 最大 CDS / 位置容量 | 1500 / 1536 | 1500 / 1536 |
| 输出类别 | 64 个原始 codon logits | 64 个原始 codon logits |

```text
CDS tokens [B,L] ── Codon Embedding ─┐
                                   + → Encoder → μ、logσ² [B,L,d_z]
目标蛋白 a [B,L] ── AA Embedding ───┘                 ↓ 重参数采样
                                            z [B,L,d_z]
                                                   ↓ Linear
目标蛋白 a [B,L] ── AA Embedding ──── + → Decoder → D(z,a) [B,L,64]
                                                        ↓ + αΔ
                                               同义硬掩码 → softmax / argmax
```

Encoder 与 Decoder 共用 AA embedding。RoPE 作用于每层注意力的 Q/K，不再叠加绝对位置 embedding。每个 block 为 Pre-LN、双向 self-attention、残差与 GELU FFN。无原 CDS 绕过 z 的连接。PAD 不参与注意力 key、损失或概率输出；输出 head 不含 PAD/MASK 类别。CDS 上限含末端 STOP；1536 是 RoPE 容量预留，当前 tokenizer 不额外添加 BOS/EOS，超过 1500 时拒绝，不截断。

为了兼容训练入口，`encode(tokens, target_tokens)`、`decode(z, target_tokens)` 接口保留。`target_tokens` 必须是有效 codon ID 或 PAD；模型内部仅由其查表得到对齐 AA（含末端 STOP）和 PAD mask。Decoder 不读取其 codon embedding；同义替换 target 不改变固定 z 的原始输出。数据入口仍由统一 tokenizer 检查起始/终止和内部 STOP。

## 双空间接口

```python
model.eval().requires_grad_(False)
z = model.encode(tokens, tokens)[0].detach().requires_grad_(True)
delta = torch.zeros(*tokens.shape, 64, device=tokens.device, requires_grad=True)
logits = model.decode(z, tokens)  # 连续原始 logits；不在此处 hard mask / argmax
probs = codon_probabilities(logits, tokens, temperature=1.0, delta=delta, alpha=0.5)
# probs: [B,L,64]；再传给可微性质预测器，候选优化器只更新 z、delta。
```

`codon_probabilities` 执行 `softmax((D(z,a)+alpha*delta+M(a))/temperature)`；非法 codon 为 `-inf`，PAD 输出全零。Δ 支持 `[B,L,64]` 或单目标/共享 `[L,64]`。`alpha=0` 或 `delta=0` 退化为纯潜在分支；以零 latent logits 加 Δ 可做直接空间对照。原始 decode logits 为有限值，硬掩码后的 `-inf` 是有意的合法性约束。

当前仅提供融合和可微概率映射。完整性质预测器接线、搜索优化器、离散后重评分和 RNAfold 评估尚未实现于此模块。最终生成时须移除 PAD，并复核离散 CDS 的翻译结果；测试中的蛋白保真不代表全量优化候选已验收。

## 损失与训练

`loss = full_loss_weight * CE_full + masked_loss_weight * CE_mask + beta(t) * KL_per_token`。

CE 在目标 AA 的同义集合内归一化，full 项覆盖所有有效位置，masked 项只覆盖被遮挡的有效位置。KL 对每个位置的 latent 维度求和，再按有效位置数平均。日志同时报告 `kl_per_token` 和 `kl_per_dimension`。

**旧版 KL 先对 latent 维度取平均，新版改为求和；相同 μ/logσ² 下，新口径等于旧口径乘 latent_dim。** 默认 β 暂沿用 0.001、前 20% 优化步 warm-up，需在验证集重新标定，不能直接比较旧版 loss 或认定 β 已调优。

其余默认：20 epochs、关闭早停、batch 4、累积 8、AdamW lr=1e-4、FP32、25% batch 不遮挡，其余以 15% 概率遮挡可变 sense codon。25–30% mask 留作单独鲁棒性实验。MVP 保留相同长度上限便于兼容已有冻结数据；先做短/中长 CDS 预实验时应另建有记录的数据子集，不能静默过滤冻结版。

正式版入口读取 [联合数据集](../dataset/README.md)，44,001 条 CDS，train/val/test = 35,710/3,949/4,342。数据与 tokenizer 未因本次架构改动重新划分。近同源隔离尚未认证，因此入口保留 `--allow-development-split`，不据此宣称正式泛化通过。

```bash
# 在当前项目代码目录执行；不要使用旧 Release 包代替新代码。
python -m unittest latent_decoder.test_architecture latent_decoder.test_decoder latent_decoder.test_combined latent_decoder.test_repartition -v

# 正式版：输出 runs/vmae_rope_6x512_20ep
bash latent_decoder/run_server_20e.sh
# 仅恢复该新版 run，累计至 20 轮
bash latent_decoder/run_server_20e.sh resume

# MVP 使用独立输出目录
python -m latent_decoder.train --data dataset --allow-development-split \
  --config latent_decoder/config_mvp.json --output runs/vmae_rope_mvp_20ep
```

启动会核对数据/配置，并对最长训练 batch 做前向、反向预检；不更新权重，恢复随机状态。新版在 V100 上的显存和速度尚未实测；如 OOM，应另存配置降低 batch，并从新目录开始，不截断 CDS。现有环境是否可用以服务器预检为准，不要求覆盖原 RiboDecode 环境。

延长正在运行的实验时，先在完整轮次保存后停止进程并备份 run，再使用 checkpoint 的原配置、原输出目录和 `--resume <run>/last.pt --extend-epochs 50`，将目标改为累计 50 轮。模型、优化器和随机状态接续；学习率从 checkpoint 当前值平滑 cosine 衰减到新终点，KL warm-up 保持原步数。已衰减到零的 checkpoint 会拒绝此操作。首次延长后，后续普通恢复使用 run 中更新后的 `config.json`，不再传 `--extend-epochs`；若尚未保存首个延长后的 checkpoint，应仍使用原配置重做延长命令。该策略与从头按 50 轮规划学习率不同。

排查参数是否更新时，可使用独立回放：`python -m latent_decoder.replay_single --data dataset --checkpoint <备份的last.pt> --output <新的诊断目录> --steps 60`。该诊断选取一条训练 CDS，复制为原 batch 大小并保留原梯度累积、AdamW 状态与损失配置；分别对比零学习率、7.596e-7、1e-4，以及 1e-4 下动态 mask。固定组复用相同掩码和 dropout/z 随机数，逐步记录真实参数差与梯度，并用 eval/μ 模式的固定探针比较。它会有意覆盖诊断副本中的学习率，不使用 scheduler；不写入原 run，输出权重仅用于诊断，不能当作完整续训 checkpoint。单样本拟合不代表全量数据泛化。软件自检：`python -m unittest latent_decoder.test_replay_single -v`。

评估在验收规则锁定后单独运行，训练期间 test 不参与模型选择：

```bash
python -m latent_decoder.train --data dataset --allow-development-split \
  --config runs/vmae_rope_6x512_20ep/config.json \
  --evaluate runs/vmae_rope_6x512_20ep/best.pt --split test \
  --output runs/vmae_rope_6x512_20ep/test_metrics.json
```

旧 4×256 / latent 64 的 checkpoint 和 `vmae-20ep-20261016` Release 属于历史实验，不能用于本架构续训。恢复旧实验须配套旧代码、旧配置、旧数据；本次没有修改旧权重或 Release。

## 本次检查与剩余验收

2026-10-17，本地 PyTorch 2.14.0+cpu，22 项测试通过；正式版与 MVP 均完成最大 1500 codons 前向和短序列 padding 检查。新梯度及断点续训测试使用小型合成模型；未执行正式规模完整训练。

| 文档项目 | 当前证据与边界 |
|---|---|
| G1 | 已有联合数据全量校验、token roundtrip、固定 split 测试通过；近同源泄漏审计未完成 |
| G2 | 正式/MVP 最大长度、隐层/潜变量/输出形状、padding、越界拒绝测试通过 |
| G3 | 合成样本及大幅随机残差下同义 hard mask 保真；尚无全量训练后生成/优化结果 |
| G4 / G5 | 冻结参数时 z/Δ 梯度有限且可变位非零；PAD 梯度零；纯潜在/直接空间退化检查通过 |
| G6 | 合成前后向、clean loss、KL、非法 token/all-PAD 行拒绝、序列化和断点恢复测试通过；GPU 长训练未验证 |
| Q1–Q6 | 未完成新版训练、基线比较、Top-k/长度分层/置信区间、多样性与潜在连续性报告 |
| I1–I5 | 未完成预测器集成、soft/discrete gap、公平消融和独立评估 |

`evaluate` 中已有的 `decoder_pilot_gate` 仅是早期诊断，不能替代文档完整 Q/I 验收，也不能据此声称表达量改善。
