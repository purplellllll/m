# 上下文消融与潜在空间优化原始记录

本目录保存 2026-10-19 对 VMAE 第 50 轮权重进行的只读诊断记录。实验冻结 VMAE 和评分器，不包含模型权重、评分器权重或凭据。

- `context_ablation/`：128 条验证 CDS 的固定掩码上下文消融、逐序列记录和配对分析。
- `latent_optimization/`：16 条验证 CDS、两种 z 学习率、直接 logits 对照和随机 z 对照的逐步记录；包括官方评分适配核对、RNAfold 后处理和几何检查。
- `prior_sampling/`：条件后验和标准正态先验采样记录。

主结论和实验边界见上级 [VMAE_context_latent_tests_2026-10-19.md](../../VMAE_context_latent_tests_2026-10-19.md)。原始记录仅支持复查本次实验，不能替代独立表达评价或完整双空间验收。
