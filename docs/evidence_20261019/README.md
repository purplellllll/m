# 2026-10-19 诊断原始记录

- `single_record_replay/`：四组回放的逐步 JSONL、单条 CDS 与掩码说明、汇总、源权重校验。
- `mask_plateau/`：五组短程对照的逐步 JSONL、样本 ID、汇总、遮挡率对照及源权重校验。
- `vgpu_maskratio_probe.py`、`cds_ambiguity_audit.py`、`plot_mask_plateau.py`：原实验辅助脚本，保留当时的相对路径和设备设置。应从项目根目录运行，并先按脚本准备对应权重、数据和输出路径；它们不是通用训练入口。

Git 版本不包含 `.pt` 权重、单样本 tensor 缓存或完整部署归档；这些文件已在本地/服务器保留。原始归档中的公共诊断代码在仓库 `latent_decoder/` 中维护，不重复保存。

完整 50 轮指标见上一级 `lsco_lr_50ep_metrics_snapshot_2026-10-19.jsonl`；新配对评估见 `vmae_prior_paired_metrics_20261019.json`。历史 clean-input 的 shuffle CE 与新增 masked-input shuffle CE 不应混用。
