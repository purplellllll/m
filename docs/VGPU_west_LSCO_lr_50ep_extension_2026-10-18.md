# 当前 VMAE 实验延长至累计 50 轮

用户要求将正在运行的 20 轮实验扩展为累计 50 轮。本次从已完成第 19 轮的 checkpoint 继续第 20 轮，不重新初始化模型。

- 服务器：`ssh -p 32412 root@connect.westc.seetacloud.com`。
- 项目：`/root/autodl-tmp/ds_ribodecode_lr_0001_5ep_2026-10-18`。
- checkpoint：`runs/vmae_lsco_lr_20ep_20261018/last.pt`，原 epoch=19、step=21204。
- 22:29:38 +0800 在完整 checkpoint 保存后暂停旧 PID 14366；备份完成后替换该进程。其他 GPU 进程未改动。
- 新 PID：20973，2026-10-18 22:34 +0800 启动。
- 新日志：`vmae_lsco_lr_50ep.log`；PID 文件：`vmae_lsco_lr_50ep.pid`。
- 结果仍写入原 run 目录，目录名中的 `20ep` 是历史名称，实际目标以 config 和日志为准。

## 恢复与学习率

保留模型、AdamW、scheduler、scaler、Python / PyTorch / CUDA 随机状态，以及此前最佳权重。原数据、tokenizer、训练/验证/测试划分、batch 和损失配置保持不变。

第 19 轮 checkpoint 的当前学习率为 `7.596123493895991e-7`。续训首个更新使用此值，此后平滑 cosine 衰减，到累计第 50 轮共 55800 次更新时归零。新增 checkpoint 保存 `lr_extension`，普通断点恢复继续使用相同曲线。

不重新执行学习率 warm-up。KL warm-up 保持原 4464 次更新（4 轮），不因目标延长而重新计算。训练步从 21204 接续，剩余 31 轮。先前不完整第 20 轮的少量日志由恢复逻辑清理，原日志副本保留在备份目录。

此策略与一开始就按 50 轮规划 LR 不同。当前学习率已很低，延长轮数不保证明显改善验证指标。

## 备份、配置与命令

`runs/vmae_lsco_lr_20ep_20261018/extension_to_50_backup/` 保留 last.pt、best.pt、原 config、metrics、steps、原训练与测试代码，复制后核对 SHA256，并记录 `backup_manifest.json`。

`latent_decoder/config_lsco_lr_50ep.json` 仅将目标由 20 改成 50，其他字段与 20 轮配置一致。首次延长必须提供旧 checkpoint 对应的 20 轮配置；训练入口在严格核对后应用新目标。

首次延长使用：

```bash
bash latent_decoder/run_server_lsco_lr_50ep.sh extend
```

保存首个延长后的 checkpoint 后，后续恢复使用：

```bash
bash latent_decoder/run_server_lsco_lr_50ep.sh resume
```

若首次延长尚未保存下一轮 checkpoint 就中断，应仍使用 `extend`。不要同时启动两份训练。

查看日志：

```bash
cd /root/autodl-tmp/ds_ribodecode_lr_0001_5ep_2026-10-18
tail -f vmae_lsco_lr_50ep.log
```

## 验证

本地 decoder / LR 测试 10 项通过；服务器 decoder、LR、architecture、combined、repartition 共 25 项测试通过。覆盖学习率接续、延长后再次断点恢复的权重与日志一致、KL warm-up 保持，以及零学习率拒绝且不修改 run 配置。

2026-10-18 22:35:51 +0800 核验：GPU 最大 batch 前向/反向预检通过；日志确认 `from_epochs=20`、`to_epochs=50`、`resume_epoch=20`、`resume_step=21204`、`resume_lr=7.596123493895991e-7`、`kl_warmup_steps=4464`。实际已更新至 step 21230，梯度范数 0.4025，学习率 7.596113706665307e-7，进程正常运行。50 轮尚未完成，未使用测试集选模型。
