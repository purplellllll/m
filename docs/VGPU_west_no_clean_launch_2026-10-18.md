# 全 batch 遮挡训练启动记录

服务器：`ssh -p 32412 root@connect.westc.seetacloud.com`。口令不保存。

项目目录：`/root/autodl-tmp/ds_ribodecode_lr_0001_5ep_2026-10-18`。

2026-10-18 16:03 +0800，用户明确要求启动此前准备的新配置。启动前检查没有发现旧训练进程，GPU 空闲显存 32230 MiB / 32760 MiB。

配置 `latent_decoder/lr_sweep_configs/lr_0.0001_no_clean_batches.json`：clean_batch_probability=0.0，mask_probability=0.15，learning_rate=0.0001 恒定，epochs=5。与原配置逐项比较，唯一变化为 clean batch 概率从 0.25 改为 0.0；正式架构、数据、seed、Full/Masked CE 权重、KL 调度、FP32、batch 4、累积 8 均保持一致。

从零初始化新实验，不使用旧权重续训。启动 PID 为 10110，记录在 `lr_0.0001_no_clean_5ep.pid`。

启动命令：`nohup bash latent_decoder/run_server_lr_0001_no_clean_5ep.sh > lr_0.0001_no_clean_5ep.log 2>&1 < /dev/null &`。

日志：`/root/autodl-tmp/ds_ribodecode_lr_0001_5ep_2026-10-18/lr_0.0001_no_clean_5ep.log`。

指标与权重：项目目录内 `runs/vmae_lr_0.0001_no_clean_5ep/`。原实验输出保持在原目录。训练使用相同冻结划分，test 不参与选择；验证仍为固定种子下的 15% 遮挡评估。

```bash
cd /root/autodl-tmp/ds_ribodecode_lr_0001_5ep_2026-10-18
tail -f lr_0.0001_no_clean_5ep.log
```

窗口日志应显示 clean_batches=0、masked_batch_fraction=1；clean_mode_loss=null 是该窗口没有 clean batch 的正常表示。新训练的总体 loss、日志 masked_ce 与旧版混合均值不具有直接可比性，应优先比较固定验证 masked_mutable_ce、z 消融及相同更新预算。

本记录不表示五轮训练已经完成。

2026-10-18 16:04:48 +0800 确认：最长 batch 4 / 1500 codons GPU 预检通过，已实际更新至 epoch 1 / step 30。窗口总 loss=1.42848，masked CE=1.17166，lr=0.0001；clean_batches=0、masked_batch_fraction=1.0，梯度有限非零。GPU 显存占用约 11726 MiB，新配置已生效。
