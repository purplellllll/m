# VGPU 学习率实验部署

服务器：`ssh -p 31463 root@connect.nmb1.seetacloud.com`。口令不写入此记录。

项目目录：`/root/autodl-tmp/ds_ribodecode_lr_sweep_5ep_2026-10-17`。

实际设备报告：NVIDIA GeForce RTX 4080 SUPER，32760 MiB 显存；Python 3.8.10，PyTorch 2.0.0+cu118，CUDA 11.8。24 项软件测试在该服务器通过，CUDA 前后向探测通过。

启动命令：`nohup bash latent_decoder/run_server_lr_sweep.sh > lr_sweep_launcher.log 2>&1 < /dev/null &`。启动 PID 为 1687；训练子进程由该进程依次创建。SSH 断开不影响后台任务。

配置：正式版 6×512 Encoder / Decoder、latent 256；FP32、batch 4、累积 8；学习率依次为 0.01、0.1、0.001，各 5 epochs，恒定 LR，使用同一冻结数据和初始化 seed。

启动确认：0.01 组已通过 batch 4、1500 codons 的 GPU 前后向预检，峰值 allocated 显存 10,986,852,864 bytes（约 10.23 GiB）。已观察到 epoch 1、step 10（batch 80/8928）的正常有限 loss/梯度日志，实际参数更新已开始。该窗口 loss 为 3.0962，仅是训练早期值，不表示最终性能。

总日志：`/root/autodl-tmp/ds_ribodecode_lr_sweep_5ep_2026-10-17/lr_sweep_launcher.log`。

各组日志、指标和权重：

```text
runs/vmae_lr_sweep_5ep/
  lr_0.01.log
  lr_0.1.log
  lr_0.001.log
  status_0.01.json
  status_0.1.json
  status_0.001.json
  lr_0.01/{steps.jsonl,metrics.jsonl,last.pt,best.pt}
  lr_0.1/{steps.jsonl,metrics.jsonl,last.pt,best.pt}
  lr_0.001/{steps.jsonl,metrics.jsonl,last.pt,best.pt}
```

组结束后自动生成 `loss_comparison.png`、`.svg`、`comparison.csv` 和 `summary.json`。失败组会保留日志并继续下一组。每组目录/文件在对应阶段创建；epoch 权重和汇总指标须等完整 epoch 结束才出现。

```bash
cd /root/autodl-tmp/ds_ribodecode_lr_sweep_5ep_2026-10-17
tail -f lr_sweep_launcher.log
```

训练状态以服务器日志和进程为准；该记录不是训练完成报告。
