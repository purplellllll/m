# 2026-10-18 新服务器 LR=0.0001 五轮实验

用户指定服务器：`ssh -p 32412 root@connect.westc.seetacloud.com`。口令不写入文档。

项目目录：`/root/autodl-tmp/ds_ribodecode_lr_0001_5ep_2026-10-18`。

实际设备报告：NVIDIA GeForce RTX 4080 SUPER，32760 MiB 显存；Python 3.8.10，PyTorch 2.0.0+cu118，CUDA 11.8。数据包及所有文件 SHA-256 校验通过；CUDA 矩阵前后向探测和 24 项软件测试在此服务器通过。

从零初始化正式 VMAE（6×512 Encoder / Decoder、latent 256），seed=20261016。使用 `latent_decoder/lr_sweep_configs/lr_0.0001.json`：恒定学习率 0.0001、5 epochs、FP32、batch 4、累积 8，其余训练目标和数据与前次相同。固定开发数据划分为 35710/3949/4342；本次不做 test 模型选择。

唯一训练代码改动是补充分项及 clean/masked 模式计数日志。窗口 loss 分量相加检查、按模式数量合并检查、断点恢复确定性检查均通过；没有改梯度、优化器更新或采样分布。

启动：`nohup bash latent_decoder/run_server_lr_0001_5ep.sh > lr_0.0001_5ep.log 2>&1 < /dev/null &`。PID 为 1440，记录在 `lr_0.0001_5ep.pid`。SSH 断开不影响任务。

日志：`/root/autodl-tmp/ds_ribodecode_lr_0001_5ep_2026-10-18/lr_0.0001_5ep.log`。

权重和指标：项目内 `runs/vmae_lr_0.0001_5ep/`，每个完整 epoch 后保存 metrics.jsonl、last.pt、验证最佳 best.pt。steps.jsonl 记录每个日志窗口的总 loss、CE/KL 分项和模式占比。

```bash
cd /root/autodl-tmp/ds_ribodecode_lr_0001_5ep_2026-10-18
tail -f lr_0.0001_5ep.log
```

服务器包：`deliverables/ds_ribodecode_lr_0001_5ep_2026-10-18.tar.gz`，SHA-256 为 `d7ac134fb1c5dcab85aa67db32975737be3020b50edfa6285efc02f019a1d105`。

此记录是部署记录，不表示训练已完成。

2026-10-18 10:58 +0800 启动确认：最长 1500 codons、batch 4 FP32 GPU 前后向预检通过，已实际更新到 epoch 1 / step 10。首个更新 loss=2.26610，step 10 窗口 loss=1.64524，梯度范数有限且非零，lr=0.0001。新分项日志已出现在总日志和 steps.jsonl；GPU 显存约 11716 MiB。首轮验证尚未完成。

2026-10-18 11:58:02 +0800，用户要求暂停并准备新的配置，暂不开训。已对核验后的 PID 1440 发送 SIGSTOP，确认状态 T (stopped)，最后日志 epoch 5 / step 5370。记录在 `runs/vmae_lr_0.0001_5ep/pause_manifest.json`。SIGSTOP 保留内存和约 14748 MiB 显存；当前 GPU 利用率为 0%。原日志、checkpoint、配置保持原状。

本地及服务器新增 `latent_decoder/lr_sweep_configs/lr_0.0001_no_clean_batches.json`，与原 0.0001 配置逐项比较，唯一变化为 `clean_batch_probability: 0.25 -> 0.0`。mask_probability=0.15、Full/Masked CE 权重、KL、恒定学习率 1e-4、5 epochs 等均不变。

配套新启动脚本为 `latent_decoder/run_server_lr_0001_no_clean_5ep.sh`，指向独立输出目录 `runs/vmae_lr_0.0001_no_clean_5ep`。已检查 JSON 差异、shell 语法及远端进程状态；脚本没有执行，新输出目录不存在。原进程仍运行旧配置且已暂停；直接 SIGCONT 只会恢复旧实验，不会应用新配置。未来采用新配置须新开实验，不能用不一致的配置直接 --resume 旧 checkpoint。
