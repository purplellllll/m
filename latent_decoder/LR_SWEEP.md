# 三组学习率 × 5 epochs

本实验用于观察学习率对 loss 下降和验证集表现的影响。三组依次运行，学习率顺序为 **0.01、0.1、0.001**，每组从零初始化、训练 **5 epochs**，不早停。实际数据训练尚未启动。

## 固定条件

- 架构：正式版 Encoder/Decoder 各 6 层，hidden 512，8 heads，FFN 2048，latent 256，RoPE，显式 AA 条件，共 38,302,784 参数。
- 同一 seed `20261016`，同一随机初始化、数据顺序和随机数设置；GPU 不同内核/环境不承诺逐位一致。
- 优化器 AdamW，weight decay 0.01；microbatch 4，梯度累积 8，grad clip 1.0，FP32。
- **恒定学习率**：关闭 LR warm-up/cosine；三个配置只在 learning_rate 上不同。项目原有 20 轮配置不变。
- full/masked CE 权重均为 1，KL 最终 β=0.001，前 20% 更新步（约第 1 个 epoch）warm-up。25% batch 不遮挡，其余以 15% 概率遮挡可变 sense codon。
- 每组约 8,928 microbatches/epoch、1,116 optimizer steps/epoch，共 5,580 更新；最后一个累积组不足 8 个时按实际组大小缩放。
- 使用完整冻结的联合 CDS 库：train **35,710**，val **3,949**，test **4,342**。所有组共用 `dataset/` 和 Codon Tokenizer，不再随机划分。test 只做完整性检查，不参与训练、验证或学习率选择。
- 本数据版本未通过近同源隔离认证，按开发实验使用。当前联合库来源是 GENCODE v28 + RiboDecode；GEMORNA 尚未并入。

配置保存在 `latent_decoder/lr_sweep_configs/lr_0.01.json`、`lr_0.1.json`、`lr_0.001.json`。

## 服务器启动

服务器包 `ds_ribodecode_lr_sweep_5ep_2026-10-17.tar.gz` 包含当前代码、三组配置和完整 `dataset/`，不需要重新下载或处理原始来源。

```bash
tar -xzf ds_ribodecode_lr_sweep_5ep_2026-10-17.tar.gz
cd ds_ribodecode_lr_sweep_5ep_2026-10-17
sha256sum -c SHA256SUMS

# 使用现有可用的 CUDA/PyTorch 环境。
python -c "import torch, matplotlib; print(torch.__version__, torch.cuda.is_available())"
# 仅在缺少 matplotlib 时安装绘图依赖：
# python -m pip install matplotlib

nohup bash latent_decoder/run_server_lr_sweep.sh > lr_sweep_launcher.log 2>&1 &
tail -f lr_sweep_launcher.log
```

在 tmux 内前台运行也可：`bash latent_decoder/run_server_lr_sweep.sh`。不要同时启动两份任务到相同输出目录。

脚本先验证数据与配置，然后串行运行三组。每组启动时对最长 batch 做 FP32 前后向预检。出现非有限 loss/梯度、OOM 等异常时保存 traceback 和失败状态，继续下一组；不会把不足 5 轮的组标成成功。用户 Ctrl+C 会停止整个 sweep。5 轮内 loss 不改善本身不会触发早停。

默认输出：`runs/vmae_lr_sweep_5ep/`。已有目录会拒绝重新开始，以防覆盖。另起一套实验可传 `--output runs/另一个目录`。

## 恢复与实时汇总

```bash
# 继续未完成的组，跳过已成功完成 5 轮的组
bash latent_decoder/run_server_lr_sweep.sh --resume

# 训练过程中或结束后重新生成图表
python -m latent_decoder.lr_sweep --summarize
```

每个完整 epoch 保存 `last.pt` 与验证最佳 `best.pt`。中断后的不完整 epoch 从上次 checkpoint 重跑，恢复时去掉未提交 epoch 的重复指标；配置和数据必须与原实验一致。若某组在第一轮内失败且目录已存在但无 checkpoint，不能从该目录续训；检查日志后另选输出目录重新开始，脚本不会删除失败证据。

若显存不足，三份配置应同步调整 batch/累积以保持可比性，并在新输出目录重跑三组，不混用不同 batch 的曲线。

## 看哪些输出

| 文件 | 内容 |
|---|---|
| `loss_comparison.png` / `.svg` | 六幅对比曲线，包含 epoch loss、固定 β 验证 loss、重建 CE、KL、masked CE、更新步 loss |
| `comparison.csv` | 所有已完成 epoch 的指标，可用 Excel 打开 |
| `summary.json` | 各组完成/失败状态、已记录轮数、最低验证 loss 及对应 epoch |
| `lr_*/metrics.jsonl` | 每个 epoch 的完整指标 |
| `lr_*/steps.jsonl` | 第 1 步、每 10 个 optimizer steps 和 epoch 末的窗口平均 loss、实际 LR、β、裁剪前梯度范数 |
| `lr_0.01.log` 等 | 每组日志和错误 traceback |
| `status_*.json` | 每组状态、命令和退出码 |
| `sweep_manifest.json` | 三组配置、冻结数据 manifest 的 SHA-256 |

曲线纵轴使用 symlog：0 附近线性，大值对数，便于同时观察高 LR 的尖峰和低 loss 段。空白/较短曲线表示没有完成对应 epoch，应结合状态文件阅读。

`train_loss` 是实际训练目标的 microbatch 平均，含训练时 mask、dropout、z 采样和 β warm-up；同时报告 `train_full_ce`、`train_masked_ce`、`train_kl_per_token`，便于分清 loss 变化来自重建还是正则项。不同长度 microbatch 等权，非整个 epoch 按 token 加权。

`val_loss_fixed_beta` 在固定验证遮挡、关闭 dropout、z=μ 条件下计算：

```text
val_loss_fixed_beta = masked_input_full_ce
                    + masked_mutable_ce
                    + 0.001 * masked_input_kl_per_token
```

full CE/KL 按有效位置汇总，masked CE 按实际遮挡位置汇总；β 从第一个 epoch 起就固定为最终值，便于跨 epoch 比较。验证只使用同一份 val 数据及固定评估种子。它与随机混合输入下的 train loss 不是完全相同的统计口径，因此不直接将两者的数值差视为过拟合程度。`selection_loss` 和 best.pt 仍沿用原训练入口的 clean/masked 可变位 CE 之和；`summary.json` 的最低固定 β 验证 loss 是另一项统计。

选学习率时同时观察验证 masked CE、固定 β 验证 loss、训练 loss 的下降速度及稳定性；不要只按训练 loss 最低选择。当前没有实际三组训练结果，不能预先判定哪个学习率最佳。

## 本地验证

```bash
python -m unittest latent_decoder.test_lr_sweep latent_decoder.test_decoder latent_decoder.test_combined latent_decoder.test_repartition latent_decoder.test_architecture -v
```

2026-10-17：24 项 CPU 测试通过，包括完整数据校验、小型合成模型三组各 5 轮、恒定 LR、相同初始化、失败继续、图表/CSV、断点恢复指标一致性。合成训练结果不作为真实语料的学习率选择依据。正式架构在服务器 GPU 上的训练显存、时长及数值稳定性仍需实测。
