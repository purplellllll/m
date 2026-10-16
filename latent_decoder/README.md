# RiboDecode CDS-VMAE：V100 32 GB，20 epochs

本轮训练encoder＋decoder，使用RiboDecode官方数据派生的CDS，固定20 epochs，关闭早停。不扩展GENCODE/MANE/OAS语料；GENCODE v28只用于核对原CDS的真实终止密码子。

## 1. 下载与环境

[服务器包：代码＋清洗数据](https://github.com/purplellllll/m/releases/download/vmae-20ep-20261016/ds_ribodecode_20ep_v100_2026-10-16.tar.gz)

```bash
wget -c https://github.com/purplellllll/m/releases/download/vmae-20ep-20261016/ds_ribodecode_20ep_v100_2026-10-16.tar.gz
tar -xzf ds_ribodecode_20ep_v100_2026-10-16.tar.gz
cd ds_ribodecode_20ep_v100_2026-10-16
sha256sum -c SHA256SUMS
conda create -n ds-ribodecode python=3.10 pip -y
conda activate ds-ribodecode
python -m pip install numpy==1.24.4
python -m pip install torch==2.0.1 --index-url https://download.pytorch.org/whl/cu118
nvidia-smi
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO CUDA')"
python -m unittest latent_decoder.test_decoder latent_decoder.test_repartition -v
```

包内直接包含`data/latent_decoder/corpus.jsonl`、`proteins.fasta`、`audit.json`和框架，不需要重新下载官方122 MB原始zip。Git仓库保存代码、说明与数据审计，语料在Release附件中。

使用独立环境，保留原RiboDecode环境。安装来源：[PyTorch官方历史版本](https://pytorch.org/get-started/previous-versions/)。本轮不要求MMseqs2、RNAfold、ModernBERT或性质预测器权重。默认FP32；V100不使用BF16，本地CPU测试不等同于Linux/CUDA实测。

## 2. 启动20轮

```bash
nohup bash latent_decoder/run_server_20e.sh > launcher_20ep.log 2>&1 &
tail -f launcher_20ep.log
```

也可在tmux内前台执行`bash latent_decoder/run_server_20e.sh`，两种方式选择一种。脚本另外保存带时间和PID的`logs/vmae_20ep_*.log`。Ctrl+C退出tail不会结束后台训练。

入口要求config的`epochs=20`、`patience=0`。验证指标暂时不改善也继续至第20轮，同时保存验证集最佳checkpoint。异常、OOM或手动停止不会自动重试。

启动时验证哈希、划分和CUDA，然后用最长的4条训练CDS做一次FP32前向/反向。该检查不执行optimizer更新，不改变参数或随机状态；通过会记录`preflight_pass`和显存，失败不创建新的训练目录。它不能保证整次GPU任务绝无错误。

| 配置 | 值 |
|---|---:|
| epochs / early stopping | 20 / 关闭 |
| encoder / decoder | 各4层双向Transformer |
| hidden / latent | 256 / 64 |
| 参数量 | 6,407,616 |
| microbatch / 累积 | 4 / 8，近似32条/更新 |
| Optimizer / lr | AdamW / 1e-4 |
| 长度 / 精度 | 最多1,500 codons（含stop）/ FP32 |
| KL β | 最终.001，前20%优化步数升温 |
| 输出 | `runs/vmae_ribodecode_20ep` |

mask为15%可变sense codons，25%训练batch保持完整输入。不同长度microbatch均权，不严格等于合并后token均权的大batch。首次GPU检查尚未执行，不预报20轮小时数；第一轮的`epoch_seconds`包含验证，可用于估算。

如OOM，停止后将batch降为2、累积升为16并从新run开始；不要截断序列或混用不同配置。已有输出目录会拒绝重新开始，以免覆盖结果。

## 3. 本轮数据与结论边界

| split | CDS | gene | 来源 |
|---|---:|---:|---|
| train | 26,764 | 9,291 | 全部属于官方10,552个训练基因 |
| val | 2,744 | 965 | 从官方训练基因中留出 |
| test | 3,288 | 1,148 | 1,138个官方测试基因＋10个保护/组件迁移的训练基因 |

train/val中官方测试基因数为0。共有32,796条唯一CDS，gene及exact-AA跨split无重叠。来源与校验和见[数据卡](../data/latent_decoder/DATA_CARD.md)。

本轮为**开发训练**，保留`homology_certified=false`，启动脚本显式加入`--allow-development-split`。近同源聚类尚未执行，不能用结果宣称未见同源蛋白的正式泛化。20轮期间test不参与训练或checkpoint选择。

## 4. 续训和评估

```bash
nohup bash latent_decoder/run_server_20e.sh resume > launcher_20ep_resume.log 2>&1 &
```

从本run的`last.pt`恢复，数据和配置须一致。目标为**累计20轮**，不是再加20轮。每个完整epoch保存一次，中途停止会重跑未完成的epoch。旧40轮配置的checkpoint不能直接混用。

`best.pt`与`last.pt`均含encoder、decoder及训练状态；`metrics.jsonl`记录每epoch验证CE、准确率、KL、latent诊断、耗时和峰值CUDA分配显存。

配置及验收规则冻结后显式评估test：

```bash
python -m latent_decoder.train \
  --data data/latent_decoder --allow-development-split \
  --config runs/vmae_ribodecode_20ep/config.json \
  --evaluate runs/vmae_ribodecode_20ep/best.pt --split test \
  --output runs/vmae_ribodecode_20ep/test_metrics.json
```

初步验收：硬mask保真100%、clean可变位点accuracy≥95%、masked CE比训练频率基线低≥5%、zero-z及同AA内部shuffle-z使CE恶化≥0.10 nats。以上是提案门槛，尚无训练结果；接入性质预测器仍需latent连续性、梯度和公平优化对照验证。

## 5. 可选同源划分

本步骤不属于当前默认启动。安装MMseqs2后，可执行`bash latent_decoder/server_cluster.sh`，从相同清洗库聚类并调用`latent_decoder.repartition`；不引入新序列，不再要求原始zip、v28参考或WT文件。所有旧test及其新关联组件保持test，另写`data/latent_decoder_clustered`。

输出train/val/test蛋白FASTA供跨split敏感搜索审计。`homology_certified=true`仅表示已用完整cluster TSV划分，`homology_search_audited=false`明确保留；不能把它称作零同源证明。有跨split命中时先合并组件、重划分，再从零训练新run。

## 范围与来源

数据：Yupeng Li，[RiboDecode Figshare v3](https://doi.org/10.6084/m9.figshare.28916288.v3)，[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)。数据卡注明本项目派生处理。方法参考[RiboDecode](https://doi.org/10.1038/s41467-025-64894-x)和[LSCO](https://arxiv.org/html/2610.03098v1)；本实现为小型条件VMAE，有明确结构和损失差异。

本轮只完成VMAE预训练准备，尚无训练完成权重或经过验证的双空间优化器。原方案40轮建议由本轮20轮配置取代；完整GENCODE/MANE扩展不在本轮范围。
