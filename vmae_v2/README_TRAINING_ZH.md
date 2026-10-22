# DS-RiboDecode VMAE V2：服务器训练包

这是可通过网页文件管理器上传的完整数据与代码包。无需 SSH，也不需要访问原来的服务器。先解压，再通过服务器的网页终端或 JupyterLab 运行。

**本包默认从头训练。旧任务已由用户停止；本地没有旧服务器的 `last.pt` / `best.pt`，因此 ZIP 不包含已训练的 V2 权重。** `reference_previous_run/` 只有历史指标和硬件测速，不会被当成新训练结果。冻结的翻译评分器权重已包含，后续搜索不用再下载评分模型。

## 1. 包内内容

| 文件/目录 | 用途 |
|---|---|
| `run_vmae_v2.py` | 统一入口：检查、测速、单次训练、15次训练、完整实验、查看状态 |
| `START_IN_JUPYTER.ipynb` | 没有网页终端时，用 Notebook 检查并启动完整实验 |
| `data/corpus.jsonl.gz` | 全部44,001条天然CDS，含编码序列及标识 |
| `data/folds.json` | 已生成的正式五折及每折训练/验证/测试ID，直接复用 |
| `data/proteins.fasta` | 对应蛋白序列，供核查 |
| `data/export.json`、`data/homology_provenance.json` | 数据校验和、同源检索版本与结果摘要 |
| `latent_decoder/v2_train.py` | V2训练、按长度控制LR、checkpoint与逐轮指标 |
| `latent_decoder/v2_search.py` | 潜变量/残差步长系数搜索及RiboDecode对照 |
| `latent_decoder/v2_experiment.py` | 五折训练→内层选择→外层评估的完整队列 |
| `latent_decoder/v2_data.py` | 数据读取和分折方法；本次直接使用现成折表 |
| `latent_decoder/model.py`、`tokenizer.py` | 256维VMAE架构与密码子编码 |
| `configs/VMAE_V2_EFFECTIVE_SETTINGS.json` | 有效配置清单，供阅读和核查，不是另一个运行配置入口 |
| `score_model/` | 冻结的HEK293T评分器、配置和条件数据 |
| `external/`、`licenses/` | RiboDecode基线所需源文件及原许可 |
| `docs/EXPERIMENT_PROTOCOL.md` | 分折、学习率、缩放、预算与统计口径 |
| `PACKAGE_MANIFEST.json` | 全包逐文件SHA-256及打包说明 |

为保留此前的代码哈希与恢复兼容性，底层模块和旧基础配置文件名保持原样。其他历史辅助模块是依赖/源码快照；使用顶层入口即可。

不含：旧服务器V2检查点、Python/CUDA安装包、MMseqs可执行程序及原始命中TSV。**本次训练不需要重新运行MMseqs**；折表已经确定，分组校验可以在包内完成。若要改变同源阈值重划折，需要另行重跑同源检索。

## 2. 上传和环境

1. 通过服务器网页文件管理器上传ZIP，解压到持久数据盘。进入解压后的 `DS_RiboDecode_VMAE_V2_5Fold_ServerBundle_20261022` 文件夹。
2. 建议预留至少30 GB可用磁盘、16 GB主机内存，使用单张NVIDIA GPU。当前microbatch=8已在16 GB T4上验证；更换GPU后先测速。
3. 使用 **Python 3.8–3.11 + PyTorch 2.0.0+cu118**。已实际训练的环境是Python3.8.10、PyTorch2.0.0+cu118；新环境可用Python3.10。不要直接用Python3.12安装这套旧版PyTorch。

如果平台能选择镜像，优先选择上述版本；如果提供Conda，可在**网页终端**中建立独立环境：

```bash
conda create -n ribodecode_v2 python=3.10 -y
conda activate ribodecode_v2
python -m pip install torch==2.0.0 --index-url https://download.pytorch.org/whl/cu118
python -m pip install -r requirements.txt
```

PyTorch安装命令依据[官方历史版本说明](https://pytorch.org/get-started/previous-versions/#v200)，这里只安装项目实际需要的torch。已装好匹配PyTorch的镜像可跳过torch安装。依赖安装需要软件源网络；ZIP本身不含离线wheel。

运行以下命令确认当前Python的实际环境和GPU，而不只看系统的 `nvidia-smi`：

```bash
python run_vmae_v2.py check --gpu
python -m unittest latent_decoder.test_v2 -v
```

`check` 校验全部文件哈希、44,001条记录、五折覆盖，以及训练/验证/测试同源组无交叉。`--gpu` 还检查PyTorch版本、CUDA和实际CUDA运算。纯文件检查可在没有GPU的机器上运行 `python run_vmae_v2.py check`。

## 3. 先测速，再正式运行

```bash
python run_vmae_v2.py benchmark
```

会用短/中/长三档的中位和最长序列检查前向、反向、优化器及显存。结果在 `results/hardware_benchmark/benchmark.json`。测速会丢弃模型权重，不会替代正式训练或影响模型初始化。

T4原测最长1500 codon、microbatch=8峰值约11.8 GiB；正式运行还会有CUDA缓存，`nvidia-smi`占用可更高。原服务器完整epoch约14.1分钟。若显存不足，先停下，不要在已有对照中途混用批次配置；小显存方案需要统一调整整个新实验。

### A. 推荐：直接执行完整实验

这条命令自动完成15次训练、每折内部系数选择，以及5%/10%/20%预算的外层RiboDecode对照：

```bash
python -u run_vmae_v2.py full
```

在网页终端后台执行并保存日志：

```bash
mkdir -p logs
CUDA_VISIBLE_DEVICES=0 nohup python -u run_vmae_v2.py full > logs/full_experiment.log 2>&1 < /dev/null &
echo $! > logs/full_experiment.pid
```

不要同时启动多个full或train-all任务到同一个结果目录。普通关闭浏览器页面与关闭GPU实例是两回事：实例停止后训练也会停止；平台若会清理会话子进程，应使用平台的后台任务功能。下一节说明如何恢复。

### B. 只训练模型，暂不跑搜索

```bash
python -u run_vmae_v2.py train-all
```

按fold_0到fold_4、每折uniform→short_fast→long_fast完成15次训练。之后用相同的输出目录执行 `full`，会识别已完成的训练并继续搜索。

只训练一个版本：

```bash
python -u run_vmae_v2.py train --fold 0 --profile uniform
```

`--fold` 使用 **0到4**，0代表第1折。`--profile` 可选 `uniform`、`short_fast`、`long_fast`。

所有命令可指定其他持久输出目录，例如：

```bash
python -u run_vmae_v2.py full --output /mnt/data/vmae_v2_results
python run_vmae_v2.py status --output /mnt/data/vmae_v2_results
```

相对输出路径以解压目录为基准。数据路径由入口自动定位，不依赖原来Windows路径或旧服务器地址。

### C. 只能使用JupyterLab

在解压目录打开 `START_IN_JUPYTER.ipynb`，选择已安装上述依赖的Python内核，按顺序运行。Notebook包含文件/GPU检查、测速、后台启动和状态查看，不需要SSH。依赖缺失时可在该内核执行 `%pip install -r requirements.txt`；PyTorch仍需按上面的版本安装并重启内核。

## 4. 默认训练设置

- 全量44,001条CDS按11,533个同源/身份分组做五折外层CV；各折内部另留出验证组。每条序列恰好做一次外层测试。折表在各版本间保持一致，不能每次训练随意重分。
- 长度不含终止密码子：短≤300 aa、中301–699 aa、长≥700 aa。
- 架构：hidden512、encoder6层、decoder6层、每位置latent256；15%mask。
- 基础LR：uniform=`1e-4/1e-4/1e-4`；short_fast=`2e-4/1e-4/5e-5`；long_fast=`5e-5/1e-4/2e-4`，顺序均为短/中/长。
- 每次更新只使用一个长度档，最多32条序列；microbatch=8、最多累积4次。AdamW、weight_decay=.01；FP16 autocast、FP32损失、GradScaler；梯度裁剪1。
- 最多50 epochs，最少10；验证目标连续8轮不改善则早停。LR前10%计划步数warmup；KL权重升至.001，前20%计划步数warmup。
- 每折/版本都从头初始化；不能用已经看过外层测试数据的旧epoch50模型热启动。
- 模型checkpoint由内层验证目标选择；最终训练版本和步长系数由内层合法Hard Gain选择，外层测试不参与调参。
- 搜索在标准化潜变量位移和实际logits残差坐标中优化；两个步长系数各为0.5/1/2，共9组，物理正则强度固定。主分析JS≤.05，次分析在5/10/20%预算分别用JS≤.05/.1/.3。

配置清单用于核对。实际执行设置来自冻结的 `v2_train.py` / `v2_experiment.py` 及基础架构JSON；**不要只修改旧 `config_lsco_lr_50ep.json` 就认为V2所有参数都会改变**。本包入口会检查源文件哈希，以免把不同方案混在同一实验中。若要改协议，应复制成新实验、更新校验清单并使用新的输出目录。

## 5. 进度、文件和停止/恢复

```bash
python run_vmae_v2.py status
tail -n 30 logs/full_experiment.log
```

主要结果路径：

```text
results/
  plan.json                       完整实验的固定方案与哈希
  status.json                     full队列当前阶段（需结合日志/进程判断是否仍活跃）
  summary.json                    外层搜索汇总；中途是部分结果
  complete.json                   全部5折完成时才出现
  failure.json                    full入口捕获到异常时的错误信息
  fold_0/
    uniform/                      每个版本各自一个目录
      plan.json                   本次训练的有效配置及哈希
      progress.json               每25次更新刷新，step为累计更新数
      epoch_01.json               完成一轮后的训练/验证指标
      last.pt                     最近完整epoch：模型+优化器+scaler+RNG
      best.pt                     内层验证最优模型权重
      complete.json               该训练版本已结束
    short_fast/
    long_fast/
    selection.json                内层选出的版本及各长度档步长系数
    inner/                        内层搜索逐任务结果
    outer/                        外层对照逐任务结果
    outer_encoder.json            外层编码器尺度和潜变量诊断
    complete.json                 该折训练及评估均完成
  fold_1/ ... fold_4/
```

停止前尽量等当前epoch完成并写出 `last.pt`；突然停机只会保留最近已成功保存的epoch，**不会从某个microbatch精确接着跑**。保存结果目录到持久盘，再次使用相同环境、相同代码和相同命令即可自动恢复。不要在同一目录同时启动两份任务。训练完成标记和已保存的搜索任务用于跳过已完成工作。

若另外拿到了旧服务器完整的 `results/` 目录，可通过网页上传到解压目录下再运行；保留各层plan、last、best、epoch日志及其他结果文件，不要只拿 `last.pt`。本包底层源码和正式折表保留了旧哈希，但必须匹配原来的PyTorch版本和配置。遇到 `Resume configuration/source mismatch` 或 `Experiment changed on resume`，先检查环境/文件，不要删除plan或强行忽略检查。本包**未包含**这些旧权重，不能仅凭历史日志恢复到第16轮。

## 6. 时间与结果解释

原T4实测约14.1分钟/epoch；当前包的15次训练若都跑满50轮，训练约7.3天，加搜索对照粗估总计8–9天。新GPU速度不同，早停也会缩短时间。这个时间是原硬件估计，不是新服务器的实测保证。

原任务最后本地记录为第1折uniform已完成15轮、正在第16轮；内层AU约42–43/256，仅为历史指标。活跃维度阈值不等于独立信息容量；最终要看外层合法Hard Gain和Filter Gap。

当前没有V2最佳系数结论，也未完成五折外层对照。冻结评分器结果不直接等同真实翻译提升；尚无新的表达实验验证。原始数据来源/校验见`data/export.json`，同源检索有灵敏度限制，分组保证限于已检测的相似边和原有身份分组。
