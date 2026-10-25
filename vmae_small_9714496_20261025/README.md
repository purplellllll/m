# VMAE：9,714,496 参数冻结版本

这是 2026-10-25 训练完成、最优 epoch35 的上一版小模型。宽度256、FF1024、encoder/decoder各6层、8 heads、每密码子latent256。它与后来的2,498,240参数模型分开保存。

代码位于仓库 `purplellllll/m` 的 `vmae-v2` 分支，本目录为独立快照。训练源及配置按原训练清单SHA核验；搜索代码对应当前30条完整测试锁定的版本。详见 `SOURCE_PROVENANCE.json` 和 `SNAPSHOT_MANIFEST.json`。

## 检查与环境

先进入本目录。`python verify_snapshot.py` 仅用标准库检查全部文件SHA、语法和固定配置，不启动训练。

训练环境为Linux、Python3.12、Torch2.5.1+cu124，支持BF16的CUDA GPU。先安装匹配的CUDA版Torch，再安装 `requirements.txt`。原训练还使用NumPy1.26.4。完整原生流程使用ViennaRNA2.7.2；seq2ribo另外需要官方源码、匹配Torch/CUDA/ABI的Mamba2.2.4与权重，这些不属于单纯VMAE训练依赖。

```bash
python -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu124
python -m pip install -r requirements.txt
python verify_snapshot.py
```

不将数据、权重及原生完整输出存入Git。准备 `ASSETS_REQUIRED.json` 指明的 `corpus.jsonl.gz`、`folds.json`；两份资产须按SHA保持不变。原同源分组split0固定为TRAIN23812 / VAL2592 / TEST6392；目录名fold_0保留以兼容，**不循环执行五折**。

## 重新训练同一规模

必须从本目录启动，才能读到这里的256/1024配置。使用新的输出路径，已有完整训练无需重跑。

```bash
python verify_snapshot.py --data /absolute/path/to/data
mkdir -p logs
nohup python -u -m latent_decoder.ribodecode_only_train \
  --data /absolute/path/to/data \
  --output /absolute/path/to/new_run/fold_0/uniform \
  --fold 0 --profile uniform --microbatch 4 \
  > logs/train_9714496.log 2>&1 < /dev/null &
tail -F logs/train_9714496.log
```

日志的`event=train`包含loss、ce、full_ce、kl、LR和梯度；`event=epoch`包含验证ce/fixed_loss及分长度AU。`progress.json`记录当前训练窗口，`epoch_*.json`、`diagnostics_*.json`保存完整验证诊断。

训练实现原样保留：seed20261023、uniform峰值LR1e-4、micro4/effective32、MASK0.15、AdamW WD0.01、grad_clip1、BF16、最多50轮/最少10轮/patience8。训练损失masked CE + full CE + beta×KL/token，最终beta0.001。实际warmup、cosine和beta定义在`ribodecode_only_train.py`中，不能只根据JSON推断。相同种子不保证不同设备和软件栈的轨迹完全一致。

继续中断训练时使用相同源文件、数据和输出路径；原代码从`last.pt`恢复，并校验已有plan。不要删除plan绕过不一致检查，不要覆盖已经完成的权重。

## 加载已完成模型

最优权重SHA256：`d79605bcee93be8b39e5e773a15be1aacd762230a25b8f9e0630200197614547`，38,906,827 bytes；位置见资产清单。

```bash
python verify_snapshot.py --checkpoint /absolute/path/to/fold0_uniform.pt
```

该命令在CPU上严格加载权重并确认9,714,496参数，不启动搜索。程序中可用：

```python
from latent_decoder.latent_information_probe import load_model
model = load_model('/absolute/path/to/fold0_uniform.pt', 'cpu')
model.eval()
```

43轮训练后早停；按固定192条验证诊断面板的fixed_loss选择epoch35。masked CE=0.9818691983819008、fixed_loss=1.130951371975243。原大模型对应CE=0.9820680162、loss=1.1300193779，尚未同时优于两项。训练完成不代表翻译设计验收通过。

## 搜索及当前后台实验

`tools/run_vmae_30cds_20261025.py`、原生wrapper、恢复队列、备份脚本及对应检查一并保留。它们是**原服务器目录结构的运行快照**，使用既有评分器、条件、案例清单、冻结plan及外部源码，不能当作在任意目录克隆后即可启动的一键命令。

冻结搜索：80步、AdamLR0.05、标准化半径8、T1，全可修改位MASK，无修改率预算/JS约束/direct L2；按RiboDecode Hard Score选候选，再计算独立seq2ribo。原生对照是用户要求的20轮×40000候选，保留默认MFE联合优化。两侧在线预算不同，时间与查询数分别报告。

30条完整测试仍在后台，阶段状态见 `HANDOFF_ZH.md`，之后以远端实时状态为准。当前测试CDS不得用于未披露的调参或seq2ribo候选重选。完整结果最终应有90行指标和30组配对。

`python -m unittest tools.test_vmae_30cds_20261025` 可运行抽样与归档保护的4项检查。它不执行官方完整预算。GPU数值一致性和断点恢复已在原环境完成；交接说明内给出证据位置。

## 目录与外部资产

- `latent_decoder/`：训练、模型、隐空间诊断、搜索及实际导入依赖。
- `tools/`：完整流程、原生GPU适配/恢复、评分核验及备份。
- `score_model/`：原有RiboDecode评分器推理源码和配置，作者来源为 `wangfanfff/RiboDecode`；作者权重另行准备，不重新训练评分器。本快照不新增第三方许可授权。
- `MODEL_CARD.json`：模型身份、配置、验证数值、权重SHA。
- `ASSETS_REQUIRED.json`：未随代码上传的资产及位置/哈希。
- `HANDOFF_ZH.md`：当前任务、结果边界、服务器目录和接续步骤。

历史JS消融和多种recipe因实际依赖关系保留在源码中；当前训练使用BASE recipe，不能把存在于文件中的其他选项当成本次启用的设置。
