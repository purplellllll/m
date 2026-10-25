# VMAE 971 万参数版本：下一轮对话交接

更新时间：2026-10-25，Asia/Shanghai。本文记录已确认的状态；进程号、测试轮数和完成数必须在下一轮重新核查。

## 先读这几条

1. 用户本次要求把上一版 **9,714,496 参数**模型代码上传 GitHub，并写好交接文档。仓库为 `https://github.com/purplellllll/m`，沿用 `vmae-v2` 分支，独立目录 `vmae_small_9714496_20261025/`。提交及逐文件校验见本地 `runs/vmae_small_size_only_20261025/github_publication_receipt.json`。
2. 当前主要后台任务是 **30 条 CDS 的完整五指标对照**。小模型 30/30 完成；原生 RiboDecode 1/30 完成，第二条 69 aa 已从第 6 轮检查点恢复。20:42 实测正在第 9/20 轮。不要重复启动、减少预算或拿历史 80 步原生结果替代。
3. 971 万参数模型训练已结束，最优 epoch35；后来 **2,498,240 参数**模型也已训练完成，CE/loss 更差，未替换本轮模型。不要重新启动已结束的训练。
4. 最新训练要求是 **固定训练/验证/测试划分，不做 K 折**；规模对照只改 `d_model` 和 `ff_dim`，其他条件保持。旧文件中的五折计划与训练中状态属于历史，不作为当前指令。
5. 用户长期目标尚未达成：同时改善两个翻译评分器、CAI，以及相对原生参考的 GC/MFE，并解决有效隐空间不足。预测分数提升不等于实测表达提升。

## 工作区与服务器

本地工作区：`F:\科研实验复现\新建文件夹`，Windows PowerShell。工作区根不是 Git 仓库；不要对整个工作区执行 `git add .`。本次上传使用冻结的独立交付目录，不含原始数据、二进制权重、登录信息或大量中间产物。

当前 SSH：`ssh -p 31026 root@183.147.142.40`。登录凭据已在原对话提供，不写入交接文档或 GitHub；若新对话无法取得凭据，请使用用户已有 SSH 配置，确有需要再索取。实例会关机/改端口，先确认连接再报告。

服务器 Python：`/root/gpufree-data/vmae_gpu_env/bin/python`；RTX4090 24GB，Torch2.5.1+cu124、NumPy1.26.4、ViennaRNA2.7.2。原生完整任务需要 Linux fork，不能直接在 Windows 上执行。

## 971 万参数模型的准确身份

| 项目 | 冻结值 |
|---|---|
| 模型类 | `latent_decoder.latent_recovery_gpu.RecoveryVMAE`，`BASE` recipe |
| 参数量 | 9,714,496 |
| 模型宽度 / FF | 256 / 1024 |
| Encoder / Decoder | 各 6 层，8 heads，dropout0.1 |
| 隐变量 | 每个密码子位置 256 维；不是整条 CDS 仅一个 256 维向量 |
| 长度上限 | max_codons1500（含终止），max_positions1536 |
| 训练 | 随机初始化，seed20261023，uniform peakLR1e-4，micro4 / effective32，BF16 |
| 正则 | AdamW WD0.01，grad_clip1，最终 KL 系数0.001 |
| MASK / CE | MASK概率0.15，masked CE + full CE + beta×KL/token |
| 调度 / 早停 | 原实现 warmup cosine；最多50轮、最少10轮、patience8 |
| 实际结束 / 最优 | 43轮结束；按验证 fixed_loss 选 epoch35 |
| 验证 masked CE / fixed loss | 0.9818691983819008 / 1.130951371975243 |

训练实现以冻结代码为准：`ribodecode_only_train.py` 内 LR、warmup 和 beta 调度有直接定义；不能只改 JSON 后就声称实际训练已改变。

最优权重本地：

`runs/vmae_small_size_only_20261025/checkpoints/fold0_uniform.pt`

远端：

`/root/gpufree-data/vmae_small_size_only_20261025/results/fold_0/uniform/best.pt`

大小 **38,906,827 bytes**；SHA256：

`d79605bcee93be8b39e5e773a15be1aacd762230a25b8f9e0630200197614547`

冻结训练源目录：`/root/gpufree-data/vmae_small_size_only_20261025/source`。本次 GitHub 配置文件直接从此目录取得，SHA256 `edb7ee2693f5a6459990ceddc699607f2de670ce0e5e9449563ae1fbf8898369`；不要误用工作区仍为大模型的同名配置。训练代码 SHA256 `94b5d7d3eac01b57a2673d128d89df70609d6e296e65f3185b0278ea4131175e`。

与原 38,302,784 参数模型相比，仅宽度512→256、FF2048→1024。原大模型验证 CE0.9820680162、loss1.1300193779：小模型 CE 略低、loss 略高，**没有同时通过 CE/loss 更优的验收**。250 万参数模型最优 epoch42：CE0.9838190881、loss1.1352500385，两项更差。

## 固定数据与复现

仅用经过官方序列核验的 RiboDecode CDS：32,796 条、8,221 个同源组。TRAIN23,812 / VAL2,592 / TEST6,392；沿用原 split0 的同源分组，目录仍名 `fold_0` 只为兼容，不再轮换折。

远端数据目录：`/root/gpufree-data/vmae_adjustment_20261024/data`。

- `corpus.jsonl.gz` SHA256：`6a1d10d95130cbfd81a35ba0eac7ba16db1429b33d47664976841aa6ac4eece1`
- `folds.json` SHA256：`229ffce99a401f8741c68a1bd11b590dae5160075547e644ce16c6ee4b30ec28`
- 本地原始资产：`runs/ribodecode_only_vmae_20261023/data/`；使用前按 GitHub 的 `ASSETS_REQUIRED.json` 核对。

模型选择只看固定192条验证诊断面板（短、中、长各64），不是6392条测试集。验证 loss 固定使用最终beta0.001，训练阶段 beta 逐渐预热。

从 GitHub 对应目录重新训练的命令见该目录 `README.md`。使用新输出目录；不要覆盖现有最优权重，不要直接启动旧五折 runner。训练代码原样保留，本次没有再训练。

## 当前 30 CDS 完整测试：下一轮优先接续

远端输出：`/root/gpufree-data/vmae_30cds_20261025`。

远端执行源：`/root/gpufree-data/vmae_transfer_assessment_20261024` 下的 `tools/` 和 `latent_decoder/`。

本地自动备份：`runs/vmae_30cds_20261025`。关键文件：`metrics.csv`、`paired_comparison.csv`、`candidates.fasta`、`summary.json`、`native_status.json`、`ours_status.json`、`plan.json`、`panel.json`、`native_receipts/`、`native_archives/`。

协议：TEST 内短≤300aa、中301–699aa、长≥700aa各10条，30个独立同源组，事先哈希抽样；与TRAIN/VAL无组重叠。30条长度为：43、69、69、95、163、192、200、231、236、272、311、339、545、550、558、565、613、665、673、677、708、711、726、748、955、1104、1113、1267、1295、1455 aa。

- 小模型：80步、AdamLR0.05、标准化半径8、温度1，所有可修改密码子MASK；无修改率预算、无JS惩罚/硬筛选、无direct L2。仅由RiboDecode Hard Score选候选，锁定后再算seq2ribo。
- 原生：用户明确选择官方默认 **20轮×每轮40000候选**，保留MFE联合优化；MFE权重0.7、batch20、40000次更新，每条1,600,000目标/负条件评分序列查询。比较 **第20轮实际返回值**，不是历史最高档案。
- 预算不相同，必须分别报告分数、运行时间、查询数量；不可宣称等预算胜出。
- 原始CDS、小模型、完整原生各30条，最终90行指标、30组配对。现在61行、1组配对，不能当作最终30条结论。

最近实测（20:42）：恢复队列PID580、workerPID706均存活；第2条69aa正在第9轮optim。PID仅供定位，下一轮务必核对 `/proc/<pid>/cmdline` 和events，再决定动作。

```bash
cat /root/gpufree-data/vmae_30cds_20261025/summary.json
cat /root/gpufree-data/vmae_30cds_20261025/native_status.json
cat /root/gpufree-data/vmae_30cds_20261025/active_native_process.json
tail -F /root/gpufree-data/vmae_30cds_20261025/native_recovery_20261025.log
nvidia-smi
```

当前第2条ID：`cds_4ab87a0ce372b115df9eb5f66d5d4f5885428c2ac5ff228fca0c97d6921b22f9`；细日志为 `native/<ID>.log`，阶段与轮数为 `native/<ID>/events.jsonl`。

服务器此前重启导致进程消失。已增加独立恢复脚本 `native_full_gpu_resume_20261025.py` 和 `resume_vmae_30cds_queue_20261025.py`，保持原锁定源文件不变。从完整第6轮状态恢复所有模型及Python/NumPy/Torch CPU/CUDA RNG，再运行第7轮。恢复前实测两轮MFE流程：80候选DNA、最终序列、分布完全一致，状态张量误差0，全部RNG逐位相等。证据：`recovery_validation/completion_receipt.json`。

中断第7轮产物保留在该CDS的 `interrupted/`，额外废弃算力不计入名义完整预算，不能据此声称物理总查询精确等于名义数。恢复时间累计中断前已记录计算时间和恢复后时间，不含停机时间。若恰好在第20轮完成后、最终receipt写出前中断，当前恢复wrapper的`resume_round < rounds`边界需另行核验处理，不能盲目重跑。

仅在确认队列/worker都已消失后，才从原执行源目录用原Python启动恢复队列；它检查完整结果、断点、重复进程与源SHA。不在GitHub拷贝目录直接启动原有任务，否则相对路径会改变。

本地备份监视器 `tools/watch_vmae_30cds_20261025.py` 每60秒备份CSV、JSON、FASTA和已完成原生归档，归档SHA核验后落盘。它是本地进程，关闭应用/电脑可能停止；下一轮核查存活。若需重开：

```powershell
python tools/watch_vmae_30cds_20261025.py --host 183.147.142.40 --port 31026
```

密码通过交互输入。不要把密码放在命令、脚本、文档或提交中。原生完整工作目录完成后先打包并逐成员校验，再清理可恢复原始文件；不要删部分检查点。

## 已完成结果与解释边界

30条小模型相对**原始CDS**：RiboDecode分数30/30提高，平均+1.65255151；seq2ribo三种子均分28/30提高，平均+1.19543959；CAI23/30提高，平均+0.02545655。这不是对完整原生RiboDecode的胜率。

完整原生目前仅43aa首条完成，耗时4083.77秒。该条小模型vs完整原生：Hard3.891781vs3.336234，TE2.212956vs1.195354，CAI0.723429vs0.661210；但GC54.5455%vs66.6667%、MFE−46.3vs−98，联合偏好范围失败。仅一条不能外推。

历史6条重用验证CDS、80步原生对照：小模型Hard0/6胜出，平均差−0.11061744。独立seq2ribo：3/6三种子均分更高，其中仅65aa和1171aa全部三种子更高；无双评分器同时超过原生的条目。这是探索性验证结果，不是本轮全预算30条测试。

已有官方评分器一致性核验：RiboDecode18条共同输入最大评分误差4.77e−7；seq2ribo真实CUDA与官方CPU参考11条最大差6.56e−7，通过既定容差。无需重训RiboDecode评分器，也不需重复安装Mamba。

## 五指标定义与验收

RiboDecode：作者预训练评分器，HEK293T 10552维背景实际已加入，目标mRNA丰度固定4.5，终止密码子从共同评分输入排除；不使用每条CDS实测丰度，也没有独立物种参数。不要说没加入细胞背景，也不要说固定4.5等于真实表达量。

seq2ribo：官方HEK293、CDS-only、TE头，8次sTASEP，init0.01，seeds20261020/21/22，逆minmax标签尺度；可为负，不能直接当TE倍数。仓库commit `66c80c653e6db0d0b19a737f5d9fc8009db252b2`。先冻结候选再打分，禁止按该测试评分重新选候选。

GC：完整CDS百分比；CAI：仅固定TRAIN计算参考权重；MFE：全长RNAfold、37℃、dangles2，报告kcal/mol和每nt。用户以**同条CDS原生实际返回序列**为结构参考：GC偏差≤2个百分点；全长MFE偏差尽量≤10kcal/mol，是偏好范围而非删样本硬筛选。Hard/TE/CAI越高越好。

所有结果保留，包括失败者；原生同义终止密码子变化保留并标记。模型和评分器权重状态不应在搜索/评估中变化。没有实测表达标签，不新算或虚构预测R²。

## 隐空间问题已知结论

971万模型短/中/长 clean坐标AU(var>0.01)约151/106/93，原大模型66/50/46；但within-AA协方差参与率仅7.16/7.04/6.98，原模型5.21/4.75/4.59，坐标SNR>1仅2/2/2。坐标AU增多不能称256维已充分利用，也不能把21/256直接等同于8%密码子修改率。

以前把205个低方差维度整体重置，会让759aa候选Hard下降约1.416，说明弱坐标整体仍可有功能。需要区分坐标方差、AA条件内独立维数、信噪比、解码器敏感性和搜索可达性。

既有文献与诊断见 `docs/latent_molecular_optimization_MLM_research_20261024.md`、`docs/latent_search_diagnosis_20261024.md`。短期性质引导研究：简单gradient dot外推失败；冻结物理量的条件前向在12个TRAIN亲本/120变体上RMSE0.243、符号正确90.83%，有进一步研究价值，但并未加入本次只改规模的实验，不能称完整物理梯度已解决。

## 下一轮从这里开始

先读取本交接、本次GitHub发布回执和本地30条任务状态，连接远端确认进程与events推进，保留后台完整预算队列。待原生逐条完成，核对90行/30配对、AA/序列哈希、时间/查询字段、两个评分器和GC/CAI/MFE，再按长短档报告。

如果用户要继续调整模型，应基于固定验证集设计新对照；当前30条测试集已经用于这次冻结评估，若据此调参需明确披露其开发用途并使用新的最终评估集。尚无证据宣称小模型全面胜出，也没有“256维已充分利用”的结论。

详细历史与研究验收状态：`runs/latent_residual_recovery_gpu_20261023/HANDOFF.md`、`runs/latent_research_20261024/ACCEPTANCE_STATE.json`。本交接时间更近的状态优先，旧文档内的任务安排仅作背景。
