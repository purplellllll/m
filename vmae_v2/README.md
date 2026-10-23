# VMAE V2：五折训练与长度条件学习率

本目录保留2026-10-22服务器训练包的代码与文件校验，训练从头初始化；历史指标单独放在reference_previous_run，不代表本轮进度。原main分支的代码和实验记录保留。

## 环境与数据

Python 3.10、PyTorch 2.0.0+cu118；其余固定依赖见requirements.txt。完整安装、恢复、训练和评估说明见[README_TRAINING_ZH.md](README_TRAINING_ZH.md)。

训练数据、折表、蛋白FASTA、评分器权重和条件数组不提交到Git。将此前提供的DS_RiboDecode_VMAE_V2_5Fold_ServerBundle_20261022.zip解压，复制本目录缺少的文件至相同位置；缺失清单及SHA-256见ASSETS_REQUIRED.json。数据和权重准备好后，PACKAGE_MANIFEST.json应全项通过。本目录不包含服务器登录信息。

## 运行

```bash
cd vmae_v2
python run_vmae_v2.py check --gpu
python run_vmae_v2.py benchmark
mkdir -p logs
nohup python -u run_vmae_v2.py train-all > logs/train_all.log 2>&1 < /dev/null &
echo $! > logs/train_all.pid
python run_vmae_v2.py status
```

train-all执行5折×3种学习率配置，共15次训练；每个最多50轮，最少10轮，验证连续8轮不改善早停。长度分档为≤300、301–699、≥700 aa。三个版本的短/中/长基础学习率分别是uniform=(1e-4,1e-4,1e-4)、short_fast=(2e-4,1e-4,5e-5)、long_fast=(5e-5,1e-4,2e-4)。模型latent256、hidden512、encoder/decoder各6层；microbatch8，累积至最多32条。

训练和步长系数选择分开：训练结束后执行full可继续既定内层系数选择及外层对照，也可直接执行full从训练开始运行全队列。外层测试不参与模型或系数选择；尚无V2最优系数结论。详细协议见docs/EXPERIMENT_PROTOCOL.md。

启动记录写入结果目录和日志；git分支提交不包含训练输出、checkpoint、数据或登录凭据。不要将源码改动混入已有结果目录后强行恢复。

## ??????

??deploy_gpu.sh??Python3.10???????????????????????????????/GPU???????????????train-all????VMAE_HOME?VMAE_PYTHON?VMAE_RESULTS???????????SSH???????????????????????PyTorch???????????????cu118 wheel?


## Post-training coefficient experiments

After all 15 training runs finish, use `tools/run_vmae_v2_followup_20261022.sh` to queue the inner-selected fine-tuning experiments. See [the protocol](docs/VMAE_V2_POSTTRAIN_PROTOCOL_20261022.md) for the gain, learning-rate, KL, weight-decay and free-bits controls, paired V1 CE/loss acceptance, and latent interventions. The V1 epoch-50 reference weight is a separate private runtime asset; its required SHA-256 is in the protocol.


## Early follow-up after an explicit training pause

Use `tools/run_vmae_v2_followup_early_20261023.sh` only after pausing the upstream training queue. It starts from completed uniform folds and labels the result as a partial cross-validation pilot. See [the early-start protocol](docs/VMAE_V2_EARLY_FOLLOWUP_20261023.md).
