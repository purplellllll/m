# V3: VMAE 2,498,240 参数冻结版本

本目录保存实际完成的 250 万参数模型：d_model128、FF512、Encoder/Decoder各6层、8 heads、每密码子latent256。只相对971万版改变两个宽度字段。源码、数据与权重身份由JSON清单记录。

固定TRAIN23812 / VAL2592 / TEST6392，按同源分组固定split0，不轮换五折。训练50轮，按固定192条验证诊断面板选择epoch42；masked CE0.9838190881，fixed loss1.1352500385。两项均未优于971万版。

环境：Linux、Python3.12、Torch2.5.1+cu124、NumPy1.26.4。安装匹配CUDA的Torch及requirements.txt。进入本目录后运行：

```bash
python verify_snapshot.py
python verify_snapshot.py --data /absolute/path/to/data --checkpoint /absolute/path/to/fold0_uniform.pt
python -m latent_decoder.ribodecode_only_train --data /absolute/path/to/data --output /absolute/path/to/new_run/fold_0/uniform --fold 0 --profile uniform --microbatch 4
```

使用新输出目录，保留完成的checkpoint。实际训练参数以MODEL_CARD.json和训练源为准；配置JSON内部分字段被训练入口固定覆盖。最优权重作为同仓库Release资产提供，下载后核验ASSETS_REQUIRED.json中的SHA256。训练数据不随Git发布。

加载权重：

```python
from latent_decoder.latent_information_probe import load_model
model = load_model('/absolute/path/to/fold0_uniform.pt', 'cpu').eval()
```

截至2026-10-26，两个VMAE、CodonTransformer和FPPGroup CodonBERT的30条统一评分已经完成，原生完整预算6/30；详见BENCHMARK_REPORT.md。GEMORNA、RNop与LinearDesign扩展单独记录，未完成的项目不得视为已有结果。GC即GC含量，MFE同时报告全长和每nt；不同模型的原生分数不混成一个统一尺度。

tools中的搜索和服务器恢复脚本保留原运行目录依赖，并非任意目录开箱即用。共同评分需要官方RiboDecode、seq2ribo及其权重、锁定面板、背景向量、ViennaRNA2.7.2和匹配CUDA的Mamba2.2.4。输入面板来自人类天然CDS。

当前预测评分不代表实测表达，原生完整对照尚未完成，搜索预算不相同。第三方评分器保留原作者来源与许可边界。
