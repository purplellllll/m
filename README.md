# DS-RiboDecode

研究预训练潜在空间与直接密码子空间是否具有互补性。当前可运行部分为**RiboDecode CDS-VMAE预训练框架**，训练encoder和decoder。

## 本轮启动：V100 32 GB，20 epochs

- [服务器下载、环境、启动与续训说明](latent_decoder/README.md)
- [完整服务器包：框架＋已清洗数据](https://github.com/purplellllll/m/releases/download/vmae-20ep-20261016/ds_ribodecode_20ep_v100_2026-10-16.tar.gz)
- [数据来源、划分和校验和](data/latent_decoder/DATA_CARD.md)

数据沿用RiboDecode官方数据派生库：train 26,764条、val 2,744条、test 3,288条。train/val全部来自原官方训练gene集合，官方测试gene不参与训练。GENCODE v28仅核对末端stop，本轮不扩展GENCODE/MANE/OAS语料。

配置固定20轮、关闭早停，microbatch4、累积8、FP32。每个完整epoch保存checkpoint；恢复后累计到第20轮。开始前自动执行最长训练batch的前向/反向检查。

```bash
# 下载并解压服务器包、激活环境后，从包根目录执行
bash latent_decoder/run_server_20e.sh
# 中断恢复
bash latent_decoder/run_server_20e.sh resume
```

训练与重划分共12项合成测试已通过；未运行真实语料训练或V100实测。当前为开发split，近同源聚类尚未完成，不能据此宣称未见同源蛋白泛化。双空间优化器、通用MFE代理及其效果验证仍在后续研究阶段。

语料通过Release附件提供，Git仓库保存代码、数据审计和方案。数据许可与派生处理见数据卡；当前没有训练完成的checkpoint。

## 研究方案

1. [整体方案与解码器验收标准](docs/DS_RiboDecode_plan_2026-10-16.md)
2. [双空间数学与实验设计审阅](docs/ds_ribodecode_math_review.md)
3. [数据来源与GENCODE／MANE审阅](docs/ds_ribodecode_data_review.md)

上述方案保留研究讨论历史；其中40轮及扩展GENCODE/MANE的建议，由本轮明确的“RiboDecode来源、20轮”训练配置取代。以`latent_decoder/config.json`及启动说明为当前执行依据。

## 主要参考

- [RiboDecode论文](https://doi.org/10.1038/s41467-025-64894-x)及[官方代码](https://github.com/wangfanfff/RiboDecode)
- [RiboDecode官方数据 v3](https://doi.org/10.6084/m9.figshare.28916288.v3)
- [LSCO](https://arxiv.org/abs/2610.03098)
