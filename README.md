# DS-RiboDecode 研究方案

探索预训练潜在空间与直接密码子空间协同优化：在保持目标蛋白质氨基酸序列不变的前提下，研究生成先验与受约束直接搜索是否具有互补性。

当前阶段：**方案设计与实验准备**。文档版本：2026-10-16。

## 阅读顺序

1. [整体方案与解码器验收标准](docs/DS_RiboDecode_plan_2026-10-16.md)：数据准备、VMAE配置、验收门槛、双空间目标、权重与温度调度、公平对照及评价体系。
2. [双空间数学与实验设计审阅](docs/ds_ribodecode_math_review.md)：残差尺度、正则、温度的耦合，以及初始化和计算预算的控制。
3. [数据来源与GENCODE／MANE审阅](docs/ds_ribodecode_data_review.md)：现有开发库的来源、版本对齐、MANE交集与同源划分的限制。

## 本次发布范围

本仓库当前上传研究方案及审阅文档。文中提到的训练代码、清洗数据、参考序列、模型权重和服务器训练包不包含在本次提交中；这些文件路径指配套的本地研究工作区，不能直接在本仓库执行相关训练命令。

当前已整理的开发库来自RiboDecode官方数据，含32,796条CDS；它不等于完整GENCODE／MANE主库。真实语料训练、近同源聚类、GPU性能测量和双空间效果验证尚未完成。文中的验收阈值与超参数为实验提案，不代表已经取得的结果。

## 主要参考

- [RiboDecode论文](https://doi.org/10.1038/s41467-025-64894-x)及[官方代码](https://github.com/wangfanfff/RiboDecode)
- [LSCO：Predictor-Guided Latent Space Codon Optimization for Maximizing Protein Expression](https://arxiv.org/abs/2610.03098)
- [GENCODE Human](https://www.gencodegenes.org/human/)
- [NCBI MANE](https://www.ncbi.nlm.nih.gov/refseq/MANE/)
