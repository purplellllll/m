# LCL、RPE-1、iPSC环境准备

评分器采用seq2ribo，RPE-1接口名为`rpe`。与现有HEK293评价保持一致：CDS-only TE、逆变换输出、随机种子20261020/20261021/20261022，每个种子8次sTASEP，init_p=0.01。Mean不在本次范围内。

服务器根目录：`/root/gpufree-data/seq2ribo_multicell_20261027`。

源码固定在官方提交`66c80c653e6db0d0b19a737f5d9fc8009db252b2`，沿用现有经过验证的CUDA/Mamba运行时。新增权重置于独立目录，不更换现有HEK293权重或评估锁文件。

每种环境包含：

- `<cell>_mamba_te_full_final_cds.pt`：官方CDS-only TE权重。
- `<cell>_te_transform_cds.json`：与权重配套的TE输出逆变换。
- `environment_parameters.json`中的对应条目：官方拟合的61个有义密码子等待时间、4类配对计数参数、4类角度参数、3个CDS区段参数；来自固定源码constants.py。

以上是seq2ribo推理需要的环境参数与模型资产，并非新增原始RNA-seq/Ribo-seq数据集，也不构造RiboDecode的RPKM环境向量。使用seq2ribo评分无需那类环境向量。

准备脚本：`tools/prepare_seq2ribo_multicell_20261027.py`。

准备过程中校验官方Git blob哈希，记录文件SHA256；严格加载模型，在同一批真实CDS上运行3个随机种子，检验有限输出、相同种子重复结果、批次逆序一致性、模型权重不变及结构几何非零。

只有`status.json`为`ready`且`cuda_preflight_passed=true`时，才能视为准备成功。`assets.json`、`scorer_lock.json`及`preflight.json`分别保存来源、评价设置和实测推理记录。

后续若将这些评分器用于隐空间优化，它们的得分应列为目标评分，不能再称为相对于该优化任务的独立评估。保留未参与优化的评价模型或实验结果用于独立验证。
