# 人源HEK293 TE有限开发结果

256训练、96验证，80–500密码子，两个种子；未预测测试集。

|模型|验证Pearson|
|---|---:|
|finetune2_no_physics两种子集成|0.4315|
|finetune2_no_stats两种子集成|0.4350|
|finetune2_properties两种子集成|0.4379|
|finetune2_sequence_only两种子集成|0.4326|
|finetune4_properties两种子集成|0.4239|
|frozen_c_properties两种子集成|0.4324|
|native_seq2ribo_paper_HEK293|0.5118|
|RiboDecode_fixed_RNA_proxy|0.5368|
|length_only_ridge10|0.3670|
|codon_frequency_ridge10|0.3443|
|GC_MFE_CAI_ridge10|0.0814|

作者seq2ribo历史权重与当前原生对齐划分的训练交叉尚未认证，只作描述性比较。RiboDecode固定RNA分数不是专门训练的TE头。六支线都在验证集开发，不能当独立确认。扩大训练和确认集评估前，不替换旧实验权重或冒称已实现全面人源校准。
