# 本轮数据：RiboDecode官方训练来源

配置：V100 32 GB、20 epochs，使用现有CDS开发划分，不扩展GENCODE/MANE/OAS语料。

## 来源与许可

- Yupeng Li，RiboDecode，Figshare v3：[10.6084/m9.figshare.28916288.v3](https://doi.org/10.6084/m9.figshare.28916288.v3)，[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)。
- 原zip：`dataset_for_translation_model.zip`，121,954,830 bytes，MD5 `9e73fdb5b25aa063a8152b541ac302d5`。
- CDS来源：`human_coding_transcript_sequences.csv`；使用`mjIsoform_320.csv`的isoform集合和`genes.csv`的官方训练/测试gene集合。
- 本项目进行了筛选、参考核对、stop恢复、去重和开发划分；派生划分不冒充原作者原始split。
- GENCODE v28仅核对对应transcript的真实末端stop：只有原CDS逐字等于参考完整CDS去掉末3nt且参考stop有效时恢复。不盲加TAA，不加入新的GENCODE转录本。
- RNA/RPF数值不是VMAE重建标签，未使用预测高分或优化产生的序列训练。

## 划分核验

| split | CDS数 | gene数 | 官方训练gene | 官方测试gene |
|---|---:|---:|---:|---:|
| train | 26,764 | 9,291 | 9,291 | 0 |
| val | 2,744 | 965 | 965 | 0 |
| test | 3,288 | 1,148 | 10 | 1,138 |

train/val没有官方测试gene或官方集合外gene。test中3,278条CDS涉及官方测试gene，另10条为保护目标/连通组件迁移产生的隔离；test不参与20轮训练或checkpoint选择。

总计32,796条唯一CDS、32,746个不同蛋白；长度90–4,500nt（含stop），中位数1,032nt。相同gene、transcript、exact-AA跨split无重叠。

近同源聚类尚未执行，`homology_certified=false`，当前是开发划分。未见gene只针对VMAE，不等于原translation predictor或全系统从未见过相近蛋白。

## 校验和

| 文件 | SHA-256 |
|---|---|
| corpus.jsonl | `29ec4ab16cef559e7733cfa3d310a9a9c288bffe4c0ab52a3df6bc00bc472222` |
| proteins.fasta | `f3354284ac59d744a8b68ef9d9b938304a4f50c6e28d1008198479df4e756e33` |

`audit.json`保留来源、处理计数和保护记录。原材料路径是来源记录，不表示所有原zip/参考文件都包含在轻量附件内。训练和可选repartition使用现有清洗库。
