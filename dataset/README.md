# Frozen CDS dataset: GENCODE v28 + RiboDecode

版本：`ds-ribodecode-gencode28-v1`，冻结日期：2026-10-17。

本目录是新增的联合 CDS 数据集；历史 `data/latent_decoder` 数据说明仍描述之前的 RiboDecode-only 实验。新数据不包含优化生成序列，也不使用 RNA/RPF 表达值进行筛选。

## 文件与数量

```text
dataset/
  train/part-00000.jsonl.gz ... part-00007.jsonl.gz
  val/part-00000.jsonl.gz
  test/part-00000.jsonl.gz
  tokenizer.json
  split_assignments.tsv.gz
  manifest.json
  README.md
```

| 集合 | 唯一 CDS | 基因 | 不同蛋白 | 同时具有 RiboDecode 来源的 CDS |
|---|---:|---:|---:|---:|
| train | 35,710 | 15,327 | 35,631 | 22,283 |
| val | 3,949 | 1,735 | 3,934 | 2,303 |
| test | 4,342 | 1,883 | 4,327 | 2,727 |
| 合计 | 44,001 | 18,945 | 43,892 | 27,313 |

每个压缩分片最多 5,000 条记录。完整、顺序固定的分片列表在 `manifest.json` 的 `split_files` 字段中。

GENCODE 接受 53,673 条转录本记录，RiboDecode 接受 32,628 条记录；按完整 CDS 全局去重，合并 42,300 条重复记录，最终保留 44,001 条。27,313 条 CDS 同时具有两个来源，16,688 条仅有 GENCODE 来源；本次筛选后没有 RiboDecode 独有 CDS。两个来源不能被视为彼此独立的数据集。

每行 JSON 含 `sequence_id`、`cds_dna`、`protein_sequence`、`codon_ids`、`n_codons`、`gene_ids`、`transcript_ids`、`sources`、`source_transcript_ids`、`split`、`cluster_id`。DNA 字母表示与 RNA 的区别仅为 T/U；CDS 和 token 均包含末端 stop，蛋白字符串不含 stop。

## 提取、筛选与去重

1. 固定使用人类 GENCODE v28（GRCh38.p12）的 `gencode.v28.pc_transcripts.fa.gz` 与同版本 `gencode.v28.annotation.gtf.gz`。按 FASTA 标头中的 1-based inclusive `CDS:start-end` 提取 CDS，按精确版本 transcript ID 关联 GTF。
2. GTF 必须为 `transcript_type=protein_coding`，排除 `cds_start_NF` / `cds_end_NF` 和线粒体转录本。不会把 FASTA 中的 NMD 等其他转录本类型混作标准蛋白编码转录本。
3. RiboDecode 使用官方 Figshare v3 的 `human_coding_transcript_sequences.csv`，保留 `mjIsoform_320.csv` 中的转录本且基因属于 `genes.csv` 官方 train/test 集合；再应用相同 GTF 和 CDS 规则。
4. 仅当 RiboDecode CDS 与同版本 GENCODE 完整 CDS 去掉末 3 nt 后逐字一致，且参考末 3 nt 是合法 stop 时恢复该真实 stop；不盲加 TAA。
5. 两个来源统一要求：仅 A/C/G/T、ATG 起始、一个末端 stop、无内部 stop、长度为 3 的倍数，且为 30–1,500 codons（90–4,500 nt，包含 stop）；不截断。
6. 按完整 CDS 字符串去重，合并所有基因、转录本和来源信息。不同同义编码但蛋白相同的 CDS 保留，因为它们是不同的密码子样本；划分时将其绑定。

各来源逐项排除计数、原始文件 SHA-256、RiboDecode MD5、旧数据集哈希均在 `manifest.json`。旧库 32,796 条中，27,357 条 CDS 在新库中保留，5,439 条未通过新增的注释筛选或未由其他合格转录本保留。保留记录的原始 split 均未改变；少数记录因其他合格 GENCODE 转录本保留了 CDS，但不再具有合格 RiboDecode 转录本来源。

## 固定划分

将共享基因 ID、去版本号 transcript ID 或完全相同蛋白的记录做传递连通合并。`cluster_id` 表示这个身份连通组件，不是近同源聚类结果。

- 组件含官方测试基因或旧 test 身份：整个组件进入 test。
- 否则含旧 val 身份：进入 val；否则含旧 train 身份：进入 train。
- 全新增组件：使用固定字符串 `ds-ribodecode-gencode28-v1` 和组件 ID 的 SHA-256，对 10 取模；0 为 test，1 为 val，其余为 train。比例是组件层面约 80/10/10，最终样本数不强制精确比例。
- `split_assignments.tsv.gz` 保存所有序列的最终归属，训练时直接读取，不重新随机划分。组件 ID 基于排序后的成员 ID，输入顺序不影响划分。

已全量验证：CDS 无重复；基因、去版本号转录本、完全相同蛋白和组件在集合之间无重叠；官方测试基因不进入 train/val；所有 token 可以逐字还原 CDS。

**本版是固定开发划分，`homology_certified=false`。尚未执行近同源聚类和跨集合敏感搜索，不能据此宣称未见同源蛋白泛化。** 如需正式同源隔离版本，应另建数据版本并重新训练，不覆盖此冻结版。已有 epoch-10 checkpoint 来自旧语料和划分，不能作为此版正式评估的训练完成模型。

## 统一 Codon Tokenizer

实现位于 `latent_decoder/tokenizer.py`，数据与 VMAE 共用同一实现；冻结词表在 `tokenizer.json`。

- 有效 codon ID 为 0–63，逐项兼容原 RiboDecode `my_vocab`。
- VMAE 特殊 token：`PAD=64`，`MASK=65`。数据文件不保存 padding 或 mask。
- 输入大小写归一化，U 转 T；非法 CDS 抛出异常，不截断。
- 原评分器的 `NNN=64` 是另一种接口语义，不能直接拿 VMAE 的 PAD 当作 NNN 或可设计密码子；有效位置适配 65 通道评分器时追加全零 NNN 概率通道。

```python
from latent_decoder.prepare_combined import iter_split
from latent_decoder.tokenizer import encode_cds, decode_cds

record = next(iter_split("dataset", "train"))
assert encode_cds(record["cds_dna"]) == record["codon_ids"]
assert decode_cds(record["codon_ids"]) == record["cds_dna"]
```

## 校验与训练读取

在仓库根目录使用 Python 3.10+：

```bash
python -m latent_decoder.prepare_combined --verify dataset
python -m unittest latent_decoder.test_combined latent_decoder.test_decoder latent_decoder.test_repartition -q
```

首次校验不需要 PyTorch；单元测试和训练读取需要原项目 PyTorch 环境。训练入口已支持 `--data dataset`，并保留非同源隔离数据的显式开发标志：

```bash
python -m latent_decoder.train --data dataset --config latent_decoder/config.json --output runs/combined_v1 --allow-development-split
```

以上仅为使用说明，本次交付未启动训练。数据哈希绑定 `manifest.json`，旧数据的 checkpoint 不会被静默续训到新数据上。

复现构建需要在 `--root` 指定目录准备以下原始材料；脚本拒绝覆盖已有输出：

```text
data/official/dataset_for_translation_model.zip
data/reference/gencode_v28/gencode.v28.pc_transcripts.fa.gz
data/reference/gencode_v28/gencode.v28.annotation.gtf.gz
data/latent_decoder/corpus.jsonl
data/latent_decoder/proteins.fasta
data/latent_decoder/audit.json
```

```bash
python -m latent_decoder.prepare_combined --root /path/to/source_workspace --output dataset_rebuilt
```

最后三个文件是冻结旧 split 的锚点，可从已有 [20-epoch 工程包 release](https://github.com/purplellllll/m/releases/tag/vmae-20ep-20261016) 获取并核对 manifest 中记录的哈希；本脚本不使用其中的训练 checkpoint。压缩文件固定时间戳、记录排序和词表，便于相同环境下逐字节复现。

## 来源、署名和使用条件

- GENCODE Consortium：[GENCODE Human Release 28](https://www.gencodegenes.org/human/release_28.html)，[公开数据访问说明](https://www.gencodegenes.org/pages/data_access.html)，[EMBL-EBI 使用条款](https://www.ebi.ac.uk/about/terms-of-use/)。原始文件位于官方 FTP 的 `pub/databases/gencode/Gencode_human/release_28/`。本目录只发布筛选后的派生 CDS，不为 GENCODE 原始材料另行指定许可证。
- Yupeng Li，RiboDecode 数据：[Figshare v3 / DOI 10.6084/m9.figshare.28916288.v3](https://doi.org/10.6084/m9.figshare.28916288.v3)，[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)。原 ZIP 为 121,954,830 bytes，MD5 `9e73fdb5b25aa063a8152b541ac302d5`。
- 本项目对原数据进行了 CDS 提取、注释筛选、参考核对后的 stop 恢复、去重、tokenization 和新划分；这里的划分不是原作者完整原始 split。
