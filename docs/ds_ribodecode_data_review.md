# Human CDS / MANE 与 RiboDecode 数据准备审查

审查日期：2026-10-16。范围：只读检查现有准备脚本、audit、GENCODE v28 GTF及官方版本资料；新增此说明及一个固定版本 MANE summary 小文件和交集清单，没有修改代码、现有语料/split或下载新的完整序列库，没有启动训练。

## 结论

现有 32,796 条语料是 **RiboDecode 官方 major-isoform/gene-set 派生的人完整 CDS 开发库，使用 GENCODE v28 恢复精确匹配的终止密码子**。它可用于流程开发，但不等于“全 Human GENCODE CDS 主库”，也不是“MANE Select 核心子集”。

GENCODE v28 早于 MANE 的首次正式 Ensembl 注释集成，不能直接从 v28 标签抽取原生 MANE Select 子集。保留 v28 序列、增加固定 MANE 版本的选择注释是可行方案，但必须精确核对 transcript 版本或 CDS 序列，并如实命名为版本交集。

现有 `prepare_data.py --clusters ...` 需要原始官方 zip、GENCODE FASTA/index 等材料，不能只凭 `corpus.jsonl + audit.json + proteins.fasta` 重划分。

## Prepared：已经存在并核实的材料

| 项目 | 现状 |
|---|---|
| 官方数据 | `data/official/dataset_for_translation_model.zip`，121,954,830 B，MD5 `9e73fdb5b25aa063a8152b541ac302d5` |
| 主要来源 | zip 中 `human_coding_transcript_sequences.csv`；按 `mjIsoform_320.csv` 中实际出现的 isoforms 和 `genes.csv` 官方基因集合过滤 |
| 注释来源 | `data/reference/gencode_v28/gencode.v28.pc_transcripts.fa.gz` 和 `.cds_index.tsv` |
| 终止密码子 | 官方 CDS 通常不含 stop；仅当官方序列严格等于同版本 GENCODE transcript CDS 去掉末 stop 后的序列时，恢复该真实 stop；不盲加 TAA |
| 完整性 | ATGC、ATG 起始、末 stop、无内部 stop、长度整除 3；30–1500 codons，含 stop，不截断 |
| 有效记录 | 39,163 transcript rows；相同 CDS 合并 6,367 条，得到 32,796 unique CDS / 32,746 unique AA |
| 长度 | 90–4500 nt，中位数 1032 nt |
| 开发 split | train 26,764 / val 2,744 / test 3,288；11,376 gene/exact-AA 连通组件 |
| 保护 | 官方 test genes；SVBP `ENST00000372522.5` / `ENSG00000177868`；已有 Gluc、SVBP、NGF、H1N1 WT 的 exact-AA 组件保护 |
| 禁用材料 | 未使用优化后的 Gluc/SVBP 候选作主训练集；未根据预测表达量筛选天然 CDS |
| 认证 | `source_verified=true`；`homology_certified=false`。仅验证 gene/transcript/exact-AA 分离，尚未完成近同源簇 gate |

现有数据的适当短名可用 `RiboDecode-v3-derived human CDS / GENCODE-v28-stop-verified / development split`。这里的 v3 是 Figshare article version，不是 GENCODE 版本，也不是 RiboDecode 软件版本。

不能把这 32,796 条称为“GENCODE v28 的全部完整 CDS”：脚本额外限制了作者 major isoforms、官方 11,725-gene 集合及长度。也不能将“从作者 major-isoform 映射出现”解释为“每个基因只保留一个代表 isoform”；样本间 major isoforms 可以不同。

`reference_verified_stop_restorations=47,631` 是过滤流程中成功恢复 stop 的 transcript rows 数，包含后来因完整性/长度被排除的记录，不等于最终独特 CDS 数。

### Prepared extension：MANE v1.4 完整版本 transcript-ID 交集

已从 [NCBI 官方固定 release_1.4 summary](https://ftp.ncbi.nlm.nih.gov/refseq/MANE/MANE_human/release_1.4/MANE.GRCh38.v1.4.summary.txt.gz) 下载单个小文件，未下载 MANE 全序列库：

- `data/reference/mane/MANE.GRCh38.v1.4.summary.txt.gz`：1,113,014 B，SHA-256 `4b3992457556e302a5e47e18a305bb4763718377696ca37d5a6b34df7db630d2`。
- summary 共 19,404 行：19,338 MANE Select，66 MANE Plus Clinical。本次只使用 Select。
- 精确规则为：summary 的完整 `Ensembl_nuc`（包括 `.version`）等于 corpus 的一个 `transcript_ids` 条目；同时断言 Ensembl gene base ID 与该 corpus row 的 `gene_ids` 相符。未去掉 transcript version 强匹配。
- 匹配 **235 unique CDS / 235 transcripts / 235 genes**，占当前 32,796-CDS 库的 **0.7166%**，占 v1.4 MANE Select transcript IDs 的 **1.2152%**。
- 沿用原 split：train 183 / val 17 / test 35；未改动原 corpus/split。
- ID 清单：`data/reference/mane/corpus_mane_v1.4_exact_transcript_intersection.tsv`；SHA-256 `86bc2f956811e64adc43185161349f0926a0b7bd04b75fc5537e9a598d861944`。
- 完整覆盖率、下载 URL/时间、source/corpus/audit hashes 及限制：`data/reference/mane/corpus_mane_v1.4_intersection_manifest.json`。

这批扩展材料只能称 **MANE v1.4 transcript-ID intersection**：summary 不提供实际 CDS/UTR 序列，本次没有独立验证 MANE CDS sequence equivalence。235 条不是全 MANE core，也不是全 GENCODE 的覆盖估计。如此严格的版本交集较小，不应为扩大数量偷偷改为无版本 ID 匹配；需要更多核心样本时，下一步是带版本差异审计的真实 CDS 逐字比对。

## Official evidence：GENCODE v28 与 MANE 的时间及版本关系

1. GENCODE 官方发布历史将 v28 对应到 **2018 年 4 月、GRCh38.p12、Ensembl 92**。v28 发布页提供 protein-coding transcript FASTA，但该集合本身还包含 NMD、non-stop decay、IG/TR 等 transcript biotypes，并非每条都是常规完整、可翻译的主转录本。[GENCODE release history](https://www.gencodegenes.org/human/releases.html)、[GENCODE v28](https://www.gencodegenes.org/human/release_28.html)
2. Ensembl 的官方发布说明将首次 MANE Select 注释集成对应到 **Ensembl 96 / GENCODE 30，2019 年**；不能将它回写为 v28 原生标签。[Ensembl 96 release plan](https://www.ensembl.info/2019/02/07/whats-coming-in-ensembl-96-ensembl-genomes-43/)
3. MANE 定义要求匹配的 RefSeq 和 Ensembl/GENCODE transcripts 在 exon coordinates、coding region 和 sequence 上相同。相同基因、无版本 ENST 相同或蛋白 AA 相同，均不足以证明完整转录本为同一个 MANE transcript。[NCBI MANE](https://www.ncbi.nlm.nih.gov/refseq/MANE/)
4. NCBI 官方说明提醒 MANE 建立过程中更新了 transcript annotations，导致 accession version changes。这正是不能去掉 ENST 的 `.version` 后直接认定“MANE Select 匹配”的原因。[NCBI MANE Select v0.5 announcement](https://ncbiinsights.ncbi.nlm.nih.gov/2019/03/12/mane-select-v0-5/)

本次实际扫描本地 `gencode.v28.annotation.gtf.gz`：`MANE_Select` 和 `MANE_Plus_Clinical` 标签均为 0。APPRIS、GENCODE basic 或最长 CDS 都不能冒名为 MANE Select。

## Planned：如何满足“GENCODE 主库，优先 MANE Select 核心”

建议分开保存三种数据视图，避免新需求被旧语料名称掩盖：

| 视图 | 来源与选择规则 | 当前状态 |
|---|---|---|
| GENCODE-v28 Human CDS 主库 | 从固定 v28 annotation/FASTA 提取完整核基因 CDS，明确 protein_coding/biotype、染色体范围、标准遗传密码表、长度和 stop 规则；不限定作者表达过滤后的 gene set | Planned：现有 FASTA/GTF 可复用，独立构建入口及审计未完成 |
| v28 序列与 MANE Select 的核心交集 | 保留上述 v28 DNA；使用固定 MANE release 的 Select 记录进行 full-version ENST 匹配，或逐字相同 CDS 对照；只加入有证据的交集 | Prepared extension：当前作者派生库的 235 条 v1.4 transcript-ID 交集及 manifest；Planned：全 GENCODE 主库交集、CDS 序列等价核查 |
| RiboDecode 派生兼容库 | 当前 32,796 条；与作者 processed 数据及官方 split 兼容，用于复现、开发和对照 | Prepared；近同源 split 尚未认证 |

不混序列版本的两种严谨实现：

- **完整 transcript 身份匹配**：MANE summary 中带版本的 Ensembl transcript accession 与 v28 的 ENST.version 完全一致，并核对相关序列来源；命名为“GENCODE v28 ∩ MANE Select release X, exact transcript-version intersection”。不因缺失而回退到另一条 isoform。
- **仅 CDS 等价匹配**：若 MANE transcript 版本更新但 CDS DNA 与 v28 完全相同，可标记“MANE-selected CDS-equivalent v28 subset”。必须记录 MANE ENST.version、v28 ENST.version、CDS hash 和匹配类型；不能据此声称 UTR/exon 全转录本也匹配。蛋白 AA 一样但 CDS 同义差异存在时不算此类等价。

这会使用较新的 MANE 选择元数据，但序列仍全部来自 v28；应公开两个版本，不能称“原生 MANE v28”。如果改为直接采用较新 MANE DNA，就必须单列新版本数据域，并与旧 RiboDecode 评分器的输入分布偏移分开验证。

仍缺的具体材料：

1. MANE v1.4 官方 summary/Select 标记、带版本 ENST/RefSeq 对应表、下载日期和校验和已准备；尚缺将这一步接入独立 GENCODE 主库构建的可复现入口。
2. 对版本不一致但想保留的候选，提供该 MANE release 的实际 CDS 序列或能重建 CDS 的注释/转录本材料，以做逐字比较；不能仅凭基因名或 accession base ID 推断。
3. 主库/核心交集的行级 membership manifest、未匹配及版本冲突原因统计。匹配后再给出样本数，不预报“一基因一条”的完整覆盖量。
4. 主库及 core 的 MMseqs2 聚类、参数/版本/命令、输入 hash 和跨 split 同源审计；保留官方 test 和目标 WT 保护。

## 服务器重划分：现有脚本的真实依赖

`build()` 先打开并校验官方 zip，再读取 GENCODE FASTA/index，然后读 WT 文件，最后应用 `--clusters`。现有 CLI 没有 `--input-corpus` 或只重划分模式；因此只携带已清洗三文件无法执行该命令。

### Prepared 路径：不改代码的最小完整重建包

从仓库根目录运行时至少携带下列文件，保持相对路径：

```text
latent_decoder/prepare_data.py
data/official/dataset_for_translation_model.zip                 121,954,830 B
data/reference/gencode_v28/gencode.v28.pc_transcripts.fa.gz       41,509,543 B
data/reference/gencode_v28/gencode.v28.pc_transcripts.cds_index.tsv 5,550,273 B
data/gluc_cds/train_data.txt
data/svbp_cds/train_data.txt
data/NGF_cds/train_data.txt
data/H1N1_cds/train_data.txt
data/latent_decoder/corpus.jsonl                                69,787,257 B
data/latent_decoder/proteins.fasta                               16,046,360 B
data/latent_decoder/audit.json
```

原始材料约 169 MB；清洗三文件另约 86 MB。原始 zip 内含官方 split，无需额外解压矩阵。重新清洗不使用 GTF，因此仅为这一步无需携带 38 MB GTF；构建新的全 GENCODE 主库则另需 GTF。

四个 WT 文件虽然在代码中按“存在则读取”，为了保持与当前审计相同的防泄漏保护，它们是此打包方案的必带材料，不能当作可选附件省略。对 MANE 主库的新候选也应保留这些保护。

服务器先用已有 proteins.fasta 聚类，保存 TSV 和 provenance，再执行：

```bash
python latent_decoder/prepare_data.py \
  --clusters path/to/mmseqs_rep_member.tsv \
  --output data/latent_decoder_clustered
```

TSV 需覆盖全部当前 sequence IDs，包括 singleton self-pairs，且不能混入另一语料版本的 ID。脚本不覆盖现有输出，需使用新目录。

**认证局限**：当前代码把提供且覆盖完整的 TSV 记作 `homology_certified=true`，但仅由该 TSV 无法验证算法、identity、coverage、敏感度或命令。正式 gate 还必须检查外部 clustering provenance，不能把任意 identity/self-pair TSV 当成合格同源认证。

### Planned 路径：轻量服务器包

后续最小改动可增加单独的 corpus repartition 入口：验证 corpus/proteins hash → 读取完整 cluster TSV → union gene + exact-AA + cluster → 保留原 test 的所有组件及新增同源关联 → 从剩余组件固定划分 val/train → 新目录写 corpus/audit。

这样仅需现有三文件、cluster TSV/provenance 和该入口，不再需要重复携带官方 122 MB zip与GENCODE参考。但这一步**尚未实现或测试**，不能把它写成当前可执行命令。按原 test 优先传播是保守保护方案；不得在重划分时意外把旧 test/WT 放回 train。

## 限制

本次准备了 235 条 MANE transcript-ID 交集的 membership 清单，但未声称验证其 CDS 等价，也未另建核心训练语料；未更换现有 corpus 数据来源，未运行同源聚类、训练或 GPU。现有库可用于开发；新草案要求的全 GENCODE 主库及更完整 MANE 核心需要独立构建/序列审计后才能标记为 prepared。
