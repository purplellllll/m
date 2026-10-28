# 跨环境评分对比：仅评分

用户明确指定不启动优化。本次工作仅对原固定30条TEST身份的已有候选进行seq2ribo LCL、RPE-1、iPSC评分，不训练、不搜索、不重新选择序列，也不修改第6轮权重。此前已启动的第6轮优化/HEK293独立评价任务继续运行；其测试候选产出后自动补评。

启动时冻结305个已有候选：原始序列30、9.7M模型30、V3原Hard方法30、V3同奖励搜索30、V4第2轮30、第3轮30、GEMORNA30、CodonTransformer30、FPP-CodonBERT30、LinearDesign λ0共27、完整RiboDecode共8。只有第6轮新增测试候选会自动加入，其他基线保持启动时快照。

每个候选在三个环境分别使用随机种子20261020/20261021/20261022，各8次sTASEP，CDS-only TE逆变换输出。对同环境、完全相同序列复用缓存评分。不同环境的原始分数尺度不直接横向排名，未完成N不与完整30条均值直接排名。报告提供相同身份与原始序列、第6轮的配对差值和胜率计数。

HEK293分数以及GC、CAI、MFE、MFE/nt、RiboDecode分数复用原来锁定的同一候选记录。没有构造Mean综合分。

服务器输出目录：`/root/gpufree-data/seq2ribo_multicell_comparison_20261027`。

工作区自动备份目录：`runs/seq2ribo_multicell_comparison_20261027`。

- `candidate_manifest.json`：锁定候选及序列哈希、源文件和原有指标。
- `plan.json`：评分配置、环境锁文件哈希及脚本哈希。
- `metrics/`：每个候选三环境分种子得分、均值和种子标准差。
- `metrics.csv`：已有指标及三环境均值。
- `summary.json`、`COMPARISON_ZH.md`：各方法完成N、均值和同身份配对对比。
- `status.json`、`complete.json`：运行状态和正式完成凭据。

配对统计的独立单元测试已通过。实际CUDA评分已启动并确认产出结果；RNop因权重不可获取仍缺项。
