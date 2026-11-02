# 多任务训练与特征路径独立审查

## Material Passport

- 日期：2026-11-02，Asia/Shanghai。
- 只读审查 `train.py`、`features.py`、`model.py` 与 `dynamics.py`；未修改被审查模块，未启动GPU训练，未操作旧队列。
- CPU检查：模型原自检、缺失UTR、长CDS压缩、右padding不变、缺失标签梯度、三任务真实记录batch对齐、目标训练统计复算均通过。

## 需要处理或明确限制的发现

### [P1] 当前neutral动力学没有密码子或结构辨别能力

`features.py`的`prepare()`调用`dynamics.extract(...)`未传入参数或geometry，`dynamics.py:29`默认所有sense-codon等待值为2，配对/角度/位置等待全零。相同长度且同seed的序列会得到相同模拟概率及全部事件输出；host只记在审计中，不改变数值。已直接比较缓存中两个不同DNA的mRFP训练序列：`node`及`global_features`逐位完全相同。

因此这个dynamics arm只能检验固定窗口事件观测和长度派生信号，不能验证同义密码子的动力学增量，更不能宣称已完成宿主动力学校准。全部mRFP长226codons，新增物理输入在该任务中没有样本间差异。协议已明确neutral，但验收解释必须保留这一限制。

建议在训练前确认：将neutral视作明确消融对照；如要检验密码子效应，增加由合法同宿主train-only位置RPF拟合的参数或有来源的固定密码子代理，并明确后者不是真实校准。缺少训练来源时不能由TE标量制造已识别动力学结论。需要改变特征源时，先停止相关特征进程并保存旧协议，再建新缓存/锁，不能静默复用。

### [P2] 训练入口尚未校验派生数据与缓存的预期哈希

`train.prepare_protocol()`记录`derived_data_protocol.json`本身的SHA，却未将其中`derived_sha256['train_validation.jsonl']`与实际训练文件比较；实际行文件若在密封后变动，当前训练仍会继续。`target_statistics.json`也未作为训练协议直接锁定的文件，后续逆变换必须防止统计文件漂移。

`features.load()`只核验缓存的DNA SHA，未检查每份缓存的`protocol_sha256`是否等于当前`features_protocol_locked.json`的SHA。虽然`features.prepare()`正常完成前做过检查，但直接运行训练、或特征协议变化后读旧缓存，都不能仅靠当前load断言证明来源一致。

建议训练前增加只读验证：派生行文件SHA、目标统计SHA、特征锁中实际source SHA、scaler与`features_complete.json`匹配，以及各缓存protocol SHA。验证可放在训练入口，避免为这一检查改动已锁定特征生成脚本。

### [P2] 缓存写入不是原子替换，异常中断后需要显式恢复

`features.prepare()`直接以目标`.npz`路径写入。中断可能留下存在但不完整的文件；下一次恢复会尝试`np.load()`并失败。它不会悄悄接受坏缓存，因此不是静默污染，但应记录损坏路径后仅重建该缓存，或新版本采用临时文件完成后替换。不要一遇到存在文件就跳过来源检查。

## 已核查无问题的项目

|项目|证据与判断|
|---|---|
|真实数据batch及mask|三条真实训练记录的mask恰为expression、HL+TE、solubility；对应四列位置正确，279维stats匹配模型|
|目标归一化|仅train值复算四任务均值/总体SD，与`target_statistics.json`误差0；val/test不参与|
|输入归一化|`features.prepare()`的s/global/node统计仅来自split=train；所有序列的无标签缓存不等于用测试标签训练|
|同DNA不同UTR|缓存只缓存CDS动力学，`base_features()`仍逐记录编码各自UTR及全区组成，不会因共享DNA把UTR覆盖|
|重复测量键|bank以record_id而非uid存储；同split重复CDS不丢失独立标签行|
|warmstart|只载入codons/context，逐键形状及完整state严格加载；新头/UTR/AA/物理另训。旧人源TE训练蛋白与本轮1188测试记录的蛋白交集为0|
|公平两臂初始化|相同seed、相同构造器和载入顺序；参数存在于两臂，sequence臂只绕开物理计算，样本顺序由独立相同numpy RNG驱动|
|序列压缩|超过512时对全部有效codon embedding做adaptive平均池化，不截断后段；会损失单密码子分辨率，是能力限制而非错位|
|UTR限制|>1024时保留首尾各512且保留全区1–3mer组成；不能描述成完整UTR序列注意力|
|模型维度|2×192 CDS池化＋256 UTR＋128 stats＋16 host＋64 physics＋3 availability正确对应head输入|
|缺失标签|masked_loss把未观测target先置零后屏蔽；CPU检查未观测列梯度严格0，不形成零标签监督|
|训练任务权重|三primary轮转；HL批次内部HL/TE各半，因此三个primary监督总权重并非完全相等，协议已有平均观测任务说明，不应另称严格等primary损失|
|checkpoint选择|只用三个实测primary的validation macro Pearson，每500步选择；不使用TE或test指标选checkpoint|
|训练/验证模式|predict设eval，下一次更新重新train；bf16仅训练，验证全float32|
|断点状态|保存numpy、torch及CUDA RNG、优化器、step与best；恢复核对训练协议SHA|
|GPU并存|设置25% allocator额度及2CPU线程，未停止旧队列；额度不等于设备级硬隔离。审查时GPU8188MiB总、2549MiB使用，未实际验证新长批次峰值|

## 结论

数据对齐、缺失标签、维度和验证选择没有发现阻塞错误。主要科学风险是neutral物理对同长同义序列完全无辨别能力；应先决定物理参数实验定义，再启动最终验收训练。代码来源与缓存绑定建议在训练入口补强。模型变化带来的提升仍须三项实测指标分别验证，且没有同样本三标签联合面板。

未进行GPU训练或修改已锁源码；新模型的预测收益、GPU峰值及宿主动力学校准尚未被本次审查证明。
