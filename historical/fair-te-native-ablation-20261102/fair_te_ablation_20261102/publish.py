"""Publish only completed, audited results on an isolated branch and prerelease."""
from pathlib import Path
import os,sys,json,hashlib,shutil,subprocess,tarfile,time
import torch
import numpy as np
HERE=Path(__file__).resolve().parent;WORK=HERE.parent.parent;TE=WORK/'runs/te_stability_20261102'
DEST=WORK/'repos/riboseq_fair_evidence_20261102';ART=WORK/'deliverables/fair_te_ablation_20261102'
BRANCH='results/fair-te-ablation-20261102';TAG='fair-te-native-ablation-20261102';REPOSITORY='purplellllll/m'
ENV={k:v for k,v in os.environ.items() if k.lower() not in ['http_proxy','https_proxy','all_proxy']}
ENV['GIT_TERMINAL_PROMPT']='0'
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,v):p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,indent=2,ensure_ascii=False),encoding='utf-8')
def copy(src,dst):dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,dst)
def fmt(value):return 'NA' if value is None else f'{value:.4f}'
def run(cmd,cwd=DEST):
    r=subprocess.run(cmd,cwd=cwd,env=ENV,capture_output=True,text=True,encoding='utf-8',errors='replace',creationflags=subprocess.CREATE_NO_WINDOW)
    if r.returncode:raise RuntimeError(str(cmd[:3])+': '+r.stderr[-5000:])
    return r.stdout.strip()
def git(*args):return run(['git','-c','http.proxy=','-c','https.proxy=','-c','http.version=HTTP/1.1',*args])
def gh(*args):return run(['gh',*args])
def main():
    assert read(HERE/'complete.json')['status']=='completed'
    assert read(HERE/'independent_audit.json')['status']=='passed'
    assert read(TE/'native_compare/complete.json')['all_endpoints_cross_scored']
    assert read(TE/'te_100000/training_complete.json')['models']==2
    assert read(HERE/'full_length_100000/complete.json')['reused_test']
    assert git('branch','--show-current')==BRANCH
    assert not git('diff','--cached','--name-only'),'Unexpected pre-staged changes; preserve them for review'
    if (HERE/'publication_complete.json').exists():return
    if (ART/'committed_package.json').exists():
        local=read(ART/'committed_package.json');assert git('rev-parse','HEAD')==local['commit']
        assert sha(ART/'fair-te-own-models-20261102.tar.gz')==local['archive_sha256']
        return upload(local['commit'])
    for path,digest in read(HERE/'protocol_locked.json')['source_sha256'].items():assert sha(path)==digest,path
    assert sha(HERE/'evaluate.py')==read(HERE/'metric_protocol_locked.json')['evaluator_sha256']
    assert sha(DEST/'physics_scorer/fair_te_ablation.py')==read(HERE/'portable_module_parity.json')['portable_source_sha256']
    save(HERE/'publication_status.json',dict(stage='building_audited_package',unix=time.time()))
    ART.mkdir(parents=True,exist_ok=True);evidence=DEST/'evidence'/TAG;code=DEST/'historical'/TAG
    sources=[]
    def evidence_copy(root,names,prefix):
        for name in names:
            source=root/name;assert source.is_file(),source;target=evidence/prefix/name;copy(source,target)
            sources.append(dict(source=str(source.relative_to(WORK)).replace('\\','/'),destination=str(target.relative_to(DEST)).replace('\\','/'),sha256=sha(source)))
    evidence_copy(HERE,['protocol_locked.json','metric_protocol_locked.json','inputs_locked.json','summary.json','metrics.csv','per_gene_predictions.csv','test_predictions_locked.json',
        'all_checkpoints_locked.json','independent_audit.json','complete.json','公平基线与三路消融.md','all_preflight_complete.json',
        'ribogl_backend_parity.json','portable_module_parity.json','initial_PyG_import_crash_archived.json','preflight_cudnn_failure_archived.json'],'fair_te')
    for path in sorted((HERE/'preflight').glob('*.json')):evidence_copy(HERE,[str(path.relative_to(HERE))],'fair_te')
    for path in sorted((HERE/'models').glob('*/history.json')):evidence_copy(HERE,[str(path.relative_to(HERE))],'fair_te')
    evidence_copy(TE/'native_compare',['protocol_locked.json','summary.json','endpoint_metrics.json','原生优化对照.md','complete.json',
        'preflight_audit.json','loading_path_correction.json','case3_recovery_execution.json','native_surrogate_fit_audit.json','native_surrogate_holdout_audit.json',
        'own_search/protocol_locked.json','own_search/reload_check.json'],'native_compare')
    for i in [1,2,3]:evidence_copy(TE/'native_compare',[f'case{i}_native/completion_receipt.json',f'case{i}_native/runtime_protocol.json'],'native_compare')
    for i in [1,2,3]:evidence_copy(TE/'native_compare',[f'own_search/case{i}.json'],'native_compare')
    if (TE/'native_compare/own_search/memory_recovery.json').exists():evidence_copy(TE/'native_compare',['own_search/memory_recovery.json'],'native_compare')
    if (TE/'native_compare/case3_native_interrupted_io_1/failure.json').exists():evidence_copy(TE/'native_compare',['case3_native_interrupted_io_1/failure.json'],'native_compare')
    evidence_copy(TE/'te_100000',['protocol_locked.json','selection_locked.json','training_complete.json','requested_plan.json'],'human_te_100000')
    evidence_copy(TE/'te_24000',['protocol_locked.json','selection_locked.json','training_complete.json'],'human_te_search24k')
    for path in sorted((TE/'te_100000/models').glob('*/history.json')):evidence_copy(TE/'te_100000',[str(path.relative_to(TE/'te_100000'))],'human_te_100000')
    evidence_copy(TE/'stability',['protocol_locked.json','summary.json','persist_units_audit.json','model.npz'],'half_life')
    rescoring=[{k:r[k] for k in ['uid','gene','dna','utr5','utr3','n_codons']} for r in read(TE/'lambda_scan/inputs_locked.json')]
    save(evidence/'native_compare/inputs_for_rescoring.json',dict(scope='Three public-reference development inputs only, not the bulk training dataset',records=rescoring))
    evidence_copy(HERE/'full_length_100000',['protocol_locked.json','summary.json','metrics.csv','predictions.json','own_predictions_locked.json','complete.json'],'human_te_100000_reused903')
    scripts={HERE:['models.py','experiment.py','evaluate.py','evaluate_100000.py','publish.py','pipeline.py','preflight_queue.py',
        'verify_ribogl_backend.py','verify_portable_module.py','audit_native_surrogates.py','audit_native_holdout_linux.py'],
        TE:['continue_te.py','stability.py','locked_scorer.py','scan.py','evaluate_scan.py','train_to_24000.py','train_to_30000.py','train_to_100000.py',
            'native_compare_queue.py','search_continued_native_panel.py','finish_native_comparison.py','locked_scorer_24000.py','recover_native_case3.py']}
    for root,names in scripts.items():
        for name in names:
            path=root/name;target=code/root.name/name;copy(path,target)
            sources.append(dict(source=str(path.relative_to(WORK)).replace('\\','/'),destination=str(target.relative_to(DEST)).replace('\\','/'),sha256=sha(path),role='source_snapshot'))
    for source in [TE/'native_compare/native_bounded_cpu.py',WORK/'tools/native_default_cpu.py']:
        target=code/'native_compare'/source.name;copy(source,target)
        sources.append(dict(source=str(source.relative_to(WORK)).replace('\\','/'),destination=str(target.relative_to(DEST)).replace('\\','/'),sha256=sha(source),role='native_runtime_source'))
    runtime_text=(TE/'native_compare/native_bounded_cpu.py').read_text(encoding='utf-8')
    package=(WORK/'external/ribodecode_1_3_0/RiboDecode').resolve()
    linux_package='/mnt/'+package.drive[0].lower()+'/'+package.relative_to(package.anchor).as_posix()
    old='package=Path('+repr(linux_package)+')';assert runtime_text.count(old)==1
    runtime_text=runtime_text.replace(old,"package=Path('/home/lc666/.cache/te_native_20261102_case3_retry1/RiboDecode')")
    target=code/'native_compare/case3_native_bounded_ext4.py';target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(runtime_text.encode('utf-8'))
    assert sha(target)==read(TE/'native_compare/case3_native/runtime_protocol.json')['wrapper_sha256']
    sources.append(dict(source='Linux_ext4_runtime_wrapper_reconstructed_and_hash_verified',destination=str(target.relative_to(DEST)).replace('\\','/'),sha256=sha(target),role='native_runtime_source'))
    save(evidence/'SOURCE_MANIFEST.json',dict(copied_evidence=sources,raw_train_sequences_not_redistributed=True,third_party_checkpoints_not_redistributed=True))
    # Own trained weights only, tensor states plus plain metadata; no optimizer/RNG/raw data.
    sys.path.insert(0,str(HERE));import experiment as ex
    from models import build
    from physics_scorer.human_te_model import HumanTEScalar
    import importlib.util
    spec=importlib.util.spec_from_file_location('physics_scorer.fair_te_ablation',DEST/'physics_scorer/fair_te_ablation.py')
    portable_module=importlib.util.module_from_spec(spec);spec.loader.exec_module(portable_module)
    torch.set_num_threads(2);torch.backends.mha.set_fastpath_enabled(False)
    models=ART/'own_models';models.mkdir(exist_ok=True);manifest=[];parity=[]
    selection=read(TE/'te_100000/selection_locked.json')
    for stage in ['te_100000','te_24000']:
        chosen=read(TE/stage/'selection_locked.json')
        assert sha(TE/stage/'train_validation.json')==sha(TE/'te_100000/train_validation.json')
        if stage=='te_24000':assert [r['checkpoint_sha256'] for r in chosen['selected_models']]==read(TE/'native_compare/own_search/protocol_locked.json')['own_checkpoint_sha256']
        for record in chosen['selected_models']:
            source=Path(record['best_path']);assert sha(source)==record['checkpoint_sha256'];raw=torch.load(source,map_location='cpu',weights_only=True)
            state={'backbone.'+k:v for k,v in raw['backbone'].items()};state.update({'head.'+k:v for k,v in raw['head'].items()})
            state.update(stats_mean=torch.tensor(raw['stats_mean']),stats_scale=torch.tensor(raw['stats_scale']),target_mean=torch.tensor(raw['target_mean']),target_scale=torch.tensor(raw['target_scale']))
            config=dict(backbone_config=raw['config'],context_layers=2,use_properties=False,sequence_only=False)
            model=HumanTEScalar(**config).eval();model.load_state_dict(state,strict=True)
            val=[r for r in read(TE/'te_100000/train_validation.json') if r['split']=='val'][:4];features={}
            for row in val:
                with np.load(TE/'te_100000/features'/f'{row["uid"]}.npz',allow_pickle=False) as z:features[row['uid']]={k:z[k].copy() for k in z.files}
            with torch.no_grad():pred=model(tuple(t.cpu() for t in ex.batch(val,features))).numpy()
            error=float(np.max(np.abs(pred-np.array(raw['validation_predictions'][:4]))));assert error<2e-5
            parity.append(dict(model='human_'+stage,seed=raw['seed'],max_error=error,n=4))
            out=models/f'human_{stage}_no_stats_{raw["seed"]}.pt'
            metadata=dict(spec=dict(kind='HumanTEScalar',config=config),role='validation_selected_'+stage,seed=raw['seed'],step=raw['step'],
                source_checkpoint_sha256=sha(source),validation_pearson=raw['validation_pearson'],independent_test_evaluated=False,reused_test_evaluated=(stage=='te_100000'))
            torch.save(dict(state_dict=state,metadata=metadata),out);torch.load(out,map_location='cpu',weights_only=True)
            manifest.append(dict(path=out.name,sha256=sha(out),bytes=out.stat().st_size,metadata=metadata))
    for arm in ['full','no_physics','no_graph','no_context','seq_transformer']:
        for seed in [30101,30102]:
            source=HERE/'models'/f'{arm}_{seed}'/'best.pt';raw=torch.load(source,map_location='cpu',weights_only=True)
            config=dict(arm=arm,backbone_config=dict(width=192,sequence_v2=True,structure_gate=True,anchored_head=False,physics=True))
            state=dict(raw['model']);state.update(stats_mean=torch.zeros(3),stats_scale=torch.ones(3),target_mean=torch.tensor(raw['target_mean']),target_scale=torch.tensor(raw['target_scale']))
            model=portable_module.FairTEAblation(**config).cpu().eval();model.load_state_dict(state,strict=True)
            val=[r for r in read(HERE/'train_validation.json') if r['split']=='val'][:4];features=ex.load(val)
            with torch.no_grad():
                values=[]
                for start in [0,2]:
                    group=val[start:start+2];inputs=tuple(t.cpu() for t in ex.batch(group,features))
                    values.extend(model(inputs).numpy().tolist())
            error=float(np.max(np.abs(np.array(values)-np.array(raw['validation_predictions'][:4]))));assert error<2e-5
            parity.append(dict(model=arm,seed=seed,max_error=error,n=4))
            out=models/f'fair_te_{arm}_{seed}.pt';metadata=dict(spec=dict(kind='FairTEAblation',config=config),kind='FairTEAblation',arm=arm,seed=seed,target_mean=raw['target_mean'],target_scale=raw['target_scale'],
                validation_pearson=raw['validation_pearson'],step=raw['step'],source_checkpoint_sha256=sha(source),
                reconstruction='physics_scorer.checkpoint_io.load_exported_checkpoint; output already in native TE label scale')
            torch.save(dict(state_dict=state,metadata=metadata),out);torch.load(out,map_location='cpu',weights_only=True)
            manifest.append(dict(path=out.name,sha256=sha(out),bytes=out.stat().st_size,metadata=metadata))
    assert len(manifest)==14
    save(ART/'model_export_verification.json',dict(status='passed',models=14,real_validation_sequences_per_model=4,
        maximum_error=max(r['max_error'] for r in parity),checks=parity,strict_state_loading=True,weights_only_loading=True))
    copy(ART/'model_export_verification.json',evidence/'model_export_verification.json')
    save(models/'manifest.json',dict(models=manifest,no_third_party_weights=True,no_optimizer_or_raw_training_data=True))
    verifier=DEST/'scripts/verify_fair_evidence_models.py'
    run([str(Path(sys.base_prefix)/'python.exe'),str(verifier),'--models-root',str(models),'--output',str(evidence/'portable_cpu_verification.json')])
    archive=ART/'fair-te-own-models-20261102.tar.gz'
    with tarfile.open(archive,'w:gz') as tar:tar.add(models,arcname='own_models')
    with tarfile.open(archive,'r:gz') as tar:
        assert len([m for m in tar.getmembers() if m.isfile() and m.name.endswith('.pt')])==14
        for r in manifest:
            f=tar.extractfile('own_models/'+r['path']);assert hashlib.sha256(f.read()).hexdigest()==r['sha256']
    save(ART/'model_manifest.json',dict(models=manifest,archive_sha256=sha(archive),archive_bytes=archive.stat().st_size))
    copy(ART/'model_manifest.json',evidence/'model_manifest.json')
    (ART/'SHA256SUMS').write_text(sha(archive)+'  '+archive.name+'\n',encoding='utf-8')
    fair=read(HERE/'summary.json');native=read(TE/'native_compare/summary.json')
    val=selection['validation_mean_per_seed']['no_stats']
    repeated=read(HERE/'full_length_100000/summary.json')
    lines=['# 人源TE公平基线、三路消融与原生优化对照','',
        f'100000步训练已完成（每种子累计100000步，验证选最佳）；验证两种子Pearson均值为{val:.6f}。该模型未新增独立测试；另做903条历史测试复测，和短CDS从头训练表分列。','',
        '|100000步模型及冻结外部基线：903条历史复测|Pearson|95%条件区间|','|---|---:|---|']
    for arm,r in repeated['metrics'].items():lines.append(f"|{arm}|{r['pearson']:.4f}|[{r['ci95'][0]:.4f}, {r['ci95'][1]:.4f}]|")
    lines+=['','该903条测试已在此前研究迭代中打开，属于探索性复测，不是新独立确认。外部公开权重训练成员未完全核验；RiboDecode为固定RNA的TE代理。','',
        '本次公平实验：1951训练、262验证、227条历史测试子集，80–300密码子；8组×2种子，均从头训练3000更新、batch2，按同一验证Pearson选权重。所有组均报告。','',
        '|模型|TE Pearson|95%条件区间|','|---|---:|---|']
    for arm,r in fair['metrics'].items():lines.append(f"|{arm}|{fmt(r['pearson'])}|[{fmt(r['ci95'][0])}, {fmt(r['ci95'][1])}]|")
    lines+=['','227条是903条的子集，两表不是独立重复验证。未定义的Pearson保留为NA，不按0代替或删除模型。','']
    lines+=['','RiboGL是官方GCN_LSTM前向加TE标量读出；SAGE/GraphNorm采用Torch算子后端，与保存的官方PyG预测核对6例最大误差约2.4e-8，避开无关可选库导入崩溃。seq2ribo使用官方Mamba/TE头的连续输出适配和模拟均值；RiboDecode使用官方CNN固定RNA/环境条件的TE适配。这里不是三篇原论文原生任务的SOTA比较。训练样本和更新数匹配，不代表FLOPs、容量、耗时或各架构最优配方相同。','',
        '三路消融去除全部物理嵌入通路、图/结构表示通路或整个Transformer；额外给出仅序列Transformer。主标量头未加入GC、CAI、MFE或显式长度统计。当前物理输入是两维模拟聚合特征，不是原始71维的逐维消融。','',
        'RiboDecode原生优化：同3条开发CDS，每条3轮×800提案、MFE权重0.7，独立训练蛋白特异MFE代理；自有V3固定λ0.75、每条2400提案。使用原生末轮终点，不按外部指标挑轮。两边提案数相同，但计算量不同。','',
        '|方法|自有TE分数|RiboDecode分数|seq2ribo分数|MFE/nt|半衰期对数预测|','|---|---:|---:|---:|---:|---:|']
    for r in native['methods']:lines.append('|'+r['method']+'|'+'|'.join(f'{r[k]:.4f}' for k in ['own_score','ribodecode_score','seq2ribo_score','evaluation_mfe_per_nt','stability_log_score'])+'|')
    lines+=['',f"自有终点五项均值同时支配原生终点：{native['mean_five_metric_dominance']}；逐条为{native['case_dominance_count']}/3。",'',
        '原生比较固定使用24000步阶段模型，未混入后续30000或100000步权重。原生优化允许同义终止密码子变化，自有搜索固定原终止密码子；精确终止密码子约束不同，未悄悄重写原生输出。第3条原生任务因WSL挂载盘I/O失败重跑；自有搜索因内存退出用同种子缓存重放，额外成本披露在证据中。','',
        '原生日志R²混合了参数更新过程中产生的预测，不能代替最终冻结代理的验证。已另用每条32个未参加代理训练的同义序列诊断，见native_surrogate_holdout_audit.json。原生代理对超过768nt的序列使用分块MFE训练标签，最终终点评估统一为全文MFE；不能把本轮MFE优势泛化为优于充分训练的原生优化器。','',
        '这些搜索指标都是模型预测；生成序列没有实测TE或半衰期，不能计算其真实表型Pearson。MFE不等于细胞内半衰期。半衰期使用独立辅助ridge，586验证基因Pearson0.4519，输出为原目标对数尺度，不宣称小时。','',
        '测试子集已在历史实验打开，属于探索性复测。尚缺等协议的新独立数据确认和生成候选实验验证；自然基因间相关性不能替代同蛋白同义变体的实测验证。不能据单项高分宣称所有下游任务均优于外部模型。','',
        f'完整证据：[evidence/{TAG}](../evidence/{TAG})；源脚本：[historical/{TAG}](../historical/{TAG})。自己的14份精选模型将作为同名prerelease资产发布，第三方原始权重、优化器与原始训练数据不上传。']
    lines+=['','模型包含搜索使用的24000步阶段、自有100000步阶段及公平消融权重。下载并解压后可用`python scripts/verify_fair_evidence_models.py --models-root own_models --output verification.json`核验；推理入口为`physics_scorer.checkpoint_io.load_exported_checkpoint`，输出已还原到原TE标签尺度。','',
        '训练源脚本按原工作区结构保存，重跑需要相应公开数据、冻结特征和路径配置；不将其描述为无需数据的一键复现包。']
    report=DEST/'docs/FAIR_TE_NATIVE_ABLATION_20261102_ZH.md';report.write_text('\n'.join(lines)+'\n',encoding='utf-8')
    attributes=DEST/'.gitattributes';rules=attributes.read_text(encoding='utf-8')
    for rule in [f'evidence/{TAG}/** -text',f'historical/{TAG}/** -text']:
        if rule not in rules:rules+='\n'+rule+'\n'
    attributes.write_text(rules,encoding='utf-8')
    readme=DEST/'README.md';text=readme.read_text(encoding='utf-8');link='[人源TE公平基线、三路消融与原生优化对照](docs/FAIR_TE_NATIVE_ABLATION_20261102_ZH.md)'
    if link not in text:readme.write_text(link+'\n\n'+text,encoding='utf-8')
    # Verify exact copied bytes before publishing, including all negative outcomes.
    for r in sources:assert sha(DEST/r['destination'])==r['sha256']
    assert not git('diff','--cached','--name-only'),'Unexpected concurrent staging; preserve changes'
    git('diff','--check');git('add','--','.gitattributes','README.md','docs/FAIR_TE_NATIVE_ABLATION_20261102_ZH.md',
        'physics_scorer/fair_te_ablation.py','physics_scorer/checkpoint_io.py','scripts/verify_fair_evidence_models.py',str(evidence.relative_to(DEST)),str(code.relative_to(DEST)))
    staged=[p for p in git('diff','--cached','--name-only','-z').split('\0') if p]
    allowed={'.gitattributes','README.md','docs/FAIR_TE_NATIVE_ABLATION_20261102_ZH.md','physics_scorer/fair_te_ablation.py','physics_scorer/checkpoint_io.py','scripts/verify_fair_evidence_models.py'}
    assert staged and all(not p.endswith('.pt') and (p in allowed or p.startswith('evidence/'+TAG+'/') or p.startswith('historical/'+TAG+'/')) for p in staged)
    for record in sources:
        blob=subprocess.run(['git','show',':'+record['destination']],cwd=DEST,env=ENV,capture_output=True,check=True).stdout
        assert hashlib.sha256(blob).hexdigest()==record['sha256']
    git('commit','-m','Report controlled human TE baselines, branch ablations and native optimization')
    pr_body=ART/'pr_body.md';pr_body.write_text('完成同预算人源TE适配基线与三路消融，并报告有限预算RiboDecode原生优化、100000步训练及全部指标。\n\n验证：完整模型数值复现、消融梯度阻断、全部检查点重载、逐基因Pearson及配对区间独立复算、资产哈希核验。\n\n限制：历史测试复测、短CDS、TE任务适配、有限原生优化预算；无生成候选实测标签，不宣称全面领先。\n',encoding='utf-8')
    notes=ART/'release_notes.md';notes.write_text('\n'.join(lines)+'\n',encoding='utf-8')
    commit=git('rev-parse','HEAD')
    save(ART/'committed_package.json',dict(commit=commit,archive_sha256=sha(archive)))
    upload(commit)

def upload(commit):
    archive=ART/'fair-te-own-models-20261102.tar.gz';pr_body=ART/'pr_body.md';notes=ART/'release_notes.md'
    git('push','-u','origin',BRANCH)
    assert git('ls-remote','origin','refs/heads/'+BRANCH).split()[0]==commit
    save(HERE/'publication_status.json',dict(stage='branch_pushed',branch=BRANCH,commit=commit,unix=time.time()))
    prs=json.loads(gh('pr','list','--repo',REPOSITORY,'--head',BRANCH,'--state','all','--json','url'))
    pr_url=prs[0]['url'] if prs else gh('pr','create','--repo',REPOSITORY,'--draft','--base','results/human-te-scorer-20261102','--head',BRANCH,
        '--title','Human TE: controlled baselines, three-branch ablations and native optimization','--body-file',str(pr_body))
    releases=json.loads(gh('api',f'repos/{REPOSITORY}/releases?per_page=100'))
    tagrefs=[line.split() for line in git('ls-remote','--tags','origin','refs/tags/'+TAG,'refs/tags/'+TAG+'^{}').splitlines() if line.strip()]
    if tagrefs:
        actual=next((ref[0] for ref in tagrefs if ref[1].endswith('^{}')),tagrefs[0][0]);assert actual==commit,'Existing tag points elsewhere; do not overwrite it'
    release=next((r for r in releases if r['tag_name']==TAG),None)
    if release is None:gh('release','create',TAG,'--repo',REPOSITORY,'--target',commit,'--draft','--prerelease','--latest=false','--title','Human TE controlled evidence — 2026-11-02','--notes-file',str(notes))
    assets=[archive,ART/'SHA256SUMS',ART/'model_manifest.json',ART/'model_export_verification.json',HERE/'independent_audit.json']
    for asset in assets:
        current=next(r for r in json.loads(gh('api',f'repos/{REPOSITORY}/releases?per_page=100')) if r['tag_name']==TAG)
        existing=next((a for a in current['assets'] if a['name']==asset.name),None)
        if existing and existing.get('digest')=='sha256:'+sha(asset) and existing['size']==asset.stat().st_size:continue
        assert current['draft'],'Never overwrite published release assets'
        for attempt in range(3):
            save(HERE/'publication_status.json',dict(stage='uploading_release_asset',asset=asset.name,bytes=asset.stat().st_size,attempt=attempt+1,unix=time.time()))
            try:gh('release','upload',TAG,str(asset),'--repo',REPOSITORY,'--clobber');break
            except RuntimeError:
                if attempt==2:raise
                time.sleep(2)
    metadata=next(r for r in json.loads(gh('api',f'repos/{REPOSITORY}/releases?per_page=100')) if r['tag_name']==TAG)
    uploaded={a['name']:a for a in metadata['assets']}
    for asset in assets:assert uploaded[asset.name]['digest']=='sha256:'+sha(asset) and uploaded[asset.name]['size']==asset.stat().st_size
    if metadata['draft']:gh('release','edit',TAG,'--repo',REPOSITORY,'--draft=false','--prerelease','--latest=false')
    published=json.loads(gh('api',f'repos/{REPOSITORY}/releases/tags/{TAG}'));assert not published['draft'] and published['prerelease']
    receipt=dict(status='published',branch=BRANCH,commit=commit,pr_url=pr_url,release_url=f'https://github.com/{REPOSITORY}/releases/tag/{TAG}',
        own_models=14,archive_sha256=sha(archive),server_sha256_verified=True,old_branches_not_modified=True)
    failure=HERE/'publication_failure.json'
    if failure.exists():failure.rename(HERE/('publication_failure_recovered_'+str(time.time_ns())+'.json'))
    save(HERE/'publication_complete.json',receipt);save(ART/'publication_receipt.json',receipt);print(json.dumps(receipt),flush=True)
    save(HERE/'publication_status.json',dict(stage='published',receipt=receipt,unix=time.time()))
if __name__=='__main__':
    try:main()
    except BaseException as error:save(HERE/'publication_failure.json',dict(error=repr(error),unix=time.time()));raise
