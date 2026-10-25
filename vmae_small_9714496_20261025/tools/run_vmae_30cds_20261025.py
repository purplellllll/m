"""Locked 30-CDS full-native comparison, with incremental five-metric outputs."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil
import statistics
import subprocess
import sys
import tarfile
import time

BASE = Path(__file__).resolve().parents[1]
OUT = BASE.parent/'vmae_30cds_20261025'
sys.path.insert(0, str(BASE))
SEEDS = [20261020, 20261021, 20261022]
GROUPS = ('short', 'medium', 'long')


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda:handle.read(2**20), b''):
            digest.update(block)
    return digest.hexdigest()


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    temporary.replace(path)


def length_bin(record):
    n = record['n_codons']-1
    return 'short' if n<=300 else 'medium' if n<700 else 'long'


def select_panel(records, per_length=10):
    selected, used = [], set()
    for group in reversed(GROUPS):
        eligible = sorted((r for r in records if length_bin(r)==group), key=lambda r:
            hashlib.sha256(('test30-20261025:'+r['sequence_id']).encode()).hexdigest())
        chosen = []
        for record in eligible:
            if record['v2_group'] not in used:
                chosen.append(record)
                used.add(record['v2_group'])
            if len(chosen)==per_length:
                break
        assert len(chosen)==per_length, ('Insufficient independent test groups', group)
        selected.extend(chosen)
    assert len(selected)==len(used)==3*per_length
    return sorted(selected, key=lambda r:(r['n_codons'], r['sequence_id']))


def prepare():
    from latent_decoder.v2_train import corpus
    from latent_decoder.train import codon_frequency
    from latent_decoder.tokenizer import AA, CODONS
    data_path = BASE.parent/'vmae_adjustment_20261024/data'
    data = corpus(data_path, 0)
    records = select_panel(data['test'])
    assert records==select_panel(list(reversed(data['test'])))
    groups = {r['v2_group'] for r in records}
    assert not groups & {r['v2_group'] for r in data['train']+data['val']}
    checkpoint = BASE.parent/'vmae_small_size_only_20261025/results/fold_0/uniform/best.pt'
    assert sha(checkpoint)=='d79605bcee93be8b39e5e773a15be1aacd762230a25b8f9e0630200197614547'
    calibration_ids = json.loads((BASE.parent/'vmae_transfer_20261024/cases.json').read_text())['calibration_ids']
    counts = codon_frequency(data['train'])
    weights = {c:float(counts[i]/max(counts[j] for j,a in enumerate(AA) if a==AA[i])) for i,c in enumerate(CODONS)}
    OUT.mkdir(exist_ok=True)
    assert not (OUT/'plan.json').exists(), 'Panel is immutable after preparation'
    save(OUT/'cai_weights.json', weights)
    save(OUT/'panel.json', records)
    save(OUT/'plan.json', dict(records=records, counts={g:sum(length_bin(r)==g for r in records) for g in GROUPS},
        split='Previously unused fixed held-out test subset; 30 distinct homology groups',
        sampling_salt='test30-20261025:', checkpoint=str(checkpoint), checkpoint_sha256=sha(checkpoint),
        corpus_sha256=sha(data_path/'corpus.jsonl.gz'), folds_sha256=sha(data_path/'folds.json'),
        calibration_ids=calibration_ids, source_sha256=sha(Path(__file__)), native_wrapper_sha256=sha(BASE/'tools/native_full_gpu_20261025.py'),
        dependencies={n:sha(BASE/'latent_decoder'/n) for n in ('model.py','latent_search_probe_cpu.py','latent_information_probe.py','latent_optimization.py','analyze_random_validation.py','v2_train.py')},
        search=dict(steps=80, lr=.05, radius=8., temperature=1., initialization='Mask every mutable codon',
                    edit_budget=None, js_penalty=0., js_hard_limit=None),
        native=dict(full_default=True, outer_rounds=20, proposals_per_round=40000, batch_size=20,
                    mfe_weight=.7, num_seqs_train_gen=100000, num_seqs_select_mfe=20000, num_seqs_train_mfe=500000,
                    proposals_per_cds=800000, native_target_negative_queries_per_cds=1600000),
        comparison='User-requested full native budget versus frozen 80-step VMAE pipeline; unequal online budgets, log costs separately',
        seq2ribo=dict(cell_line='hek293', task='te', use_utr=False, seeds=SEEDS, n_stasep_runs=8, return_scaled_te=False),
        gc_mfe_reference='Actual last-round native returned sequence, never archive maximum',
        gc_tolerance_percentage_points=2., mfe_preferred_tolerance_kcal_mol=10., mfe_is_hard_filter=False,
        current_tiny_training_unchanged=True, checkpoint_selection_or_tuning_on_this_panel=False))
    for r in records:
        save(OUT/'tasks'/(r['sequence_id']+'.json'), dict(r, n_aa=r['n_codons']-1, seed=20261025))
    save(OUT/'status.json', dict(state='prepared', cds=30, expected_metric_rows=90, goal_complete=False))
    print(json.dumps(dict(event='prepared', counts={g:sum(length_bin(r)==g for r in records) for g in GROUPS},
                         lengths=[r['n_codons']-1 for r in records])), flush=True)


def environment():
    sys.path.insert(0, str(BASE/'.runtime/mamba_cuda_v2_2_4/packages'))
    repository = BASE/'tmp/seq2ribo_source_20261020/seq2ribo-main'
    sys.path.insert(0, str(repository))
    import torch
    import RNA
    import mamba_ssm
    from seq2ribo import Seq2Ribo
    from latent_decoder.dual_space_ablation import CountedScorer
    from latent_decoder.latent_optimization import TranslationObjective, state_hash
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.manual_seed(20261024)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    assert torch.cuda.is_available() and RNA.__version__=='2.7.2' and mamba_ssm.__version__=='2.2.4'
    lock_path = BASE/'docs/seq2ribo_evaluator_lock_20261020.json'
    lock = json.loads(lock_path.read_text())
    for name, digest in {**lock['source_sha256'], **lock['weights_sha256']}.items():
        assert sha(repository/name)==digest
    parity = json.loads((BASE/'seq2ribo_cuda_parity/completion_receipt.json').read_text())
    assert parity['cuda_parity_verified']
    assert Path(mamba_ssm.__file__).resolve().is_relative_to((BASE/'.runtime/mamba_cuda_v2_2_4/packages').resolve())
    scorer = CountedScorer(TranslationObjective(BASE/'score_model')).cuda().eval().requires_grad_(False)
    canonical = BASE.parent/'vmae_transfer_20261024/canonical_condition.pt'
    subset = json.loads((BASE/'scorer_subset_receipt.json').read_text())
    assert sha(canonical)==subset['canonical_condition_sha256']
    with torch.no_grad():
        scorer.scorer.condition.copy_(torch.load(canonical, map_location='cpu', weights_only=True).cuda())
    assert state_hash(scorer.scorer)==subset['model_state_hash']
    predictor = Seq2Ribo(cell_line=lock['cell_line'], device='cuda', weights_dir=str(repository/'weights'), cache_dir=str(OUT/'geometry'))
    teacher = predictor._load_model('te', use_utr=False).eval().requires_grad_(False)
    assert state_hash(teacher)==parity['teacher_state_hash']
    return scorer, predictor, teacher, lock


def measure(record, method, dna, scorer, predictor, lock, generation_seconds, queries):
    import torch
    from torch.nn import functional as F
    from latent_decoder.analyze_random_validation import sequence_metrics, fold_candidate
    from latent_decoder.model import collate_records
    from latent_decoder.tokenizer import AA, encode_cds
    start = time.monotonic()
    before, after = encode_cds(record['cds_dna']), encode_cds(dna)
    assert [AA[i] for i in before]==[AA[i] for i in after]
    assert dna[:3]==record['cds_dna'][:3]
    if method=='small':
        assert dna[-3:]==record['cds_dna'][-3:]
    weights = json.loads((OUT/'cai_weights.json').read_text())
    properties = dict(**sequence_metrics(dna, weights), **fold_candidate(dna))
    assert properties['viennarna_version']=='2.7.2' and properties['temperature_celsius']==37 and properties['dangles']==2
    tokens = collate_records([dict(cds_dna=dna)]).cuda()
    with torch.no_grad():
        hard = float(scorer(F.one_hot(tokens,64).float(), tokens))
    te = {str(seed):float(predictor.predict(dna.replace('T','U'), task='te', n_stasep_runs=8,
        init_p=lock['init_p'], seed=seed, return_scaled_te=False, batch_size=8)[0]) for seed in SEEDS}
    row = dict(sequence_id=record['sequence_id'], group=length_bin(record), n_aa=record['n_codons']-1,
        method=method, ribodecode_score=hard, seq2ribo_score=statistics.mean(te.values()),
        seq2ribo_seed_20261020=te['20261020'], seq2ribo_seed_20261021=te['20261021'], seq2ribo_seed_20261022=te['20261022'],
        seq2ribo_seed_sd=statistics.pstdev(te.values()), gc_percent=100*properties['gc_fraction'],
        cai=properties['train_reference_cai'], mfe_kcal_mol=properties['mfe'], mfe_per_nt=properties['mfe_per_nt'],
        changed_codons=sum(x!=y for x,y in zip(before,after)), changed_fraction=sum(x!=y for x,y in zip(before,after))/len(before),
        generation_seconds=generation_seconds, evaluation_seconds=time.monotonic()-start,
        generation_scorer_queries=queries, native_full_default=(method=='native'),
        protein_preserved=True, start_preserved=True, stop_preserved=dna[-3:]==record['cds_dna'][-3:],
        cds_dna=dna, sequence_sha256=hashlib.sha256(dna.encode()).hexdigest())
    assert all(torch.isfinite(torch.tensor(row[k])) for k in ('ribodecode_score','seq2ribo_score','gc_percent','cai','mfe_kcal_mol'))
    save(OUT/'metrics'/record['sequence_id']/(method+'.json'), row)
    refresh()
    return row


def refresh():
    import fcntl
    with (OUT/'summary.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        rows = [json.loads(p.read_text()) for p in sorted((OUT/'metrics').glob('*/*.json'))]
        if not rows:
            return
        with (OUT/'metrics.csv.tmp').open('w', encoding='utf-8-sig', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader(); writer.writerows(rows)
        (OUT/'metrics.csv.tmp').replace(OUT/'metrics.csv')
        lookup = {(r['sequence_id'],r['method']):r for r in rows}
        comparisons=[]
        for identity in sorted({r['sequence_id'] for r in rows}):
            if (identity,'small') not in lookup or (identity,'native') not in lookup:
                continue
            ours,native=lookup[identity,'small'],lookup[identity,'native']
            delta={k:ours[k]-native[k] for k in ('ribodecode_score','seq2ribo_score','gc_percent','cai','mfe_kcal_mol')}
            all_te=all(ours['seq2ribo_seed_'+str(s)]-native['seq2ribo_seed_'+str(s)]>1e-5*(1+abs(native['seq2ribo_seed_'+str(s)])) for s in SEEDS)
            preferred=delta['ribodecode_score']>1e-6 and all_te and delta['cai']>1e-8 and abs(delta['gc_percent'])<=2 and abs(delta['mfe_kcal_mol'])<=10
            comparisons.append(dict(sequence_id=identity, group=ours['group'], n_aa=ours['n_aa'],
                **{'delta_'+k:v for k,v in delta.items()}, seq2ribo_positive_all_seeds=all_te, joint_preferred=preferred))
        save(OUT/'paired_comparison.json', comparisons)
        if comparisons:
            with (OUT/'paired_comparison.csv.tmp').open('w', encoding='utf-8-sig', newline='') as stream:
                writer=csv.DictWriter(stream,fieldnames=list(comparisons[0]));writer.writeheader();writer.writerows(comparisons)
            (OUT/'paired_comparison.csv.tmp').replace(OUT/'paired_comparison.csv')
        summary=dict(metric_rows=len(rows), expected_rows=90, completed_full_native_cases=sum(r['method']=='native' for r in rows),
            completed_small_cases=sum(r['method']=='small' for r in rows), paired_cases=len(comparisons),
            joint_preferred=sum(r['joint_preferred'] for r in comparisons), complete=len(rows)==90 and len(comparisons)==30,
            source_weights_unchanged=True, native_full_default=True, equal_online_budgets=False, goal_complete=False)
        save(OUT/'summary.json', summary)
        stages={name:json.loads((OUT/(name+'_status.json')).read_text()) if (OUT/(name+'_status.json')).exists() else {'state':'pending'} for name in ('ours','native')}
        state='failed' if any(v['state']=='failed' for v in stages.values()) else 'completed' if summary['complete'] and all(v['state']=='completed' for v in stages.values()) else 'running'
        save(OUT/'status.json',dict(state=state,**summary,stages=stages))
        text=''.join('>'+r['sequence_id']+'|'+r['method']+'|aa='+str(r['n_aa'])+'\n'+r['cds_dna']+'\n' for r in rows)
        (OUT/'candidates.fasta.tmp').write_text(text)
        (OUT/'candidates.fasta.tmp').replace(OUT/'candidates.fasta')


def ours():
    import torch
    from latent_decoder.latent_information_probe import load_model
    from latent_decoder.latent_collapse_cpu import Pool
    from latent_decoder.latent_optimization import state_hash
    from latent_decoder.latent_search_probe_cpu import search
    from latent_decoder.model import collate_records, mutable_positions, MASK, PAD
    from latent_decoder.v2_train import corpus
    plan=json.loads((OUT/'plan.json').read_text())
    assert sha(plan['checkpoint'])==plan['checkpoint_sha256']
    scorer,predictor,teacher,lock=environment()
    model=load_model(Path(plan['checkpoint']),'cuda')
    hashes=[state_hash(m) for m in (model,scorer,teacher)]
    data=corpus(BASE.parent/'vmae_adjustment_20261024/data',0)
    calibration=[r for r in data['train'] if r['sequence_id'] in plan['calibration_ids']]
    assert len(calibration)==12
    pool=Pool(256,covariance=True)
    with torch.no_grad():
        for r in calibration:
            tokens=collate_records([r]).cuda(); mu,_=model.encode(tokens,tokens)
            pool.add(mu.cpu(),tokens.masked_fill(~mutable_positions(tokens),PAD).cpu())
    weak=torch.tensor(pool.summary()['within_aa']['variance_by_dim'])<=.01
    save(OUT/'train_geometry.json',pool.summary())
    state=dict(state='running',completed=0,total=30)
    try:
        for record in plan['records']:
            state.update(sequence_id=record['sequence_id'],group=length_bin(record))
            save(OUT/'ours_status.json',state)
            destination=OUT/'search'/record['sequence_id'];destination.mkdir(parents=True,exist_ok=False)
            tokens=collate_records([record]); inputs=tokens.masked_fill(mutable_positions(tokens),MASK)
            result=search(model,scorer,record,80,weak,destination,arms=('latent_radius8',),encoder_tokens=inputs)[0]
            save(destination/'locked_winner.json',dict(sequence_id=record['sequence_id'],best=result['best'],selection='RiboDecode hard score only; locked before any seq2ribo measurement'))
            measure(record,'original',record['cds_dna'],scorer,predictor,lock,0.,0)
            measured=measure(record,'small',result['best']['cds_dna'],scorer,predictor,lock,result['seconds'],result['search_scorer_queries'])
            assert abs(measured['ribodecode_score']-result['best']['hard_score'])<2e-5
            state['completed']+=1
            save(OUT/'ours_status.json',state)
        assert hashes==[state_hash(m) for m in (model,scorer,teacher)]
        state.update(state='completed',source_weights_unchanged=True)
    except BaseException as error:
        state.update(state='failed',error=repr(error));raise
    finally:
        save(OUT/'ours_status.json',state)
        refresh()


def native_command(task, output, mode, all_history=False):
    command=[sys.executable,'-u',str(BASE/'tools/native_full_gpu_20261025.py'),
        '--package',str(OUT/'external/RiboDecode'),'--task',str(task),'--output',str(output),'--mode',mode]
    if all_history:
        command.append('--retain-all-history')
    with output.with_suffix('.log').open('xb') as log:
        child=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,cwd=BASE)
        save(OUT/'active_native_process.json',dict(pid=child.pid,task=str(task),output=str(output),mode=mode))
        assert child.wait()==0, ('Inspect native log; no automatic retry', str(output))
    receipt=json.loads((output/'completion_receipt.json').read_text())
    assert receipt['state']=='completed' and receipt['scorer_weights_unchanged']
    return receipt


def parity(first, second):
    import torch
    a,b=OUT/'preflight'/first,OUT/'preflight'/second
    x,y=[json.loads((p/'completion_receipt.json').read_text()) for p in (a,b)]
    assert x['all_round_selected']==y['all_round_selected']
    assert x['native_target_negative_queries']==y['native_target_negative_queries']
    maxerror=0.
    for i in range(x['rounds']):
        suffix='results_natural/HEK293T_optim_mfe_dist-optim/'+str(i)+'/samples/optim_results.txt'
        left,right=(a/suffix).read_text().splitlines(),(b/suffix).read_text().splitlines()
        assert len(left)==len(right)
        for l,r in zip(left,right):
            l,r=l.split('\t'),r.split('\t');assert l[0]==r[0]
            assert all(abs(float(u)-float(v))<=1e-5*(1+abs(float(u))) for u,v in zip(l[1:],r[1:]))
        states=[torch.load(p/('round_%02d_state.pt'%(i+1)),map_location='cpu',weights_only=False) for p in (a,b)]
        for module in ('z','mfe'):
            for key,value in states[0][module].items():
                err=float((value.double()-states[1][module][key].double()).abs().max());maxerror=max(maxerror,err)
                assert torch.allclose(value,states[1][module][key],atol=1e-5,rtol=1e-5)
    save(OUT/'preflight'/(first+'_parity.json'),dict(passed=True,all_candidate_dna_equal=True,max_tensor_error=maxerror))


def native_queue():
    plan=json.loads((OUT/'plan.json').read_text())
    (OUT/'preflight').mkdir(exist_ok=False)
    (OUT/'native').mkdir(exist_ok=False)
    state=dict(state='preflight',completed=0,total=30)
    save(OUT/'native_status.json',state)
    try:
        small=plan['records'][0]; large=plan['records'][-1]
        task=lambda r:OUT/'tasks'/(r['sequence_id']+'.json')
        for name,mode,retain in [('parity_tail','parity',False),('parity_all','parity',True),
                                 ('mixed_tail','preflight',False),('mixed_all','preflight',True)]:
            native_command(task(small),OUT/'preflight'/name,mode,retain)
        parity('parity_tail','parity_all');parity('mixed_tail','mixed_all')
        native_command(task(large),OUT/'preflight/long_preflight','preflight')
        save(OUT/'preflight/complete.json',dict(passed=True,source_algorithm_unchanged=True,short_both_history_modes_verified=True,
            longest_sequence_gpu_pipeline_verified=True,full_native_default_settings_not_reduced=True))
        for record in plan['records']:
            state.update(state='full_native_running',sequence_id=record['sequence_id'],n_aa=record['n_codons']-1)
            save(OUT/'native_status.json',state)
            assert shutil.disk_usage(OUT).free>12*2**30, 'Preserve existing results; insufficient space for next full native workspace'
            output=OUT/'native'/record['sequence_id']
            receipt=native_command(task(record),output,'full')
            assert receipt['full_default'] and receipt['rounds']==20 and receipt['native_target_negative_queries']==1600000
            save(OUT/'native_receipts'/(record['sequence_id']+'.json'),receipt)
            subprocess.run([sys.executable,'-u',str(Path(__file__)),'--stage','assess-native','--identity',record['sequence_id']],check=True,cwd=BASE)
            archive_native(output)
            state['completed']+=1
            save(OUT/'native_status.json',state)
        state.update(state='completed',goal_complete=False)
    except BaseException as error:
        state.update(state='failed',error=repr(error));raise
    finally:
        save(OUT/'native_status.json',state)
        refresh()


def archive_native(output):
    # Completed raw workspaces are recoverable from checked archives; keep shared weights once.
    assert output.resolve().parent==(OUT/'native').resolve() and output.name.startswith('cds_')
    assert json.loads((output/'completion_receipt.json').read_text())['full_default']
    archive=OUT/'native_archives'/(output.name+'.tar.gz');archive.parent.mkdir(exist_ok=True)
    assert not archive.exists()
    entries={}
    with tarfile.open(archive.with_suffix('.partial'),'w:gz',compresslevel=1) as bundle:
        for p in sorted(output.rglob('*')):
            if p.is_file() and p.relative_to(output).as_posix()!='score_model/best_model.p' and '__pycache__' not in p.parts:
                name=p.relative_to(output).as_posix();entries[name]=sha(p);bundle.add(p,arcname=name)
    archive.with_suffix('.partial').replace(archive)
    with tarfile.open(archive) as bundle:
        for member in bundle:
            stream=bundle.extractfile(member)
            digest=hashlib.sha256()
            for block in iter(lambda:stream.read(2**20),b''):digest.update(block)
            assert digest.hexdigest()==entries[member.name]
    save(archive.with_suffix('.receipt.json'),dict(bytes=archive.stat().st_size,sha256=sha(archive),members=entries,
        source_weights_location=str(OUT/'external/RiboDecode/score_model/best_model.p'),all_members_verified=True))
    # Only the completed, exact verified experiment workspace is removed; archive retains its contents.
    assert output.resolve()==(OUT/'native'/output.name).resolve() and not output.is_symlink()
    shutil.rmtree(output)


def assess_native(identity):
    from latent_decoder.latent_optimization import state_hash
    plan=json.loads((OUT/'plan.json').read_text())
    record=next(r for r in plan['records'] if r['sequence_id']==identity)
    receipt=json.loads((OUT/'native_receipts'/(identity+'.json')).read_text())
    scorer,predictor,teacher,lock=environment(); hashes=[state_hash(m) for m in (scorer,teacher)]
    measure(record,'native',receipt['native_returned_cds'],scorer,predictor,lock,
            receipt['elapsed_seconds'],receipt['native_target_negative_queries'])
    assert hashes==[state_hash(m) for m in (scorer,teacher)]


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage',choices=('prepare','ours','native','assess-native'),required=True)
    parser.add_argument('--identity')
    args=parser.parse_args()
    if args.stage!='prepare':
        pinned=json.loads((OUT/'plan.json').read_text())
        assert sha(Path(__file__))==pinned['source_sha256']
        assert sha(BASE/'tools/native_full_gpu_20261025.py')==pinned['native_wrapper_sha256']
        assert all(sha(BASE/'latent_decoder'/n)==digest for n,digest in pinned['dependencies'].items())
    if args.stage=='prepare':prepare()
    elif args.stage=='ours':ours()
    elif args.stage=='native':native_queue()
    else:assess_native(args.identity)
