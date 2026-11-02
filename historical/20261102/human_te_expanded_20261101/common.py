"""Locked inputs and resumable feature preparation for expanded native human TE."""
from pathlib import Path
import hashlib
import importlib.util
import json
import time

ROOT=Path(__file__).resolve().parent
WORK=ROOT.parent.parent
REPO=WORK/'repos/riboseq_scorer_20261031'
PILOT=WORK/'runs/human_te_pilot_20261101'
SOURCE=WORK/'runs/graph_polish_20261031/gpu_result/completed_snapshot/graph_polish_20261031/source'
PARAM=SOURCE.parent/'data/physics_parameters_train_only.json'
SEEDS=[30101,30102]
ARMS=['no_stats','properties']
MAX_STEPS=6000
EVERY=250

def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def module(name,p):
    spec=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
def save(p,data):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix('.tmp')
    tmp.write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    for attempt in range(30):
        try:tmp.replace(p);return
        except PermissionError:
            if attempt==29:raise
            time.sleep(.1)
def verify_sources():
    lock=read(ROOT/'protocol_locked.json')
    for name,digest in lock['source_sha256'].items():assert sha(Path(name))==digest,name
    return lock

def prepare():
    if (ROOT/'protocol_locked.json').exists():
        locked=verify_sources()
        for name,key in [('train_validation.json','train_validation_sha256'),('test_inputs.json','test_inputs_sha256'),('sealed_test_targets.json','test_targets_sha256')]:assert sha(ROOT/name)==locked[key]
        return
    import gzip
    import sys
    sys.path.insert(0,str(WORK))
    from latent_decoder.prepare_data import GENETIC_CODE
    cohort=WORK/'runs/scorer_path_alignment_20261031/native_te_cohort.json'
    gene={}
    with gzip.open(WORK/'runs/scorer_path_alignment_20261031/sources/gencode.v48.pc_transcripts.fa.gz','rt',encoding='utf-8') as handle:
        for line in handle:
            if line.startswith('>'):
                fields=line[1:].split('|');gene[fields[0]]=fields[1]
    used_genes=set();used_proteins=set();selected=[];counts={}
    for split in ['train','val','test']:
        eligible=[]
        for r in read(cohort)['records']:
            if r['cell']!='hek293' or r['split']!=split:continue
            dna=r['dna'];n=len(dna)//3
            if not (80<=n<=1500 and len(dna)%3==0 and dna.startswith('ATG') and dna[-3:] in {'TAA','TAG','TGA'}):continue
            if any(dna[j:j+3] in {'TAA','TAG','TGA'} for j in range(0,len(dna)-3,3)):continue
            protein=''.join(GENETIC_CODE[dna[j:j+3]] for j in range(0,len(dna)-3,3))
            gid=gene.get(r['tx_id']);phash=hashlib.sha256(protein.encode()).hexdigest()
            if not gid or gid in used_genes or phash in used_proteins:continue
            uid=hashlib.sha256(dna.replace('T','U').encode()).hexdigest();assert uid==r['cds_sha256']
            eligible.append(dict(uid=uid,dna_sha256=hashlib.sha256(dna.encode()).hexdigest(),dna=dna,
                split=split,transcript=r['tx_id'],gene=gid,protein_sha256=phash,target=r['target'],n_codons=n))
        counts[split]=len(eligible);selected+=eligible
        used_genes|={r['gene'] for r in eligible};used_proteins|={r['protein_sha256'] for r in eligible}
    for key in ['uid','gene','protein_sha256']:
        sets={s:{r[key] for r in selected if r['split']==s} for s in counts}
        assert not (sets['train']&sets['val'] or sets['train']&sets['test'] or sets['val']&sets['test'])
    assert counts==dict(train=7373,val=922,test=903)
    trainval=[r for r in selected if r['split']!='test'];test=[r for r in selected if r['split']=='test']
    save(ROOT/'train_validation.json',trainval)
    save(ROOT/'test_inputs.json',[{k:v for k,v in r.items() if k!='target'} for r in test])
    save(ROOT/'sealed_test_targets.json',{r['uid']:r['target'] for r in test})
    protocol=dict(task='human_HEK293_native_CDS_only_TE',counts=counts,length_codons=[80,1500],seeds=SEEDS,arms=ARMS,
        max_steps=MAX_STEPS,validation_every=EVERY,early_stop_stale_checks=8,lr_encoder=1e-5,lr_head=3e-4,warmup=200,
        optimizer='AdamW',weight_decay=.01,clip=1.,max_batch=8,max_padded_codons=4096,max_attention_cells=4500000,
        target='Provided native TE labels, train mean/std normalization only',explicit_length_feature=False,
        properties=['GC_fraction','MFE_per_nt','CAI'],selection='Mean per-seed validation Pearson; lexical tie; freeze all checkpoints before test prediction',
        test_access='Separate target file; training reads only train_validation.json. Test inference only after selection_locked.json',
        independence='Native splits retained, gene/exact-CDS/exact-protein disjoint; no near-homology clustering',
        physics='Frozen mouse-proxy train-fitted parameters,4 simulations; not claimed human kinetic calibration',
        native_baselines='Local actual predictions on identical903 targets; no paper coefficients. seq2ribo historical-weight training membership not certified. RiboDecode fixed-RNA score is TE proxy.',
        canonical_uid='SHA256 of RNA; seed human_te:<RNA_SHA>. DNA SHA separate; old feature dna_sha256 field is legacy RNA UID',
        source_sha256={str(p):sha(p) for p in list(ROOT.glob('*.py'))+[cohort,PARAM,
            REPO/'physics_scorer/reliable_graph_context.py',REPO/'physics_scorer/scalar_downstream.py',
            REPO/'historical/drivers/run_aligned_profile_ablation_20261030.py',PILOT/'run_pilot.py',
            WORK/'runs/vmae_own_full30_cpu_20261031/evaluate_cpu.py',WORK/'runs/vmae_tiny_30cds_20261025/cai_weights.json']+
            [WORK/f'runs/scorer_compare_search_20261031/search_outputs/scorer_backbones/gated_graph_{s}.pt' for s in SEEDS]+
            list((SOURCE/'evaluator/seq2ribo').glob('*.py'))},
        train_validation_sha256=sha(ROOT/'train_validation.json'),test_inputs_sha256=sha(ROOT/'test_inputs.json'),test_targets_sha256=sha(ROOT/'sealed_test_targets.json'))
    if (ROOT/'protocol_locked.json').exists():assert read(ROOT/'protocol_locked.json')==protocol
    else:save(ROOT/'protocol_locked.json',protocol)
    save(ROOT/'split_audit.json',dict(status='passed',counts=counts,gene_disjoint=True,exact_CDS_disjoint=True,exact_protein_disjoint=True,
        unique_genes={s:len({r['gene'] for r in selected if r['split']==s}) for s in counts},test_predictions_opened=False))

_helper=None
def worker_init():
    global _helper
    import torch
    torch.set_num_threads(1)
    _helper=module('expanded_human_features',REPO/'historical/drivers/run_aligned_profile_ablation_20261030.py')
def feature_worker(row):
    import numpy as np
    path=ROOT/'features'/f'{row["uid"]}.npz'
    f=_helper.structure_job((str(SOURCE),row['dna']))
    seed=int(hashlib.sha256(('human_te:'+row['uid']).encode()).hexdigest()[:8],16)
    f.update(_helper.simulation_job((str(SOURCE),row['dna'],f,read(PARAM),seed)))
    f.update(dna_sha256=row['uid'],canonical_rna_sha256=row['uid'],sequence_dna_sha256=row['dna_sha256'])
    tmp=path.with_suffix('.tmp')
    with tmp.open('wb') as handle:np.savez_compressed(handle,**f)
    tmp.replace(path)
    return row['uid']

def ensure_features(rows,stage):
    import shutil
    from concurrent.futures import ProcessPoolExecutor,wait,FIRST_COMPLETED
    verify_sources();folder=ROOT/'features';folder.mkdir(exist_ok=True)
    old=read(PILOT/'protocol_locked.json');assert old['physics_sha256']==sha(PARAM) and old['source_sha256']==sha(PILOT/'run_pilot.py')
    pending_rows=[];completed=0
    for row in rows:
        path=folder/f'{row["uid"]}.npz';old_path=PILOT/'features'/path.name
        if not path.exists() and old_path.exists():shutil.copy2(old_path,path)
        if path.exists():completed+=1
        else:pending_rows.append(row)
    save(ROOT/f'{stage}_status.json',dict(stage=stage,completed=completed,total=len(rows),workers=2))
    iterator=iter(pending_rows);pool=ProcessPoolExecutor(max_workers=2,initializer=worker_init);pending=set()
    def submit():
        row=next(iterator,None)
        if row is not None:pending.add(pool.submit(feature_worker,row))
    for _ in range(4):submit()
    try:
        while pending:
            done,_=wait(pending,return_when=FIRST_COMPLETED)
            for future in done:
                pending.remove(future);future.result();completed+=1;submit()
            save(ROOT/f'{stage}_status.json',dict(stage=stage,completed=completed,total=len(rows),workers=2))
            if completed%25==0:print(stage,completed,len(rows),flush=True)
    finally:pool.shutdown(wait=True,cancel_futures=True)
    save(ROOT/f'{stage}_complete.json',dict(status='completed',count=completed,total=len(rows)))
