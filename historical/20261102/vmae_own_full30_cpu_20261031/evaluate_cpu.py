"""CPU-only same-protocol evaluation of original, historical V3 and own-guided V3."""
import argparse
import contextlib
import csv
import gc
import hashlib
import importlib.util
import io
import json
import math
from pathlib import Path
import sys
import time

OUT=Path(__file__).resolve().parent
WORK=OUT.parent.parent
SRC=WORK/'tmp/seq2ribo_source_20261020/seq2ribo-main'
REF=WORK/'deliverables/public_riboseq_validation_20261028'
SEEDS=[20261020,20261021,20261022]


def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def save(path,value):
    path=Path(path);path.parent.mkdir(exist_ok=True,parents=True)
    temporary=path.with_suffix('.tmp');temporary.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8');temporary.replace(path)


def source_protocol():
    lock=read(WORK/'docs/seq2ribo_evaluator_lock_20261020.json')
    for name,digest in {**lock['source_sha256'],**lock['weights_sha256']}.items():assert sha(SRC/name)==digest,name
    return dict(device='cpu',torch_threads=1,seq2ribo=dict(task='te',cell='hek293',use_utr=False,
        runs_per_seed=8,seeds=SEEDS,init_p=.01,batch_size=8,source_lock=lock,
        backend='Official Mamba2.2.4 class and selective_scan_ref, loaded by verified AST; no CUDA extensions'),
        ribodecode=dict(checkpoint_sha256=sha(REF/'ribodecode_inference_only.pt'),
            inference_source_sha256=sha(REF/'source/inference.py'),config_sha256=sha(REF/'source/model_config.json'),
            canonical_condition_sha256=sha(WORK/'runs/vmae_adjustment_gpu_20261024/transfer_assets/canonical_condition.pt'),
            rna_log_input=4.5,stop_excluded=True,padded_nt=4500),
        properties=dict(viennarna='2.7.2',temperature=37,dangles=2,
            cai_weights_sha256=sha(WORK/'runs/vmae_tiny_30cds_20261025/cai_weights.json')),
        search_fold_version='2.7.0 for own-scorer training-input compatibility; evaluation properties use2.7.2 uniformly',
        historical_reference='Original and V3_2.5M_Hard from locked30 panel; re-evaluated by same local CPU pipeline',
        generated_sequences_have_no_measured_expression=True,own_scorer_outside_mRFP_domain=True)


def setup():
    sys.path.insert(0,str(WORK));sys.path.insert(0,str(WORK/'.runtime/viennarna_2_7_2'))
    import torch
    import numpy as np
    import RNA
    assert RNA.__version__=='2.7.2'
    torch.set_num_threads(1)
    return torch,np,RNA


def ribodecode_model(torch):
    spec=importlib.util.spec_from_file_location('vmae_cpu_ribodecode',REF/'source/inference.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    model=module.CNN_Encoder_Model(read(REF/'source/model_config.json'),module.InferenceModel.input_shape)
    payload=torch.load(REF/'ribodecode_inference_only.pt',map_location='cpu',weights_only=True)
    model.load_state_dict(payload['model_state_dict'],strict=True);model.eval().requires_grad_(False)
    condition=torch.load(WORK/'runs/vmae_adjustment_gpu_20261024/transfer_assets/canonical_condition.pt',map_location='cpu',weights_only=True).reshape(1,-1)
    assert condition.shape==(1,10552)
    return model,condition


def hard_score(torch,model,condition,dna):
    ids=torch.tensor([dict(A=0,C=1,G=2,T=3)[c] for c in dna[:-3]],dtype=torch.long)
    assert len(ids)<=4500
    x=torch.nn.functional.one_hot(ids,4).float().T[None]
    x=torch.nn.functional.pad(x,(0,4500-x.shape[-1]))
    with torch.no_grad():value=float(model(x,condition,torch.tensor([[4.5]])).flatten()[0])
    assert math.isfinite(value);return value


def seq2_model():
    from latent_decoder.seq2ribo_cpu_reference import install
    backend=install();sys.path.insert(0,str(SRC))
    from seq2ribo import Seq2Ribo
    predictor=Seq2Ribo(cell_line='hek293',device='cpu',weights_dir=str(SRC/'weights'),cache_dir=str(OUT/'evaluation_geometry_2_7_2'))
    return predictor,backend.reference_manifest


def seq2_score(predictor,dna):
    values=[]
    for seed in SEEDS:
        with contextlib.redirect_stdout(io.StringIO()):
            value=float(predictor.predict([dna],task='te',use_utr=False,n_stasep_runs=8,init_p=.01,
                seed=seed,batch_size=8,return_scaled_te=False)[0])
        assert math.isfinite(value);values.append(value)
    return dict(seq2ribo_score=sum(values)/3,seq2ribo_by_seed=dict(zip(map(str,SEEDS),values)))


def first_case():
    panel=read(WORK/'runs/vmae_tiny_30cds_20261025/panel.json')
    manifest=read(WORK/'runs/seq2ribo_multicell_comparison_20261027/candidate_manifest.json')
    row=min(panel,key=lambda r:r['n_codons'])
    return row,next(v for v in manifest.values() if v['sequence_id']==row['sequence_id'] and v['method']=='original')


def preflight():
    protocol=source_protocol();torch,np,RNA=setup();row,cached=first_case();started=time.monotonic()
    model,condition=ribodecode_model(torch);hard=hard_score(torch,model,condition,row['cds_dna'])
    hard_seconds=time.monotonic()-started;del model,condition;gc.collect()
    start=time.monotonic();predictor,backend=seq2_model();score=seq2_score(predictor,row['cds_dna']);seq_seconds=time.monotonic()-start
    del predictor;gc.collect()
    output=dict(status='completed',sequence_id=row['sequence_id'],n_codons=row['n_codons'],torch=torch.__version__,viennarna=RNA.__version__,
        ribodecode_cpu=hard,ribodecode_historical=cached['ribodecode_score'],hard_abs_error=abs(hard-cached['ribodecode_score']),
        seq2ribo_cpu=score['seq2ribo_score'],seq2ribo_historical=cached['seq2ribo_score'],
        seq2_abs_error=abs(score['seq2ribo_score']-cached['seq2ribo_score']),
        hard_load_and_score_seconds=hard_seconds,seq2_load_and_score_seconds=seq_seconds,
        backend=backend,source_protocol=protocol,scope='Shortest original sequence only; no complete30 claim')
    save(OUT/'evaluation_preflight.json',output);print(json.dumps(output,ensure_ascii=False,indent=2),flush=True)


def endpoints():
    path=OUT/'final_search_results.json'
    if path.exists():rows=read(path)
    else:rows=[read(p) for p in sorted((OUT/'outputs').glob('*.json'))]
    assert len(rows)==30 and len({r['sequence_id'] for r in rows})==30,'Need all30 completed search cases'
    records=[]
    for row in rows:
        assert row['steps']==80
        for field,method in [('original','original'),('old_v3','V3_RiboDecode_search'),('best','V3_own_scorer_search')]:
            value=row[field];dna=value['dna'];digest=hashlib.sha256(dna.encode()).hexdigest()
            assert digest==value['sequence_sha256']
            records.append(dict(sequence_id=row['sequence_id'],group=row['group'],n_codons=row['n_codons'],method=method,dna=dna,
                sequence_sha256=digest,own_score=value['own_score']))
    return records


def evaluate(reuse_historical_reference=False):
    records=endpoints();spec=source_protocol();torch,np,RNA=setup()
    spec.update(torch=torch.__version__,viennarna=RNA.__version__,driver_sha256=sha(__file__),
        reuse_historical_reference=reuse_historical_reference,
        historical_provenance='Original/oldV3 GPU metrics may be reused only after exact source/weight SHA and shortest-case same-version CPU parity; all novel endpoints computed on CPU')
    lock=OUT/'evaluation_protocol_locked.json'
    if lock.exists():assert read(lock)==spec
    else:save(lock,spec)
    digest=sha(lock);bank={r['sequence_sha256']:r for r in records};begun=time.monotonic()
    hard_folder=OUT/'evaluation/ribodecode';seq_folder=OUT/'evaluation/seq2ribo'
    hard_folder.mkdir(parents=True,exist_ok=True);seq_folder.mkdir(parents=True,exist_ok=True)
    if reuse_historical_reference:
        check=read(OUT/'evaluation_preflight.json')
        assert check['status']=='completed' and check['viennarna']=='2.7.2'
        assert check['hard_abs_error']<1e-5 and check['seq2_abs_error']<1e-5
        assert check['source_protocol']==source_protocol()
        path=WORK/'runs/vmae_tiny_30cds_20261025/metrics.csv'
        with path.open(encoding='utf-8-sig',newline='') as handle:historical=list(csv.DictReader(handle))
        lookup={(r['sequence_id'],r['method']):r for r in historical}
        for row in records:
            if row['method']=='V3_own_scorer_search':continue
            method='original' if row['method']=='original' else 'small'
            old=lookup[row['sequence_id'],method];assert row['dna']==old['cds_dna'] and row['sequence_sha256']==old['sequence_sha256']
            provenance=dict(kind='historical_GPU_reference',source=str(path),source_sha256=sha(path),
                cpu_parity_sha256=sha(OUT/'evaluation_preflight.json'),same_source_weights_and_viennarna=True)
            save(hard_folder/f'{row["sequence_sha256"]}.json',dict(sequence_sha256=row['sequence_sha256'],
                ribodecode_score=float(old['ribodecode_score']),protocol_sha256=digest,provenance=provenance))
            save(seq_folder/f'{row["sequence_sha256"]}.json',dict(sequence_sha256=row['sequence_sha256'],
                seq2ribo_score=float(old['seq2ribo_score']),
                seq2ribo_by_seed={str(s):float(old[f'seq2ribo_seed_{s}']) for s in SEEDS},
                protocol_sha256=digest,provenance=provenance))
    missing_hard=any(not (hard_folder/f'{key}.json').exists() for key in bank)
    if missing_hard:model,condition=ribodecode_model(torch)
    for number,(key,row) in enumerate(bank.items(),1):
        dest=hard_folder/f'{key}.json'
        if dest.exists():assert read(dest)['protocol_sha256']==digest;continue
        value=hard_score(torch,model,condition,row['dna']);save(dest,dict(sequence_sha256=key,ribodecode_score=value,protocol_sha256=digest,
            provenance=dict(kind='local_CPU_recomputed',torch=torch.__version__,viennarna=RNA.__version__)))
        save(OUT/'evaluation_progress.json',dict(stage='ribodecode',completed=number,total=len(bank),seconds=time.monotonic()-begun))
    if missing_hard:del model,condition
    gc.collect()
    missing_seq=any(not (seq_folder/f'{key}.json').exists() for key in bank)
    if missing_seq:
        predictor,backend=seq2_model()
        save(OUT/'evaluation_backend.json',backend)
    for number,(key,row) in enumerate(bank.items(),1):
        dest=seq_folder/f'{key}.json'
        if dest.exists():assert read(dest)['protocol_sha256']==digest;continue
        started=time.monotonic();values=seq2_score(predictor,row['dna'])
        save(dest,dict(sequence_sha256=key,**values,protocol_sha256=digest,seconds=time.monotonic()-started,
            provenance=dict(kind='local_CPU_recomputed',torch=torch.__version__,viennarna=RNA.__version__)))
        save(OUT/'evaluation_progress.json',dict(stage='seq2ribo',completed=number,total=len(bank),seconds=time.monotonic()-begun))
        print(json.dumps(dict(event='seq2ribo_complete',completed=number,total=len(bank),seconds=time.monotonic()-started)),flush=True)
    if missing_seq:del predictor
    gc.collect();table=[]
    cai=read(WORK/'runs/vmae_tiny_30cds_20261025/cai_weights.json')
    for row in records:
        key=row['sequence_sha256'];dna=row['dna'];md=RNA.md();md.temperature=37.;md.dangles=2
        _,mfe=RNA.fold_compound(dna.replace('T','U'),md).mfe()
        entry=dict(**row,ribodecode_score=read(hard_folder/f'{key}.json')['ribodecode_score'],
            seq2ribo_score=read(seq_folder/f'{key}.json')['seq2ribo_score'],gc_percent=100*(dna.count('G')+dna.count('C'))/len(dna),
            cai=float(np.exp(np.mean([math.log(cai[dna[i:i+3]]) for i in range(0,len(dna)-3,3)]))),
            mfe_kcal_mol=float(mfe),mfe_per_nt=float(mfe)/len(dna),
            ribodecode_provenance=read(hard_folder/f'{key}.json')['provenance']['kind'],
            seq2ribo_provenance=read(seq_folder/f'{key}.json')['provenance']['kind'],
            properties_provenance='local_ViennaRNA2.7.2_CPU_recomputed')
        table.append(entry)
    with (OUT/'comparison_metrics.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(table[0]));writer.writeheader();writer.writerows(table)
    save(OUT/'comparison_metrics.json',table)
    save(OUT/'evaluation_complete.json',dict(status='completed',n_cases=30,n_method_rows=90,n_unique_sequences=len(bank),
        metrics_sha256=sha(OUT/'comparison_metrics.csv'),protocol_sha256=digest,seconds=time.monotonic()-begun,
        same_source_weights_and_viennarna_for_all=True,all_metrics_recomputed_on_cpu=not reuse_historical_reference,
        historical_reference_reuse_explicit=reuse_historical_reference,not_measured_expression=True))
    summarize(table)
    print('FULL30 CPU EVALUATION COMPLETE',flush=True)


def summarize(table=None):
    if table is None:table=read(OUT/'comparison_metrics.json')
    import numpy as np
    keys=['own_score','ribodecode_score','seq2ribo_score','gc_percent','cai','mfe_kcal_mol','mfe_per_nt']
    methods=['original','V3_RiboDecode_search','V3_own_scorer_search'];summary=[]
    for method in methods:
        group=[r for r in table if r['method']==method]
        assert len(group)==30
        summary.append(dict(method=method,n=30,**{k:float(np.mean([r[k] for r in group])) for k in keys}))
    lookup={(r['sequence_id'],r['method']):r for r in table};paired=[]
    for source in ['original','V3_RiboDecode_search']:
        for identity in sorted({r['sequence_id'] for r in table}):
            candidate=lookup[identity,'V3_own_scorer_search'];reference=lookup[identity,source]
            paired.append(dict(sequence_id=identity,group=candidate['group'],reference=source,
                **{f'delta_{k}':candidate[k]-reference[k] for k in keys}))
    save(OUT/'comparison_summary.json',dict(summary=summary,paired=paired,
        interpretation='30 reused human CDS test cases; own scalar head fitted on bacterial mRFP, all generated outcomes are predictions',
        optimization_budget='Both80 outer iterations; historical V3 gradient240 scorer calls vs current black-box80 proposals, not equal compute',
        property_note='CAI is fixed human-training-corpus reference; GC/CAI/MFE are not monotonic expression quality'))
    with (OUT/'paired_deltas.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(paired[0]));writer.writeheader();writer.writerows(paired)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('phase',choices=['static','preflight','evaluate','report'])
    parser.add_argument('--reuse-historical-reference',action='store_true');args=parser.parse_args()
    try:
        if args.phase=='static':save(OUT/'evaluation_source_check.json',dict(status='passed',protocol=source_protocol()));print('PASS: all frozen seq2ribo source/weight hashes')
        elif args.phase=='preflight':preflight()
        elif args.phase=='evaluate':evaluate(args.reuse_historical_reference)
        else:summarize()
    except BaseException as error:save(OUT/'evaluation_failure.json',dict(phase=args.phase,error=repr(error),unix=time.time()));raise
