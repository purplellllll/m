"""Measured multi-assay labels with missing-task masks and protected holdouts."""
from pathlib import Path
import json, hashlib
from collections import Counter, defaultdict
import numpy as np
import pandas as pd
from Bio.Seq import Seq

HERE=Path(__file__).resolve().parent
WORK=HERE.parents[1]
TASKS=['expression','half_life','solubility','te']
SPLITS={'train':0,'val':1,'test':2}

def read(p):return json.loads(Path(p).read_text(encoding='utf8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,v):Path(p).write_text(json.dumps(v,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf8')

def record(task, source_id, dna, split, host, targets, gene, utr5=None, utr3=None, aa=None, **metadata):
    dna=str(dna).upper().replace('U','T')
    if len(dna)%3 or not set(dna)<=set('ACGT') or dna[-3:] not in {'TAA','TAG','TGA'}:
        raise ValueError(f'Invalid complete CDS {task}:{source_id}')
    translated=str(Seq(dna).translate(table=11 if host=='ecoli' else 1,cds=True))
    if aa is not None and translated!=aa:raise ValueError(f'Protein identity differs: {source_id}')
    aa=translated
    for k,v in targets.items():
        if k not in TASKS or not np.isfinite(v):raise ValueError('Invalid target')
    utr_ok=utr5 is not None and utr3 is not None and len(utr5)>0 and len(utr3)>0
    if utr5 is not None and not set(utr5.upper().replace('U','T'))<=set('ACGT'):raise ValueError('Invalid UTR5')
    if utr3 is not None and not set(utr3.upper().replace('U','T'))<=set('ACGT'):raise ValueError('Invalid UTR3')
    return dict(record_id=f'{task}:{source_id}',uid=hashlib.sha256(dna.encode()).hexdigest(),
        canonical_rna_sha256=hashlib.sha256(dna.replace('T','U').encode()).hexdigest(),
        dna=dna,utr5=utr5.upper().replace('U','T') if utr5 is not None else '',
        utr3=utr3.upper().replace('U','T') if utr3 is not None else '',utr_available=bool(utr_ok),
        aa=aa,protein_sha256=hashlib.sha256(aa.encode()).hexdigest(),host=host,split=split,task=task,
        gene=str(gene).split('.')[0],targets={k:float(v) for k,v in targets.items()},
        target_mask={k:k in targets for k in TASKS},source_id=str(source_id),**metadata)

def protect_splits(rows):
    """Protect every higher split; mRFP deliberately measures same-protein variants."""
    indexed={s:defaultdict(list) for s in SPLITS}
    for r in rows:
        for kind,value in [('dna',r['uid']),('protein',r['protein_sha256']),('gene',r['host']+':'+r['gene'])]:
            indexed[r['split']][(kind,value)].append(r)
    kept=[];excluded=[]
    for r in rows:
        reasons=[]
        for other in SPLITS:
            if SPLITS[other]<=SPLITS[r['split']]:continue
            for kind,value in [('dna',r['uid']),('protein',r['protein_sha256']),('gene',r['host']+':'+r['gene'])]:
                matches=indexed[other].get((kind,value),[])
                if kind in ['protein','gene'] and r['task']=='expression':
                    matches=[q for q in matches if q['task']!='expression']
                if matches:reasons.append(dict(split=other,key=kind,matched_record_ids=[q['record_id'] for q in matches[:5]]))
        if reasons:excluded.append(dict(record_id=r['record_id'],task=r['task'],split=r['split'],reasons=reasons))
        else:kept.append(r)
    # Re-run strict DNA isolation globally and protein isolation for cross-gene tasks.
    for a,b in [('train','val'),('train','test'),('val','test')]:
        assert not {r['uid'] for r in kept if r['split']==a}&{r['uid'] for r in kept if r['split']==b}
        for task in ['half_life','solubility']:
            assert not {r['protein_sha256'] for r in kept if r['task']==task and r['split']==a}&{r['protein_sha256'] for r in kept if r['task']==task and r['split']==b}
    return kept,excluded

def main():
    HERE.mkdir(parents=True,exist_ok=True)
    mrfp_path=WORK/'runs/scalar_downstream_20261031/data/protein.json'
    hl_path=WORK/'runs/te_stability_20261102/stability/development_records.json'
    test_path=WORK/'runs/joint_scalar_20261102/human_joint/test_predictions.csv'
    context_path=WORK/'runs/scorer_path_alignment_20261031/native_te_cohort.json'
    te_inputs=WORK/'runs/human_te_expanded_20261101/test_inputs.json'
    esol_path=WORK/'runs/joint_scalar_20261102/esol_v2/mapped_sequences.csv'
    sources=[mrfp_path,hl_path,test_path,context_path,te_inputs,esol_path]
    protocol=dict(status='locked_before_model_training',source_sha256={str(p.relative_to(WORK)):sha(p) for p in sources},
        script_sha256=sha(__file__),tasks=TASKS,
        primary_tasks=['expression','half_life','solubility'],auxiliary_task='te',
        transforms=dict(expression='Use source mRFP Value as supplied; no extra logarithm',
            half_life='Published Dieterich HEK293 mean of two log10(HL+1) replicates; not hours',
            solubility='Measured PURE Solubility (%) unchanged',te='Original native human HEK293 TE unchanged'),
        normalization='Mean and population SD separately for each observed target on train records only; all predictions inverse-transformed before metrics',
        missing_labels='Absent task is missing, never zero target; target_mask defines per-task supervision',
        missing_context='Unavailable UTR represented by empty string plus utr_available=False, never claimed complete context',
        holdouts='Preserve existing task splits; exclude lower-priority exact-CDS/gene/protein conflicts globally, test>val>train',
        mrfp_exception='Same mRFP protein is deliberately shared across splits: synonymous-variant evaluation, NOT protein-disjoint. Exact DNA remains split-disjoint.',
        acceptance='Evaluate three MEASURED endpoints separately. No same-construct function+HL+solubility joint acceptance because no three-label panel exists.',
        limits=['Different hosts and assays: mRFP E.coli fluorescence, HL human HEK293, eSOL PURE cell-free E.coli proteins',
                'Historical test sets previously opened; new experiment is exploratory, not untouched confirmation',
                'eSOL native W3110 reference lacks verified exact ASKA tags; near-homology split not certified',
                'TE auxiliary labels must not replace measured functional expression'])
    save(HERE/'data_protocol_locked.json',protocol)
    rows=[];source_exclusions=[]
    data=read(mrfp_path);overlap=set(data['normalized_cds_train_test_overlap_ids'])
    for i,r in enumerate(data['records']):
        if r.get('exclude_from_clean_fit',False) and r['split']=='train':
            source_exclusions.append(dict(task='expression',source_index=i,reason='prior_clean_fit_exclusion'));continue
        if r['split']=='test' and r['uid'] in overlap:
            source_exclusions.append(dict(task='expression',source_index=i,reason='prior_normalized_CDS_train_test_overlap'));continue
        rows.append(record('expression',str(i),r['dna'],'val' if r['split']=='validation' else r['split'],
            'ecoli',{'expression':r['target'][0]},'mRFP',source_dataset='mRFP_1459',
            assay='Functional fluorescent reporter expression',original_uid=r['uid']))
    for r in read(hl_path):
        rows.append(record('half_life',r['uid'],r['dna'],r['split'],'human',
            {'half_life':r['half_life_log_target'],'te':r['target']},r['gene'],r['utr5'],r['utr3'],
            source_dataset='Dieterich_HEK293',transcript=r['transcript'],assay='Measured mRNA half-life; auxiliary native TE from separate study',original_uid=r['uid']))
    contexts={r['tx_id']:r for r in read(context_path)['records'] if r['cell']=='hek293' and r['utr_ok']}
    tests={r['uid']:r for r in read(te_inputs)}
    for q in pd.read_csv(test_path).to_dict('records'):
        r=tests[q['uid']];c=contexts[r['transcript']];assert c['dna']==r['dna'] and c['split']=='test'
        rows.append(record('half_life',r['uid'],r['dna'],'test','human',{'half_life':q['HL_target'],'te':q['TE_target']},
            r['gene'],c['utr5'],c['utr3'],source_dataset='Dieterich_HEK293',transcript=r['transcript'],
            assay='Measured mRNA half-life; auxiliary native TE from separate study',original_uid=r['uid']))
    for r in pd.read_csv(esol_path).to_dict('records'):
        rows.append(record('solubility',r['JW_ID'],r['dna'],r['split'],'ecoli',{'solubility':r['solubility_percent']},
            r['gene'],aa=r['aa'],source_dataset='eSOL_W3110',assay='Measured PURE cell-free soluble/total protein percent',
            reference_accession=r['accession'],b_number=r['b_number'],construct_sequence_verified=False))
    counts_before=Counter((r['task'],r['split']) for r in rows)
    rows,conflicts=protect_splits(rows)
    assert len({r['record_id'] for r in rows})==len(rows)
    statistics={}
    for task in TASKS:
        values=np.array([r['targets'][task] for r in rows if r['split']=='train' and task in r['targets']])
        assert len(values)>100 and np.isfinite(values).all() and values.std()>0
        statistics[task]=dict(train_n=len(values),mean=float(values.mean()),scale=float(values.std()),
            min=float(values.min()),max=float(values.max()))
    for r in rows:
        r['standardized_targets']={k:(v-statistics[k]['mean'])/statistics[k]['scale'] for k,v in r['targets'].items()}
    text=''.join(json.dumps(r,ensure_ascii=False,allow_nan=False)+'\n' for r in rows)
    (HERE/'records.jsonl').write_text(text,encoding='utf8')
    counts={task:{split:sum(r['task']==task and r['split']==split for r in rows) for split in SPLITS} for task in ['expression','half_life','solubility']}
    audit=dict(status='passed',total_records=len(rows),counts=counts,
        source_counts_before_global_filter={task:{split:counts_before[task,split] for split in SPLITS} for task in counts},
        source_exclusions=source_exclusions,cross_task_or_split_conflicts_excluded=conflicts,
        global_exact_CDS_split_disjoint=True,cross_gene_tasks_exact_protein_split_disjoint=True,
        mrfp_protein_disjoint=False,mrfp_unique_proteins=len({r['protein_sha256'] for r in rows if r['task']=='expression'}),
        near_homology_checked=False,utr_available_by_task={task:sum(r['task']==task and r['utr_available'] for r in rows) for task in counts},
        true_expression_HL_joint_n=sum(all(k in r['targets'] for k in ['expression','half_life']) for r in rows),
        true_three_endpoint_joint_n=sum(all(k in r['targets'] for k in ['expression','half_life','solubility']) for r in rows),
        test_labels_used_for_training=False,test_labels_previously_opened=True,
        records_sha256=sha(HERE/'records.jsonl'),protocol_sha256=sha(HERE/'data_protocol_locked.json'))
    save(HERE/'target_statistics.json',statistics);save(HERE/'data_audit.json',audit)
    print(json.dumps(dict(counts=counts,total=len(rows),conflicts=len(conflicts),joint_n=audit['true_three_endpoint_joint_n']),indent=2))

if __name__=='__main__':main()
