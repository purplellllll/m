"""Lock real scalar labels/splits; no model scores enter data selection."""
import hashlib
import json
from pathlib import Path
import zipfile
import numpy as np
import pandas as pd

WORK=Path(__file__).resolve().parents[1]
OUT=WORK/'runs/scalar_downstream_20261031/data'
LABELS=['mean_te','TE_HEK293','TE_LCL','TE_RPE-1']


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,v):Path(p).write_text(json.dumps(v,indent=2,allow_nan=False),encoding='utf-8')
def sid(dna):return hashlib.sha256(dna.encode()).hexdigest()
def finite(v):return float(v) if np.isfinite(v) else None


def main():
    OUT.mkdir(parents=True,exist_ok=False)
    path=OUT.parent/'mRFP_Expression.csv';d=pd.read_csv(path)
    assert len(d)==1459 and d.Split.value_counts().to_dict()=={'train':1021,'val':219,'test':219}
    duplicated=d[d.Sequence.duplicated(keep=False)]
    demo=pd.read_csv(WORK/'tmp/seq2ribo_source_20261020/seq2ribo-main/notebooks/sample_data/mrfp_expr_test_samples.csv')
    lookup=d.set_index('Sequence')
    assert all(lookup.loc[r.Sequence,'Split']=='test' and np.isclose(lookup.loc[r.Sequence,'Value'],r.Value,rtol=0,atol=1e-8) for r in demo.itertuples())
    rows=[]
    for r in d.itertuples():
        dna=r.Sequence.upper().replace('U','T');assert len(dna)==678 and not set(dna)-set('ACGT')
        rows.append(dict(uid=sid(dna),dna=dna,split='validation' if r.Split=='val' else r.Split,target=[float(r.Value)]))
    test_ids={r['uid'] for r in rows if r['split']=='test'}
    overlap_ids={r['uid'] for r in rows if r['split']=='train'}&test_ids
    for r in rows:r['exclude_from_clean_fit']=r['split']=='train' and r['uid'] in overlap_ids
    save(OUT/'protein.json',dict(task='protein',records=rows,counts={'train':1021,'validation':219,'test':219},
        source_sha256=sha(path),source_url='https://github.com/Kingsford-Group/codonmoe/blob/main/datasets/mRFP_Expression.csv',
        public_seq2ribo_test_slice_exact_match=len(demo),paper_split_counts_match=True,
        duplicated_sequences=duplicated.to_dict('records'),
        normalized_cds_train_test_overlap_ids=sorted(overlap_ids),
        clean_fit_train_count=sum(r['split']=='train' and not r['exclude_from_clean_fit'] for r in rows),
        clean_test_count=sum(r['split']=='test' and r['uid'] not in overlap_ids for r in rows),
        primary_comparison='Report all219 native test variants and paired217 without normalized-CDS train overlap; omit2 overlapping train rows from our fit',
        full_seq2ribo_split_identity_certified=False,split_note='Full split from CodonMoE; all seq2ribo public held-out examples match. Full seq2ribo training manifest unavailable.',
        granularity='Pearson across synonymous mRFP variants; protein identity is deliberately shared',label_transform='train-only mean/std, invert before Pearson'))
    table=WORK/'deliverables/public_riboseq_validation_20261028/human_TE_Supplementary_Table1.xlsx'
    d=pd.read_excel(table);rows=[];excluded=[]
    for index,r in enumerate(d.itertuples(index=False)):
        sequence=str(r.tx_sequence).upper().replace('U','T')
        dna=sequence[int(r.utr5_size):int(r.utr5_size+r.cds_size)]
        if not dna or len(dna)%3 or set(dna)-set('ACGT'):
            excluded.append(dict(gene=str(r.gene_id),reason='invalid_codon_sequence'));continue
        split='test' if int(r.fold)==0 else 'validation' if int(r.fold)==1 else 'train'
        # Access columns by original spelling, not itertuples' renamed TE fields.
        labels=[finite(v) for v in d.iloc[index][LABELS].to_numpy(dtype=float)]
        rows.append(dict(uid=sid(dna),dna=dna,gene=str(r.gene_id),transcript=str(r.tx_id),fold=int(r.fold),split=split,
                         utr5=sequence[:int(r.utr5_size)],utr3=sequence[int(r.utr5_size+r.cds_size):],target=labels))
    # Keep native supplied folds; report any collisions explicitly rather than changing the test set.
    assert len({r['gene'] for r in rows})==len(rows),'Repeated gene needs grouped splitting'
    ids={s:{r['uid'] for r in rows if r['split']==s} for s in ['train','validation','test']}
    collisions={a+'__'+b:len(ids[a]&ids[b]) for a,b in [('train','validation'),('train','test'),('validation','test')]}
    save(OUT/'te.json',dict(task='te',labels=LABELS,records=rows,excluded=excluded,
        counts={s:sum(r['split']==s for r in rows) for s in ids},source_sha256=sha(table),
        split='RiboNN supplied folds: 0=test,1=validation,2..9=train; NOT seq2ribo native stratified split',
        original_paper_split_reproduced=False,exact_cds_collisions=collisions,
        label_transform='Per-output train-only mean/std with missing-label masks; report original-scale Pearson',
        full_cds_no_truncation=True))
    archive=WORK/'data/official/dataset_for_translation_model.zip'
    with zipfile.ZipFile(archive) as z:
        genes=pd.read_csv(z.open('genes.csv'));rna=pd.read_csv(z.open('RNA_RPKM_320.csv'),index_col=0)
        rpf=pd.read_csv(z.open('RPF_RPKM_320.csv'),index_col=0);meta=pd.read_csv(z.open('sample_information.csv'))
        iso=pd.read_csv(z.open('mjIsoform_320.csv'),dtype=str)
        seq=pd.read_csv(z.open('human_coding_transcript_sequences.csv'),dtype=str).set_index('transcript_id').cds_sequence.to_dict()
    train=genes.Training_set.dropna().tolist();test=genes.Test_set.dropna().tolist();assert len(train)==10552 and len(test)==1173
    assert rna.index.equals(rpf.index) and [g for g in rna.index if g in set(train)]==train
    gene_ids=rna.index.tolist();trainset=set(train)
    validation={g for g in train if int(hashlib.sha256(('scalar_rpf_val:'+g).encode()).hexdigest()[:8],16)%10==0}
    gs=np.array([2 if g not in trainset else 1 if g in validation else 0 for g in gene_ids],dtype=np.int8)
    es=np.array([0 if t=='training' else 2 for t in meta.Type],dtype=np.int8)
    environment=np.log1p(5*rna.loc[train].to_numpy(dtype=np.float32).T)
    assert environment.shape==(320,10552) and (es==0).sum()==200
    gene_lookup={g.split('.')[0]:i for i,g in enumerate(gene_ids)};sequence_rows=[];sequence_lookup={};arrays=[];excluded_counts={}
    grouped={k:v for k,v in iso.groupby('sample_RPF_id',sort=False)}
    rna_values=rna.to_numpy();rpf_values=rpf.to_numpy()
    for e,sample in enumerate(rpf.columns):
        part=grouped.get(sample);assert part is not None,sample
        columns=[[],[],[],[],[]]
        for r in part.itertuples(index=False):
            gi=gene_lookup.get(r.gene_id);dna=seq.get(r.isoform_id)
            if gi is None:continue
            reason=None
            if not isinstance(dna,str):reason='missing_major_isoform'
            elif not 90<=len(dna)<=4500 or len(dna)%3 or set(dna)-set('ACGT') or not dna.startswith('ATG'):reason='native_90_4500nt_canonical_CDS'
            x=float(rna_values[gi,e]);y=float(rpf_values[gi,e])
            if not np.isfinite([x,y]).all() or min(x,y)<0:reason='invalid_expression'
            if reason:excluded_counts[reason]=excluded_counts.get(reason,0)+1;continue
            key=sid(dna)
            if key not in sequence_lookup:
                sequence_lookup[key]=len(sequence_rows);sequence_rows.append(dict(uid=key,dna=dna))
            for col,v in zip(columns,[sequence_lookup[key],e,gi,np.log1p(5*x),np.log1p(5*y)]):col.append(v)
        arrays.append([np.asarray(v,dtype=np.int32 if j<3 else np.float32) for j,v in enumerate(columns)])
    names=['sequence_index','environment_index','gene_index','log_rna','log_rpf']
    result={name:np.concatenate([a[j] for a in arrays]) for j,name in enumerate(names)}
    result.update(gene_split=gs,environment_split=es,environment=environment)
    np.savez_compressed(OUT/'rpf_rows.npz',**result)
    save(OUT/'rpf.json',dict(task='rpf',records=sequence_rows,genes=gene_ids,context_gene_order=train,
        environments=[dict(index=i,rpf_sample=s,rna_sample=rna.columns[i],**meta.iloc[i].to_dict()) for i,s in enumerate(rpf.columns)],
        n_rows=len(result['log_rpf']),n_sequences=len(sequence_rows),excluded_counts=excluded_counts,
        source_sha256=sha(archive),row_archive_sha256=sha(OUT/'rpf_rows.npz'),
        gene_split_codes={'train':0,'validation':1,'test':2},environment_split_codes={'training':0,'test':2},
        training_rule='Only train genes in 200 author-training environments; hash-held validation genes within author train',
        label='log1p(5*RPF_RPKM)',inputs='Full CDS, actual log1p(5*RNA_RPKM),10552-gene RNA context in author order',
        primary='Pearson within each environment across author-test genes, then macro mean',
        secondary='Held-out environments on author-train/test genes; pooled Pearson and RNA-only baseline separately'))
    save(OUT/'manifest.json',{p.name:sha(p) for p in OUT.iterdir() if p.is_file()})
    print(json.dumps(dict(protein=1459,te=len(rows),rpf_rows=len(result['log_rpf']),rpf_sequences=len(sequence_rows))))


if __name__=='__main__':main()
