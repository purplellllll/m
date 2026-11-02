"""Gene-separated HEK293 half-life development model; no TE test data."""
from pathlib import Path
import json, hashlib, itertools, os
os.environ.setdefault('OMP_NUM_THREADS','2')
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from scipy.stats import spearmanr

ROOT=Path(__file__).resolve().parent
WORK=ROOT.parent.parent
OUT=ROOT/'stability';OUT.mkdir(exist_ok=True)
KMERS=[''.join(x) for k in [1,2,3] for x in itertools.product('ACGT',repeat=k)]
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def save(p,v):Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def features(dna,utr5,utr3):
    values=[]
    for region in [utr5,dna,utr3]:
        region=str(region).upper().replace('U','T')
        for k in [1,2,3]:
            counts={s:0 for s in KMERS if len(s)==k}
            for i in range(len(region)-k+1):
                s=region[i:i+k]
                if s in counts:counts[s]+=1
            den=max(1,sum(counts.values()))
            values.extend(v/den for v in counts.values())
    return np.asarray(values,dtype=np.float64)
def metrics(y,p):
    return dict(n=len(y),pearson=float(np.corrcoef(y,p)[0,1]),spearman=float(spearmanr(y,p).statistic),mse=float(np.mean((y-p)**2)))
class StabilityScorer:
    def __init__(self):
        with np.load(OUT/'model.npz',allow_pickle=False) as z:
            self.mean=z['mean'];self.scale=z['scale'];self.coef=z['coef'];self.intercept=float(z['intercept'])
    def __call__(self,dna,utr5,utr3):
        return float(((features(dna,utr5,utr3)-self.mean)/self.scale)@self.coef+self.intercept)
def main():
    rows=read(WORK/'runs/human_te_expanded_20261101/train_validation.json')
    cohort=read(WORK/'runs/scorer_path_alignment_20261031/native_te_cohort.json')['records']
    contexts={r['tx_id']:r for r in cohort if r['cell']=='hek293' and r['utr_ok']}
    source=ROOT/'sources/saluki_all_HLs_human.txt.gz'
    table=pd.read_csv(source,sep='\t').set_index('ensembl_gene_id')
    columns=['Dieterich_4sU_HEK293_1','Dieterich_4sU_HEK293_2']
    protocol=dict(target='Mean of two Dieterich HEK293 replicates after published log10(half_life+1) transform; not Saluki PC1',
        source_sha256=sha(source),transform_source_sha256=sha(ROOT/'sources/saluki_master_comparison_human.R'),
        development_rows_sha256=sha(WORK/'runs/human_te_expanded_20261101/train_validation.json'),
        inputs='Separate 5UTR/CDS/3UTR 1-3mer frequencies, no explicit length input; fixed UTRs for synonymous search',
        selection='Ridge alpha 10/100/1000 via three train-only gene folds; validation opened once',
        test='No previously opened 903 TE test samples read',source_code_sha256=sha(__file__))
    save(OUT/'protocol_locked.json',protocol)
    records=[];used=set()
    for r in rows:
        gene=r['gene'].split('.')[0]
        if gene in used or gene not in table.index or r['transcript'] not in contexts:continue
        values=table.loc[gene,columns].to_numpy(dtype=float)
        if not np.isfinite(values).all():continue
        c=contexts[r['transcript']]
        assert c['dna']==r['dna'] and c['split']==r['split']
        used.add(gene);records.append(dict(**r,utr5=c['utr5'],utr3=c['utr3'],half_life_log_target=float(values.mean())))
    train=[r for r in records if r['split']=='train'];val=[r for r in records if r['split']=='val']
    assert len(train)>1000 and len(val)>100
    assert not {r['gene'].split('.')[0] for r in train}&{r['gene'].split('.')[0] for r in val}
    save(OUT/'development_records.json',records)
    X=np.stack([features(r['dna'],r['utr5'],r['utr3']) for r in train]);y=np.array([r['half_life_log_target'] for r in train])
    V=np.stack([features(r['dna'],r['utr5'],r['utr3']) for r in val]);v=np.array([r['half_life_log_target'] for r in val])
    choices=[]
    for alpha in [10.,100.,1000.]:
        pred=np.empty(len(y))
        for ti,vi in KFold(3,shuffle=True,random_state=20261102).split(X):
            m=make_pipeline(StandardScaler(),Ridge(alpha=alpha));m.fit(X[ti],y[ti]);pred[vi]=m.predict(X[vi])
        choices.append(dict(alpha=alpha,**metrics(y,pred)))
    alpha=min(choices,key=lambda r:r['mse'])['alpha']
    model=make_pipeline(StandardScaler(),Ridge(alpha=alpha)).fit(X,y);pred=model.predict(V)
    scale,reg=model.steps[0][1],model.steps[1][1]
    np.savez(OUT/'model.npz',mean=scale.mean_,scale=scale.scale_,coef=reg.coef_,intercept=reg.intercept_)
    reload=StabilityScorer();assert np.max(np.abs(pred-np.array([reload(r['dna'],r['utr5'],r['utr3']) for r in val])))<1e-10
    rng=np.random.default_rng(20261102);boot=[]
    for _ in range(2000):
        i=rng.integers(len(v),size=len(v));boot.append(np.corrcoef(v[i],pred[i])[0,1])
    ci=np.quantile(boot,[.025,.975]).tolist()
    report=dict(status='completed',train_genes=len(train),validation_genes=len(val),alpha=alpha,train_only_cv=choices,
        validation=metrics(v,pred),pearson_ci95=ci,model_sha256=sha(OUT/'model.npz'),
        usable_for_exploratory_scan=bool(ci[0]>0),not_independent_final_confirmation=True,
        limits=['Gene-level measurements assigned to one selected transcript; isoform-specific half-life not measured',
                'HEK293 assay, not all cell types; changed CDS predictions have no measured half-life',
                'Absolute hours not claimed; transformed original time units retained',
                'Kmer ridge is an auxiliary model, not a trained half-life head in our TE network'])
    save(OUT/'validation_predictions.json',[dict(uid=r['uid'],gene=r['gene'],target=float(a),prediction=float(b)) for r,a,b in zip(val,v,pred)])
    save(OUT/'summary.json',report);print(json.dumps(report),flush=True)
    persist=pd.read_csv(ROOT/'sources/persist_constructs.tsv',sep='\t');valid=persist.dropna(subset=['half_life_in_cell','degradation_coef_in_cell'])
    save(OUT/'persist_units_audit.json',dict(n=len(valid),max_relative_error_ln2_over_k=float(np.max(np.abs(valid.half_life_in_cell-np.log(2)/valid.degradation_coef_in_cell)/valid.half_life_in_cell)),
        issue='Published R code fits log2 abundance but derives half-life as ln(2)/slope; a log2 slope implies 1/slope. Do not silently relabel these as calibrated hours.',
        decision='PERSIST retained as secondary source, not used to train human stability model'))
if __name__=='__main__':main()
