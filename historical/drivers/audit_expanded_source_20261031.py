"""Summarize public source eligibility without changing data or selecting by performance."""
import json
import numpy as np
import pandas as pd
from pathlib import Path

ROOT=Path(__file__).resolve().parent
source=Path('/root/fusion_encoders_20261031/related_liver/LIVER.csv')
frame=pd.read_csv(source);lengths=[];eligible=[];counts={}
for r in frame.itertuples(index=False):
    ids=np.fromstring(r.codon_sequence.strip('[]'),sep=',',dtype=int)
    y=np.fromstring(r.annotations.strip('[]'),sep=',')
    lengths.append(len(ids))
    if len(y)!=len(ids) or len(ids)<40 or (ids<0).any() or (ids>=64).any():continue
    mask=np.isfinite(y)
    if mask.mean()<.95 or (y[mask]<0).any() or np.mean(y[mask]>0)<.3 or np.ptp(y[mask])==0:continue
    longest=current=0
    for v in y:current=current+1 if v==0 else 0;longest=max(longest,current)
    if longest>20:continue
    eligible.append(len(ids))
result=dict(source_records=len(frame),length_quantiles=np.percentile(lengths,[0,25,50,75,90,95,99,100]).tolist(),
    quality_eligible=len(eligible),quality_eligible_up_to={n:sum(v<=n for v in eligible) for n in [250,512,1024,2048,10000]},
    quality='>=95% finite, >=30% positive, longest zero <=20, length >=40; no CDS/genetic dedup yet')
(ROOT/'source_audit.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
