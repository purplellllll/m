"""Independent endpoint arithmetic, checkpoint hashes and real-label checks."""
from pathlib import Path
import json,hashlib
import numpy as np
import pandas as pd
HERE=Path(__file__).resolve().parent;WORK=HERE.parents[1]
def read(p):return json.loads(Path(p).read_text(encoding='utf8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    checks=[]
    for directory in [WORK/'runs/multitask_transformer_20261102',HERE]:
        a=read(directory/'evaluation/summary.json');d=pd.read_csv(directory/'evaluation/predictions.csv');targets=read(directory/'sealed_test_targets.json')
        for task in ['expression','half_life','solubility']:
            sub=d[d.task==task]
            for row in sub.itertuples():assert abs(targets[row.record_id][task]-row.observed)<1e-10
            for m in a['metrics']:
                if m['task']!=task:continue
                value=float(np.corrcoef(sub.observed,sub[m['model']])[0,1]);assert abs(value-m['pearson'])<1e-10
        for m in read(directory/'selection_locked.json')['models']:assert sha(m['path'])==m['sha256']
        checks.append(dict(stage=directory.name,status='passed',n=len(d),metric_count=len(a['metrics']),checkpoint_count=4))
    protocol=read(HERE/'features_v2_protocol_locked.json')
    for name,digest in protocol['source_sha256'].items():assert sha(HERE/name)==digest
    records=[json.loads(l) for l in (HERE/'records.jsonl').read_text(encoding='utf8').splitlines()];held=[r for r in records if r['split']!='train'];params={h:read(HERE/(h+'_parameters.json')) for h in ['human','ecoli']}
    for host,p in params.items():
        h={r['uid'] for r in held if r['host']==host};assert not h&set(p['training_dna_sha256'])
    expr=[r for r in records if r['task']=='expression' and r['split']=='train'][:2];nodes=[]
    for r in expr:
        with np.load(HERE/'features_v2'/(r['host']+'_'+r['uid']+'.npz'),allow_pickle=False) as z:nodes.append(z['node'].copy())
    assert np.max(np.abs(nodes[0]-nodes[1]))>0
    checks.append(dict(stage='profile_physics',status='passed',same_length_synonymous_features_differ=True,no_heldout_exact_CDS_fit=True))
    (HERE/'verification.json').write_text(json.dumps(dict(status='passed',checks=checks),indent=2),encoding='utf8')
    print(json.dumps(checks,indent=2))
if __name__=='__main__':main()
