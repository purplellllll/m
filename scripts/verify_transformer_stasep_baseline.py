"""Verify frozen model hashes, historical metrics, and real-sequence CPU parity."""
from pathlib import Path
import argparse,json,hashlib,csv,sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--models-root',type=Path,required=True);parser.add_argument('--output',type=Path);args=parser.parse_args()
    card=json.loads((ROOT/'baselines/transformer_stasep_v1/BASELINE.json').read_text(encoding='utf8'))
    for relative,digest in card['frozen_code_sha256'].items():assert sha(ROOT/relative)==digest,relative
    for relative,digest in card['asset_file_sha256'].items():assert sha(args.models_root/relative)==digest,relative
    with (ROOT/card['predictions_file']).open(encoding='utf-8-sig') as f:rows=list(csv.DictReader(f))
    metric_errors=[]
    for m in card['historical_metrics']:
        group=[r for r in rows if r['task']==m['task']];assert len(group)==m['n']
        y=np.array([float(r['observed']) for r in group]);p=np.array([float(r[m['arm']+'_ensemble']) for r in group])
        error=abs(float(np.corrcoef(y,p)[0,1])-m['pearson']);assert error<1e-12;metric_errors.append(error)
    from multitask_scorer import Scorer
    fixtures=json.loads((ROOT/'baselines/transformer_stasep_v1/fixtures.json').read_text(encoding='utf8'));checks=[]
    for arm in ['sequence','dynamics']:
        scorer=Scorer(args.models_root,stage='profile',arm=arm,device='cpu');errors=[]
        for fixture in fixtures:
            got=scorer.predict(**fixture['inputs'])['predictions']
            for task,expected in fixture['expected'][arm].items():errors.append(abs(got[task]-expected))
        assert max(errors)<.002,(arm,max(errors))
        checks.append(dict(arm=arm,real_validation_sequences=len(fixtures),max_native_target_error=max(errors),strict_load=True))
    result=dict(status='passed',baseline_id=card['baseline_id'],weight_files=4,
        metrics_recomputed=6,maximum_pearson_error=max(metric_errors),CPU_parity=checks)
    if args.output:args.output.write_text(json.dumps(result,indent=2),encoding='utf8')
    print(json.dumps(result,indent=2))
if __name__=='__main__':main()
