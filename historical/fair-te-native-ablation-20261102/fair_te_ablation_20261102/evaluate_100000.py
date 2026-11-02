"""Repeat the historically opened903 test after100000 training; never call it fresh confirmation."""
from pathlib import Path
import sys,json,hashlib,time,csv
import numpy as np
import torch
HERE=Path(__file__).resolve().parent;WORK=HERE.parent.parent;TE=WORK/'runs/te_stability_20261102/te_100000';DATA=WORK/'runs/human_te_expanded_20261101'
OUT=HERE/'full_length_100000';OUT.mkdir(exist_ok=True)
sys.path.insert(0,str(HERE));import experiment as ex
from physics_scorer.human_te_model import HumanTEScalar
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,v):Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
def load(rows):
    result={}
    for r in rows:
        assert hashlib.sha256(r['dna'].replace('T','U').encode()).hexdigest()==r['uid']
        with np.load(DATA/'features'/f'{r["uid"]}.npz',allow_pickle=False) as z:f={k:z[k].copy() for k in z.files}
        assert str(f['dna_sha256'])==r['uid'];result[r['uid']]=f
    return result
def main():
    ex.configure();selection=read(TE/'selection_locked.json');rows=read(DATA/'test_inputs.json');assert len(rows)==903
    assert all('target' not in r for r in rows)
    val=[r for r in read(TE/'train_validation.json') if r['split']=='val'][:4]
    features=load(rows+val);predictions=[];checks=[]
    protocol=dict(task='Human_HEK293_native_TE',test_scope='Previously opened903-gene test, explicitly exploratory repeated evaluation',
        selection_sha256=sha(TE/'selection_locked.json'),inputs_sha256=sha(DATA/'test_inputs.json'),
        frozen_baseline_predictions_sha256=sha(DATA/'test_predictions.json'),old_summary_sha256=sha(DATA/'test_summary.json'),
        test_used_for_gradient_or_checkpoint_selection=False,project_test_not_fresh=True,source_sha256=sha(__file__))
    protocol['feature_sha256']={r['uid']:sha(DATA/'features'/f'{r["uid"]}.npz') for r in rows}
    save(OUT/'protocol_locked.json',protocol)
    for record in selection['selected_models']:
        path=Path(record['best_path']);assert sha(path)==record['checkpoint_sha256']
        raw=torch.load(path,map_location='cpu',weights_only=True)
        model=HumanTEScalar(dict(raw['config']),use_properties=False).cuda().eval()
        state={'backbone.'+k:v for k,v in raw['backbone'].items()};state.update({'head.'+k:v for k,v in raw['head'].items()})
        state.update(stats_mean=torch.tensor(raw['stats_mean']),stats_scale=torch.tensor(raw['stats_scale']),target_mean=torch.tensor(raw['target_mean']),target_scale=torch.tensor(raw['target_scale']))
        model.load_state_dict(state,strict=True)
        def predict(group):
            values=[]
            with torch.no_grad():
                for start in range(0,len(group),2):
                    chunk=group[start:start+2];values.extend(model(ex.batch(chunk,features)).cpu().numpy().tolist())
            return np.array(values)
        error=float(np.max(np.abs(predict(val)-np.array(raw['validation_predictions'][:4]))));assert error<2e-5
        p=predict(rows);assert np.isfinite(p).all();predictions.append(p)
        checks.append(dict(seed=raw['seed'],best_step=raw['step'],checkpoint_sha256=record['checkpoint_sha256'],reload_max_error=error))
        del model;torch.cuda.empty_cache()
    own=np.mean(predictions,axis=0)
    save(OUT/'own_predictions_locked.json',dict(uids=[r['uid'] for r in rows],predictions=own.tolist(),seed_predictions=[r.tolist() for r in predictions],checks=checks,protocol_sha256=sha(OUT/'protocol_locked.json')))
    targets=read(DATA/'sealed_test_targets.json');y=np.array([targets[r['uid']] for r in rows]);old={r['uid']:r for r in read(DATA/'test_predictions.json')}
    assert set(old)=={r['uid'] for r in rows} and all(abs(targets[uid]-r['target'])<1e-12 for uid,r in old.items())
    arrays={'ours100000':own,'seq2ribo':np.array([old[r['uid']]['seq2ribo_score'] for r in rows]),'RiboDecode_fixedRNA_TEproxy':np.array([old[r['uid']]['ribodecode_score'] for r in rows])}
    rng=np.random.default_rng(20261102);indices=rng.integers(len(y),size=(2000,len(y)));metrics={};boot={}
    for arm,v in arrays.items():
        b=np.array([np.corrcoef(y[i],v[i])[0,1] for i in indices]);assert np.isfinite(b).all();boot[arm]=b
        metrics[arm]=dict(n=len(y),pearson=float(np.corrcoef(y,v)[0,1]),ci95=np.quantile(b,[.025,.975]).tolist(),mse=float(np.mean((y-v)**2)))
    previous=read(DATA/'test_summary.json')['summary']
    assert abs(metrics['seq2ribo']['pearson']-previous['seq2ribo']['pearson'])<1e-12
    assert abs(metrics['RiboDecode_fixedRNA_TEproxy']['pearson']-previous['ribodecode']['pearson'])<1e-12
    paired={arm:dict(delta=metrics['ours100000']['pearson']-metrics[arm]['pearson'],ci95=np.quantile(boot['ours100000']-boot[arm],[.025,.975]).tolist()) for arm in arrays if arm!='ours100000'}
    report=dict(status='completed',metrics=metrics,paired=paired,reused_test=True,independent_fresh_confirmation=False,
        limitations=['Prior903 results informed project iterations; do not call this fresh confirmatory testing',
            'Frozen external published weights have uncertified training-set overlap; different training resources',
            'RiboDecode is fixed-RNA TE proxy, not its native RPF task',
            'No RiboGL native TE score; its explicitly adapted TE comparison is reported separately',
            'No near-homology or cross-study confirmation; intervals exploratory'],reload_checks=checks)
    save(OUT/'summary.json',report)
    with (OUT/'metrics.csv').open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.writer(f);w.writerow(['model','n','pearson','ci_low','ci_high','mse'])
        for arm,r in metrics.items():w.writerow([arm,r['n'],r['pearson'],*r['ci95'],r['mse']])
    save(OUT/'predictions.json',dict(uids=[r['uid'] for r in rows],target=y.tolist(),predictions={a:v.tolist() for a,v in arrays.items()}))
    save(OUT/'complete.json',dict(status='completed',n=903,strict_backbone_and_head_load=True,reused_test=True))
if __name__=='__main__':
    try:main()
    except BaseException as error:save(OUT/'failure.json',dict(error=repr(error),unix=time.time()));raise
