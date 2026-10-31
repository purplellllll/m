"""Independent post-run checks, including train-only fitting and reloaded inference."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main(args):
    root=Path(args.run);sys.path.insert(0,str(root));sys.path.insert(1,args.assets)
    import run_aligned_profile_ablation_20261030 as run
    from physics_scorer.profile_model import AlignedProfileModel
    from latent_decoder.prepare_data import GENETIC_CODE
    torch.set_num_threads(1);os.nice(10)
    assert json.loads((root/'status.json').read_text())['stage']=='completed'
    manifest=json.loads((root/'run_manifest.json').read_text());panel=json.loads((root/'panel.json').read_text())
    locked=json.loads((root/'checkpoints_locked.json').read_text());rows=panel['records']
    assert sha(root/'panel.json')==manifest['panel_sha256']==locked['panel_sha256']
    assert sha(root/'physics_scorer/profile_model.py')==manifest['model_sha256']
    assert sha(root/'physics_scorer/model.py')==manifest['graph_layer_sha256']
    assert sha(root/'run_aligned_profile_ablation_20261030.py')==manifest['driver_sha256']
    for filename,digest in manifest['simulator_hashes'].items():
        assert sha(Path(args.assets)/'evaluator/seq2ribo'/filename)==digest
    assert len(locked['models'])==24 and len(rows)==600
    proteins=[];features=[]
    for r in rows:
        proteins.append(''.join(GENETIC_CODE[r['dna'][i:i+3]] for i in range(0,len(r['dna']),3)))
        with np.load(root/'features'/f'{r["gene"]}.npz',allow_pickle=False) as z:f={k:z[k].copy() for k in z.files}
        n=len(r['dna'])//3
        assert f['physics'].shape==(n,2) and f['structure'].shape==(n,5)
        assert np.array_equal(f['codon_start_nt0'],np.arange(n)*3)
        assert ''.join(run.native(args.assets)[0].CODONS[c] for c in f['cod'])==r['dna'].replace('T','U')
        assert ''.join(run.ORIGINAL_CODONS[c] for c in r['source_ids'])==r['dna']
        y=np.array([np.nan if v is None else v for v in r['y']],dtype=np.float32)
        assert np.array_equal(f['observed'],np.isfinite(y))
        np.testing.assert_allclose(f['label'],np.log1p(np.nan_to_num(y,nan=0)),atol=1e-7)
        features.append(f)
    assert len(set(proteins))==len({r['gene'] for r in rows})==len({r['dna'] for r in rows})==600
    # Deliberately replace every validation/test target: physics fit must stay identical.
    changed=[dict(r,y=[999.]*len(r['y'])) if r['split']!='train' else r for r in rows]
    refit=run.fit_physics(changed,features)
    params=json.loads((root/'physics_parameters_train_only.json').read_text())
    for key in ['waits','pair','angle','bucket']:np.testing.assert_allclose(refit[key],params[key],atol=1e-10,rtol=1e-10)
    for item in locked['models']:assert sha(item['path'])==item['sha256']
    metrics=pd.read_csv(root/'metrics.csv');per=pd.read_csv(root/'per_transcript_metrics.csv')
    testrows=[r for r in rows if r['split']=='test'];test=[f for r,f in zip(rows,features) if r['split']=='test']
    differences=[]
    with np.load(root/'test_predictions.npz',allow_pickle=False) as predictions:
        for record in metrics.to_dict('records'):
            name=record['model'];correlations=[]
            for i,e in enumerate(test):
                p=predictions[f'{name}__{i}'];mask=e['observed'];y=e['label']
                assert p.shape==y.shape and np.isfinite(p).all()
                a=y[mask].astype(float);b=p[mask].astype(float)
                if np.ptp(a)==0 or np.ptp(b)==0:r=np.nan
                else:
                    a-=a.mean();b-=b.mean();r=float(a@b/np.sqrt(a@a*(b@b)))
                correlations.append(r)
                item=per[(per.model==name)&(per.gene==testrows[i]['gene'])]
                assert len(item)==1
                reported=float(item.iloc[0].pearson_log)
                if np.isfinite(r):differences.append(abs(r-reported));assert abs(r-reported)<2e-6
            assert abs(np.nanmean(correlations)-record['mean_transcript_pcc_log'])<2e-6
        item=next(i for i in locked['models'] if i['arm']=='full' and i['seed']==30101)
        ckpt=torch.load(item['path'],map_location='cpu',weights_only=True)
        model=AlignedProfileModel(**ckpt['config']);model.load_state_dict(ckpt['state_dict'])
        actual=run.predict(model,test)
        max_prediction_error=max(float(np.max(np.abs(p-predictions[f'full_30101__{i}']))) for i,p in enumerate(actual))
        assert max_prediction_error<2e-6
        for arm in run.ARMS:
            for i in range(120):
                expected=np.mean([predictions[f'{arm}_{s}__{i}'] for s in run.SEEDS],axis=0)
                np.testing.assert_allclose(expected,predictions[f'{arm}_ensemble__{i}'],atol=1e-7)
    result=dict(status='verified',n_alignment_records=600,n_locked_checkpoints=24,n_models_scored=len(metrics),
        n_test_transcripts=120,max_independent_pearson_error=max(differences),max_reloaded_prediction_error=max_prediction_error,
        train_only_physics_fit_invariant_to_changed_validation_test_labels=True,all_ensembles_recomputed=True,
        source_panel_sha256=sha(root/'panel.json'),test_predictions_sha256=sha(root/'test_predictions.npz'))
    (root/'verification.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',required=True);p.add_argument('--assets',required=True);main(p.parse_args())
