"""Full original V3 30-case panel, 80 black-box latent iterations per case on CPU."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time
import numpy as np
import torch

OUT=Path(__file__).resolve().parent
WORK=OUT.parent.parent
REPO=WORK/'repos/riboseq_scorer_20261031'
sys.path.insert(0,str(REPO));sys.path.insert(0,str(WORK))
from physics_scorer.reliable_graph_context import ReliableGraphContext
from physics_scorer.scalar_downstream import TaskScalarModel
from latent_decoder.model import CodonVMAE,collate_records,mutable_positions,synonymous_logits,MASK,AA_IDS
from latent_decoder.tokenizer import CODONS

SOURCE=WORK/'runs/graph_polish_20261031/gpu_result/completed_snapshot/graph_polish_20261031/source'
PARAM=SOURCE.parent/'data/physics_parameters_train_only.json'
VMAE=WORK/'runs/vmae_tiny_size_only_20261025/checkpoints/fold0_uniform.pt'
PROBE=WORK/'runs/mrfp_branch_probe_20261031/results/heads'
BACKBONE=WORK/'runs/scorer_compare_search_20261031/search_outputs/scorer_backbones'


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def save(path,data):
    path=Path(path);temporary=path.with_suffix('.tmp')
    temporary.write_text(json.dumps(data,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf-8');temporary.replace(path)
def load_file(name,path):
    spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module


def input_rows():
    panel=read(WORK/'runs/vmae_tiny_30cds_20261025/panel.json')
    manifest=read(WORK/'runs/seq2ribo_multicell_comparison_20261027/candidate_manifest.json')
    lookup={(r['sequence_id'],r['method']):r for r in manifest.values()}
    rows=[]
    for row in panel:
        identity=row['sequence_id'];old=lookup[identity,'V3_2.5M_Hard'];original=lookup[identity,'original']
        assert row['cds_dna']==original['cds_dna']
        rows.append(dict(sequence_id=identity,original=original,old_v3=old,n_codons=row['n_codons'],
                         group='short' if row['n_codons']-1<=300 else 'medium' if row['n_codons']-1<700 else 'long'))
    assert len(rows)==30 and len({r['sequence_id'] for r in rows})==30
    return rows


class OwnScorer:
    def __init__(self):
        import RNA
        self.rna_version=RNA.__version__;self.parameters=read(PARAM)
        self.implementation=load_file('frozen_feature_driver',REPO/'historical/drivers/run_aligned_profile_ablation_20261030.py')
        self.probe=load_file('frozen_probe_driver',WORK/'runs/mrfp_branch_probe_20261031/run_probe.py')
        self.cai=read(WORK/'runs/vmae_tiny_30cds_20261025/cai_weights.json');self.models=[]
        self.cache={};self.queries=0
        for seed in [30101,30102]:
            raw=torch.load(BACKBONE/f'gated_graph_{seed}.pt',map_location='cpu',weights_only=True)
            base=ReliableGraphContext(**raw['metadata']['spec']['config']);base.load_state_dict(raw['state_dict'],strict=True)
            model=TaskScalarModel(base,'protein').eval().requires_grad_(False)
            meta=torch.load(PROBE/f'gated_graph_{seed}_c.pt',map_location='cpu',weights_only=True)
            head=self.probe.Probe(meta['mask'],meta['nstats']).eval().requires_grad_(False);head.load_state_dict(meta['state_dict'],strict=True)
            self.models.append((model,head,meta))

    @torch.no_grad()
    def __call__(self,dna):
        key=hashlib.sha256(dna.encode()).hexdigest()
        if key in self.cache:return self.cache[key]
        path=OUT/'features'/f'{key}.npz';record_path=OUT/'scores'/f'{key}.json'
        if record_path.exists():
            value=read(record_path);assert value['sequence_sha256']==key and value['feature_sha256']==sha(path)
            self.cache[key]=value;return value
        f=self.implementation.structure_job((str(SOURCE),dna))
        seed=int(hashlib.sha256(('scalar_features:'+key).encode()).hexdigest()[:8],16)
        f.update(self.implementation.simulation_job((str(SOURCE),dna,f,self.parameters,seed)));f['dna_sha256']=key
        x,g=self.probe.batch([dict(dna=dna,uid=key)],{key:f},'cpu');values=[]
        for model,head,meta in self.models:
            pooled,_=model.representations(x)
            stats=((g-np.asarray(meta['global_mean']))/np.asarray(meta['global_scale'])).astype(np.float32)
            values.append(float(head(pooled,torch.from_numpy(stats))[0])*meta['y_scale']+meta['y_mean'])
        temporary=path.with_suffix('.tmp')
        with temporary.open('wb') as handle:np.savez_compressed(handle,**f)
        temporary.replace(path)
        value=dict(own_score=float(np.mean(values)),own_seed_scores=values,
            gc_percent=100*(dna.count('G')+dna.count('C'))/len(dna),
            cai=float(np.exp(np.mean([np.log(self.cai[dna[j:j+3]]) for j in range(0,len(dna)-3,3)]))),
            mfe=float(f['mfe']),mfe_per_nt=float(f['mfe'])/len(dna),sequence_sha256=key,
            feature_sha256=sha(path),dna=dna)
        self.cache[key]=value;self.queries+=1;save(record_path,value);return value


def protocol(rows):
    import RNA
    return dict(scope='All original30 V3 panel cases; frozen V3 original2498240; own mRFP scorer OOD on human CDS',
        steps_per_case=80,proposals_per_step=1,black_box=True,original_fallback=True,
        latent='Masked mutable-codon encoder; mu+sigma*offset; Gaussian improvement-only hill climb',
        radius=8.,noise_schedule='max(0.25,0.95**(step//10))',seed=20261031,
        query_budget='80 decoded proposals plus initial decoder, original and old-V3 references per case; cache deduplicated',
        fairness='Same V3 generator, panel and80 outer iterations; original gradient search used more scorer calls and a different optimizer, not equal compute',
        original_v3_search='Historical80 Adam steps lr0.05 radius8, RiboDecode objective, 240 scorer forward sequences',
        device='cpu',torch=torch.__version__,viennarna=RNA.__version__,threads=2,
        generator_sha256=sha(VMAE),params_sha256=sha(PARAM),
        backbone_sha256={str(s):sha(BACKBONE/f'gated_graph_{s}.pt') for s in [30101,30102]},
        heads_sha256={str(s):sha(PROBE/f'gated_graph_{s}_c.pt') for s in [30101,30102]},
        source_sha256={p.name:sha(p) for p in [Path(__file__),REPO/'physics_scorer/reliable_graph_context.py',REPO/'physics_scorer/scalar_downstream.py']},
        feature_source_sha256={p.name:sha(p) for p in (SOURCE/'evaluator/seq2ribo').glob('*.py')},
        data_inputs_sha256=hashlib.sha256(json.dumps(rows,sort_keys=True).encode()).hexdigest(),
        no_retraining=True,no_test_metric_candidate_selection=True,
        meaning='Predicted search objective and property comparison; no measured expression for generated sequences')


@torch.no_grad()
def one_case(model,scorer,row,steps):
    identity=row['sequence_id'];started=time.monotonic()
    tokens=collate_records([dict(cds_dna=row['original']['cds_dna'])]);active=mutable_positions(tokens)
    mu,lv=model.encode(tokens.masked_fill(active,MASK),tokens);sigma=(.5*lv).exp()
    rng=torch.Generator().manual_seed(20261031+int(identity[4:12],16))
    original=scorer(row['original']['cds_dna']);old=scorer(row['old_v3']['cds_dna'])
    def evaluate(offset,step):
        logits=synonymous_logits(model.decode(mu+sigma*offset,tokens),tokens);ids=logits.argmax(-1)
        ids[~active]=tokens[~active]
        assert torch.equal(AA_IDS[ids],AA_IDS[tokens])
        dna=''.join(CODONS[i] for i in ids[0].tolist())
        assert dna[:3]==row['original']['cds_dna'][:3] and dna[-3:]==row['original']['cds_dna'][-3:]
        return dict(step=step,**scorer(dna))
    offset=torch.zeros_like(mu);initial=evaluate(offset,0);latent_best=initial['own_score']
    best=dict(step=-1,**original);trace=[initial]
    if initial['own_score']>best['own_score']:best=initial
    for step in range(1,steps+1):
        trial=offset+torch.randn(mu.shape,generator=rng)*active[...,None]*max(.25,.95**(step//10))
        trial*=active[...,None];rms=trial[active].square().mean().sqrt();trial*=min(1.,8/float(rms.clamp_min(1e-12)))
        entry=evaluate(trial,step);trace.append(entry)
        if entry['own_score']>latent_best:latent_best=entry['own_score'];offset=trial
        if entry['own_score']>best['own_score']:best=entry
        save(OUT/'progress.json',dict(stage='searching',sequence_id=identity,group=row['group'],step=step,steps=steps,
            seconds_current_case=time.monotonic()-started,original_own_score=original['own_score'],best_own_score=best['own_score'],
            completed_cases=len(list((OUT/'outputs').glob('*.json'))),total_cases=30))
        if step%10==0:print(json.dumps(dict(sequence_id=identity,step=step,best=best['own_score'],seconds=time.monotonic()-started)),flush=True)
    return dict(sequence_id=identity,group=row['group'],n_codons=row['n_codons'],steps=steps,
        original=original,old_v3=old,initial_decoder=initial,best=best,trace=trace,
        gain_from_original=best['own_score']-original['own_score'],gain_vs_old_v3=best['own_score']-old['own_score'],
        changed_codons=sum(best['dna'][j:j+3]!=original['dna'][j:j+3] for j in range(0,len(original['dna']),3)),
        seconds=time.monotonic()-started)


def main(smoke=False):
    torch.set_num_threads(2);torch.set_num_interop_threads(1)
    for name in ['features','scores','outputs']:(OUT/name).mkdir(exist_ok=True)
    rows=input_rows();spec=protocol(rows)
    if (OUT/'protocol_locked.json').exists():assert read(OUT/'protocol_locked.json')==spec,'Source or protocol changed; inspect before continuation'
    else:save(OUT/'protocol_locked.json',spec);save(OUT/'inputs_locked.json',rows)
    raw=torch.load(VMAE,map_location='cpu',weights_only=False)
    model=CodonVMAE(**raw['config']['model']).eval().requires_grad_(False);model.load_state_dict(raw['model'],strict=True)
    assert sum(p.numel() for p in model.parameters())==2498240
    scorer=OwnScorer();begun=time.monotonic();results=[]
    for row in rows[:1] if smoke else rows:
        path=OUT/'outputs'/f'{row["sequence_id"]}.json'
        if path.exists():
            result=read(path);assert result['steps']==80
        else:
            result=one_case(model,scorer,row,2 if smoke else 80)
            save(OUT/'smoke.json' if smoke else path,result)
        results.append(result)
        save(OUT/'status.json',dict(stage='smoke_complete' if smoke else 'searching',completed=len(results),total=30,seconds=time.monotonic()-begun))
    assert all(p.grad is None for p in model.parameters()) and sha(VMAE)==spec['generator_sha256']
    if not smoke:
        save(OUT/'final_search_results.json',results)
        save(OUT/'search_complete.json',dict(status='completed',cases=len(results),steps_per_case=80,
            no_generator_retraining=True,seconds=time.monotonic()-begun,unique_candidate_features=len(list((OUT/'features').glob('*.npz')))))
    print('SMOKE COMPLETE' if smoke else 'FULL30 SEARCH COMPLETE',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--smoke',action='store_true');args=parser.parse_args()
    try:main(args.smoke)
    except BaseException as error:save(OUT/'failure.json',dict(error=repr(error),unix=time.time()));raise
