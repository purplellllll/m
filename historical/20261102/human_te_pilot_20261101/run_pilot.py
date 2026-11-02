"""Native human HEK293 TE development pilot; no test prediction or production replacement."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time
import numpy as np
import torch
from torch import nn

ROOT=Path(__file__).resolve().parent
WORK=ROOT.parent.parent
REPO=WORK/'repos/riboseq_scorer_20261031'
sys.path.insert(0,str(REPO));sys.path.insert(0,str(WORK))
from physics_scorer.reliable_graph_context import ReliableGraphContext
from physics_scorer.scalar_downstream import TaskScalarModel

def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,v):
    p=Path(p);tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    for attempt in range(20):
        try:tmp.replace(p);break
        except PermissionError:
            if attempt==19:raise
            time.sleep(.1)
def module(name,p):
    s=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
def corr(y,p):return float(np.corrcoef(y,p)[0,1]) if np.ptp(p)>0 else 0.

class Head(nn.Module):
    def __init__(self):
        super().__init__();self.norm=nn.LayerNorm(384)
        self.net=nn.Sequential(nn.Linear(387,128),nn.GELU(),nn.Dropout(.1),nn.Linear(128,1))
    def forward(self,c,stats):return self.net(torch.cat((self.norm(c),stats),-1)).squeeze(-1)

def main():
    torch.set_num_threads(2);assert torch.cuda.is_available();torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.mha.set_fastpath_enabled(False)
    cohort=WORK/'runs/scorer_path_alignment_20261031/native_te_cohort.json'
    source=WORK/'runs/graph_polish_20261031/gpu_result/completed_snapshot/graph_polish_20261031/source'
    params=read(source.parent/'data/physics_parameters_train_only.json')
    helper=module('human_feature_helper',REPO/'historical/drivers/run_aligned_profile_ablation_20261030.py')
    probe=module('existing_batch_helper',WORK/'runs/mrfp_branch_probe_20261031/run_probe.py')
    records=read(cohort)['records'];selected=[]
    for split,n in [('train',256),('val',96)]:
        eligible=[r for r in records if r['cell']=='hek293' and r['split']==split and 80<=len(r['dna'])//3<=500 and len(r['dna'])%3==0 and r['dna'][-3:] in {'TAA','TAG','TGA'}]
        selected+=sorted(eligible,key=lambda r:hashlib.sha256(('human_te_pilot:'+r['cds_sha256']).encode()).hexdigest())[:n]
        assert len([r for r in selected if r['split']==split])==n
    assert not ({r['cds_sha256'] for r in selected if r['split']=='train'}&{r['cds_sha256'] for r in selected if r['split']=='val'})
    for r in selected:r['uid']=r['cds_sha256']
    arms=['frozen_c_properties','finetune2_properties','finetune4_properties','finetune2_no_stats','finetune2_no_physics','finetune2_sequence_only']
    protocol=dict(task='Human HEK293 native CDS-only TE',train=256,validation=96,length_codons=[80,500],seeds=[30101,30102],arms=arms,
        max_steps=300,validation_every=50,batch_size=4,optimizer='AdamW',lr_encoder=1e-5,lr_head=3e-4,weight_decay=.01,
        properties=['GC_fraction','MFE_per_nt','CAI'],explicit_length_feature=False,statistics_normalization='Train only',
        physics='Frozen existing mouse-proxy parameters, four simulations; domain mismatch retained and ablated, not human-calibrated',
        expansion='Duplicate pretrained context layers2->4; no width change; all other architecture and budgets matched',
        limits='Small length-restricted development pilot; test never predicted; no claim of full human calibration, SOTA or generator improvement',
        source_sha256=sha(__file__),cohort_sha256=sha(cohort),physics_sha256=sha(source.parent/'data/physics_parameters_train_only.json'))
    lock=ROOT/'protocol_locked.json'
    if lock.exists():assert read(lock)==protocol
    else:save(lock,protocol)
    save(ROOT/'selected_development_rows.json',selected)
    features={};folder=ROOT/'features';folder.mkdir(exist_ok=True)
    for i,r in enumerate(selected):
        path=folder/f'{r["uid"]}.npz'
        if not path.exists():
            f=helper.structure_job((str(source),r['dna']))
            seed=int(hashlib.sha256(('human_te:'+r['uid']).encode()).hexdigest()[:8],16)
            f.update(helper.simulation_job((str(source),r['dna'],f,params,seed)));f['dna_sha256']=r['uid']
            with path.open('wb') as handle:np.savez_compressed(handle,**f)
        with np.load(path,allow_pickle=False) as z:features[r['uid']]={k:z[k].copy() for k in z.files}
        assert str(features[r['uid']]['dna_sha256'])==r['uid']
        save(ROOT/'status.json',dict(stage='features',completed=i+1,total=len(selected)))
    cai=read(WORK/'runs/vmae_tiny_30cds_20261025/cai_weights.json')
    def stats(r):
        dna=r['dna'];return [(dna.count('G')+dna.count('C'))/len(dna),float(features[r['uid']]['mfe'])/len(dna),
            np.exp(np.mean([np.log(cai[dna[i:i+3]]) for i in range(0,len(dna)-3,3)]))]
    train=[r for r in selected if r['split']=='train'];val=[r for r in selected if r['split']=='val']
    g=np.array([stats(r) for r in train]);gm=g.mean(0);gs=g.std(0).clip(1e-6)
    y=np.array([r['target'] for r in train]);ym=y.mean();ys=max(y.std(),1e-6)
    def batch(rows,no_stats=False):
        x,_=probe.batch(rows,features,'cuda');g=(np.array([stats(r) for r in rows])-gm)/gs
        if no_stats:g*=0
        return x,torch.tensor(g,dtype=torch.float32,device='cuda')
    results=[];(ROOT/'checkpoints').mkdir(exist_ok=True)
    for seed in [30101,30102]:
        payload=torch.load(WORK/f'runs/scorer_compare_search_20261031/search_outputs/scorer_backbones/gated_graph_{seed}.pt',map_location='cpu',weights_only=True)
        for arm in arms:
            torch.manual_seed(seed);base=ReliableGraphContext(**payload['metadata']['spec']['config']);base.load_state_dict(payload['state_dict'],strict=True)
            if arm=='finetune4_properties':base.context.layers.extend([copy.deepcopy(layer) for layer in base.context.layers]);base.context.num_layers=4
            if arm=='finetune2_no_physics':base.use_physics=False
            if arm=='finetune2_sequence_only':
                base.input_fusion.register_forward_pre_hook(lambda module,args:(torch.cat((args[0][...,:192],torch.zeros_like(args[0][...,192:])),dim=-1),))
            model=TaskScalarModel(base,'te').cuda();head=Head().cuda()
            frozen=arm=='frozen_c_properties'
            if frozen:model.requires_grad_(False)
            optimizer=torch.optim.AdamW([{'params':head.parameters(),'lr':3e-4}]+([] if frozen else [{'params':model.parameters(),'lr':1e-5}]),weight_decay=.01)
            def predict(rows):
                model.eval();head.eval();values=[]
                with torch.no_grad():
                    for start in range(0,len(rows),4):
                        x,g=batch(rows[start:start+4],arm=='finetune2_no_stats');p,_=model.representations(x)
                        values.extend((head(p[:,2],g)*ys+ym).cpu().tolist())
                return np.asarray(values)
            best=-float('inf');best_state=None;best_step=0;rng=np.random.default_rng(seed);history=[]
            for step in range(1,301):
                group=[train[j] for j in rng.choice(len(train),4,replace=False)];x,g=batch(group,arm=='finetune2_no_stats')
                model.eval() if frozen else model.train();head.train();optimizer.zero_grad(set_to_none=True)
                p,_=model.representations(x);prediction=head(p[:,2],g);target=torch.tensor([(r['target']-ym)/ys for r in group],dtype=torch.float32,device='cuda')
                loss=(prediction-target).square().mean();assert torch.isfinite(loss);loss.backward()
                torch.nn.utils.clip_grad_norm_(list(head.parameters())+list(model.parameters()),1.);optimizer.step()
                if step%50==0:
                    predictions=predict(val);actual=np.array([r['target'] for r in val]);r=corr(actual,predictions)
                    history.append(dict(step=step,validation_pearson=r,validation_mse=float(np.mean((actual-predictions)**2))))
                    if r>best:
                        best=r;best_step=step;best_state=dict(backbone=copy.deepcopy(base.cpu().state_dict()),head=copy.deepcopy(head.cpu().state_dict()));model.cuda();head.cuda()
                    save(ROOT/'status.json',dict(stage='training',arm=arm,seed=seed,step=step,validation_pearson=r,completed_models=len(results),total_models=12))
            base.load_state_dict(best_state['backbone']);head.load_state_dict(best_state['head']);pred=predict(val)
            path=ROOT/'checkpoints'/f'{arm}_{seed}.pt';torch.save(dict(**best_state,arm=arm,seed=seed,stats_mean=gm.tolist(),stats_scale=gs.tolist(),target_mean=float(ym),target_scale=float(ys),protocol=protocol),path)
            results.append(dict(arm=arm,seed=seed,best_step=best_step,validation_pearson=best,parameters=sum(p.numel() for p in base.parameters())+sum(p.numel() for p in head.parameters()),validation_predictions=pred.tolist(),history=history,checkpoint_sha256=sha(path)))
            save(ROOT/'results.json',results);del model,base,head,optimizer;torch.cuda.empty_cache()
    save(ROOT/'complete.json',dict(status='completed',models=len(results),train=256,validation=96,test_predicted=False))

if __name__=='__main__':
    try:main()
    except BaseException as error:save(ROOT/'failure.json',dict(error=repr(error),unix=time.time()));raise
