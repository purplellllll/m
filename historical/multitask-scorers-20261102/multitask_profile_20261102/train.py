"""Bounded joint training; validation selects checkpoints, sealed labels stay closed."""
from pathlib import Path
import os,sys,json,time,hashlib,random,gc,math,argparse
os.environ['OMP_NUM_THREADS']='2';os.environ['MKL_NUM_THREADS']='2'
import numpy as np
import torch
from model import MultiTaskTransformer,TASKS,masked_loss
import features as ft
HERE=Path(__file__).resolve().parent;WORK=HERE.parents[1]
PRIMARY=['expression','half_life','solubility'];SEEDS=[30101,30102];ARMS=['sequence','dynamics'];STEPS=6000;EVERY=500;BATCH=8
def verify_inputs(rows):
    derived=ft.read(HERE/'derived_data_protocol.json')
    for name,digest in derived['derived_sha256'].items():assert ft.sha(HERE/name)==digest,name
    protocol=ft.read(HERE/'features_v2_protocol_locked.json')
    for name,digest in protocol['source_sha256'].items():assert ft.sha(HERE/name)==digest,name
    expected=ft.sha(HERE/'features_v2_protocol_locked.json');complete=ft.read(HERE/'features_complete.json')
    assert ft.sha(HERE/'input_scaler.npz')==complete['scaler_sha256']
    for r in {q['host']+'_'+q['uid']:q for q in rows}.values():
        with np.load(ft.cache_path(r),allow_pickle=False) as z:
            assert str(z['protocol_sha256'])==expected and str(z['dna_sha256'])==r['uid']
def load_bank(rows,name):
    path=HERE/(name+'_tensor_cache.pt');digest=ft.sha(HERE/'features_v2_protocol_locked.json')+ft.sha(HERE/'input_scaler.npz')
    if path.exists():
        cache=torch.load(path,map_location='cpu',weights_only=True);assert cache['digest']==digest
        assert cache['record_ids']==[r['record_id'] for r in rows]
        result={}
        for key,record in cache['bank'].items():result[key]={k:([t.numpy() for t in v] if isinstance(v,list) else v.numpy() if isinstance(v,torch.Tensor) else v) for k,v in record.items()}
        return result
    bank=ft.load(rows);payload={}
    for key,record in bank.items():payload[key]={k:([torch.from_numpy(t) for t in v] if isinstance(v,list) else torch.from_numpy(v) if isinstance(v,np.ndarray) else v) for k,v in record.items()}
    temp=path.with_suffix('.tmp');torch.save(dict(digest=digest,record_ids=[r['record_id'] for r in rows],bank=payload),temp);temp.replace(path)
    return bank
def setup():
    torch.set_num_threads(2);torch.backends.mha.set_fastpath_enabled(False)
    if not torch.cuda.is_available():raise RuntimeError('Use .runtime/scorer_cuda/Scripts/python.exe')
    torch.cuda.set_per_process_memory_fraction(.25)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
def batch(rows,bank,device='cuda',stats=None):
    b=len(rows);n=max(len(bank[r['record_id']]['codons']) for r in rows);u=max(3,max((len(x) for r in rows for x in bank[r['record_id']]['utrs']),default=0))
    x=dict(codons=np.full((b,n),64,np.int64),amino=np.full((b,n),21,np.int64),valid=np.zeros((b,n),bool),physics=np.zeros((b,n,12),np.float32),
        utr=np.full((b,2,u),4,np.int64),utr_present=np.zeros((b,2),np.float32),physics_present=np.ones(b,np.float32),physics_calibrated=np.zeros(b,np.float32),
        host=np.array([0 if r['host']=='human' else 1 for r in rows],dtype=np.int64))
    target=np.zeros((b,4),np.float32);mask=np.zeros((b,4),bool);ss=[];gg=[]
    for i,r in enumerate(rows):
        f=bank[r['record_id']];length=len(f['codons']);x['codons'][i,:length]=f['codons'];x['amino'][i,:length]=f['amino'];x['valid'][i,:length]=True;x['physics'][i,:length]=f['physics']
        x['physics_calibrated'][i]=f.get('physics_calibrated',0.)
        for j,a in enumerate(f['utrs']):x['utr'][i,j,:len(a)]=a;x['utr_present'][i,j]=float(len(a)>0)
        ss.append(f['stats']);gg.append(f['physical_global'])
        for j,t in enumerate(TASKS):
            if t in r.get('standardized_targets',{}):target[i,j]=r['standardized_targets'][t];mask[i,j]=True
    x.update(stats=np.stack(ss),physical_global=np.stack(gg))
    x={k:torch.from_numpy(v).to(device) for k,v in x.items()}
    return x,torch.from_numpy(target).to(device),torch.from_numpy(mask).to(device)
def corr(y,p):return None if np.ptp(y)<1e-12 or np.ptp(p)<1e-12 else float(np.corrcoef(y,p)[0,1])
def predict(model,rows,bank):
    model.eval();out=[]
    with torch.inference_mode():
        for start in range(0,len(rows),BATCH):
            x,_,_=batch(rows[start:start+BATCH],bank);out.extend(model(x).float().cpu().numpy().tolist())
    p=np.array(out);assert np.isfinite(p).all();return p
def validation(model,rows,bank):
    p=predict(model,rows,bank);metrics={}
    for task in PRIMARY:
        indices=[i for i,r in enumerate(rows) if r['task']==task];j=TASKS.index(task);actual=np.array([rows[i]['standardized_targets'][task] for i in indices]);score=p[indices,j]
        metrics[task]=dict(n=len(indices),pearson=corr(actual,score),mse=float(np.mean((score-actual)**2)))
    macro=float(np.mean([m['pearson'] if m['pearson'] is not None else -1. for m in metrics.values()]))
    return macro,metrics,p
def warmstart(model,seed):
    selection=ft.read(WORK/'runs/te_stability_20261102/te_100000/selection_locked.json');rec=next(r for r in selection['selected_models'] if r['seed']==seed)
    path=Path(rec['best_path']);assert ft.sha(path)==rec['checkpoint_sha256']
    ck=torch.load(path,map_location='cpu',weights_only=True);state=model.state_dict();used=[]
    for key,v in ck['backbone'].items():
        if key.startswith(('codons.','context.')):
            assert key in state and state[key].shape==v.shape,key;state[key]=v;used.append(key)
    model.load_state_dict(state,strict=True);assert len(used)>20
    return dict(source_sha256=rec['checkpoint_sha256'],keys=used,scope='Own human TE pretrained codon/context only; all new heads and UTR/amino/physics modules trained jointly')
def prepare_protocol():
    files=[HERE/n for n in ['model.py','train.py','features.py','dynamics_v2.py','data_protocol_locked.json','derived_data_protocol.json','features_v2_protocol_locked.json','input_scaler.npz','target_statistics.json']]
    protocol=dict(seeds=SEEDS,arms=ARMS,max_steps=STEPS,batch_size=BATCH,validation_every=EVERY,
        task_sampling='Round robin expression/half_life/solubility, equal8 examples per update; humanHL also has TE auxiliary label',
        loss='MSE of train-standardized observed labels; average observed tasks; missing labels masked',
        model=dict(width=192,context_layers=2,heads=8,max_context_tokens=512,CDS='Keep all codons; average-pool representations only when >512',UTR='Separate nucleotide convolution encoders, retain ends plus full composition',AA='Learned per-residue embeddings aligned to codons and20 composition features',host='Explicit human/ecoli embedding conditions both codon context and readout; cell-free assay semantics retained in record metadata',fold='Six RNAfold short-window MFE/nt sequence features shared by both arms'),
        optimization=dict(optimizer='AdamW',encoder_lr=1e-5,new_lr=3e-4,weight_decay=.01,clip=1.,warmup=200,amp='bfloat16 training only; full float32 validation/inference'),
        selection='Max validation macro Pearson of3 primary measured endpoints, not TE or test; freeze all4 checkpoints before opening test labels',
        physics='Fixed-window3-rate event features, observed host-specific train-only RPF relative-occupancy proxy; 5prime150 RNAfold pairing; no mouse parameters and not identified absolute kinetics',
        comparison='Same initialization, sample order, data, steps, selection and architecture except physics paths disabled in sequence arm',
        resource='CUDA allocator capped25% of8GB; prior fair experiment remains running;2CPU threads',
        acceptance='Measured fluorescence, observed HL, observed eSOL solubility separately; TE auxiliary cannot satisfy expression acceptance; no paired-expression-HL/triple labels available',
        history='Historical test sets already used; explicitly exploratory, no fresh independent confirmation',
        hashes={str(p.name):ft.sha(p) for p in files})
    path=HERE/'training_protocol_locked.json'
    if path.exists():assert ft.read(path)==protocol,'Training sources/protocol changed'
    else:ft.save(path,protocol)
    return protocol
def train(arm,seed):
    setup();protocol=prepare_protocol();out=HERE/'models'/f'{arm}_{seed}';out.mkdir(parents=True,exist_ok=True)
    if (out/'complete.json').exists():return
    rows=ft.records(HERE/'train_validation.jsonl');verify_inputs(rows);bank=load_bank(rows,'development');tr=[r for r in rows if r['split']=='train'];val=[r for r in rows if r['split']=='val'];groups={t:[r for r in tr if r['task']==t] for t in PRIMARY}
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
    model=MultiTaskTransformer(len(next(iter(bank.values()))['stats']),use_physics=arm=='dynamics').cuda();warm=warmstart(model,seed)
    fast=[];slow=[]
    for name,p in model.named_parameters():(slow if name.startswith(('codons.','context.')) else fast).append(p)
    optimizer=torch.optim.AdamW([dict(params=slow,lr=1e-5),dict(params=fast,lr=3e-4)],weight_decay=.01)
    rng=np.random.default_rng(seed);best=-np.inf;begin=1;started=time.monotonic()
    if (out/'latest.pt').exists():
        ck=torch.load(out/'latest.pt',map_location='cpu',weights_only=True);assert ck['protocol_sha256']==ft.sha(HERE/'training_protocol_locked.json')
        model.load_state_dict(ck['model'],strict=True);optimizer.load_state_dict(ck['optimizer']);begin=ck['step']+1;best=ck['best'];rng.bit_generator.state=json.loads(ck['numpy_rng'])
        torch.set_rng_state(ck['torch_rng']);torch.cuda.set_rng_state(ck['cuda_rng'])
    for step in range(begin,STEPS+1):
        task=PRIMARY[(step-1)%3];indices=rng.integers(len(groups[task]),size=BATCH);group=[groups[task][j] for j in indices]
        x,y,mask=batch(group,bank);model.train();optimizer.zero_grad(set_to_none=True)
        factor=min(1.,step/200)*(.1+.9*.5*(1+math.cos(math.pi*step/STEPS)))
        optimizer.param_groups[0]['lr']=1e-5*factor;optimizer.param_groups[1]['lr']=3e-4*factor
        with torch.autocast('cuda',dtype=torch.bfloat16):p=model(x);loss=masked_loss(p,y,mask)
        if not torch.isfinite(loss):raise FloatingPointError('Nonfinite loss')
        loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step()
        if step%100==0:ft.save(out/'status.json',dict(step=step,max_steps=STEPS,loss=float(loss.detach()),best=best if np.isfinite(best) else None,elapsed=time.monotonic()-started))
        if step%EVERY==0 or step==STEPS:
            score,metrics,vp=validation(model,val,bank)
            if score>best:
                best=score
                torch.save(dict(model={k:v.detach().cpu() for k,v in model.state_dict().items()},config=dict(stats_dim=len(next(iter(bank.values()))['stats']),use_physics=arm=='dynamics'),seed=seed,arm=arm,step=step,validation_macro=score,validation_metrics=metrics,validation_record_ids=[r['record_id'] for r in val],validation_predictions=torch.tensor(vp),warmstart=warm,protocol_sha256=ft.sha(HERE/'training_protocol_locked.json')),out/'best.pt')
            torch.save(dict(model={k:v.detach().cpu() for k,v in model.state_dict().items()},optimizer=optimizer.state_dict(),step=step,best=best,numpy_rng=json.dumps(rng.bit_generator.state),torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state(),protocol_sha256=ft.sha(HERE/'training_protocol_locked.json')),out/'latest.pt')
            ft.save(out/'validation_latest.json',dict(step=step,macro=score,metrics=metrics,best=best));print(json.dumps(dict(arm=arm,seed=seed,step=step,macro=score,metrics=metrics)),flush=True)
    ft.save(out/'complete.json',dict(status='completed',arm=arm,seed=seed,steps=STEPS,best_macro=best,best_sha256=ft.sha(out/'best.pt'),seconds=time.monotonic()-started))
    del model,optimizer,bank;gc.collect();torch.cuda.empty_cache()
def preflight():
    setup();rows=ft.records(HERE/'train_validation.jsonl');sample=[next(r for r in rows if r['task']==t and r['split']=='train') for t in PRIMARY];verify_inputs(sample);bank=ft.load(sample)
    torch.manual_seed(30101);model=MultiTaskTransformer(len(next(iter(bank.values()))['stats'])).cuda();warm=warmstart(model,30101);x,y,mask=batch(sample,bank)
    p=model(x);loss=masked_loss(p,y,mask);loss.backward()
    assert model.context.layers[0].self_attn.in_proj_weight.grad.abs().sum()>0
    for task in PRIMARY:assert model.heads[task][-1].weight.grad.abs().sum()>0
    assert model.node_physics.weight.grad.abs().sum()>0
    ft.save(HERE/'gpu_preflight.json',dict(status='passed',loss=float(loss.detach()),torch=torch.__version__,warmstart=warm,primary_heads_and_encoder_receive_gradient=True,physics_projection_receives_gradient=True,memory_allocated=torch.cuda.max_memory_allocated()))
    print('GPU preflight passed',flush=True)
if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('command',choices=['preflight','train']);parser.add_argument('--arm',choices=ARMS);parser.add_argument('--seed',type=int,choices=SEEDS);a=parser.parse_args()
    try:
        if a.command=='preflight':preflight()
        else:train(a.arm,a.seed)
    except BaseException as e:ft.save(HERE/'failure_latest.json',dict(command=vars(a),error=repr(e),unix=time.time()));raise
