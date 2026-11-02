"""Controlled short-CDS TE benchmark; reused test is explicitly exploratory."""
from pathlib import Path
import sys,os,time,json,hashlib,math,random,copy,gc
os.environ.setdefault('OMP_NUM_THREADS','2');os.environ.setdefault('MKL_NUM_THREADS','2')
import numpy as np
import torch
HERE=Path(__file__).resolve().parent;WORK=HERE.parent.parent;DATA=WORK/'runs/human_te_expanded_20261101'
sys.path.insert(0,str(WORK));sys.path.insert(0,str(HERE))
from models import ARMS,build,Ours
SEEDS=[30101,30102];STEPS=3000;EVERY=250
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,v):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(v,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf-8');tmp.replace(p)
def feature_path(uid):
    p=DATA/'features'/f'{uid}.npz';assert p.exists();return p
def prepare():
    if (HERE/'inputs_locked.json').exists():return
    (HERE/'graphs').mkdir(exist_ok=True)
    rows=[r for r in read(DATA/'train_validation.json') if 80<=r['n_codons']<=300]
    tests=[r for r in read(DATA/'test_inputs.json') if 80<=r['n_codons']<=300]
    assert all('target' not in r for r in tests)
    for r in tests:r['split']='test'
    assert sum(r['split']=='train' for r in rows)==1951 and sum(r['split']=='val' for r in rows)==262
    for split in ['train','val','test']:
        subset=[r for r in rows+tests if r['split']==split]
        for other in ['train','val','test']:
            if split>=other:continue
            rhs=[r for r in rows+tests if r['split']==other]
            for key in ['gene','uid','protein_sha256']:assert not {r[key] for r in subset}&{r[key] for r in rhs},(split,other,key)
    save(HERE/'train_validation.json',rows);save(HERE/'test_inputs.json',tests)
    protocol=dict(task='Human_HEK293_TE',length_domain_codons=[80,300],seeds=SEEDS,arms=ARMS,
        counts={s:sum(r['split']==s for r in rows+tests) for s in ['train','val','test']},
        recipe=dict(max_optimizer_steps=STEPS,batch_size=2,validation_every=EVERY,optimizer='AdamW',lr=3e-4,weight_decay=.01,clip=1.,warmup=100,cosine_floor=.1,early_stop=False),
        initialization='All models random from scratch; no published weights loaded; identical full-backbone initialization across own ablations within seed',
        selection='Same validation Pearson selection every250steps; all arms reported, no test-based winner selection',
        loss='MSE of train-standardized native TE label; all models see same two rows per step',
        physics='Same frozen mouse train-proxy4-run simulations across branches; current own physics encoder receives2 aggregate features, not the original71 dimensions; seq2ribo uses log1p simulation mean, not its published per-run inference protocol',
        adaptations=dict(RiboGL='Official GCN_LSTM forward with Torch SAGE/GraphNorm backend matched to original saved PyG predictions and cuDNN disabled for LSTM; hook pre-linear256-dim nodes, mean/max pool and512->128->1 scalar head',
            seq2ribo='Official RiboPolisherMamba + official TEHeadFullCounts positive_output=False for standardized TE; explicit official Torch reference scan on CUDA',
            RiboDecode='Official CNN_Encoder_Model from scratch; constant original HEK293T context and RNA4.5; TE-supervised task adaptation, not native RPF-label training'),
        ablations=dict(no_physics='Remove simulated physics embedding from all scalar paths',no_graph='Remove graph and structural representation from fusion; retain sequence and physics',
            no_context='Bypass entire Transformer, pool its input; retain sequence/physics/graph fusion',seq_transformer='Remove both physics and graph, retain sequence Transformer'),
        limits=['Finite common-recipe short-CDS TE adaptation benchmark, not each original paper task or full-data optimum',
            '903-sample source test already opened historically; this filtered reuse is exploratory, not fresh independent confirmation',
            'Training samples, updates, batch and selection matched; FLOPs, wall time, capacity and feature availability differ',
            'No near-homology or cross-study split; no universal/SOTA superiority claim'],
        source_sha256={str(HERE/'models.py'):sha(HERE/'models.py'),str(Path(__file__)):sha(__file__),
            str(DATA/'train_validation.json'):sha(DATA/'train_validation.json'),str(DATA/'test_inputs.json'):sha(DATA/'test_inputs.json')})
    additional_sources=[
        WORK/'runs/profile_model_comparison_20261030/completed_snapshot/frozen_sources/ribogl/gnn_utils.py',
        WORK/'tmp/seq2ribo_source_20261020/seq2ribo-main/seq2ribo/models.py',
        WORK/'deliverables/public_riboseq_validation_20261028/source/inference.py',
        WORK/'deliverables/public_riboseq_validation_20261028/source/model_config.json',
        WORK/'runs/vmae_adjustment_gpu_20261024/transfer_assets/canonical_condition.pt',
        WORK/'repos/riboseq_human_te_release_20261102/physics_scorer/reliable_graph_context.py',
        WORK/'repos/riboseq_human_te_release_20261102/physics_scorer/human_te_model.py',
        WORK/'latent_decoder/seq2ribo_cpu_reference.py',
        WORK/'runs/mrfp_branch_probe_20261031/run_probe.py']
    protocol['source_sha256'].update({str(p):sha(p) for p in additional_sources})
    assert read(HERE/'ribogl_backend_parity.json')['status']=='passed'
    protocol['ribogl_backend_parity_sha256']=sha(HERE/'ribogl_backend_parity.json')
    protocol['source_sha256'][str(HERE/'verify_ribogl_backend.py')]=sha(HERE/'verify_ribogl_backend.py')
    protocol['previously_opened_test_targets_sha256']=sha(DATA/'sealed_test_targets.json')
    save(HERE/'protocol_locked.json',protocol)
    # Exact official 32-step random-walk return probabilities, cached once per graph.
    from scipy import sparse
    import itertools
    codons=[''.join(x) for x in itertools.product('ATCG',repeat=3)];hashes={}
    for i,r in enumerate(rows+tests,1):
        path=HERE/'graphs'/f'{r["uid"]}.npz'
        if not path.exists():
            with np.load(feature_path(r['uid']),allow_pickle=False) as z:f={k:z[k].copy() for k in z.files}
            n=len(f['cod']);a=np.array([np.arange(n-1),np.arange(1,n)]);c=f['edges']
            edges=np.unique(np.concatenate((c,c[::-1].copy(),a,a[::-1].copy()),axis=1),axis=1)
            matrix=sparse.csr_matrix((np.ones(edges.shape[1]),(edges[0],edges[1])),shape=(n,n))
            degree=np.asarray(matrix.sum(1)).ravel();matrix=sparse.diags(1/np.maximum(degree,1))@matrix
            power=matrix.copy();pe=[power.diagonal()]
            for _ in range(31):power=power@matrix;pe.append(power.diagonal())
            ids=np.array([codons.index(r['dna'][j:j+3]) for j in range(0,len(r['dna']),3)],dtype=np.int64)
            np.savez_compressed(path,ribogl_ids=ids,ribogl_edges=edges,ribogl_pe=np.stack(pe,axis=1).astype(np.float32))
        hashes[r['uid']]=dict(features=sha(feature_path(r['uid'])),graph=sha(path))
        if i%100==0:
            save(HERE/'prepare_status.json',dict(completed=i,total=len(rows+tests)));print('prepare',i,len(rows+tests),flush=True)
    save(HERE/'inputs_locked.json',dict(status='completed',protocol_sha256=sha(HERE/'protocol_locked.json'),files=hashes,
        train_validation_sha256=sha(HERE/'train_validation.json'),test_inputs_sha256=sha(HERE/'test_inputs.json')))
def load(rows):
    manifest=read(HERE/'inputs_locked.json');features={}
    for r in rows:
        path=feature_path(r['uid']);gp=HERE/'graphs'/f'{r["uid"]}.npz'
        assert sha(path)==manifest['files'][r['uid']]['features'] and sha(gp)==manifest['files'][r['uid']]['graph']
        with np.load(path,allow_pickle=False) as z:f={k:z[k].copy() for k in z.files}
        with np.load(gp,allow_pickle=False) as z:f.update({k:z[k].copy() for k in z.files})
        features[r['uid']]=f
    return features
def batch(rows,features):
    from importlib.util import spec_from_file_location,module_from_spec
    if not hasattr(batch,'probe'):
        s=spec_from_file_location('fair_batch',WORK/'runs/mrfp_branch_probe_20261031/run_probe.py');m=module_from_spec(s);s.loader.exec_module(m);batch.probe=m
    return batch.probe.batch(rows,features,'cuda')[0]
def configure():
    torch.set_num_threads(2);assert torch.cuda.is_available();torch.cuda.set_per_process_memory_fraction(.65)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.mha.set_fastpath_enabled(False)
def corr(a,b):
    a=np.asarray(a,dtype=float);b=np.asarray(b,dtype=float)
    return None if np.ptp(a)==0 or np.ptp(b)==0 else float(np.corrcoef(a,b)[0,1])
@torch.no_grad()
def predictions(model,rows,features,mean,scale):
    model.eval();values=[]
    for start in range(0,len(rows),2):
        group=rows[start:start+2];v=model(group,features,batch(group,features)).cpu().numpy()*scale+mean
        assert np.isfinite(v).all();values.extend(v.tolist())
    return np.array(values)
def preflight(arm):
    configure();torch.manual_seed(30101);rows=[r for r in read(HERE/'train_validation.json') if r['split']=='train']
    rows=sorted(rows,key=lambda r:-r['n_codons'])[:2];features=load(rows);model=build(arm).cuda();x=batch(rows,features)
    model.eval();start=time.monotonic()
    with torch.no_grad():p=model(rows,features,x)
    assert p.shape==(2,) and torch.isfinite(p).all()
    model.train();training_output=model(rows,features,x);assert torch.isfinite(training_output).all();training_output.square().mean().backward();model.eval()
    total_gradient=sum(float(q.grad.abs().sum()) for q in model.parameters() if q.grad is not None);assert total_gradient>0
    grads={name:sum(float(q.grad.abs().sum()) for n,q in model.named_parameters() if n.startswith(name) and q.grad is not None) for name in ['backbone.physics.','backbone.graph.','backbone.context.']}
    if arm=='full':
        assert all(v>0 for v in grads.values())
        from physics_scorer.human_te_model import HumanTEScalar
        reference=HumanTEScalar(dict(width=192,sequence_v2=True,structure_gate=True,anchored_head=False,physics=True)).cuda().eval()
        reference.backbone.load_state_dict(model.backbone.state_dict(),strict=True);reference.head.load_state_dict(model.head.state_dict(),strict=True)
        with torch.no_grad():error=float((reference(x)-p).abs().max())
        assert error<2e-5
    else:error=None
    if arm=='no_physics':assert grads['backbone.physics.']==0
    if arm=='no_graph':assert grads['backbone.graph.']==0
    if arm=='no_context':assert grads['backbone.context.']==0
    if arm=='seq_transformer':assert grads['backbone.graph.']==grads['backbone.physics.']==0
    save(HERE/'preflight'/f'{arm}.json',dict(status='passed',arm=arm,largest_codons=[r['n_codons'] for r in rows],
        seconds=time.monotonic()-start,gradient_totals=grads,total_gradient=total_gradient,full_vs_portable_scalar_max_error=error,
        parameters=sum(q.numel() for q in model.parameters()),cuda_peak_bytes=torch.cuda.max_memory_allocated(),torch=str(torch.__version__),cuda=torch.version.cuda,cudnn=torch.backends.cudnn.version()))
def train(arm,seed):
    configure();torch.manual_seed(seed);torch.cuda.manual_seed_all(seed);np.random.seed(seed);random.seed(seed)
    protocol=read(HERE/'protocol_locked.json')
    for path,digest in protocol['source_sha256'].items():assert sha(path)==digest,path
    out=HERE/'models'/f'{arm}_{seed}';out.mkdir(parents=True,exist_ok=True)
    if (out/'complete.json').exists():return
    rows=read(HERE/'train_validation.json');features=load(rows);tr=[r for r in rows if r['split']=='train'];val=[r for r in rows if r['split']=='val']
    mean=float(np.mean([r['target'] for r in tr]));scale=float(np.std([r['target'] for r in tr]));assert scale>0
    model=build(arm).cuda();optimizer=torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=.01)
    state=dict(step=0,epoch=0,pointer=0,best=-2.,history=[],seen=0)
    if (out/'latest.pt').exists():
        ck=torch.load(out/'latest.pt',map_location='cpu',weights_only=False);assert ck['protocol_sha256']==sha(HERE/'protocol_locked.json')
        model.load_state_dict(ck['model'],strict=True);optimizer.load_state_dict(ck['optimizer']);state=ck['training_state']
        torch.set_rng_state(ck['rng']);torch.cuda.set_rng_state_all(ck['cuda_rng'])
    def order(epoch):return np.random.default_rng(seed+epoch).permutation(len(tr)).tolist()
    indices=order(state['epoch']);begun=time.monotonic()
    for step in range(state['step']+1,STEPS+1):
        ids=[]
        for _ in range(2):
            if state['pointer']==len(indices):state['epoch']+=1;state['pointer']=0;indices=order(state['epoch'])
            ids.append(indices[state['pointer']]);state['pointer']+=1
        group=[tr[i] for i in ids];model.train();optimizer.zero_grad(set_to_none=True)
        factor=min(1.,step/100)*(.1+.45*(1+math.cos(math.pi*max(0,step-100)/(STEPS-100))))
        for param in optimizer.param_groups:param['lr']=3e-4*factor
        pred=model(group,features,batch(group,features));truth=torch.tensor([(r['target']-mean)/scale for r in group],device='cuda',dtype=torch.float32)
        loss=(pred-truth).square().mean();assert torch.isfinite(loss);loss.backward()
        grad=torch.nn.utils.clip_grad_norm_(model.parameters(),1.);assert torch.isfinite(grad);optimizer.step();state['seen']+=2
        if step%EVERY==0 or step==STEPS:
            p=predictions(model,val,features,mean,scale);r=corr([x['target'] for x in val],p);assert r is not None
            state['step']=step;state['history'].append(dict(step=step,validation_pearson=r,validation_mse=float(np.mean((p-np.array([x['target'] for x in val]))**2))))
            improved=r>state['best']
            if improved:state['best']=r
            payload=dict(model={k:v.detach().cpu().clone() for k,v in model.state_dict().items()},arm=arm,seed=seed,step=step,
                validation_pearson=r,validation_predictions=p.tolist(),target_mean=mean,target_scale=scale,protocol_sha256=sha(HERE/'protocol_locked.json'))
            if improved:torch.save(payload,out/'best.pt')
            torch.save(dict(**payload,optimizer=optimizer.state_dict(),training_state=copy.deepcopy(state),rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all()),out/'latest.pt')
            save(out/'history.json',state['history']);save(out/'status.json',dict(step=step,max_steps=STEPS,best=state['best'],validation_pearson=r,elapsed_seconds=time.monotonic()-begun))
            print(arm,seed,step,r,flush=True)
    save(out/'complete.json',dict(status='completed',arm=arm,seed=seed,steps=STEPS,seen=state['seen'],best_validation=state['best'],
        checkpoint_sha256=sha(out/'best.pt'),parameters=sum(q.numel() for q in model.parameters()),elapsed_seconds=time.monotonic()-begun))
if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['prepare','preflight','train']);parser.add_argument('--arm',choices=ARMS);parser.add_argument('--seed',type=int,default=30101);args=parser.parse_args()
    try:
        if args.mode=='prepare':prepare()
        elif args.mode=='preflight':preflight(args.arm)
        else:train(args.arm,args.seed)
    except BaseException as e:
        save(HERE/'failures'/f'{args.mode}_{args.arm}_{args.seed}.json',dict(error=repr(e),unix=time.time()));raise
