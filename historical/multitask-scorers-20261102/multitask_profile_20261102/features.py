"""Label-free sequence/dynamics cache with training-only input normalization."""
from pathlib import Path
import os,json,hashlib,itertools,time
os.environ.setdefault('OMP_NUM_THREADS','1');os.environ.setdefault('MKL_NUM_THREADS','1')
import numpy as np
import sys
from collections import Counter
from Bio.Seq import Seq
import dynamics_v2 as dynamics
HERE=Path(__file__).resolve().parent;WORK=HERE.parents[1];CACHE=HERE/'features_v2';CACHE.mkdir(exist_ok=True)
sys.path.insert(0,str(WORK/'.runtime/viennarna_2_7_2'))
import RNA
assert RNA.__version__=='2.7.2'
AA='ACDEFGHIKLMNPQRSTVWY';AI={a:i for i,a in enumerate(AA)}
KMERS=[''.join(x) for k in [1,2,3] for x in itertools.product('ACGT',repeat=k)]
def read(p):return json.loads(Path(p).read_text(encoding='utf8'))
def save(p,v):
    p=Path(p);t=p.with_suffix('.tmp');t.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf8');t.replace(p)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def records(path):
    with Path(path).open(encoding='utf8') as f:return [json.loads(line) for line in f]
def composition(s):
    s=s.upper().replace('U','T');cs={k:Counter(s[i:i+k] for i in range(len(s)-k+1)) for k in [1,2,3]}
    return [cs[len(k)][k]/max(1,len(s)-len(k)+1) for k in KMERS]
def base_features(r):
    dna=r['dna'];cod=np.array([dynamics.CODON_INDEX[dna[i:i+3]] for i in range(0,len(dna),3)],dtype=np.int64)
    aa=r['aa'];am=np.array([AI.get(a,20) for a in aa]+[20],dtype=np.int64)
    if len(am)!=len(cod):raise ValueError('CDS/protein length mismatch')
    stats=[]
    for region in [r['utr5'],dna,r['utr3']]:stats.extend(composition(region))
    stats.extend(aa.count(a)/max(1,len(aa)) for a in AA)
    stats.extend([np.log1p(len(dna)),np.log1p(len(r['utr5'])),np.log1p(len(r['utr3'])),np.log1p(len(aa))])
    stats.extend(sum(b in 'GC' for b in dna[i:i+30])/max(1,len(dna[i:i+30])) for i in [0,30,60])
    # Short-window RNAfold values are computed from sequence, never labels.
    for seq in [dna[:n] for n in [30,60,90,150]]+[r['utr5'][-80:]+dna[:120],r['utr5'][:150]]:
        stats.append(float(RNA.fold(seq.replace('T','U'))[1])/len(seq) if seq else 0.)
    utrs=[]
    for seq in [r['utr5'],r['utr3']]:
        seq=seq.upper().replace('U','T')
        # Full-region composition is kept; sequence encoder retains both ends.
        if len(seq)>1024:seq=seq[:512]+seq[-512:]
        utrs.append(np.array([{'A':0,'C':1,'G':2,'T':3}.get(b,4) for b in seq],dtype=np.int64))
    return dict(codons=cod,amino=am,stats=np.array(stats,dtype=np.float32),utrs=utrs)
def cache_path(r):return CACHE/(r['host']+'_'+r['uid']+'.npz')
def geometry(dna):
    seq=dna[:150].replace('T','U');bracket,_=RNA.fold(seq)
    paired=np.zeros(len(dna)//3,dtype=np.int64)
    for i in range(len(bracket)//3):paired[i]=sum(c!='.' for c in bracket[3*i:3*i+3])
    return dict(paired=paired,abin=np.zeros_like(paired))
def prepare():
    rows=records(HERE/'records.jsonl');unique={(r['host'],r['uid']):r for r in rows};total=len(unique)
    parameters={h:read(HERE/(h+'_parameters.json')) for h in ['human','ecoli']}
    assert all(p['host']==h and p['status'].startswith('train_only_relative_occupancy_proxy') for h,p in parameters.items())
    protocol=dict(source_sha256={str(p.name):sha(p) for p in [HERE/'dynamics_v2.py',Path(__file__),HERE/'data_protocol_locked.json',HERE/'human_parameters.json',HERE/'ecoli_parameters.json']},
        parameters='New observed host-specific RPF train-only relative-occupancy proxy; not identified true kinetics; no mouse transfer',
        seed=20261102,n_runs=2,burn_steps=8192,observe_steps=4096,init_ps=[.003,.01,.03],
        cache='host+DNA SHA; each cache verifies its provenance hash; all sequence features label-free',
        labels_used_for_input_scaling='train records only, no target values',
        geometry='RNAfold2.7.2 first150nt CDS dotbracket paired-base count per codon; remaining positions zero and angles missing0, not claimed3D measurements',
        fold_statistics='MFE/nt of CDS first30/60/90/150nt; last80nt UTR5+first120nt CDS; first150nt UTR5; both arms receive same6 features',
        utr_encoder='First512+last512 bases if >1024; full UTR1-3mer composition retained; missing UTR explicit')
    lock=HERE/'features_v2_protocol_locked.json'
    if lock.exists():assert read(lock)==protocol,'Feature protocol changed'
    else:save(lock,protocol)
    digest=sha(lock);start=time.monotonic()
    for i,r in enumerate(unique.values(),1):
        path=cache_path(r)
        if path.exists():
            with np.load(path,allow_pickle=False) as z:assert str(z['protocol_sha256'])==digest
        else:
            v=dynamics.extract(r['dna'],r['host'],params=parameters[r['host']],geometry=geometry(r['dna']),seed=20261102,n_runs=2,burn_steps=8192,observe_steps=4096)
            temporary=path.with_suffix('.tmp')
            with temporary.open('wb') as f:np.savez_compressed(f,node=v['node'],global_features=v['global'],
                protocol_sha256=digest,audit_json=json.dumps(v['audit']),dna_sha256=r['uid'])
            temporary.replace(path)
        if i%100==0 or i==total:
            save(HERE/'features_status.json',dict(completed=i,total=total,elapsed=time.monotonic()-start));print('features',i,total,flush=True)
    # Normalize by training examples/positions only; constants keep scale1.
    train=[r for r in rows if r['split']=='train'];s=[];g=[];count=0;psum=np.zeros(12);psq=np.zeros(12)
    for r in train:
        s.append(base_features(r)['stats'])
        with np.load(cache_path(r),allow_pickle=False) as z:
            node=z['node'].astype(float);g.append(z['global_features']);psum+=node.sum(0);psq+=(node**2).sum(0);count+=len(node)
    s=np.array(s,dtype=float);g=np.array(g,dtype=float);nm=psum/count;ns=np.sqrt(np.maximum(0,psq/count-nm**2))
    def norm(x):
        mean=x.mean(0);scale=x.std(0);return mean,np.where(scale<1e-6,1.,scale)
    sm,ss=norm(s);gm,gs=norm(g);ns=np.where(ns<1e-6,1.,ns)
    temp=HERE/'input_scaler.tmp'
    with temp.open('wb') as handle:np.savez(handle,stats_mean=sm,stats_scale=ss,global_mean=gm,global_scale=gs,node_mean=nm,node_scale=ns)
    temp.replace(HERE/'input_scaler.npz')
    save(HERE/'features_complete.json',dict(status='completed',unique_sequences=total,train_records=len(train),stats_dim=s.shape[1],protocol_sha256=digest,scaler_sha256=sha(HERE/'input_scaler.npz')))
def load(rows):
    with np.load(HERE/'input_scaler.npz',allow_pickle=False) as z:sc={k:z[k].copy() for k in z.files}
    bank={}
    for r in rows:
        b=base_features(r)
        with np.load(cache_path(r),allow_pickle=False) as z:
            assert str(z['dna_sha256'])==r['uid'];node=z['node'].copy();glob=z['global_features'].copy()
        b.update(physics=((node-sc['node_mean'])/sc['node_scale']).astype(np.float32),physical_global=((glob-sc['global_mean'])/sc['global_scale']).astype(np.float32),
            stats=((b['stats']-sc['stats_mean'])/sc['stats_scale']).astype(np.float32),physics_calibrated=1.)
        bank[r['record_id']]=b
    return bank
if __name__=='__main__':prepare()
