"""Fixed-window sTASEP event observables, in simulation steps rather than seconds.

This is a new, bounded observation model. It deliberately does not claim
numerical parity with the original completion-triggered snapshot simulator.
"""
from pathlib import Path
import hashlib,itertools,json
import numpy as np
from numba import njit
from Bio.Data.CodonTable import unambiguous_dna_by_id

CODONS=[''.join(x) for x in itertools.product('ACGT',repeat=3)]
CODON_INDEX={c:i for i,c in enumerate(CODONS)}
STOP={'TAA','TAG','TGA'}
HOSTS={'human','ecoli'}
ECOLI_START_CODONS=frozenset(unambiguous_dna_by_id[11].start_codons)
NODE_NAMES=[f'{name}_init{p:g}' for p in [.003,.01,.03]
            for name in ['occupancy_fraction','movement_per_step','blocked_per_step']]+[
                'codon_wait','structure_wait','movement_probability']
SUMMARY_BASE=['occupancy_mean','occupancy_sd','completion_per_step',
              'initiation_success_per_step','initiation_attempt_per_step',
              'initiation_blocked_fraction','movement_per_step',
              'movement_blocked_fraction','occupancy_first30','occupancy_last30',
              'occupancy_peak15','completion_run_sd']
GLOBAL_NAMES=[f'{name}_init{p:g}' for p in [.003,.01,.03] for name in SUMMARY_BASE]

def neutral_parameters(host):
    if host not in HOSTS:raise ValueError('host must be human or ecoli')
    return dict(host=host,status='neutral_assumption_not_host_calibrated',
        source='Equal sense-codon waits; no public or mouse fitted parameters',
        waits=[2.]*64,pair=[0.]*4,angle=[0.]*4,bucket=[0.]*3,
        units='arbitrary_simulation_steps',initiation_calibrated=False)

def _prepare(dna,host,params,geometry):
    dna=str(dna).upper().replace('U','T')
    if host not in HOSTS:raise ValueError('host must be human or ecoli')
    if len(dna)<6 or len(dna)%3 or set(dna)-set('ACGT'):
        raise ValueError('Expected unambiguous complete in-frame CDS')
    codons=[dna[i:i+3] for i in range(0,len(dna),3)]
    # Table11 includes rare valid starts used by annotated reference constructs.
    if codons[0] not in ({'ATG'} if host=='human' else ECOLI_START_CODONS):
        raise ValueError('Unsupported initiation codon for host')
    if codons[-1] not in STOP or any(c in STOP for c in codons[:-1]):
        raise ValueError('Expected one terminal stop and no internal stop')
    params=neutral_parameters(host) if params is None else params
    if params.get('host')!=host:raise ValueError('Dynamics parameters belong to another host')
    if not params.get('status') or not params.get('source'):
        raise ValueError('Parameter provenance and calibration status required')
    vectors=[np.asarray(params[k],dtype=np.float64) for k in ['waits','pair','angle','bucket']]
    if any(v.shape!=(n,) or not np.isfinite(v).all() or (v<0).any()
           for v,n in zip(vectors,[64,4,4,3])):
        raise ValueError('Expected nonnegative finite64/4/4/3 wait vectors')
    ids=np.array([CODON_INDEX[c] for c in codons],dtype=np.int64);L=len(ids)
    paired=np.zeros(L,dtype=np.int64);angle=np.zeros(L,dtype=np.int64)
    if geometry is not None:
        paired=np.asarray(geometry['paired'],dtype=np.int64)
        angle=np.asarray(geometry['abin'],dtype=np.int64)
        if paired.shape!=(L,) or angle.shape!=(L,) or any((v<0).any() or (v>3).any() for v in [paired,angle]):
            raise ValueError('Geometry must align to complete CDS and use categories0..3')
        if 'cod' in geometry and not np.array_equal(ids,geometry['cod']):
            raise ValueError('Geometry uses another sequence or codon order')
    positions=np.arange(L);bucket=np.where(positions<L//3,0,np.where(positions<2*L//3,1,2))
    base=vectors[0][ids];structure=vectors[1][paired]+vectors[2][angle]+vectors[3][bucket]
    total=base+structure
    if (total[:-1]<=0).any():raise ValueError('Sense-codon total waits must be positive')
    probability=np.minimum(1.,1./np.maximum(total,1e-12))
    base[-1]=structure[-1]=probability[-1]=0.
    return dna,params,base,structure,probability

@njit(cache=True)
def _run(probability,init_p,burn,observe,exclusion,seed):
    np.random.seed(seed)
    L=len(probability);positions=np.zeros(L,dtype=np.int64);count=0
    occupancy=np.zeros(L,dtype=np.int64);moves=np.zeros(L,dtype=np.int64);blocked=np.zeros(L,dtype=np.int64)
    attempted=np.zeros(L,dtype=np.int64);entries=0;exits=0;init_attempts=0;init_blocked=0
    start_count=0
    for step in range(burn+observe):
        observed=step>=burn
        if step==burn:start_count=count
        if observed:
            for i in range(count):occupancy[positions[i]]+=1
        if np.random.random()<init_p:
            if observed:init_attempts+=1
            # Match original injection clearance; movement keeps >=exclusion spacing.
            clear=count==0 or positions[count-1]>=exclusion+1
            if clear:
                positions[count]=0;count+=1
                if observed:entries+=1
            elif observed:init_blocked+=1
        # Stored downstream-first. Downstream updates precede upstream updates.
        i=0
        while i<count:
            pos=positions[i]
            if np.random.random()<probability[pos]:
                if observed:attempted[pos]+=1
                if pos==L-2:
                    for j in range(i,count-1):positions[j]=positions[j+1]
                    count-=1
                    if observed:exits+=1
                    continue
                clear=i==0 or positions[i-1]-(pos+1)>=exclusion
                if clear:
                    positions[i]=pos+1
                    if observed:moves[pos]+=1
                elif observed:blocked[pos]+=1
            i+=1
    min_gap=L
    for i in range(1,count):min_gap=min(min_gap,positions[i-1]-positions[i])
    return occupancy,moves,blocked,attempted,np.array([
        start_count,count,entries,exits,init_attempts,init_blocked,min_gap],dtype=np.int64)

def extract(dna,host,params=None,geometry=None,seed=20261102,init_ps=(.003,.01,.03),
            n_runs=1,burn_steps=8192,observe_steps=4096,exclusion=3):
    """Return node[L,12], global[36], audit; no train/test labels consumed.

    Occupancy is the fraction of fixed observation steps with an A-site at
    a codon. Current is completed ribosomes/observation steps, never per second.
    Fixed windows can be transient for long/slow sequences; audit records it.
    """
    if not isinstance(n_runs,int) or n_runs<1 or not isinstance(burn_steps,int) or burn_steps<0 or not isinstance(observe_steps,int) or observe_steps<1:
        raise ValueError('Positive integer run/observation budget and nonnegative burn required')
    if exclusion<1 or int(exclusion)!=exclusion:raise ValueError('Positive integer exclusion required')
    if tuple(init_ps)!=(.003,.01,.03):raise ValueError('Fixed three-rate schema requires(.003,.01,.03)')
    dna,params,base,structure,probability=_prepare(dna,host,params,geometry)
    nodes=[];summaries=[];audits=[]
    for k,init in enumerate(init_ps):
        occs=[];ms=[];bs=[];attempts=[];counts=[]
        for run in range(n_runs):
            # Same supplied seed across synonymous sequences supports common RNG seeds.
            run_seed=(int(seed)+1009*k+104729*run)%(2**32-1)
            occ,m,b,a,c=_run(probability,float(init),burn_steps,observe_steps,int(exclusion),run_seed)
            if c[0]+c[2]-c[3]!=c[1]:raise AssertionError('Ribosome conservation failed')
            if c[6]<exclusion or occ[-1] or m[-1] or b[-1]:raise AssertionError('Exclusion/stop invariant failed')
            if not np.array_equal(a,m+b+np.r_[np.zeros(len(probability)-2,dtype=np.int64),c[3],0]):
                raise AssertionError('Movement-attempt accounting failed')
            occs.append(occ/observe_steps);ms.append(m/observe_steps);bs.append(b/observe_steps);attempts.append(a);counts.append(c)
        occ=np.mean(occs,axis=0);mov=np.mean(ms,axis=0);block=np.mean(bs,axis=0);c=np.sum(counts,axis=0);a=np.sum(attempts,axis=0)
        nodes.extend([occ,mov,block]);window=min(15,len(occ))
        values=[occ.mean(),occ.std(),c[3]/(n_runs*observe_steps),c[2]/(n_runs*observe_steps),
                c[4]/(n_runs*observe_steps),c[5]/max(1,c[4]),mov.sum(),
                np.sum(bs)*observe_steps/max(1,a.sum()),occ[:30].mean(),occ[-30:].mean(),
                np.convolve(occ,np.ones(window)/window,mode='valid').max(),
                np.std([q[3]/observe_steps for q in counts])]
        summaries.extend(values)
        audits.append(dict(init_p=init,conservation_per_run=[q.tolist() for q in counts],
                           completion_events=int(c[3]),observed_steps_per_run=observe_steps))
    nodes.extend([base,structure,probability]);node=np.stack(nodes,axis=1).astype(np.float32);glob=np.array(summaries,dtype=np.float32)
    if node.shape!=(len(probability),12) or glob.shape!=(36,) or not np.isfinite(node).all() or not np.isfinite(glob).all():raise AssertionError('Invalid output schema')
    return dict(node=node,**{'global':glob},audit=dict(host=host,parameter_status=params['status'],parameter_source=params['source'],
        parameter_sha256=hashlib.sha256(json.dumps(params,sort_keys=True).encode()).hexdigest(),
        canonical_dna_sha256=hashlib.sha256(dna.encode()).hexdigest(),geometry_provided=geometry is not None,
        seed=int(seed),runs=n_runs,burn_steps=burn_steps,observe_steps=observe_steps,exclusion=int(exclusion),
        observables=audits,terminal_model='Last sense codon completes according to its own wait; terminal stop never occupied',
        fixed_window_may_be_transient=burn_steps<float(np.sum(1./np.maximum(probability[:-1],1e-12))),
        units='simulation_step_not_seconds',legacy_snapshot_parity_claimed=False))

def fit_profile_parameters(records,host,source):
    """Fit a nonnegative relative-occupancy proxy from explicit train RPF only.

    This statistical proxy does not identify true codon waiting times.
    Expected records: {split:'train', host, dna, position_rpf:[L], optional
    geometry:{paired:[L],abin:[L]}}. Nontrain records are rejected, not ignored.
    """
    from scipy.optimize import nnls
    if host not in HOSTS or not source or not records:raise ValueError('Host, source and nonempty training records required')
    gram=np.zeros((75,75));rhs=np.zeros(75);uids=[];positions_used=0
    for record in records:
        if record.get('split')!='train' or record.get('host')!=host:raise ValueError('Only exact host train records accepted')
        dna,_,_,_,_=_prepare(record['dna'],host,None,record.get('geometry'))
        cod=np.array([CODON_INDEX[dna[i:i+3]] for i in range(0,len(dna),3)]);L=len(cod)
        y=np.asarray(record['position_rpf'],dtype=float)
        if y.shape!=(L,):raise ValueError('Position label/complete CDS mismatch')
        mask=np.isfinite(y)&(y>=0);mask[-1]=False
        if mask.sum()<10 or y[mask].mean()<=0:continue
        geo=record.get('geometry');pair=np.zeros(L,dtype=int) if geo is None else np.asarray(geo['paired'],dtype=int)
        angle=np.zeros(L,dtype=int) if geo is None else np.asarray(geo['abin'],dtype=int)
        pos=np.arange(L);bucket=np.where(pos<L//3,0,np.where(pos<2*L//3,1,2))
        design=np.zeros((L,75))
        for col in [cod,64+pair,68+angle,72+bucket]:design[pos,col]=1.
        a=design[mask];target=np.clip(y[mask]/y[mask].mean(),0,10);gram+=a.T@a;rhs+=a.T@target
        positions_used+=int(mask.sum());uids.append(hashlib.sha256(dna.encode()).hexdigest())
    if not positions_used:raise ValueError('No usable observed training positions')
    ridge=25.;prior=np.r_[np.full(64,2.),np.zeros(11)];gram+=ridge*np.eye(75);rhs+=ridge*prior
    chol=np.linalg.cholesky(gram);weights,_=nnls(chol.T,np.linalg.solve(chol,rhs),maxiter=10000)
    weights[:64]=np.maximum(weights[:64],.05)
    return dict(host=host,status='train_only_relative_occupancy_proxy_not_identified_kinetics',source=source,
                waits=weights[:64].tolist(),pair=weights[64:68].tolist(),angle=weights[68:72].tolist(),bucket=weights[72:].tolist(),
                units='arbitrary_simulation_steps',initiation_calibrated=False,ridge=ridge,
                training_dna_sha256=sorted(set(uids)),training_positions=positions_used)
