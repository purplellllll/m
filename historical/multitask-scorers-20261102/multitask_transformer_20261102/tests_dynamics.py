"""Determinism, accounting, exclusion, provenance and bounded runtime checks."""
from pathlib import Path
import json,time,hashlib
import numpy as np
from dynamics import extract,neutral_parameters,fit_profile_parameters,NODE_NAMES,GLOBAL_NAMES

HERE=Path(__file__).resolve().parent
def reject(call):
    try:call()
    except ValueError:return
    raise AssertionError('Expected ValueError')
def main():
    tests=[];dna='ATG'+'GCT'*99+'TAA'
    args=dict(seed=71,n_runs=2,burn_steps=128,observe_steps=512)
    start=time.perf_counter();a=extract(dna,'human',**args);compile_seconds=time.perf_counter()-start
    b=extract(dna,'human',**args)
    np.testing.assert_array_equal(a['node'],b['node']);np.testing.assert_array_equal(a['global'],b['global'])
    assert a['audit']==b['audit'];tests.append('bitwise repeated-seed determinism')
    assert a['node'].shape==(101,12) and a['global'].shape==(36,) and len(NODE_NAMES)==12 and len(GLOBAL_NAMES)==36
    assert np.isfinite(a['node']).all() and np.isfinite(a['global']).all()
    assert np.all(a['node'][-1]==0);tests.append('schema, finite outputs, terminal stop zero')
    for p in a['audit']['observables']:
        for c in p['conservation_per_run']:
            assert c[0]+c[2]-c[3]==c[1] and c[6]>=3
            assert c[4]==c[2]+c[5]
    tests.append('entry/exit conservation, minimum spacing, attempted initiation accounting')
    slow=neutral_parameters('human');slow['waits']=[100.]*64
    jam=extract('ATG'+'GCT'*12+'TAA','human',params=slow,seed=71,burn_steps=1000,observe_steps=10000)
    assert sum(p['completion_events'] for p in jam['audit']['observables'])>=0
    assert jam['node'][:,[2,5,8]].sum()>0;tests.append('actual movement-blocking events observed in congested chain')
    # Neutral codon waits should not manufacture synonymous biology.
    c=extract('ATG'+'GCC'*99+'TAA','human',**args)
    np.testing.assert_array_equal(a['node'],c['node']);np.testing.assert_array_equal(a['global'],c['global'])
    tests.append('neutral synonymous sequences share dynamics under same RNG seed')
    reject(lambda:extract(dna,'ecoli',params=neutral_parameters('human')))
    reject(lambda:extract('ATGGCT','human'))
    reject(lambda:extract('ATGTAGGCTTAA','human'))
    reject(lambda:extract(dna,'human',observe_steps=0))
    reject(lambda:extract(dna,'human',geometry=dict(paired=np.zeros(5),abin=np.zeros(5))))
    bad=neutral_parameters('human');bad['waits'][0]=float('nan')
    reject(lambda:extract(dna,'human',params=bad))
    tests.append('cross-host/invalid CDS/budget/geometry/nonfinite parameters rejected')
    ecoli=extract('GTG'+'GCT'*10+'TAA','ecoli',burn_steps=0,observe_steps=100)
    assert ecoli['node'].shape==(12,12);tests.append('valid bacterial alternative start accepted')
    record=dict(split='train',host='human',dna=dna,position_rpf=np.r_[np.ones(100),0.])
    params=fit_profile_parameters([record],'human','synthetic_test_only')
    assert params['training_positions']==100 and params['host']=='human'
    reject(lambda:fit_profile_parameters([dict(record,split='test')],'human','synthetic_test_only'))
    reject(lambda:fit_profile_parameters([dict(record,host='ecoli')],'human','synthetic_test_only'))
    tests.append('fit API rejects validation/test and mismatched-host supervision')
    perf=[]
    for length in [100,500,1500]:
        s='ATG'+'GCT'*(length-2)+'TAA';start=time.perf_counter()
        for seed in range(10):extract(s,'human',seed=seed)
        perf.append(dict(codons=length,seconds_per_sequence=(time.perf_counter()-start)/10,
                         runs=1,initiation_rates=3,burn=8192,observe=4096))
    convergence=[]
    for length in [100,500,1500]:
        s='ATG'+'GCT'*(length-2)+'TAA';runs={}
        for count in [1,2,8]:
            start=time.perf_counter();runs[count]=extract(s,'human',seed=71,n_runs=count)
            runs[count]['seconds']=time.perf_counter()-start
        for count in [1,2,8]:
            r=runs[count];reference=runs[8]
            convergence.append(dict(codons=length,runs=count,seconds=r['seconds'],
                occupancy_MAE_vs_8run_mean=float(np.mean(np.abs(r['node'][:,[0,3,6]]-reference['node'][:,[0,3,6]]))),
                current_by_init=r['global'][[2,14,26]].tolist(),
                caution='8-run reference is another Monte Carlo estimate, not ground truth; MAE need not improve monotonically'))
    source=HERE.parents[1]/'external/seq2ribo_66c80c6/seq2ribo/simulation.py'
    result=dict(status='passed',checks=tests,compile_seconds=compile_seconds,benchmark=perf,convergence=convergence,
                legacy_simulator_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                legacy_parity='Not claimed: fixed observation window and wait-aware last-sense completion deliberately change semantics',
                species_calibration='No certified human/Ecoli train positional-RPF source found in local audited panels; neutral is explicit control, not repair of biological calibration')
    (HERE/'dynamics_verification.json').write_text(json.dumps(result,indent=2,allow_nan=False),encoding='utf8')
    print(json.dumps(result,indent=2))
if __name__=='__main__':main()
