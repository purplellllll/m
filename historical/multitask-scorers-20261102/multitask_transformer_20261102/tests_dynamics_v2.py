"""Table11 initiation coverage and bitwise parity with frozen v1."""
from pathlib import Path
import hashlib,json
import numpy as np
from Bio.Data.CodonTable import unambiguous_dna_by_id
import dynamics as v1
import dynamics_v2 as v2

HERE=Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    assert sha(HERE/'dynamics.py')=='a2d90918be662aea960c3a465c9235e62598e13f213bb78ab485dfed126aa604'
    official=set(unambiguous_dna_by_id[11].start_codons)
    assert v2.ECOLI_START_CODONS==official
    for start in sorted(official):
        r=v2.extract(start+'GCT'*30+'TAA','ecoli',seed=71,n_runs=2,burn_steps=128,observe_steps=256)
        assert r['node'].shape==(32,12) and r['global'].shape==(36,) and not r['node'][-1].any()
    parity=[]
    for host,starts in [('human',['ATG']),('ecoli',['ATG','GTG','TTG'])]:
        for start in starts:
            dna=start+'GCT'*30+'TAA';params=v1.neutral_parameters(host)
            params['waits']=np.linspace(.8,3.,64).tolist()
            args=dict(params=params,seed=71,n_runs=2,burn_steps=128,observe_steps=256)
            a=v1.extract(dna,host,**args);b=v2.extract(dna,host,**args)
            np.testing.assert_array_equal(a['node'],b['node']);np.testing.assert_array_equal(a['global'],b['global']);assert a['audit']==b['audit']
            parity.append(dict(host=host,start=start))
    for host,start in [('human','GTG'),('ecoli','TGC')]:
        try:v2.extract(start+'GCT'*30+'TAA',host,burn_steps=0,observe_steps=1)
        except ValueError:pass
        else:raise AssertionError('Invalid host start accepted')
    result=dict(status='passed',official_table=11,official_starts=sorted(official),
        legacy_supported_starts_bitwise_parity=parity,invalid_host_starts_rejected=True,
        dynamics_v1_sha256=sha(HERE/'dynamics.py'),dynamics_v2_sha256=sha(HERE/'dynamics_v2.py'),
        scope='Only E.coli initiation validation expanded; event kernel and all output semantics unchanged')
    (HERE/'dynamics_v2_verification.json').write_text(json.dumps(result,indent=2,allow_nan=False),encoding='utf8')
    print(json.dumps(result,indent=2))
if __name__=='__main__':main()
