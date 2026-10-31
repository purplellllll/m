"""Independent exact-overlap audit of the chosen full-source partition."""
import hashlib
import itertools
import json
from pathlib import Path

WORK=Path(__file__).resolve().parents[1]
ROOT=WORK/'runs/riboseq_graph_v2_20261031/data'
OLD=WORK/'runs/graph_polish_20261031/gpu_result/completed_snapshot/graph_polish_20261031/data'
CODE=dict(zip(map(''.join,itertools.product('TCAG',repeat=3)),'FFLLSSSSYY**CC*WLLLLPPPPHHQQRRRRIIIMTTTTNNKKSSRRVVVVAAAADDEEGGGG'))


def protein_sha(dna):
    amino=''.join(CODE[dna[i:i+3]] for i in range(0,len(dna),3))
    return hashlib.sha256(amino.encode()).hexdigest()


def main():
    panel=json.loads((ROOT/'panel.json').read_text());rows=panel['records']
    assert len(rows)==6956
    for key in ['gene','dna','protein_sha256']:assert len({r[key] for r in rows})==len(rows),key
    old=json.loads((OLD/'exclusions.json').read_text())['records']
    old += [r for r in json.loads((OLD/'panel.json').read_text())['records'] if r['split']=='fresh_test']
    forbidden=[{r['gene'] for r in old},{r['dna'] for r in old},{protein_sha(r['dna']) for r in old}]
    for r in rows:
        assert protein_sha(r['dna'])==r['protein_sha256']
        if r['split']=='test':
            for key,values in zip(['gene','dna','protein_sha256'],forbidden):assert r[key] not in values
    result=dict(status='passed',n_records=len(rows),counts=panel['counts'],prior_records=len(old),
        no_exact_gene_cds_protein_overlap=True,near_homology_checked=False)
    (ROOT/'local_split_audit.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))


if __name__=='__main__':main()
