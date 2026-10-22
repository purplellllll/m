"""Freeze a score-blind, component-disjoint test set for B4/B5/B6 validation."""
import argparse
import csv
import gzip
import hashlib
import json
from pathlib import Path

SEEDS=[20261026,20261027,20261028]
STRATA=[(30,256,64),(257,512,64),(513,1024,64),(1025,1500,64)]


def select_formal(records,excluded):
    selected=[];used=set(excluded)
    for lo,hi,count in STRATA:
        candidates=[r for r in records if lo<=r['n_codons']<=hi and r['cluster_id'] not in used]
        candidates.sort(key=lambda r:hashlib.sha256(('b456-formal-20261020:'+r['sequence_id']).encode()).hexdigest())
        group=[]
        for r in candidates:
            if r['cluster_id'] in used:continue
            group.append(r);used.add(r['cluster_id'])
            if len(group)==count:break
        if len(group)!=count:raise ValueError('Not enough independent components in length stratum')
        selected.extend(group)
    assert len(selected)==len({r['cluster_id'] for r in selected})==256
    return selected


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',default='dataset');p.add_argument('--output',default='docs/b456_formal_validation_plan_20261020.json')
    args=p.parse_args()
    source=Path('docs/b456_random_validation_plan_20261020.json')
    original=json.loads(source.read_text());excluded={r['cluster_id'] for r in original['records']}
    records=[]
    for path in sorted((Path(args.data)/'test').glob('*.gz')):
        with gzip.open(path,'rt') as f:records.extend(json.loads(line) for line in f)
    selected=select_formal(records,excluded)
    evaluator=Path('docs/seq2ribo_evaluator_lock_20261020.json')
    plan=dict(original,version='b456-formal-validation-20261020',split='test',seeds=SEEDS,
        selection='256 test CDS, 64 per stratum, hash-ranked, unique existing cluster_id; no scores used',
        length_strata=STRATA,evaluation_ids=[r['sequence_id'] for r in selected],
        records=[{k:r[k] for k in ['sequence_id','n_codons','cluster_id']} for r in selected],
        excluded_clusters=sorted(excluded),pilot_plan_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        independent_evaluator_lock_sha256=hashlib.sha256(evaluator.read_bytes()).hexdigest(),
        rnafold_version='2.7.2',
        pilot_matching_feasibility='fixed calipers match 31/64 pilot CDS, 74/192 same-seed triples; formal coverage must be reported',
        long_definition='n_codons > 512',
        primary_length_tests=['B5-B4: mean long advantage minus mean short advantage',
                              'B5-B6: mean long advantage minus mean short advantage'],
        secondary_length_test='continuous log2(codons/512) association; no claim of monotonicity from bins',
        matching=dict(unit='same CDS and same optimization seed, all three arms together',
            max_pairwise_gc_fraction=.01,max_pairwise_mfe_per_nt=.01,max_pairwise_mutable_hamming=.02,
            selection='all eligible same-seed triples, seed average within CDS; never use independent scores to match',
            minimum_cds_coverage=.5,coverage_failure='inconclusive controlled validation; never widen calipers'),
        primary_external_tests=['seq2ribo B5-B4 on matched CDS','seq2ribo B5-B6 on matched CDS'],
        multiplicity='four primary endpoints, Bonferroni 98.75% cluster-bootstrap intervals; report all endpoints',
        analysis='CDS is unit; average all seeds, no best-seed selection; full-cohort and matched-cohort results both retained',
        success_rule='length attenuation only if long-minus-short upper corrected CI <0; external transfer only if both matched contrasts lower corrected CI >0 and >=50% CDS match coverage',
        sample_size='256 unique existing components, 128 long and 128 short; fixed precision-oriented sample, no guaranteed power claim',
        evaluator_independence='independent of optimization; training-data independence not certified; HEK293 vs HEK293T condition differs')
    plan.pop('predictor_evaluation',None)
    with Path(args.output).open('x') as f:json.dump(plan,f,indent=2)
    print(json.dumps(dict(records=len(selected),long=sum(r['n_codons']>512 for r in selected),seeds=SEEDS)))


if __name__=='__main__':main()
