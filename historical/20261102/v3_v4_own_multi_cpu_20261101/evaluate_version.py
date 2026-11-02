"""Reuse the verified CPU evaluator, with explicit multiobjective reference identities."""
import argparse
import csv
import hashlib
import importlib.util
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parent
WORK=ROOT.parent.parent
PREVIOUS=WORK/'runs/vmae_own_full30_cpu_20261031'


def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def save(path,data):Path(path).write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')


def main(version):
    spec=importlib.util.spec_from_file_location('verified_cpu_evaluator',PREVIOUS/'evaluate_cpu.py')
    evaluator=importlib.util.module_from_spec(spec);spec.loader.exec_module(evaluator)
    out=ROOT/version;evaluator.OUT=out
    original_protocol=evaluator.source_protocol
    def protocol():
        value=original_protocol();value['historical_reference']=f'Original and frozen historical multiobjective endpoint for {version}'
        value['version']=version;value['historical_baseline_method']='V3_same_reward' if version=='v3_multi' else 'V4_epoch6_current'
        value['schema_note']='Legacy evaluator method aliases normalized after inference; no selection change'
        return value
    evaluator.source_protocol=protocol
    original_factory=evaluator.ribodecode_model
    def seed_cache_then_load(torch):
        digest=sha(out/'evaluation_protocol_locked.json');current=read(out/'evaluation_protocol_locked.json')
        for source in [PREVIOUS,ROOT/'v3_multi',ROOT/'v4_epoch6']:
            if source==out or not (source/'evaluation_protocol_locked.json').exists():continue
            old=read(source/'evaluation_protocol_locked.json')
            for key in ['seq2ribo','ribodecode','properties']:assert old[key]==current[key],f'Cross-run reference mismatch: {key}'
            for row in evaluator.endpoints():
                identity=row['sequence_sha256']
                for folder in ['ribodecode','seq2ribo']:
                    source_file=source/'evaluation'/folder/f'{identity}.json'
                    destination=out/'evaluation'/folder/f'{identity}.json'
                    if not source_file.exists() or destination.exists():continue
                    data=read(source_file);assert data['sequence_sha256']==identity
                    data['protocol_sha256']=digest
                    data['provenance']=dict(kind='reused_verified_identical_sequence_and_frozen_evaluation_sources',
                        source=str(source_file),source_sha256=sha(source_file),source_protocol_sha256=sha(source/'evaluation_protocol_locked.json'))
                    save(destination,data)
        return original_factory(torch)
    evaluator.ribodecode_model=seed_cache_then_load
    evaluator.summarize=lambda table=None:None
    evaluator.evaluate(False)
    table=read(out/'comparison_metrics.json')
    mapping={'original':'original','V3_RiboDecode_search':'historical_multiobjective','V3_own_scorer_search':'own_multiobjective'}
    for row in table:row['method']=mapping[row['method']]
    save(out/'comparison_metrics.json',table)
    with (out/'comparison_metrics.csv').open('w',encoding='utf-8-sig',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(table[0]));writer.writeheader();writer.writerows(table)
    complete=read(out/'evaluation_complete.json');complete['metrics_sha256']=sha(out/'comparison_metrics.csv')
    complete['method_alias_normalization']=mapping
    complete['same_sequence_previous_prediction_cache_reused']=True
    complete['all_metrics_recomputed_on_cpu']=False
    save(out/'evaluation_complete.json',complete)
    save(out/'status.json',dict(stage='search_and_evaluation_complete',completed=30,total=30))
    print(f'{version}: EVALUATION COMPLETE',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('version',choices=['v3_multi','v4_epoch6']);args=parser.parse_args()
    try:main(args.version)
    except BaseException as error:save(ROOT/f'evaluation_failure_{args.version}.json',dict(error=repr(error)));raise
