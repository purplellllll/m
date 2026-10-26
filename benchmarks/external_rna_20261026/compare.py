"""Frozen human-CDS benchmark extension; preserve official outputs and failures."""
import csv
import hashlib
import json
import os
from pathlib import Path
import re
import statistics
import subprocess
import sys
import time

ROOT = Path('/root/gpufree-data/rna_external_comparison_20261026')
BASE = ROOT.parent/'vmae_transfer_assessment_20261024'
REFERENCE = ROOT.parent/'vmae_30cds_20261025'
PANEL = json.loads((REFERENCE/'panel.json').read_text())
sys.path.insert(0, str(BASE))


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    tmp.replace(path)


def validate(record, dna):
    from latent_decoder.tokenizer import AA, encode_cds
    assert len(dna) == len(record['cds_dna']) and set(dna) <= set('ACGT')
    assert [AA[i] for i in encode_cds(dna)] == list(record['protein_sequence'])+['*']
    assert dna[:3] == record['cds_dna'][:3]


def refresh():
    rows = []
    for directory, mapping in [
        (ROOT.parent/'codon_baselines_20261025', {}), (ROOT, {})]:
        if directory == ROOT:
            rows.extend(json.loads(p.read_text()) for p in sorted((directory/'metrics').glob('*/*.json')))
        elif (directory/'metrics.csv').exists():
            with (directory/'metrics.csv').open(encoding='utf-8-sig', newline='') as f:
                for row in csv.DictReader(f):
                    for k in ('ribodecode_score','seq2ribo_score','cai','gc_percent','mfe_kcal_mol','mfe_per_nt'):
                        row[k] = float(row[k])
                    rows.append(row)
    assert len({(r['sequence_id'], r['method']) for r in rows}) == len(rows)
    methods = sorted({r['method'] for r in rows})
    keys = ('ribodecode_score','seq2ribo_score','cai','gc_percent','mfe_kcal_mol','mfe_per_nt')
    summary = {m:dict(n=sum(r['method']==m for r in rows), means={
        k:statistics.mean(r[k] for r in rows if r['method']==m) for k in keys}) for m in methods}
    lookup = {(r['sequence_id'],r['method']):r for r in rows}
    paired = {}
    for method in methods:
        paired[method] = {}
        for reference in ('original','vmae971','vmae249','native'):
            if method == reference:
                continue
            identities = [r['sequence_id'] for r in PANEL if (r['sequence_id'],method) in lookup and (r['sequence_id'],reference) in lookup]
            if not identities:
                continue
            deltas = {k:[lookup[i,method][k]-lookup[i,reference][k] for i in identities] for k in keys}
            paired[method][reference] = dict(n=len(identities),sequence_ids=identities,
                mean_delta={k:statistics.mean(v) for k,v in deltas.items()},
                higher_count={k:sum(x>0 for x in v) for k,v in deltas.items()})
    save(ROOT/'summary.json', dict(methods=summary, paired=paired, expected_cases_per_method=30,
        equal_search_budgets=False, rnop='Blocked: official weight URL requires Google authentication',
        native_reference='Partial full-budget queue; compare only identical case IDs',
        candidate_selection='One official output per fixed seed/setting; no evaluator-based reselection'))
    if rows:
        fields = sorted({k for r in rows for k in r})
        tmp = ROOT/'metrics.csv.tmp'
        with tmp.open('w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        tmp.replace(ROOT/'metrics.csv')
    text = '# Extended model comparison\n\nSame fixed 30 human CDS; incomplete methods report their own N.\n\n'
    text += '| Method | N | RiboDecode | seq2ribo | CAI | GC % | MFE kcal/mol |\n|---|---:|---:|---:|---:|---:|---:|\n'
    for m, v in summary.items():
        text += '| '+m+' | '+str(v['n'])+' | '+' | '.join(f"{v['means'][k]:.4f}" for k in keys[:-1])+' |\n'
    text += '\nRNop has no scored results until its official checkpoint is available. '
    text += 'LinearDesign: default lambda0 and predeclared lambda3; native MFE/CAI remain in generated JSON, common metrics are recomputed. '
    text += 'GEMORNA CDS model does not provide a CDS translation-expression predictor; UTR predictors cannot score CDS-only cases. '
    text += 'GC and MFE have no universal monotonic quality direction. Scores do not establish experimental expression.\n'
    text += '\n## Matched differences against original CDS\n\n| Method | Matched N | Delta RiboDecode | Delta seq2ribo | Delta CAI |\n|---|---:|---:|---:|---:|\n'
    for method, comparisons in paired.items():
        if 'original' in comparisons:
            v = comparisons['original']
            text += '| '+method+' | '+str(v['n'])+' | '+' | '.join(f"{v['mean_delta'][k]:+.4f}" for k in keys[:3])+' |\n'
    (ROOT/'REPORT.md').write_text(text)


def generate_linear():
    repo = next((ROOT/'external').glob('LinearDesignSoftware-LinearDesign-*'))
    for method, weight in [('lineardesign_l0',0), ('lineardesign_l3',3)]:
        for record in PANEL:
            path = ROOT/'generated'/record['sequence_id']/(method+'.json')
            if path.exists():
                validate(record, json.loads(path.read_text())['cds_dna'])
                continue
            start = time.monotonic()
            result = subprocess.run([str(repo/'bin/LinearDesign_2D'),str(weight),'0',
                'codon_usage_freq_table_human.csv'], input=record['protein_sequence']+'\n',
                text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, cwd=repo, timeout=3600)
            log = ROOT/'logs'/(record['sequence_id']+'_'+method+'.log')
            log.parent.mkdir(exist_ok=True)
            log.write_text(result.stdout)
            assert result.returncode == 0, result.stdout[-1000:]
            matches = re.findall(r'mRNA sequence:\s*([ACGU]+)',result.stdout)
            assert len(matches) == 1
            dna = matches[0].replace('U','T')+record['cds_dna'][-3:]
            validate(record,dna)
            save(path,dict(sequence_id=record['sequence_id'], cds_dna=dna,
                generation_seconds=time.monotonic()-start, lambda_value=weight,
                stop_policy='Append original stop to official protein-only output',
                native_output=result.stdout, command='bin/LinearDesign_2D '+str(weight)+' 0 codon_usage_freq_table_human.csv'))
            save(ROOT/'generation_status.json',dict(state='running',method=method,sequence_id=record['sequence_id']))
            print('GENERATED',method,record['sequence_id'],flush=True)


def generate_gemorna():
    repo = ROOT/'external/GEMORNA'
    executable = ROOT/'gemorna_env/bin/python'
    seed = 20261026
    for record in PANEL:
        path = ROOT/'generated'/record['sequence_id']/'gemorna.json'
        if path.exists():
            validate(record,json.loads(path.read_text())['cds_dna'])
            continue
        start = time.monotonic()
        code = "import random,numpy as np,torch,runpy,sys;random.seed(20261026);np.random.seed(20261026);torch.manual_seed(20261026);torch.set_num_threads(2);sys.path.insert(0,'src');sys.argv=['src/generate.py','--mode','cds','--ckpt_path','checkpoints/gemorna_cds.pt','--protein_seq',sys.argv[1]];runpy.run_path('src/generate.py',run_name='__main__')"
        result = subprocess.run([str(executable),'-c',code,record['protein_sequence']],cwd=repo,
            env=dict(os.environ, CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='2', MKL_NUM_THREADS='2'),
            text=True, stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=3600)
        log = ROOT/'logs'/(record['sequence_id']+'_gemorna.log')
        log.parent.mkdir(exist_ok=True)
        log.write_text(result.stdout)
        assert result.returncode == 0,result.stdout[-2000:]
        matches = re.findall(r'[ACGUTacgut]{30,}',result.stdout)
        candidates = [x.upper().replace('U','T') for x in matches if len(x) in (len(record['cds_dna']),len(record['cds_dna'])-3)]
        candidates = list(dict.fromkeys(candidates))
        assert len(candidates) == 1, ('Ambiguous official output',str(log))
        dna = candidates[0]
        if len(dna) == len(record['cds_dna'])-3:
            dna += record['cds_dna'][-3:]
        validate(record,dna)
        save(path,dict(sequence_id=record['sequence_id'],cds_dna=dna,
            generation_seconds=time.monotonic()-start,seed=seed,device='cpu',official_output=result.stdout))
        print('GENERATED','gemorna',record['sequence_id'],flush=True)


def score():
    from tools import run_vmae_30cds_20261025 as evaluation
    from latent_decoder.latent_optimization import state_hash
    evaluation.OUT = ROOT
    evaluation.refresh = refresh
    save(ROOT/'cai_weights.json',json.loads((REFERENCE/'cai_weights.json').read_text()))
    scorer,predictor,teacher,lock = evaluation.environment()
    before = [state_hash(m) for m in (scorer,teacher)]
    for record in PANEL:
        for method in ('lineardesign_l0','lineardesign_l3','gemorna'):
            source = ROOT/'generated'/record['sequence_id']/(method+'.json')
            target = ROOT/'metrics'/record['sequence_id']/(method+'.json')
            if not source.exists() or target.exists():
                continue
            value = json.loads(source.read_text())
            validate(record,value['cds_dna'])
            evaluation.measure(record,method,value['cds_dna'],scorer,predictor,lock,value['generation_seconds'],0)
            print('SCORED',method,record['sequence_id'],flush=True)
    assert before == [state_hash(m) for m in (scorer,teacher)]
    save(ROOT/'scoring_receipt.json',dict(evaluator_weights_unchanged=True,seq2ribo_seeds=evaluation.SEEDS))
    refresh()


def ribonn():
    import shutil
    baseline = ROOT.parent/'codon_baselines_20261025'
    output = ROOT/'extra_scores'
    if not output.exists():
        shutil.copytree(baseline/'extra_scores',output)
    subprocess.run([str(baseline/'extra_env/bin/python'),'-u',str(baseline/'codon_extra_ribonn.py'),
        '--repo',str(baseline/'external/RiboNN'),'--context',str(baseline/'native_context.json'),
        '--metrics',str(ROOT/'metrics.csv'),'--output',str(output),'--batch-size','4'],cwd=baseline,check=True)


if __name__ == '__main__':
    assert len(PANEL)==30 and all(all(x.startswith('ENSG') for x in r['gene_ids']) for r in PANEL)
    ROOT.mkdir(exist_ok=True)
    {'linear':generate_linear,'gemorna':generate_gemorna,'score':score,'report':refresh,'ribonn':ribonn}[sys.argv[1]]()
