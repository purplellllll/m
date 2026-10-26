"""Record source identity and live benchmark status without altering running jobs."""
import collections
import hashlib
import json
from pathlib import Path
import time

ROOT = Path('/root/gpufree-data/rna_external_comparison_20261026')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(name, value):
    path = ROOT/name
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, indent=2)+'\n')
    temp.replace(path)


def alive(pid, keyword):
    process = Path('/proc')/str(pid)
    try:
        return keyword in (process/'cmdline').read_bytes().replace(b'\0',b' ').decode() and (process/'stat').read_text().split()[2] != 'Z'
    except FileNotFoundError:
        return False


def main():
    identity = dict(
        gemorna=dict(url='https://github.com/RainaBio/GEMORNA',commit='3bba88e4c0a32377dcfac8caa492709b954f9d9e',
            checkpoint_sha256=sha(ROOT/'external/GEMORNA/checkpoints/gemorna_cds.pt'),device='cpu',seed=20261026),
        rnop=dict(url='https://github.com/HudenJear/RPLoss',commit='b1b49f4124e28d8e3cd2c9f8f6e82dc8820363da',
            status='weights_unavailable',official_weight_file_id='1RkJ-U22sH4z2_vWHPW8EeFtkkbaOE6Nf',
            reason='Official Drive URL returned Google sign-in; user confirmed marking unavailable'),
        lineardesign=dict(url='https://github.com/LinearDesignSoftware/LinearDesign',
            commit='f0126ca89a8b853088b4bccfd2cc8c378d3678be',lambdas=[0,3],
            codon_table_sha256=sha(next((ROOT/'external').glob('LinearDesignSoftware-*'))/'codon_usage_freq_table_human.csv')),
        panel_sha256=sha(ROOT.parent/'vmae_30cds_20261025/panel.json'),
        scripts={n:sha(ROOT/n) for n in ('compare.py','compare_initial.py')},
        native_scorer_applicability=dict(gemorna='CDS generator has no standalone CDS TE head; UTR heads inapplicable',
            rnop='Mechanism-aligned losses; no available checkpoint for native results',
            lineardesign='Native MFE and native human-table CAI preserved in generated JSON'),
        common_metrics=['RiboDecode','seq2ribo','GC_percent','train_reference_CAI','RNAfold_MFE','MFE_per_nt','RiboNN'])
    assert identity['gemorna']['checkpoint_sha256']=='053de4e6c164a1b24378629229cc44d957dda776231e99698f9f42cbacb6757e'
    save('source_manifest.json',identity)
    while True:
        generated = dict(collections.Counter(p.stem for p in (ROOT/'generated').glob('*/*.json')))
        scored = dict(collections.Counter(p.stem for p in (ROOT/'metrics').glob('*/*.json')))
        running = dict(linear=alive(23742,'compare.py linear'),gemorna=alive(24302,'compare.py gemorna'),
            scoring=alive(23921,'compare.py score'),ribonn=alive(25749,'compare.py ribonn'))
        expected = dict(gemorna=30,lineardesign_l0=30,lineardesign_l3=30)
        state = 'completed' if all(scored.get(k,0)==n for k,n in expected.items()) else 'running'
        errors = []
        for job, methods in [('linear',['lineardesign_l0','lineardesign_l3']),('gemorna',['gemorna'])]:
            if not running[job] and any(generated.get(m,0)<expected[m] for m in methods):
                errors.append(job+' generator stopped before all cases completed; inspect log')
        if errors:
            state = 'failed'
        native = json.loads((ROOT.parent/'vmae_30cds_20261025/summary.json').read_text())
        save('live_status.json',dict(state=state,heartbeat_unix=time.time(),generated=generated,scored=scored,
            expected=expected,running=running,errors=errors,rnop='weights_unavailable',
            full_native_completed=native['completed_full_native_cases'],full_native_total=30,
            full_comparison_complete=state=='completed' and native['complete'],
            scored_methods_only='RNop excluded by explicit user instruction because weights are unavailable'))
        if state in ('completed','failed'):
            break
        time.sleep(30)


if __name__ == '__main__':
    main()
