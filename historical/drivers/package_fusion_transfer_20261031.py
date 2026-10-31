"""Package only frozen inputs, references and source required on the new server."""
import hashlib
import itertools
import json
from pathlib import Path
import tarfile

WORK=Path(__file__).resolve().parents[1]
OUT=WORK/'runs/fusion_encoders_20261031'
SNAP=WORK/'runs/bilstm_diagnostics_20261030/completed_snapshot/bilstm_diagnostics_20261030'
PHYS=WORK/'runs/gnn_physics_recipe_20261030/completed_snapshot/gnn_physics_recipe_20261030'


def main():
    latest=json.loads((WORK/'runs/gnn_pretrain_20261031/panel.json').read_text())['records']
    early=json.loads((WORK/'runs/ribogl_reproduction_20261030/completed_snapshot/liver_pilot/frozen_panel.json').read_text())['records']
    codons=[''.join(c) for c in itertools.product('ATCG',repeat=3)]
    known=[dict(gene=r['gene'],dna=r['dna']) for r in latest]
    known += [dict(gene=r['gene'],dna=''.join(codons[int(c)] for c in r['sequence'].strip('[]').split(','))) for r in early]
    assert len(known)==1380
    selected=[r for r in latest if r['split'] in ['train','validation']]
    assert len(selected)==480
    extras={
        'data/exclusions.json':dict(records=known),
        'data/panel_train_validation.json':dict(records=selected),
        'data/supplementary_panel.json':dict(records=[r for r in latest if r['split']=='new_test']),
    }
    manifest=[];archive=OUT/'transfer_20261031.tar.gz'
    with tarfile.open(archive,'x:gz',compresslevel=3) as tar:
        for name,obj in extras.items():
            path=OUT/Path(name).name;assert not path.exists();path.write_text(json.dumps(obj,indent=2),encoding='utf-8')
            tar.add(path,arcname=name);manifest.append(dict(relative_path=name,sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
        def add(src,name):
            src=Path(src);tar.add(src,arcname=name);manifest.append(dict(relative_path=name,sha256=hashlib.sha256(src.read_bytes()).hexdigest()))
        for r in selected:add(SNAP/'frozen_inputs/features'/f'{r["gene"]}.npz',f'data/frozen_inputs/features/{r["gene"]}.npz')
        for seed in [30101,30102,30103]:add(SNAP/f'w424_native_aux1_{seed}/best.pt',f'references/bilstm_{seed}.pt')
        add(SNAP/'frozen_sources/aligned/physics_parameters_train_only.json','data/physics_parameters_train_only.json')
        for name in ['geometry.py','simulation.py','utils.py']:add(PHYS/'frozen_sources/seq2ribo'/name,'source/evaluator/seq2ribo/'+name)
        for name in ['__init__.py','model.py','profile_model.py','diagnostic_model.py','fusion_encoder_model.py','test_fusion_encoder_model.py']:
            add(WORK/'physics_scorer'/name,'physics_scorer/'+name)
        for name in ['train_fusion_encoders_20261031.py','run_aligned_profile_ablation_20261030.py','get_riboclette_liver_20261030.py']:
            add(WORK/'tools'/name,name)
        for name in ['GPU阶段待执行协议.md','三路权重与编码器替换_阶段报告.md','gain_selection.json']:
            add(OUT/name,'prior_validation/'+name)
        path=OUT/'transfer_manifest.json';path.write_text(json.dumps(manifest,indent=2),encoding='utf-8');tar.add(path,arcname='transfer_manifest.json')
    receipt=dict(path=str(archive),sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),size_bytes=archive.stat().st_size,n_files=len(manifest))
    (OUT/'transfer_receipt.json').write_text(json.dumps(receipt,indent=2),encoding='utf-8');print(json.dumps(receipt))


if __name__=='__main__':main()
