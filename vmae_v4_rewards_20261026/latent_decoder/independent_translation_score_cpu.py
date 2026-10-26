"""Separately identified seq2ribo CPU reference evaluation with official forward/step checks."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import multiprocessing
from pathlib import Path
import sys
import time
import types

import numpy as np
import torch

from .budget_js_extension import save_json
from .latent_optimization import state_hash
from .seq2ribo_cpu_reference import ROOT, install


def initialize(repository,lock,cache):
    install();torch.set_num_threads(1);sys.path.insert(0,repository)
    from seq2ribo import Seq2Ribo
    from . import independent_translation_score as official_adapter
    official_adapter.worker_lock=lock
    official_adapter.worker_predictor=Seq2Ribo(cell_line=lock['cell_line'],device='cpu',
        weights_dir=str(Path(repository)/'weights'),cache_dir=cache)


def score(sequence):
    from .independent_translation_score import score_sequence
    row=score_sequence(sequence)
    row['seq2ribo_cpu_reference_te']=row.pop('seq2ribo_te')
    return row


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--candidates',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    role=parser.add_mutually_exclusive_group()
    role.add_argument('--selection-objective',action='store_true',help='Record use in inner-development candidate selection')
    role.add_argument('--training-labels',action='store_true',help='Record teacher labels for an outer-fold training cohort')
    args=parser.parse_args()
    preflight=ROOT/'runs/latent_research_20261024/seq2ribo_cpu_reference_preflight/completion_receipt.json'
    prior=json.loads(preflight.read_text());assert prior['state']=='completed' and prior['weights_unchanged']
    assert prior['cpu_backend_source_sha256']==hashlib.sha256(Path(__file__).with_name('seq2ribo_cpu_reference.py').read_bytes()).hexdigest()
    repository=ROOT/'tmp/seq2ribo_source_20261020/seq2ribo-main'
    lock_path=ROOT/'docs/seq2ribo_evaluator_lock_20261020.json'
    lock=json.loads(lock_path.read_text())
    assert prior['evaluator_lock_sha256']==hashlib.sha256(lock_path.read_bytes()).hexdigest()
    for name,digest in {**lock['source_sha256'],**lock['weights_sha256']}.items():
        assert hashlib.sha256((repository/name).read_bytes()).hexdigest()==digest,name
    rows=[json.loads(line) for line in args.candidates.read_text().splitlines() if line]
    if args.training_labels:
        folds={r['fold'] for r in rows};assert len(folds)==1
        manifest=json.loads((ROOT/'runs/ribodecode_only_vmae_20261023/data/folds.json').read_text())
        allowed=set(manifest['splits'][str(next(iter(folds)))]['train'])
        assert {r['sequence_id'] for r in rows}<=allowed
        assert all(r['v2_group']==manifest['assignments'][r['sequence_id']]['group'] for r in rows)
    sequences=sorted({r['best']['cds_dna'].replace('T','U') for r in rows})
    assert sequences and all(len(s)%3==0 and set(s)<=set('ACGU') for s in sequences)
    args.output.mkdir(parents=True,exist_ok=False)
    backend=install();torch.set_num_threads(1);sys.path.insert(0,str(repository))
    import RNA
    from seq2ribo import Seq2Ribo
    from seq2ribo.geometry import compute_geometry_features
    assert RNA.__version__==lock['viennarna_version'] and hasattr(RNA,'PLOT_TYPE_TURTLE')
    cache=args.output/'geometry'
    toy='AUG'+('GCU'+'GCC')*30+'UAA'
    angle,pair=compute_geometry_features(toy,cache_dir=cache)
    assert np.abs(angle).sum()>0 and pair.sum()>0
    protocol=dict(parent_official_lock=lock,parent_lock_sha256=hashlib.sha256(lock_path.read_bytes()).hexdigest(),
        execution_backend=backend.reference_manifest,adapter_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        reused_official_adapter_sha256=hashlib.sha256(Path(__file__).with_name('independent_translation_score.py').read_bytes()).hexdigest(),
        preflight_sha256=hashlib.sha256(preflight.read_bytes()).hexdigest(),
        candidates_sha256=hashlib.sha256(args.candidates.read_bytes()).hexdigest(),
        role=('Teacher supervision for fold-train property surrogate; not independent evaluation' if args.training_labels else
            'Inner-development selection objective; untouched outer evaluation retained' if args.selection_objective else
            'Frozen external assessment; no optimization or candidate selection using these scores yet'),
        condition_difference='HEK293 evaluator versus HEK293T RiboDecode scorer',
        protocol_change='Explicit CPU Torch reference backend. Preserve original CUDA protocol and separate numeric columns; CUDA parity pending.')
    save_json(args.output/'protocol.json',protocol)
    started=time.monotonic();status=dict(state='validating_long_sequence',completed=0,total=len(sequences))
    try:
        save_json(args.output/'status.json',status)
        predictor=Seq2Ribo(cell_line=lock['cell_line'],device='cpu',weights_dir=str(repository/'weights'),cache_dir=str(cache))
        model=predictor._load_model('te',use_utr=False).requires_grad_(False)
        before=state_hash(model)
        sequence=max(sequences,key=len)
        options=dict(task='te',n_stasep_runs=lock['n_stasep_runs'],init_p=lock['init_p'],
            seed=lock['seed'],return_scaled_te=lock['return_scaled_te'],batch_size=8)
        full=predictor.predict(sequence,**options)
        def step_forward(self,inputs,inference_params=None):
            assert inference_params is None
            conv,state=self.allocate_inference_cache(len(inputs),inputs.shape[1]);pieces=[]
            for token in inputs.split(1,dim=1):
                value,conv,state=self.step(token,conv,state);pieces.append(value)
            return torch.cat(pieces,dim=1)
        blocks=[m for m in model.modules() if isinstance(m,backend.Mamba)];assert len(blocks)==4
        for block in blocks:block.forward=types.MethodType(step_forward,block)
        stepped=predictor.predict(sequence,**options)
        assert np.allclose(full,stepped,atol=1e-5,rtol=1e-5),(full,stepped)
        assert before==state_hash(model)
        save_json(args.output/'long_sequence_parity.json',dict(sequence_sha256=hashlib.sha256(sequence.replace('U','T').encode()).hexdigest(),
            nt=len(sequence),original_reference=full,original_step=stepped,max_abs=float(np.max(np.abs(np.array(full)-np.array(stepped)))),
            cuda_parity_verified=False,weights_unchanged=True))
        del model,predictor
        status['state']='scoring'
        with (args.output/'predictions.jsonl').open('x') as handle, ProcessPoolExecutor(max_workers=4,
            mp_context=multiprocessing.get_context('spawn'),initializer=initialize,
            initargs=(str(repository),lock,str(cache))) as pool:
            for index,row in enumerate(pool.map(score,sequences)):
                handle.write(json.dumps(row)+'\n');handle.flush()
                status['completed']=index+1;save_json(args.output/'status.json',status)
                print(json.dumps(dict(event='cpu_reference_score',**status,seconds=time.monotonic()-started)),flush=True)
        for name,digest in lock['weights_sha256'].items():
            assert hashlib.sha256((repository/name).read_bytes()).hexdigest()==digest,name
        status.update(state='completed',weights_unchanged=True,cuda_parity_verified=False,
            execution_backend=backend.__version__,seconds=time.monotonic()-started,goal_complete=False)
        save_json(args.output/'completion_receipt.json',dict(status,protocol_sha256=hashlib.sha256((args.output/'protocol.json').read_bytes()).hexdigest()))
    except BaseException as error:
        status.update(state='failed',error=repr(error));raise
    finally:save_json(args.output/'status.json',status)


if __name__=='__main__':main()
