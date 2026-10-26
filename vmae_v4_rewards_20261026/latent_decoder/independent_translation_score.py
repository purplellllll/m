"""Score frozen CDS candidates with the pinned official seq2ribo implementation."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import platform
import multiprocessing
from pathlib import Path
import sys
import time
import numpy as np


def initialize_worker(repository,lock,cache):
    global worker_predictor,worker_lock
    sys.path.insert(0,repository)
    import torch
    from seq2ribo import Seq2Ribo
    torch.set_num_threads(1)
    worker_lock=lock
    worker_predictor=Seq2Ribo(cell_line=lock['cell_line'],device='cuda',weights_dir=str(Path(repository)/'weights'),cache_dir=cache)


def score_sequence(sequence):
    result=worker_predictor.predict(sequence,task='te',n_stasep_runs=worker_lock['n_stasep_runs'],
        init_p=worker_lock['init_p'],seed=worker_lock['seed'],return_scaled_te=worker_lock['return_scaled_te'],batch_size=8)
    value=float(np.asarray(result).reshape(-1)[0]);assert np.isfinite(value)
    return dict(sequence_sha256=hashlib.sha256(sequence.replace('U','T').encode()).hexdigest(),seq2ribo_te=value)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--repository',required=True);p.add_argument('--lock',required=True)
    p.add_argument('--candidates');p.add_argument('--output',required=True)
    p.add_argument('--smoke',action='store_true')
    args=p.parse_args();repository=Path(args.repository).resolve()
    lock=json.loads(Path(args.lock).read_text())
    for name,digest in lock['weights_sha256'].items():
        assert hashlib.sha256((repository/name).read_bytes()).hexdigest()==digest,name
    for name,digest in lock['source_sha256'].items():
        assert hashlib.sha256((repository/name).read_bytes()).hexdigest()==digest,name
    sys.path.insert(0,str(repository))
    import RNA
    import torch
    import mamba_ssm
    from seq2ribo import Seq2Ribo
    from seq2ribo.geometry import compute_geometry_features
    assert RNA.__version__==lock['viennarna_version']
    assert hasattr(RNA,'PLOT_TYPE_TURTLE'),'Missing geometry API: do not silently replace features with zeros'
    assert torch.cuda.is_available(),'Use the official CUDA Mamba implementation'
    output=Path(args.output);output.mkdir(parents=True,exist_ok=False)
    toy='AUG'+('GCU'+'GCC')*30+'UAA'
    angle,pair=compute_geometry_features(toy,cache_dir=output/'geometry')
    assert len(angle)==len(pair)==len(toy)//3 and np.abs(angle).sum()>0 and pair.sum()>0
    predictor=Seq2Ribo(cell_line=lock['cell_line'],device='cuda',weights_dir=str(repository/'weights'),cache_dir=str(output/'geometry'))
    if args.smoke:
        sequences=[toy,'AUG'+('GCU'+'GCC')*60+'UAA']
    else:
        if not args.candidates:p.error('--candidates required without --smoke')
        rows=[json.loads(s) for s in Path(args.candidates).read_text().splitlines()]
        sequences=sorted({r['best']['cds_dna'].replace('T','U') for r in rows})
    before={p:hashlib.sha256((repository/p).read_bytes()).hexdigest() for p in lock['weights_sha256']}
    scores=[];start=time.monotonic()
    with (output/'predictions.jsonl').open('x') as f, ProcessPoolExecutor(max_workers=4,
        mp_context=multiprocessing.get_context('spawn'),initializer=initialize_worker,
        initargs=(str(repository),lock,str(output/'geometry'))) as pool:
        for index,row in enumerate(pool.map(score_sequence,sequences)):
            scores.append(row);f.write(json.dumps(row)+'\n');f.flush()
            print(json.dumps(dict(event='external_score',completed=index+1,total=len(sequences),elapsed_seconds=time.monotonic()-start)),flush=True)
    if args.smoke:
        reverse=predictor.predict(list(reversed(sequences)),task='te',n_stasep_runs=lock['n_stasep_runs'],
            init_p=lock['init_p'],seed=lock['seed'],return_scaled_te=lock['return_scaled_te'],batch_size=8)
        assert np.allclose([r['seq2ribo_te'] for r in scores],list(reversed(np.asarray(reverse).reshape(-1))),atol=1e-5,rtol=1e-5)
    assert before=={p:hashlib.sha256((repository/p).read_bytes()).hexdigest() for p in before}
    receipt=dict(completed=True,smoke=args.smoke,unique_sequences=len(sequences),elapsed_seconds=time.monotonic()-start,
        evaluator_lock_sha256=hashlib.sha256(Path(args.lock).read_bytes()).hexdigest(),
        candidates_sha256=None if args.smoke else hashlib.sha256(Path(args.candidates).read_bytes()).hexdigest(),
        torch=torch.__version__,viennarna=RNA.__version__,weights_unchanged=True,
        python=platform.python_version(),cuda=torch.version.cuda,gpu=torch.cuda.get_device_name(0),
        mamba_ssm=mamba_ssm.__version__,adapter_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        scoring_workers=4,
        repository_commit=lock['commit'],condition=lock['cell_line'],not_same_condition_as_ribodecode=True)
    (output/'completion_receipt.json').write_text(json.dumps(receipt,indent=2))


if __name__=='__main__':main()
