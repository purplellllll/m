"""Prepare pinned LCL/RPE-1/iPSC CDS-only TE scorers and verify actual CUDA inference."""
import hashlib
import json
from pathlib import Path
import sys
import time
import urllib.request

BASE = Path('/root/gpufree-data/vmae_transfer_assessment_20261024')
ROOT = BASE.parent/'seq2ribo_multicell_20261027'
REPOSITORY = BASE/'tmp/seq2ribo_source_20261020/seq2ribo-main'
COMMIT = '66c80c653e6db0d0b19a737f5d9fc8009db252b2'
CELLS = ('lcl', 'rpe', 'ipsc')
SEEDS = (20261020, 20261021, 20261022)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(name, value):
    target = ROOT/name
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    temporary.replace(target)


def fetch(url):
    request = urllib.request.Request(url, headers={'User-Agent': 'research-reproduction'})
    for attempt in range(3):
        try:
            with urllib.request.urlopen(request, timeout=45) as response:
                return response.read()
        except OSError:
            if attempt == 2:
                raise
            time.sleep(2)


def main():
    import fcntl
    ROOT.mkdir(exist_ok=True)
    with (ROOT/'setup.lock').open('a') as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        save('status.json', dict(state='verifying_pinned_source',heartbeat_unix=time.time()))
        lock = json.loads((BASE/'docs/seq2ribo_evaluator_lock_20261020.json').read_text())
        assert lock['commit'] == COMMIT
        for name, expected in lock['source_sha256'].items():
            assert digest(REPOSITORY/name) == expected, name
        metadata_path = ROOT/'official_weights_metadata.json'
        if not metadata_path.exists():
            metadata = json.loads(fetch('https://api.github.com/repos/Kingsford-Group/seq2ribo/contents/weights?ref='+COMMIT))
            save(metadata_path.name, metadata)
        metadata = {r['name']: r for r in json.loads(metadata_path.read_text())}
        weights = ROOT/'weights'
        weights.mkdir(exist_ok=True)
        assets = {}
        for cell in CELLS:
            for name in (cell+'_mamba_te_full_final_cds.pt', cell+'_te_transform_cds.json'):
                entry = metadata[name]
                target = weights/name
                if not target.exists():
                    save('status.json', dict(state='downloading',asset=name,heartbeat_unix=time.time()))
                    content = fetch('https://raw.githubusercontent.com/Kingsford-Group/seq2ribo/'+COMMIT+'/weights/'+name)
                    temporary = target.with_suffix('.part');temporary.write_bytes(content);temporary.replace(target)
                content = target.read_bytes()
                assert len(content) == entry['size']
                assert hashlib.sha1(b'blob '+str(len(content)).encode()+b'\0'+content).hexdigest() == entry['sha'], name
                assets[name] = dict(bytes=len(content),sha256=digest(target),official_git_blob_sha=entry['sha'])
        save('assets.json', dict(repository=lock['repository'],commit=COMMIT,source_sha256=lock['source_sha256'],assets=assets))
        sys.path.insert(0, str(BASE/'.runtime/mamba_cuda_v2_2_4/packages'))
        sys.path.insert(0, str(REPOSITORY))
        import numpy as np
        import torch
        import RNA
        import mamba_ssm
        from seq2ribo import Seq2Ribo
        from seq2ribo import constants
        from seq2ribo.geometry import compute_geometry_features
        torch.set_num_threads(1);torch.set_num_interop_threads(1)
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        assert torch.cuda.is_available() and RNA.__version__=='2.7.2' and mamba_ssm.__version__=='2.2.4'
        assert Path(mamba_ssm.__file__).resolve().is_relative_to((BASE/'.runtime/mamba_cuda_v2_2_4/packages').resolve())
        parameters = {}
        for cell in CELLS:
            parameters[cell] = {key:np.asarray(getattr(constants,cell.upper()+'_'+key)).tolist() for key in ('RATES','ALPHA','BETA','BUCKETS')}
        save('environment_parameters.json',parameters)
        panel=json.loads((BASE.parent/'vmae_30cds_20261025/panel.json').read_text())
        panel=sorted(panel,key=lambda r:(r['n_codons'],r['sequence_id']))
        records=[panel[0],next(r for r in panel if r['n_codons']>300)]
        sequences=[r['cds_dna'].replace('T','U') for r in records]
        toy='AUG'+('GCU'+'GCC')*30+'UAA'
        angle,pair=compute_geometry_features(toy,cache_dir=str(ROOT/'geometry'))
        assert np.abs(angle).sum()>0 and pair.sum()>0, 'Geometry silently became zero'
        receipts={}
        for cell in CELLS:
            save('status.json',dict(state='cuda_preflight',cell_line=cell,heartbeat_unix=time.time()))
            predictor=Seq2Ribo(cell_line=cell,device='cuda',weights_dir=str(weights),cache_dir=str(ROOT/'geometry'))
            teacher=predictor._load_model('te',use_utr=False).eval().requires_grad_(False)
            def state_hash():
                h=hashlib.sha256()
                for key,value in sorted(teacher.state_dict().items()):
                    h.update(key.encode());h.update(value.detach().cpu().contiguous().numpy().tobytes())
                return h.hexdigest()
            before=state_hash()
            scores={}
            with torch.inference_mode():
                for seed in SEEDS:
                    result=predictor.predict(sequences,task='te',use_utr=False,return_scaled_te=False,n_stasep_runs=8,init_p=.01,seed=seed)
                    values=np.asarray(result,dtype=float).reshape(-1)
                    assert len(values)==2 and np.isfinite(values).all(), repr(result)
                    scores[str(seed)]=values.tolist()
                repeated=np.asarray(predictor.predict(sequences[::-1],task='te',use_utr=False,return_scaled_te=False,n_stasep_runs=8,init_p=.01,seed=SEEDS[0]),dtype=float).reshape(-1)[::-1]
                np.testing.assert_allclose(repeated,scores[str(SEEDS[0])],rtol=1e-3,atol=1e-4)
            assert state_hash()==before
            receipts[cell]=dict(sequence_ids=[r['sequence_id'] for r in records],n_aa=[r['n_codons']-1 for r in records],
                scores_by_seed=scores,mean_scores=np.mean(list(scores.values()),axis=0).tolist(),state_sha256=before,
                strict_checkpoint_load=True,finite_outputs=True,repeated_seed_and_reversed_batch_verified=True,weights_unchanged=True)
            save('preflight.json',receipts)
            del teacher,predictor
            torch.cuda.empty_cache()
        save('scorer_lock.json',dict(repository=lock['repository'],commit=COMMIT,cell_lines=list(CELLS),task='te',use_utr=False,
            return_scaled_te=False,seeds=list(SEEDS),n_stasep_runs=8,init_p=.01,assets_sha256=digest(ROOT/'assets.json'),
            environment_parameters_sha256=digest(ROOT/'environment_parameters.json'),source_sha256=lock['source_sha256'],
            torch_version=torch.__version__,viennarna_version=RNA.__version__,mamba_version=mamba_ssm.__version__,
            setup_script_sha256=digest(__file__),optimization_launched=False,
            independence='These scorers are currently prepared for evaluation. If used as optimization rewards, they cease to be independent evaluators.'))
        save('status.json',dict(state='ready',cell_lines=list(CELLS),task='te',use_utr=False,cuda_preflight_passed=True,heartbeat_unix=time.time()))


if __name__=='__main__':
    try:
        main()
    except BaseException as error:
        save('status.json',dict(state='failed',error=repr(error),heartbeat_unix=time.time()))
        raise
