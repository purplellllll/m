"""Compare the minimal Torch operator backend to saved genuine PyG predictions."""
from pathlib import Path
import sys,json,hashlib,types
import numpy as np
import torch
HERE=Path(__file__).resolve().parent;WORK=HERE.parent.parent
sys.path.insert(0,str(HERE))
from models import RiboGLScalar
def main():
    torch.set_num_threads(2);torch.backends.cudnn.allow_tf32=False
    root=WORK/'runs/profile_model_comparison_20261030/completed_snapshot'
    rows=json.loads((root/'panel.json').read_text(encoding='utf-8'))['records'];rows=[r for r in rows if r['split']=='test'][:3]
    from scipy import sparse
    graphs=[]
    for r in rows:
        path=root/'features'/f'{r["gene"]}.npz'
        if not path.exists():path=WORK/'runs/aligned_profile_20261030/completed_snapshot/features'/f'{r["gene"]}.npz'
        with np.load(path,allow_pickle=False) as z:f={k:z[k].copy() for k in z.files}
        n=len(f['cod']);a=np.array([np.arange(n-1),np.arange(1,n)]);c=f['edges']
        edge=np.unique(np.concatenate((c,c[::-1].copy(),a,a[::-1].copy()),axis=1),axis=1)
        mat=sparse.csr_matrix((np.ones(edge.shape[1]),(edge[0],edge[1])),shape=(n,n));degree=np.asarray(mat.sum(1)).ravel();mat=sparse.diags(1/np.maximum(degree,1))@mat
        power=mat.copy();pe=[power.diagonal()]
        for _ in range(31):power=power@mat;pe.append(power.diagonal())
        graphs.append(types.SimpleNamespace(x=torch.tensor(r['source_ids'],device='cuda'),edge_index=torch.from_numpy(edge).cuda(),random_walk_pe=torch.from_numpy(np.stack(pe,1).astype(np.float32)).cuda()))
    results=[]
    with np.load(root/'predictions.npz',allow_pickle=False) as gold:
        for seed in [30101,30102]:
            raw=torch.load(root/f'ribogl_{seed}/best.pt',map_location='cpu',weights_only=True)
            model=RiboGLScalar().base.cuda().eval();model.load_state_dict(raw['state_dict'],strict=True)
            for i,g in enumerate(graphs):
                with torch.no_grad():p=model(g).cpu().numpy()
                expected=gold[f'test__ribogl_{seed}__{i}'];assert p.shape==expected.shape
                error=float(np.max(np.abs(p-expected)));assert error<2e-5,(seed,i,error)
                results.append(dict(seed=seed,gene=rows[i]['gene'],max_absolute_error=error))
            del model;torch.cuda.empty_cache()
    report=dict(status='passed',backend='Torch equations for PyG SAGE mean/project and single-graph GraphNorm; upstream GCN_LSTM forward unchanged',
        reference='Previously saved original PyG forward outputs, not newly generated substitutes',reference_predictions_sha256=hashlib.sha256((root/'predictions.npz').read_bytes()).hexdigest(),
        samples=6,results=results,max_error=max(r['max_absolute_error'] for r in results),
        limitation='Finite regression coverage; not a proof for every graph or operator setting; supports only the exact SAGE/project/single-graph configuration used here')
    (HERE/'ribogl_backend_parity.json').write_text(json.dumps(report,indent=2),encoding='utf-8');print(json.dumps(report),flush=True)
if __name__=='__main__':main()
