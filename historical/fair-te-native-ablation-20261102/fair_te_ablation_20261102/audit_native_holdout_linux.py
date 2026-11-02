"""Frozen native MFE proxies: new synonymous diagnostics, never reselect candidates."""
from pathlib import Path
import sys,json,hashlib,math,shutil
import numpy as np
import torch
import RNA
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def main():
    work=Path(sys.argv[1]);root=work/'runs/te_stability_20261102/native_compare'
    sys.path.insert(0,str(work/'external/ribodecode_1_3_0'))
    from RiboDecode.models import mfe_conv_sim
    from RiboDecode.dataset import codon_group,my_vocab,process_line,convert_onehot
    assert str(torch.__version__)=='2.0.1+cpu' and RNA.__version__=='2.6.4'
    torch.set_num_threads(2);syn={c:g for g in codon_group for c in g};results=[]
    for case in [1,2,3]:
        task=read(root/('case%d_task.json'%case));dna=task['cds_dna'];native=root/('case%d_native'%case)
        cfg=read(native/'model_config.py')['training'];sub=cfg['mfe_tool_sub_len'];excluded=set()
        for path in native.glob('results_natural/HEK293T_optim_mfe_dist-optim/*/logs/train_data_generate_mfe.txt'):
            excluded.update(line.split(',')[0] for line in path.read_text().splitlines())
        rng=np.random.default_rng(20262102+case);hold=[]
        while len(hold)<32:
            codons=[dna[i:i+3] for i in range(0,len(dna),3)];local=len(hold)>=16
            for j in range(1,len(codons)-1):
                if not local or rng.random()<.1:codons[j]=str(rng.choice(syn[codons[j]]))
            candidate=''.join(codons)
            if candidate not in excluded and candidate not in hold and candidate!=dna:hold.append(candidate)
        assert all(s[:3]==dna[:3] and s[-3:]==dna[-3:] and len(s)==len(dna) for s in hold)
        truth=np.array([sum(float(RNA.fold(s[i:i+sub])[1]) for i in range(0,len(s),sub)) for s in hold])
        full=np.array([float(RNA.fold(s)[1]) for s in hold]);x=torch.tensor(np.array([convert_onehot(process_line(s,len(s))[0]) for s in hold]),dtype=torch.float32).permute(0,2,1)
        rounds=[]
        for rnd in [0,1,2]:
            path=native/('results_natural/HEK293T_optim_mfe_dist-optim/%d/logs/epoch_temp_mfe.pb'%rnd)
            model=mfe_conv_sim(65,cfg['hidden_dim'],cfg['latent_dim']).eval();model.load_state_dict(torch.load(path,map_location='cpu',weights_only=True),strict=True)
            with torch.no_grad():pred=np.concatenate([model(x[i:i+8]).flatten().numpy() for i in range(0,32,8)])
            assert np.isfinite(pred).all()
            def stats(target):
                return dict(pearson=float(np.corrcoef(target,pred)[0,1]) if np.ptp(pred)>0 else None,
                    r2=float(1-np.sum((target-pred)**2)/np.sum((target-target.mean())**2)),mae=float(np.mean(np.abs(target-pred))))
            rounds.append(dict(round=rnd+1,checkpoint_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                native_chunk_target=stats(truth),full_CDS_MFE=stats(full),predictions=pred.tolist()))
            del model
        results.append(dict(case=case,n=32,global_synonymous=16,local_synonymous=16,no_overlap_with_native_fold_training=True,
            native_subsequence_nt=sub,native_chunks=math.ceil(len(dna)/sub),native_chunk_targets=truth.tolist(),full_MFE_targets=full.tolist(),
            holdout_dna_sha256=[hashlib.sha256(s.encode()).hexdigest() for s in hold],rounds=rounds))
        print('frozen proxy audit case',case,rounds[-1]['native_chunk_target'],flush=True)
    report=dict(status='completed',RNA=RNA.__version__,torch=str(torch.__version__),seed_rule='20262102+case',source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),cases=results,
        diagnostic_only=True,search_candidates_not_reselected=True,limits=['32 synthetic heldout synonymous sequences per CDS; not independent biological outcomes',
            'Native proxy labels use768nt chunks beyond768nt; final endpoint scoring uses full CDS',
            'Printed native trainingR2 pools predictions made throughout training, not a frozen final-model validation'])
    (root/'native_surrogate_holdout_audit.json').write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf-8')
if __name__=='__main__':main()
