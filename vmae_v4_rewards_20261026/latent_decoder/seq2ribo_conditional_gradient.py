"""Gradients through the frozen neural TE model with simulator/geometry inputs held fixed."""
import torch


def capture_inputs(predictor,model,dna,lock):
    batches=[]
    def capture(module,args,kwargs):
        batches.append((tuple(v.detach().clone() for v in args),
            {k:v.detach().clone() if isinstance(v,torch.Tensor) else v for k,v in kwargs.items()}))
    handle=model.register_forward_pre_hook(capture,with_kwargs=True)
    try:
        value=float(predictor.predict(dna,task='te',n_stasep_runs=lock['n_stasep_runs'],
            init_p=lock['init_p'],seed=lock['seed'],return_scaled_te=False,batch_size=8)[0])
    finally:handle.remove()
    rows=[]
    for args,kwargs in batches:
        for i in range(len(args[0])):
            rows.append((tuple(v[i:i+1] for v in args),
                {k:v[i:i+1] if isinstance(v,torch.Tensor) else v for k,v in kwargs.items()}))
    assert len(rows)==lock['n_stasep_runs']
    return value,rows


def conditional_value(model,args,kwargs,probabilities,embedding_table,transform):
    """Unscaled TE with an explicit codon-embedding path; not the full pipeline derivative."""
    assert probabilities.ndim==3 and probabilities.shape[-1]==64
    assert probabilities.shape[:2]==args[0].shape and embedding_table.shape[0]==64
    handle=model.base.cod_emb.register_forward_hook(lambda module,inputs,output:probabilities@embedding_table)
    try:
        _,scaled=model(*args,**kwargs)
        scaled=scaled.clamp(float(transform['eps']),1-float(transform['eps']))
        return float(transform['lo'])+(float(transform['hi'])-float(transform['lo']))*scaled
    finally:handle.remove()
