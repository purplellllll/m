"""Standalone inference-only sparse-attention prototype; no production model edits."""
import argparse
import copy
import json
from pathlib import Path
import sys
import time
import torch
from torch import nn
from torch.nn import functional as F

ROOT=Path(__file__).resolve().parent
WORK=ROOT.parent.parent
sys.path.insert(0,str(WORK/'repos/riboseq_scorer_20261031'))
from physics_scorer.reliable_graph_context import ReliableGraphContext

def sparse_attention(m,x,radius,chunk=128):
    assert not m.training and m.batch_first and m._qkv_same_embed_dim
    b,n,width=x.shape;heads=m.num_heads;dim=width//heads
    q,k,v=F.linear(x,m.in_proj_weight,m.in_proj_bias).chunk(3,-1)
    q,k,v=[t.view(b,n,heads,dim).transpose(1,2) for t in (q,k,v)]
    values=[]
    for start in range(0,n,chunk):
        end=min(n,start+chunk);left=max(0,start-radius);right=min(n,end+radius)
        query_positions=torch.arange(start,end,device=x.device)[:,None]
        key_positions=torch.arange(left,right,device=x.device)[None,:]
        allowed=(query_positions-key_positions).abs()<=radius
        values.append(F.scaled_dot_product_attention(q[:,:,start:end],k[:,:,left:right],v[:,:,left:right],attn_mask=allowed,dropout_p=0.))
    return F.linear(torch.cat(values,2).transpose(1,2).reshape(b,n,width),m.out_proj.weight,m.out_proj.bias)

def sparse_context(context,x,radius):
    for layer in context.layers:
        assert layer.norm_first and not layer.training
        x=x+layer.dropout1(sparse_attention(layer.self_attn,layer.norm1(x),radius))
        z=layer.norm2(x)
        x=x+layer.dropout2(layer.linear2(layer.dropout(layer.activation(layer.linear1(z)))))
    return context.norm(x) if context.norm is not None else x

def main(device):
    torch.set_num_threads(2);torch.manual_seed(73)
    if device=='cuda':
        # Compare identical FP32 arithmetic before attributing differences to sparsity.
        torch.set_float32_matmul_precision('highest')
        torch.backends.cuda.matmul.allow_tf32=False
        torch.backends.cudnn.allow_tf32=False
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_cudnn_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)
        torch.backends.mha.set_fastpath_enabled(False)
    model=ReliableGraphContext(width=192).to(device).eval()
    payload=torch.load(WORK/'runs/scorer_compare_search_20261031/search_outputs/scorer_backbones/gated_graph_30101.pt',map_location=device,weights_only=True)
    model.load_state_dict(payload['state_dict'],strict=True)
    context=model.context
    with torch.no_grad():
        x=torch.randn(1,33,192,device=device)
        reference=context(x);actual=sparse_context(context,x,33)
        torch.testing.assert_close(actual,reference,atol=2e-5,rtol=2e-5)
        check_error=float((actual-reference).abs().max())
        local_checks=[]
        for n,radius in [(17,1),(65,16),(257,32)]:
            x=torch.randn(1,n,192,device=device)
            positions=torch.arange(n,device=device)
            blocked=(positions[:,None]-positions[None,:]).abs()>radius
            reference=context(x,mask=blocked)
            actual=sparse_context(context,x,radius)
            torch.testing.assert_close(actual,reference,atol=2e-5,rtol=2e-5)
            local_checks.append(dict(n_codons=n,radius=radius,max_abs_error=float((actual-reference).abs().max())))
    result=dict(device=device,torch=torch.__version__,full_window_equivalence_max_abs_error=check_error,local_mask_equivalence=local_checks,
                scope='Synthetic context-only inference; fixed checkpoint; not accuracy, training, whole folding pipeline, padding or global-token support.',benchmarks=[])
    if device=='cuda':
        result['gpu']=torch.cuda.get_device_name(0)
        result['arithmetic']='FP32; TF32 disabled; math SDPA; MHA/Transformer fused fastpath disabled for both implementations'
        for n in [256,1024,4096,8192]:
            for mode in ['dense','sparse_radius128']:
                try:
                    torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats()
                    x=torch.randn(1,n,192,device=device)
                    fn=(lambda:context(x)) if mode=='dense' else (lambda:sparse_context(context,x,128))
                    with torch.no_grad():
                        y=fn();torch.cuda.synchronize();begun=time.perf_counter()
                        y=fn();torch.cuda.synchronize();seconds=time.perf_counter()-begun
                    row=dict(n_codons=n,mode=mode,seconds=seconds,peak_allocated_bytes=torch.cuda.max_memory_allocated(),finite=bool(torch.isfinite(y).all()))
                    del y,x
                except torch.cuda.OutOfMemoryError:
                    row=dict(n_codons=n,mode=mode,status='out_of_memory');torch.cuda.empty_cache()
                result['benchmarks'].append(row)
                (ROOT/f'length_probe_{device}.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    (ROOT/f'length_probe_{device}.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(dict(status='completed',device=device,equivalence_error=check_error)),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--device',choices=['cpu','cuda'],required=True);main(p.parse_args().device)
