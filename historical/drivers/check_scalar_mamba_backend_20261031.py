"""Compare compiled-kernel model outputs against upstream2.2.4 Torch reference."""
import importlib.util
import json
from pathlib import Path
import sys
import torch
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'seq2ribo_source'))


def main():
    torch.set_num_threads(1);torch.cuda.set_per_process_memory_fraction(.12);torch.manual_seed(31)
    import mamba_ssm
    version=mamba_ssm.__version__
    from seq2ribo.models import RiboPolisherMamba
    compiled=RiboPolisherMamba().cuda().eval()
    state={k:v.detach().cpu().clone() for k,v in compiled.state_dict().items()}
    inputs=[torch.randint(0,64,(2,226),device='cuda'),torch.rand(2,226,device='cuda'),torch.ones(2,226,dtype=torch.bool,device='cuda')]
    kwargs={k:torch.randint(0,n,(2,226),device='cuda') for k,n in [('angle_bin',4),('pair_bin',4),('bucket_idx',3)]}
    with torch.no_grad():expected=compiled(*inputs,**kwargs)
    for key in list(sys.modules):
        if key=='mamba_ssm' or key.startswith('mamba_ssm.'):del sys.modules[key]
    from latent_decoder.seq2ribo_cpu_reference import install
    install()
    spec=importlib.util.spec_from_file_location('official_seq2ribo_torch_reference',ROOT/'seq2ribo_source/seq2ribo/models.py')
    reference_module=importlib.util.module_from_spec(spec);spec.loader.exec_module(reference_module)
    reference=reference_module.RiboPolisherMamba().cuda().eval();reference.load_state_dict(state,strict=True)
    with torch.no_grad():actual=reference(*inputs,**kwargs)
    error=float((expected-actual).abs().max());torch.testing.assert_close(expected,actual,rtol=2e-5,atol=2e-5)
    result=dict(status='passed',compiled_mamba_version=version,reference='unmodified Mamba2.2.4 class and selective_scan_ref',
        max_absolute_error=error,torch=torch.__version__,batch=2,length=226,dtype='float32',test='random initialized full4-layer192width profile model')
    (ROOT/'mamba_backend_parity.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))


if __name__=='__main__':main()
