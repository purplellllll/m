"""Explicit CPU backend using unmodified Mamba 2.2.4 class and reference scan definitions.

This is a separately identified execution backend; CUDA parity remains a separate check.
Upstream sources and license are retained under .runtime/mamba_v2_2_4_reference_source.
"""
import ast
import hashlib
import math
from pathlib import Path
import sys
import types
from typing import Optional

import torch
from torch import Tensor, nn
from torch.nn import functional as F


ROOT=Path(__file__).resolve().parents[1]
SOURCES=ROOT/'.runtime/mamba_v2_2_4_reference_source'
EXPECTED={
    'mamba_simple.py':'a17e4c51b582dc0d4d690a649eba521cd0c1ee3dc8f0473a0967cdc9ec0874e3',
    'selective_scan_interface.py':'c79f4432536402fe4d6c1026ccf8d5fc0aaa62b790e9a622295aee14ac45dbcc',
    'LICENSE':'760939b000194d04548ede6a857bbe735d1695d8422ec85955c8e2bd7f4b95c5'}


def install():
    """Expose the official Mamba class with its official Torch scan in this process only."""
    assert 'mamba_ssm' not in sys.modules, 'CPU reference backend must be selected before importing seq2ribo'
    for name,digest in EXPECTED.items():
        assert hashlib.sha256((SOURCES/name).read_bytes()).hexdigest()==digest,name
    sys.path.insert(0,str(ROOT/'.runtime/seq2ribo_cpu_reference'))
    import einops
    assert einops.__version__=='0.8.0'
    namespace=dict(__name__='mamba_ssm_cpu_reference',math=math,Optional=Optional,
        torch=torch,nn=nn,F=F,Tensor=Tensor,rearrange=einops.rearrange,repeat=einops.repeat,
        causal_conv1d_fn=None,causal_conv1d_update=None,selective_state_update=None,
        mamba_inner_fn=None,RMSNorm=None,layer_norm_fn=None,rms_norm_fn=None)
    def definition(filename,name,kind):
        path=SOURCES/filename
        nodes=[n for n in ast.parse(path.read_text(encoding='utf-8')).body if isinstance(n,kind) and n.name==name]
        assert len(nodes)==1,name
        # Keep the complete upstream definition intact; omit imports of unavailable CUDA extensions.
        exec(compile(ast.Module(body=nodes,type_ignores=[]),str(path),'exec'),namespace)
    definition('selective_scan_interface.py','selective_scan_ref',ast.FunctionDef)
    namespace['selective_scan_fn']=namespace['selective_scan_ref']
    definition('mamba_simple.py','Mamba',ast.ClassDef)
    def unsupported_mamba2(*args,**kwargs):
        raise NotImplementedError('This explicit CPU reference backend supports Mamba1 checkpoints only')
    module=types.ModuleType('mamba_ssm')
    module.Mamba=namespace['Mamba'];module.Mamba2=unsupported_mamba2
    module.__version__='2.2.4+torch-reference-cpu'
    module.reference_manifest=dict(backend=module.__version__,source_sha256=EXPECTED,
        upstream_tag='v2.2.4',cuda_parity_verified=False,einops=einops.__version__,torch=torch.__version__)
    sys.modules['mamba_ssm']=module
    return module
