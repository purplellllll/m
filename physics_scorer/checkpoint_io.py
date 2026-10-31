"""Load the project's tensor-only GitHub exports without importing training drivers."""
import hashlib
from pathlib import Path
import torch
from .fusion_encoder_model import FusionEncoderModel
from .graph_polish_model import GraphPolishModel
from .reliable_graph_context import ReliableGraphContext
from .scalar_downstream import TaskScalarModel


def load_exported_checkpoint(path,expected_sha256=None,device='cpu'):
    path=Path(path)
    if expected_sha256 is not None:
        actual=hashlib.sha256(path.read_bytes()).hexdigest()
        if actual!=expected_sha256:raise ValueError('Checkpoint SHA256 mismatch')
    payload=torch.load(path,map_location='cpu',weights_only=True)
    metadata=payload['metadata'];spec=metadata['spec'];kind=spec['kind']
    constructors={'FusionEncoderModel':FusionEncoderModel,'GraphPolishModel':GraphPolishModel,
                  'ReliableGraphContext':ReliableGraphContext}
    if kind=='TaskScalarModel':
        if spec['backbone']['kind']!='GraphPolishModel':raise ValueError('Unsupported exported scalar backbone')
        backbone=GraphPolishModel(**spec['backbone']['config'])
        model=TaskScalarModel(backbone,spec['task'],spec['n_outputs'],spec['use_utr'])
    elif kind in constructors:model=constructors[kind](**spec['config'])
    else:raise ValueError('Unknown exported model class')
    model.load_state_dict(payload['state_dict'],strict=True)
    model.to(device).eval()
    return model,metadata
