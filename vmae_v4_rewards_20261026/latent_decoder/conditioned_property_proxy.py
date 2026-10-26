"""Delta-property features conditioned on an already measured original CDS score."""
import torch
from torch.nn import functional as F

from .joint_property_proxy import sequence_features
from .model import PAD,mutable_positions,validate_targets


def conditioned_features(probabilities,tokens,reference_tokens,reference_seq2ribo):
    validate_targets(reference_tokens)
    assert tokens.shape==reference_tokens.shape
    reference=F.one_hot(reference_tokens.clamp_max(63),64).to(probabilities)
    reference*= (reference_tokens<PAD)[...,None]
    delta=sequence_features(probabilities,tokens)-sequence_features(reference,reference_tokens)
    active=mutable_positions(reference_tokens)
    same=probabilities.gather(-1,reference_tokens.clamp_max(63)[...,None]).squeeze(-1)
    edits=((1-same)*active).sum(1)/active.sum(1).clamp_min(1)
    score=torch.as_tensor(reference_seq2ribo,device=probabilities.device,dtype=probabilities.dtype).detach().reshape(-1)
    assert len(score)==len(tokens)
    return torch.cat([delta,edits[:,None],(edits*score)[:,None]],dim=1)
