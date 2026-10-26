"""Differentiable codon descriptors and train-fold synonymous augmentation for a teacher proxy."""
import math
import random
from collections import Counter

import torch

from .model import AA, CODONS, PAD, nucleotide_probabilities, validate_targets
from .tokenizer import TOKEN, encode_cds


BASES='ACGT'
INTERNAL_DINUC=torch.tensor([[sum((codon[i]+codon[i+1])==a+b for i in (0,1))
    for a in BASES for b in BASES] for codon in CODONS],dtype=torch.float32)
SYNONYMOUS={aa:[c for c,a in zip(CODONS,AA) if a==aa] for aa in set(AA)}


def sequence_features(probabilities,tokens):
    """208 features: all/first32/last32 codon frequencies and exact expected dinucleotides."""
    validate_targets(tokens)
    assert probabilities.shape==(*tokens.shape,64)
    valid=tokens<PAD;p=probabilities*valid[...,None]
    lengths=valid.sum(-1);position=torch.arange(tokens.shape[1],device=tokens.device)[None,:]
    first=valid & (position<32)
    last=valid & (position>=(lengths-32).clamp_min(0)[:,None])
    def average(mask):return (p*mask[...,None]).sum(1)/mask.sum(1).clamp_min(1)[:,None]
    within=(p@INTERNAL_DINUC.to(p)).sum(1)
    nts=nucleotide_probabilities(p).reshape(*tokens.shape,3,4)
    boundary=nts[:,:-1,2,:,None]*nts[:,1:,0,None,:]
    boundary=boundary*(valid[:,:-1]&valid[:,1:])[...,None,None]
    dinuc=(within+boundary.sum(1).flatten(1))/(3*lengths-1)[:,None]
    return torch.cat([average(valid),average(first),average(last),dinuc],dim=1)


def synonymous_sample(dna,counts,*,power=1.,gc_bias=0.,fraction=1.,seed):
    ids=encode_cds(dna)
    assert len(ids)>=2 and AA[ids[-1]]=='*'
    assert len(counts)==64 and all(math.isfinite(float(x)) and x>0 for x in counts)
    assert 0<=fraction<=1 and math.isfinite(power) and math.isfinite(gc_bias)
    rng=random.Random(seed);result=[dna[:3]]
    for index in range(1,len(ids)-1):
        original=CODONS[ids[index]];choices=SYNONYMOUS[AA[ids[index]]]
        weights=[float(counts[TOKEN[c]])**power*math.exp(gc_bias*(c.count('G')+c.count('C'))) for c in choices]
        result.append(rng.choices(choices,weights=weights)[0] if rng.random()<fraction else original)
    result.append(dna[-3:])
    return ''.join(result)


def fit_delta_ridge(features,targets,groups,strength):
    """Equal-parent ridge fit; zero feature change always predicts zero property change."""
    assert features.ndim==targets.ndim==2 and len(features)==len(targets)==len(groups)>0
    assert strength>0 and torch.isfinite(features).all() and torch.isfinite(targets).all()
    x,y=features.double(),targets.double();counts=Counter(groups)
    weights=x.new_tensor([1/(len(counts)*counts[g]) for g in groups])
    xs=(weights[:,None]*x.square()).sum(0).sqrt().clamp_min(1e-5)
    ys=(weights[:,None]*y.square()).sum(0).sqrt().clamp_min(1e-5)
    xn,yn=x/xs,y/ys
    gram=xn.T@(xn*weights[:,None])+strength*torch.eye(x.shape[1],dtype=x.dtype,device=x.device)
    coefficient=torch.linalg.solve(gram,xn.T@(yn*weights[:,None]))
    weight=(coefficient*ys[None,:]/xs[:,None]).T
    assert torch.isfinite(weight).all()
    return weight,dict(x_scale=xs,y_scale=ys,groups=len(counts),strength=strength)
