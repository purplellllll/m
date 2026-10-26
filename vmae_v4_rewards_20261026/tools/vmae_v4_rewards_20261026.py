"""Monotonic GC, CAI and exact-RNAfold rewards for V4 fine-tuning and latent search."""
import hashlib
import math

import torch
from torch.nn import functional as F

from latent_decoder.model import SYNONYMS, mutable_positions
from latent_decoder.tokenizer import CODONS, PAD
from tools.vmae_multiobjective_core_20261026 import (
    _check_fold, _validate_probabilities, exact_fold, property_penalties,
)

GC_SCALE = CAI_SCALE = MFE_PER_NT_SCALE = .05


def property_rewards(probabilities, tokens, weights):
    """Use common validated property statistics; rewards remain monotonic without thresholds."""
    statistics = property_penalties(probabilities, tokens, weights)
    cai = statistics['log_cai'].exp()
    original_cai = statistics['original_log_cai'].exp()
    return dict(gc_reward=(statistics['gc_fraction']-statistics['original_gc_fraction'])/GC_SCALE,
        cai_reward=(cai-original_cai)/CAI_SCALE, gc_fraction=statistics['gc_fraction'],
        original_gc_fraction=statistics['original_gc_fraction'], cai=cai, original_cai=original_cai)


def mfe_reward(candidate_per_nt, original_per_nt):
    candidate, original = float(candidate_per_nt), float(original_per_nt)
    if not math.isfinite(candidate) or not math.isfinite(original):
        raise ValueError('MFE/nt must be finite')
    return (original-candidate)/MFE_PER_NT_SCALE


def scst_loss(probabilities, tokens, original_mfe, cache):
    """Minimize negative MFE reward using a detached sampled-minus-greedy advantage."""
    if probabilities.ndim == 2:
        probabilities = probabilities.unsqueeze(0)
    if tokens.ndim == 1:
        tokens = tokens.unsqueeze(0)
    _validate_probabilities(probabilities, tokens)
    if tokens.shape[0] != 1 or (tokens == PAD).any():
        raise ValueError('SCST accepts one complete, unpadded CDS')
    if not isinstance(cache, dict):
        raise ValueError('Fold cache must be a dict')
    original_per_nt = float(original_mfe['mfe_per_nt'] if isinstance(original_mfe, dict) else original_mfe)
    mfe_reward(original_per_nt, original_per_nt)
    if isinstance(original_mfe, dict) and 'sequence_sha256' in original_mfe:
        dna = ''.join(CODONS[i] for i in tokens[0].tolist())
        _check_fold(original_mfe, hashlib.sha256(dna.encode()).hexdigest(), len(dna))
    policy = probabilities*SYNONYMS.to(tokens.device)[tokens]
    active = mutable_positions(tokens).clone()
    active[:,0] = active[:,-1] = False
    policy = torch.where(active[...,None], policy, F.one_hot(tokens,64).to(probabilities.dtype))
    total = policy.sum(-1,keepdim=True)
    if (total <= 0).any():
        raise ValueError('No probability mass on an allowed synonymous codon')
    policy = policy/total
    sampled = torch.distributions.Categorical(probs=policy[0]).sample()
    greedy = policy[0].argmax(-1)
    sample_dna = ''.join(CODONS[i] for i in sampled.tolist())
    greedy_dna = ''.join(CODONS[i] for i in greedy.tolist())
    keys = {hashlib.sha256(dna.encode()).hexdigest() for dna in (sample_dna,greedy_dna)}
    missing = sum(key not in cache for key in keys)
    sample_fold, greedy_fold = exact_fold(sample_dna,cache), exact_fold(greedy_dna,cache)
    sample_reward = mfe_reward(sample_fold['mfe_per_nt'], original_per_nt)
    greedy_reward = mfe_reward(greedy_fold['mfe_per_nt'], original_per_nt)
    advantage = probabilities.new_tensor(sample_reward-greedy_reward).detach()
    selected = policy[0].gather(-1,sampled[:,None]).squeeze(-1)
    log_probability = selected[active[0]].log().sum()
    loss = -advantage*log_probability
    return loss,dict(sample_dna=sample_dna,greedy_dna=greedy_dna,
        sample_mfe=sample_fold['mfe'],greedy_mfe=greedy_fold['mfe'],
        sample_mfe_per_nt=sample_fold['mfe_per_nt'],greedy_mfe_per_nt=greedy_fold['mfe_per_nt'],
        sample_reward=sample_reward,greedy_reward=greedy_reward,reward_advantage=float(advantage),
        sample_log_probability=float(log_probability.detach()),mutable_sense_positions=int(active.sum()),fold_calls=missing)


def search_v4(model, scorer, record, weights, *, steps=80, lr=.05, radius=8., seed=20261026):
    """Optimize frozen-model latent inputs and select by the same exact combined reward."""
    from latent_decoder.model import MASK, collate_records
    from latent_decoder.latent_optimization import design_probabilities, state_hash
    from latent_decoder.latent_search_probe_cpu import candidate, project
    if steps < 0 or lr <= 0 or radius <= 0:
        raise ValueError('Invalid latent-search budget')
    if model.training or scorer.training or any(p.requires_grad for p in model.parameters()) or any(p.requires_grad for p in scorer.parameters()):
        raise ValueError('Latent search requires frozen, eval-mode models')
    before = [state_hash(m) for m in (model,scorer)]
    torch.manual_seed(seed)
    device = next(model.parameters()).device
    tokens = collate_records([record]).to(device)
    active = mutable_positions(tokens)
    cache, archive, trace = {}, {}, []
    original = exact_fold(record['cds_dna'],cache)
    with torch.no_grad():
        mu, lv = model.encode(tokens.masked_fill(active,MASK), tokens)
        sigma = (.5*lv).exp()
    variable = torch.zeros_like(mu,requires_grad=True)
    optimizer = torch.optim.Adam([variable],lr=lr)

    @torch.no_grad()
    def remember(p, step):
        dna = candidate(p,tokens)
        if dna not in archive:
            hard = F.one_hot(p.argmax(-1),64).float()
            props = property_rewards(hard,tokens,weights)
            score = float(scorer(hard,tokens).mean())
            archive[dna] = dict(cds_dna=dna,step=step,hard_score=score,
                gc_reward=float(props['gc_reward']),cai_reward=float(props['cai_reward']),
                partial_reward=score+float(props['gc_reward'])+float(props['cai_reward']))

    remember(F.one_hot(tokens,64).float(),-1)
    with torch.no_grad():
        remember(design_probabilities(model.decode(mu,tokens),tokens),0)
    for step in range(1,steps+1):
        optimizer.zero_grad(set_to_none=True)
        p = design_probabilities(model.decode(mu+sigma*variable,tokens),tokens)
        hard = F.one_hot(p.argmax(-1),64).float()
        straight = hard+p-p.detach()
        props = property_rewards(straight,tokens,weights)
        score = scorer(straight,tokens).mean()
        loss = -score-props['gc_reward'].mean()-props['cai_reward'].mean()
        feedback = None
        if step%10 == 0:
            reinforcement, feedback = scst_loss(p,tokens,original,cache)
            loss = loss+10*reinforcement
        gradient, = torch.autograd.grad(loss,variable)
        assert torch.isfinite(gradient).all()
        variable.grad = gradient
        torch.nn.utils.clip_grad_norm_([variable],1.,error_if_nonfinite=True)
        optimizer.step()
        project(variable,active,radius)
        with torch.no_grad():
            remember(design_probabilities(model.decode(mu+sigma*variable,tokens),tokens),step)
        trace.append(dict(step=step,loss=float(loss.detach()),mfe_feedback=feedback))
    for row in archive.values():
        folded = exact_fold(row['cds_dna'],cache)
        row.update(mfe=folded['mfe'],mfe_per_nt=folded['mfe_per_nt'],
            mfe_reward=mfe_reward(folded['mfe_per_nt'],original['mfe_per_nt']))
        row['total_reward'] = row['partial_reward']+row['mfe_reward']
    assert before == [state_hash(m) for m in (model,scorer)]
    return dict(best=max(archive.values(),key=lambda r:r['total_reward']),trace=trace,
        candidates=list(archive.values()),steps=steps,exact_fold_unique=len(cache),
        selection='Maximum Hard+GC_reward+CAI_reward+MFE_reward over every unique archived discrete candidate',
        model_weights_unchanged=True,original_selected=max(archive.values(),key=lambda r:r['total_reward'])['cds_dna']==record['cds_dna'])
