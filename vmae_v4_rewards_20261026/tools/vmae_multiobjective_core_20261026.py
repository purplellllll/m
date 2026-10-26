"""Native-CDS property penalties and real-fold, score-function MFE gradients."""
import hashlib
import math

import torch
from torch.nn import functional as F

from latent_decoder.model import SYNONYMS, mutable_positions
from latent_decoder.tokenizer import AA, CODONS, PAD, encode_cds


def _validate_probabilities(probabilities, tokens):
    if (tokens.ndim != 2 or tokens.dtype != torch.long or not tokens.numel()
            or probabilities.shape != (*tokens.shape, 64)
            or probabilities.device != tokens.device or not probabilities.is_floating_point()):
        raise ValueError('Expected floating probabilities [B,L,64] and int64 tokens [B,L] on one device')
    if ((tokens < 0) | (tokens > PAD)).any():
        raise ValueError('Tokens must be codons 0..63 or PAD=64')
    if not torch.isfinite(probabilities).all() or (probabilities < 0).any():
        raise ValueError('Probabilities must be finite and nonnegative')
    valid = tokens < PAD
    lengths = valid.sum(-1)
    prefix = torch.arange(tokens.shape[1], device=tokens.device)[None, :] < lengths[:, None]
    if not torch.equal(prefix, valid):
        raise ValueError('PAD is allowed only after the complete CDS')
    for row, length in zip(tokens.detach().cpu().tolist(), lengths.tolist()):
        encode_cds(''.join(CODONS[i] for i in row[:length]))
    sums = probabilities.sum(-1)[valid]
    if not torch.allclose(sums, torch.ones_like(sums), atol=1e-5, rtol=1e-5):
        raise ValueError('Each valid codon probability row must sum to one')
    return valid


def property_penalties(probabilities, tokens, weights):
    """Return per-sequence [B] GC and log-CAI statistics/penalties, retaining gradients."""
    valid = _validate_probabilities(probabilities, tokens)
    gc = probabilities.new_tensor([(c.count('G') + c.count('C')) / 3 for c in CODONS])
    log_weights = []
    for codon, aa in zip(CODONS, AA):
        value = 1.0 if aa == '*' else float(weights[codon])
        if not math.isfinite(value) or not 0 < value <= 1:
            raise ValueError(f'CAI weight must be finite and in (0,1]: {codon}')
        log_weights.append(math.log(value))
    log_weights = probabilities.new_tensor(log_weights)
    sense_table = torch.tensor([aa != '*' for aa in AA], device=tokens.device)
    ids = tokens.clamp_max(63)
    sense = valid & sense_table[ids]
    gc_fraction = ((probabilities * gc).sum(-1) * valid).sum(-1) / valid.sum(-1)
    original_gc = (gc[ids] * valid).sum(-1) / valid.sum(-1)
    log_cai = ((probabilities * log_weights).sum(-1) * sense).sum(-1) / sense.sum(-1)
    original_log_cai = (log_weights[ids] * sense).sum(-1) / sense.sum(-1)
    return dict(gc_penalty=F.relu((gc_fraction - original_gc).abs() - .05).square() / .05**2,
                cai_penalty=F.relu(original_log_cai - log_cai).square() / .05**2,
                gc_fraction=gc_fraction, log_cai=log_cai,
                original_gc_fraction=original_gc, original_log_cai=original_log_cai)


def mfe_penalty(candidate_per_nt, original_per_nt):
    """Penalize less-negative MFE/nt; becoming more negative has zero penalty."""
    candidate, original = float(candidate_per_nt), float(original_per_nt)
    if not math.isfinite(candidate) or not math.isfinite(original):
        raise ValueError('MFE/nt must be finite')
    return max(0.0, (candidate - original) / .05)**2


def _check_fold(row, key, length):
    if (row['sequence_sha256'] != key or row['viennarna_version'] != '2.7.2'
            or row['temperature_celsius'] != 37 or row['dangles'] != 2):
        raise ValueError('Fold cache identity or RNAfold protocol mismatch')
    mfe, per_nt = float(row['mfe']), float(row['mfe_per_nt'])
    if not math.isfinite(mfe) or not math.isfinite(per_nt) or not math.isclose(mfe / length, per_nt, rel_tol=1e-10, abs_tol=1e-12):
        raise ValueError('Fold cache has invalid energy or normalization')


def exact_fold(dna, cache):
    """Fold with ViennaRNA 2.7.2 at 37 C/dangles=2, or validate a SHA-keyed cache hit."""
    if not isinstance(cache, dict) or not isinstance(dna, str) or set(dna) - set('ACGT'):
        raise ValueError('Expected uppercase DNA and a dict cache')
    encode_cds(dna)
    key = hashlib.sha256(dna.encode()).hexdigest()
    if key in cache:
        _check_fold(cache[key], key, len(dna))
        return dict(cache[key])
    import RNA
    if RNA.__version__ != '2.7.2':
        raise RuntimeError(f'Requires ViennaRNA 2.7.2, found {RNA.__version__}')
    settings = RNA.md()
    settings.temperature = 37.0
    settings.dangles = 2
    structure, energy = RNA.fold_compound(dna.replace('T', 'U'), settings).mfe()
    if len(structure) != len(dna):
        raise ValueError('RNAfold returned a structure with the wrong length')
    row = dict(sequence_sha256=key, mfe=float(energy), mfe_per_nt=float(energy) / len(dna),
               viennarna_version=RNA.__version__, temperature_celsius=float(settings.temperature),
               dangles=int(settings.dangles))
    _check_fold(row, key, len(dna))
    cache[key] = row
    return dict(row)


def scst_loss(probabilities, tokens, original_mfe, cache):
    """One-sample SCST for MFE penalty minimization; pass soft policy probabilities."""
    if probabilities.ndim == 2:
        probabilities = probabilities.unsqueeze(0)
    if tokens.ndim == 1:
        tokens = tokens.unsqueeze(0)
    _validate_probabilities(probabilities, tokens)
    if tokens.shape[0] != 1 or (tokens == PAD).any():
        raise ValueError('SCST accepts exactly one complete, unpadded CDS')
    if not isinstance(cache, dict):
        raise ValueError('Fold cache must be a dict')
    original_per_nt = float(original_mfe['mfe_per_nt'] if isinstance(original_mfe, dict) else original_mfe)
    mfe_penalty(original_per_nt, original_per_nt)
    if isinstance(original_mfe, dict) and 'sequence_sha256' in original_mfe:
        original_dna = ''.join(CODONS[i] for i in tokens[0].tolist())
        _check_fold(original_mfe, hashlib.sha256(original_dna.encode()).hexdigest(), len(original_dna))
    allowed = SYNONYMS.to(tokens.device)[tokens]
    policy = probabilities * allowed
    active = mutable_positions(tokens).clone()
    active[:, 0] = False
    active[:, -1] = False
    fixed = F.one_hot(tokens, 64).to(probabilities.dtype)
    policy = torch.where(active[..., None], policy, fixed)
    total = policy.sum(-1, keepdim=True)
    if (total <= 0).any():
        raise ValueError('No probability mass on an allowed synonymous codon')
    policy = policy / total
    sampled = torch.distributions.Categorical(probs=policy[0]).sample()
    greedy = policy[0].argmax(-1)
    sample_dna = ''.join(CODONS[i] for i in sampled.tolist())
    greedy_dna = ''.join(CODONS[i] for i in greedy.tolist())
    sample_key = hashlib.sha256(sample_dna.encode()).hexdigest()
    greedy_key = hashlib.sha256(greedy_dna.encode()).hexdigest()
    missing = len({key for key in (sample_key, greedy_key) if key not in cache})
    sample_fold = exact_fold(sample_dna, cache)
    greedy_fold = exact_fold(greedy_dna, cache)
    sample_penalty = mfe_penalty(sample_fold['mfe_per_nt'], original_per_nt)
    greedy_penalty = mfe_penalty(greedy_fold['mfe_per_nt'], original_per_nt)
    advantage = probabilities.new_tensor(sample_penalty - greedy_penalty).detach()
    chosen_probabilities = policy[0].gather(-1, sampled[:, None]).squeeze(-1)
    log_probability = chosen_probabilities[active[0]].log().sum()
    loss = advantage * log_probability
    diagnostics = dict(sample_dna=sample_dna, greedy_dna=greedy_dna,
        sample_mfe=sample_fold['mfe'], greedy_mfe=greedy_fold['mfe'],
        sample_mfe_per_nt=sample_fold['mfe_per_nt'], greedy_mfe_per_nt=greedy_fold['mfe_per_nt'],
        sample_penalty=sample_penalty, greedy_penalty=greedy_penalty,
        advantage=float(advantage), sample_log_probability=float(log_probability.detach()),
        mutable_sense_positions=int(active.sum()), fold_calls=missing)
    return loss, diagnostics
