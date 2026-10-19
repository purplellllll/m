"""Protein-conditioned codon β-VMAE with bidirectional RoPE attention."""
import math

import torch
from torch import nn
from torch.nn import functional as F

from .tokenizer import AA, CODONS, MASK, PAD, TOKEN, encode_cds
AA_ALPHABET = "ACDEFGHIKLMNPQRSTVWY*"
AA_IDS = torch.tensor([AA_ALPHABET.index(a) for a in AA] + [21, 21])
SYNONYMS = torch.tensor([[a == b for b in AA] for a in AA] + [[True]*64, [True]*64])


def collate_records(records):
    ids = [encode_cds(r["cds_dna"]) for r in records]
    tokens = torch.full((len(ids), max(map(len, ids))), PAD, dtype=torch.long)
    for row, encoded in enumerate(ids):
        tokens[row, :len(encoded)] = torch.tensor(encoded)
    return tokens


def mutable_positions(tokens):
    table = SYNONYMS.to(tokens.device)
    aa_ids = AA_IDS.to(tokens.device)
    return (tokens < PAD) & (table[tokens].sum(-1) > 1) & (aa_ids[tokens] != 20)


def corrupt(tokens, probability, generator=None):
    selected = (torch.rand(tokens.shape, device=tokens.device, generator=generator) < probability)
    selected &= mutable_positions(tokens)
    corrupted = tokens.clone()
    corrupted[selected] = MASK
    return corrupted, selected


def validate_targets(tokens):
    if (tokens.ndim != 2 or not tokens.numel() or tokens.dtype != torch.long
            or ((tokens < 0) | (tokens > PAD)).any()
            or (tokens == PAD).all(-1).any()):
        raise ValueError("Targets must contain codon IDs 0..63 and PAD=64; no empty/all-PAD rows")


def synonymous_logits(logits, target_tokens):
    validate_targets(target_tokens)
    if logits.shape != (*target_tokens.shape, 64):
        raise ValueError("Expected logits [B,L,64] aligned with targets")
    # Finite logits at padded positions avoid all-minus-infinity softmax/CE NaNs.
    return logits.float().masked_fill(~SYNONYMS.to(logits.device)[target_tokens], -torch.inf)


def codon_probabilities(logits, target_tokens, temperature=1.0, delta=None, alpha=1.0):
    """Fuse raw latent/direct logits, then enforce synonymous support; PAD outputs zero."""
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be positive")
    if not math.isfinite(alpha):
        raise ValueError("alpha must be finite")
    if delta is not None:
        if delta.shape not in (logits.shape, logits.shape[1:]):
            raise ValueError("delta must have shape [B,L,64] or [L,64]")
        logits = logits + alpha * delta
    probs = torch.softmax(synonymous_logits(logits, target_tokens) / temperature, -1)
    return probs * (target_tokens < PAD).unsqueeze(-1)


def nucleotide_probabilities(probs):
    """B,L,64 -> B,3L,4 in the baseline A,C,G,T channel order; preserves gradients."""
    mapping = probs.new_tensor([[[int(nt == base) for base in "ACGT"] for nt in codon]
                               for codon in CODONS])
    return torch.einsum("blc,cnk->blnk", probs, mapping).flatten(1, 2)


class RotaryAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout, max_positions):
        super().__init__()
        self.n_heads = n_heads
        self.dropout = dropout
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out = nn.Linear(d_model, d_model)
        head_dim = d_model // n_heads
        frequencies = 10000.0 ** (-torch.arange(0, head_dim, 2).float() / head_dim)
        angles = torch.arange(max_positions).float()[:, None] * frequencies
        self.register_buffer("cos", angles.cos(), persistent=False)
        self.register_buffer("sin", angles.sin(), persistent=False)

    def rotate(self, x):
        cos = self.cos[:x.size(-2)].to(x.dtype)
        sin = self.sin[:x.size(-2)].to(x.dtype)
        even, odd = x[..., 0::2], x[..., 1::2]
        return torch.stack((even * cos - odd * sin, even * sin + odd * cos), -1).flatten(-2)

    def forward(self, hidden, pad):
        batch, length, width = hidden.shape
        q, k, v = self.qkv(hidden).reshape(batch, length, 3, self.n_heads, -1).permute(2, 0, 3, 1, 4)
        attended = F.scaled_dot_product_attention(
            self.rotate(q), self.rotate(k), v, attn_mask=(~pad)[:, None, None, :],
            dropout_p=self.dropout if self.training else 0.0, is_causal=False)
        return self.out(attended.transpose(1, 2).reshape(batch, length, width))


class RotaryBlock(nn.Module):
    def __init__(self, d_model, n_heads, ff_dim, dropout, max_positions):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attention = RotaryAttention(d_model, n_heads, dropout, max_positions)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(nn.Linear(d_model, ff_dim), nn.GELU(), nn.Dropout(dropout),
                                nn.Linear(ff_dim, d_model))
        self.dropout = nn.Dropout(dropout)

    def forward(self, hidden, pad):
        hidden = hidden + self.dropout(self.attention(self.norm1(hidden), pad))
        return hidden + self.dropout(self.ff(self.norm2(hidden)))


class RotaryStack(nn.Module):
    def __init__(self, layers, d_model, n_heads, ff_dim, dropout, max_positions):
        super().__init__()
        self.layers = nn.ModuleList([RotaryBlock(d_model, n_heads, ff_dim, dropout, max_positions)
                                     for _ in range(layers)])
        self.norm = nn.LayerNorm(d_model)
        for parameter in self.parameters():
            if parameter.dim() > 1:
                nn.init.xavier_uniform_(parameter)

    def forward(self, hidden, pad):
        for layer in self.layers:
            hidden = layer(hidden, pad)
        return self.norm(hidden)


class CodonVMAE(nn.Module):
    def __init__(self, d_model=512, n_heads=8, encoder_layers=6, decoder_layers=6,
                 ff_dim=2048, latent_dim=256, dropout=0.1, max_codons=1500, max_positions=1536):
        super().__init__()
        if (n_heads <= 0 or d_model <= 0 or d_model % n_heads or (d_model // n_heads) % 2
                or min(latent_dim, ff_dim, encoder_layers, decoder_layers) <= 0
                or not 0 < max_codons <= max_positions):
            raise ValueError("Invalid model dimensions")
        self.max_codons = max_codons
        self.latent_dim = latent_dim
        self.embedding = nn.Embedding(66, d_model, padding_idx=PAD)
        # Protein identity is known during synonymous design, including masked positions.
        self.aa_embedding = nn.Embedding(22, d_model, padding_idx=21)
        self.register_buffer("aa_ids", AA_IDS.clone())
        def stack(layers):
            return RotaryStack(layers, d_model, n_heads, ff_dim, dropout, max_positions)

        self.encoder = stack(encoder_layers)
        self.mu = nn.Linear(d_model, latent_dim)
        self.logvar = nn.Linear(d_model, latent_dim)
        self.from_latent = nn.Linear(latent_dim, d_model)
        self.decoder = stack(decoder_layers)
        self.output = nn.Linear(d_model, 64)

    def encode(self, tokens, target_tokens):
        validate_targets(target_tokens)
        if tokens.shape != target_tokens.shape or tokens.size(1) > self.max_codons:
            raise ValueError("Mismatched shapes or CDS exceeds max_codons; truncation is prohibited")
        pad = target_tokens == PAD
        if (tokens.dtype != torch.long or ((tokens < 0) | (tokens > MASK)).any()
                or ((tokens == PAD) != pad).any()):
            raise ValueError("Encoder tokens must be codons/MASK with aligned PAD positions")
        hidden = self.embedding(tokens) + self.aa_embedding(self.aa_ids[target_tokens])
        hidden = self.encoder(hidden, pad)
        valid = (~pad).unsqueeze(-1)
        return self.mu(hidden) * valid, self.logvar(hidden).clamp(-8, 8) * valid

    def decode(self, z, target_tokens):
        """Return raw D(z,a); target_tokens supplies AA identity/PAD only, never codon embeddings."""
        validate_targets(target_tokens)
        if z.shape != (*target_tokens.shape, self.latent_dim) or z.size(1) > self.max_codons:
            raise ValueError("Latent and target shapes do not match")
        hidden = self.from_latent(z) + self.aa_embedding(self.aa_ids[target_tokens])
        return self.output(self.decoder(hidden, target_tokens == PAD))

    def forward(self, tokens, target_tokens, sample=True):
        mu, logvar = self.encode(tokens, target_tokens)
        z = mu + torch.randn_like(mu) * torch.exp(0.5 * logvar) if sample else mu
        z = z * (target_tokens < PAD).unsqueeze(-1)
        return self.decode(z, target_tokens), mu, logvar


def vae_loss(logits, tokens, mu, logvar, selected, beta, masked_weight=1.0, full_weight=1.0):
    valid = tokens < PAD
    safe_targets = tokens.clamp_max(63)
    ce = F.cross_entropy(synonymous_logits(logits, tokens).transpose(1, 2),
                         safe_targets, reduction="none")
    reconstruction = ce[valid].mean()
    selected = selected & valid
    masked = ce[selected].mean() if selected.any() else ce.sum() * 0
    kl_per_token = -0.5 * (1 + logvar.float() - mu.float().square() - logvar.float().exp()).sum(-1)
    kl = kl_per_token[valid].mean()
    loss = full_weight * reconstruction + masked_weight * masked + beta * kl
    return loss, {"reconstruction_ce": reconstruction.detach(), "masked_ce": masked.detach(),
                  "kl_per_token": kl.detach(), "kl_per_dimension": kl.detach() / mu.size(-1)}
