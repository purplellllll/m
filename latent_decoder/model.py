"""Small conditional variational masked autoencoder, not an exact LSCO reproduction."""
import math

import torch
from torch import nn
from torch.nn import functional as F

# Preserve dataset.py's existing 0..63 codon order without importing its I/O side effects.
CODONS = ("TTT TTC TTA TTG CTT CTC CTA CTG ATT ATC ATA ATG GTT GTC GTA GTG "
          "TAT TAC TAA TAG CAT CAC CAA CAG AAT AAC AAA AAG GAT GAC GAA GAG "
          "TCT TCC TCA TCG CCT CCC CCA CCG ACT ACC ACA ACG GCT GCC GCA GCG "
          "TGT TGC TGA TGG CGT CGC CGA CGG AGT AGC AGA AGG GGT GGC GGA GGG").split()
AA = list("FFLLLLLLIIIMVVVVYY**HHQQNNKKDDEESSSSPPPPTTTTAAAACC*WRRRRSSRRGGGG")
PAD, MASK = 64, 65
TOKEN = {codon: i for i, codon in enumerate(CODONS)}
AA_ALPHABET = "ACDEFGHIKLMNPQRSTVWY*"
AA_IDS = torch.tensor([AA_ALPHABET.index(a) for a in AA] + [21, 21])
SYNONYMS = torch.tensor([[a == b for b in AA] for a in AA] + [[True]*64, [True]*64])


def encode_cds(sequence):
    sequence = sequence.upper().replace("U", "T")
    if len(sequence) % 3 or not sequence or set(sequence) - set("ACGT"):
        raise ValueError("CDS must contain complete unambiguous codons")
    ids = [TOKEN[sequence[i:i+3]] for i in range(0, len(sequence), 3)]
    protein = "".join(AA[i] for i in ids)
    if not sequence.startswith("ATG") or protein[-1] != "*" or "*" in protein[:-1]:
        raise ValueError("CDS must start with ATG and contain only one terminal stop")
    return ids


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


def synonymous_logits(logits, target_tokens):
    # Finite logits at padded positions avoid all-minus-infinity softmax/CE NaNs.
    return logits.float().masked_fill(~SYNONYMS.to(logits.device)[target_tokens], -1e9)


def codon_probabilities(logits, target_tokens, temperature=1.0):
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    probs = torch.softmax(synonymous_logits(logits, target_tokens) / temperature, -1)
    return probs * (target_tokens < PAD).unsqueeze(-1)


def nucleotide_probabilities(probs):
    """B,L,64 -> B,3L,4 in the baseline A,C,G,T channel order; preserves gradients."""
    mapping = probs.new_tensor([[[int(nt == base) for base in "ACGT"] for nt in codon]
                               for codon in CODONS])
    return torch.einsum("blc,cnk->blnk", probs, mapping).flatten(1, 2)


class CodonVMAE(nn.Module):
    def __init__(self, d_model=256, n_heads=8, encoder_layers=4, decoder_layers=4,
                 ff_dim=1024, latent_dim=64, dropout=0.1, max_codons=1500):
        super().__init__()
        if d_model % 2 or d_model % n_heads or latent_dim <= 0:
            raise ValueError("Invalid model dimensions")
        self.max_codons = max_codons
        self.embedding = nn.Embedding(66, d_model, padding_idx=PAD)
        # Protein identity is known during synonymous design, including masked positions.
        self.aa_embedding = nn.Embedding(22, d_model, padding_idx=21)
        self.register_buffer("aa_ids", AA_IDS.clone())
        position = torch.arange(max_codons).unsqueeze(1)
        frequencies = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.) / d_model))
        pe = torch.zeros(max_codons, d_model)
        pe[:, 0::2] = torch.sin(position * frequencies)
        pe[:, 1::2] = torch.cos(position * frequencies)
        self.register_buffer("positions", pe.unsqueeze(0))

        def stack(layers):
            layer = nn.TransformerEncoderLayer(d_model, n_heads, ff_dim, dropout,
                                               activation="gelu", batch_first=True, norm_first=True)
            result = nn.TransformerEncoder(layer, layers, nn.LayerNorm(d_model), enable_nested_tensor=False)
            # TransformerEncoder clones its prototype; initialize layers independently.
            for parameter in result.parameters():
                if parameter.dim() > 1:
                    nn.init.xavier_uniform_(parameter)
            return result

        self.encoder = stack(encoder_layers)
        self.mu = nn.Linear(d_model, latent_dim)
        self.logvar = nn.Linear(d_model, latent_dim)
        self.from_latent = nn.Linear(latent_dim, d_model)
        self.decoder = stack(decoder_layers)
        self.output = nn.Linear(d_model, 64)

    def encode(self, tokens, target_tokens):
        if tokens.shape != target_tokens.shape or tokens.size(1) > self.max_codons:
            raise ValueError("Mismatched shapes or CDS exceeds max_codons; truncation is prohibited")
        pad = target_tokens == PAD
        hidden = self.embedding(tokens) + self.aa_embedding(self.aa_ids[target_tokens])
        hidden = hidden + self.positions[:, :tokens.size(1)]
        hidden = self.encoder(hidden, src_key_padding_mask=pad)
        valid = (~pad).unsqueeze(-1)
        return self.mu(hidden) * valid, self.logvar(hidden).clamp(-8, 8) * valid

    def decode(self, z, target_tokens):
        if z.shape[:2] != target_tokens.shape or z.size(1) > self.max_codons:
            raise ValueError("Latent and target shapes do not match")
        hidden = self.from_latent(z) + self.positions[:, :z.size(1)]
        return self.output(self.decoder(hidden, src_key_padding_mask=target_tokens == PAD))

    def forward(self, tokens, target_tokens, sample=True):
        mu, logvar = self.encode(tokens, target_tokens)
        z = mu + torch.randn_like(mu) * torch.exp(0.5 * logvar) if sample else mu
        z = z * (target_tokens < PAD).unsqueeze(-1)
        return self.decode(z, target_tokens), mu, logvar


def vae_loss(logits, tokens, mu, logvar, selected, beta, masked_weight=1.0):
    valid = tokens < PAD
    safe_targets = tokens.clamp_max(63)
    ce = F.cross_entropy(synonymous_logits(logits, tokens).transpose(1, 2),
                         safe_targets, reduction="none")
    reconstruction = ce[valid].mean()
    masked = ce[selected].mean() if selected.any() else ce.sum() * 0
    kl_per_token = -0.5 * (1 + logvar.float() - mu.float().square() - logvar.float().exp()).mean(-1)
    kl = kl_per_token[valid].mean()
    loss = reconstruction + masked_weight * masked + beta * kl
    return loss, {"reconstruction_ce": reconstruction.detach(), "masked_ce": masked.detach(),
                  "kl_per_dimension": kl.detach()}
