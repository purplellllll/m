"""Small ablation model; simulator and folding features are external inputs.

This module does not make the discrete simulator or structure prediction
differentiable, and an occupancy prediction is not a protein-output estimate.
"""

import math

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class _GraphLayer(nn.Module):
    def __init__(self, width, dropout):
        super().__init__()
        self.self_proj = nn.Linear(width, width)
        self.sequence_proj = nn.Linear(width, width, bias=False)
        self.pair_proj = nn.Linear(width, width, bias=False)
        self.norm = nn.LayerNorm(width)
        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def mean_neighbors(x, edges):
        source, destination = edges
        sums = torch.zeros_like(x).index_add(0, destination, x[source])
        degrees = x.new_zeros((x.shape[0], 1)).index_add(
            0, destination, x.new_ones((destination.numel(), 1))
        )
        return sums / degrees.clamp_min(1)

    def forward(self, x, sequence_edges, pair_edges):
        update = self.self_proj(x)
        update = update + self.sequence_proj(self.mean_neighbors(x, sequence_edges))
        update = update + self.pair_proj(self.mean_neighbors(x, pair_edges))
        return self.norm(x + self.dropout(F.gelu(update)))


class GraphSequenceScorer(nn.Module):
    """Predict codon log1p RPF counts and a separate, unbounded scalar TE label.

    Inputs are right-padded tensors on one device:
      codon_ids: int64 [B,L], valid codons 0..63, padding 64.
      valid_mask: bool [B,L], a nonempty valid prefix per sequence.
      sim_feature: float [B,L], raw nonnegative sTASEP counts, log1p inside.
      geometry: int64 [B,L,3], (pair 0..3, angle 0..3, bucket 0..2).
      pair_edges: int64 [2,E], undirected structure edges in flattened B*L
        indexing. Either/both orientations are accepted and deduplicated.

    Structural edges must stay inside one sample and avoid padding. Adjacent
    sequence edges are generated here with separate message-passing weights.
    Output profile is zero at padding; TE must be trained against its own labels.
    """

    def __init__(self, encoder="transformer", use_graph=True, use_sim=True,
                 d_model=64, num_layers=2, graph_layers=2, nhead=4, dropout=0.1):
        super().__init__()
        if encoder not in {"transformer", "bilstm"}:
            raise ValueError("encoder must be transformer or bilstm")
        if d_model < 2 or d_model % 2 or d_model % nhead:
            raise ValueError("d_model must be positive, even, and divisible by nhead")
        if num_layers < 1 or (use_graph and graph_layers < 1):
            raise ValueError("enabled encoders need at least one layer")
        self.encoder_kind = encoder
        self.use_graph, self.use_sim = use_graph, use_sim
        self.codon_embedding = nn.Embedding(65, d_model, padding_idx=64)
        self.geometry_embeddings = nn.ModuleList([
            nn.Embedding(4, d_model), nn.Embedding(4, d_model), nn.Embedding(3, d_model)
        ])
        self.sim_projection = nn.Linear(1, d_model) if use_sim else None
        self.graph = nn.ModuleList([
            _GraphLayer(d_model, dropout) for _ in range(graph_layers if use_graph else 0)
        ])
        if encoder == "transformer":
            layer = nn.TransformerEncoderLayer(
                d_model, nhead, dim_feedforward=2*d_model, dropout=dropout,
                activation="gelu", batch_first=True, norm_first=True,
            )
            self.encoder = nn.TransformerEncoder(
                layer, num_layers, norm=nn.LayerNorm(d_model), enable_nested_tensor=False,
            )
        else:
            self.encoder = nn.LSTM(
                d_model, d_model//2, num_layers=num_layers, batch_first=True,
                bidirectional=True, dropout=dropout if num_layers > 1 else 0,
            )
        self.profile_head = nn.Linear(d_model, 1)
        self.te_head = nn.Sequential(nn.Linear(d_model, d_model//2), nn.GELU(),
                                     nn.Dropout(dropout), nn.Linear(d_model//2, 1))

    @staticmethod
    def _validate(codon_ids, valid_mask, sim_feature, geometry, pair_edges):
        if codon_ids.ndim != 2 or codon_ids.dtype != torch.long:
            raise ValueError("codon_ids must be int64 [B,L]")
        if valid_mask.shape != codon_ids.shape or valid_mask.dtype != torch.bool:
            raise ValueError("valid_mask must be bool [B,L]")
        if sim_feature.shape != codon_ids.shape or not sim_feature.is_floating_point():
            raise ValueError("sim_feature must be floating point [B,L]")
        if geometry.shape != (*codon_ids.shape, 3) or geometry.dtype != torch.long:
            raise ValueError("geometry must be int64 [B,L,3]")
        if pair_edges.ndim != 2 or pair_edges.shape[0] != 2 or pair_edges.dtype != torch.long:
            raise ValueError("pair_edges must be int64 [2,E]")
        tensors = (valid_mask, sim_feature, geometry, pair_edges)
        if any(t.device != codon_ids.device for t in tensors):
            raise ValueError("all inputs must be on the same device")
        lengths = valid_mask.sum(1)
        expected = torch.arange(codon_ids.shape[1], device=codon_ids.device)[None, :] < lengths[:, None]
        if not lengths.numel() or (lengths == 0).any() or not torch.equal(expected, valid_mask):
            raise ValueError("each sample needs a nonempty right-padded valid prefix")
        if ((codon_ids[valid_mask] < 0) | (codon_ids[valid_mask] >= 64)).any() or (codon_ids[~valid_mask] != 64).any():
            raise ValueError("valid codons must be 0..63 and padding must be 64")
        sim_valid = sim_feature[valid_mask]
        if not torch.isfinite(sim_valid).all() or (sim_valid < 0).any():
            raise ValueError("valid simulation counts must be finite and nonnegative")
        geo_valid = geometry[valid_mask]
        limits = geometry.new_tensor([4, 4, 3])
        if ((geo_valid < 0) | (geo_valid >= limits)).any():
            raise ValueError("geometry categories are outside pair/angle/bucket ranges")
        if pair_edges.numel():
            if (pair_edges < 0).any() or (pair_edges >= codon_ids.numel()).any():
                raise ValueError("structure edge index outside flattened batch")
            if not valid_mask.flatten()[pair_edges].all():
                raise ValueError("structure edges cannot touch padding")
            if (pair_edges[0] // codon_ids.shape[1] != pair_edges[1] // codon_ids.shape[1]).any():
                raise ValueError("structure edges cannot cross samples")
        return lengths

    @staticmethod
    def _positions(length, width, device, dtype):
        position = torch.arange(length, device=device, dtype=dtype)[:, None]
        frequency = torch.exp(torch.arange(0, width, 2, device=device, dtype=dtype)
                              * (-math.log(10000.0) / width))
        values = torch.zeros((length, width), device=device, dtype=dtype)
        values[:, 0::2], values[:, 1::2] = torch.sin(position*frequency), torch.cos(position*frequency)
        return values

    def forward(self, codon_ids, valid_mask, sim_feature, geometry, pair_edges):
        lengths = self._validate(codon_ids, valid_mask, sim_feature, geometry, pair_edges)
        batch_size, length = codon_ids.shape
        geometry = geometry.masked_fill(~valid_mask[..., None], 0)
        x = self.codon_embedding(codon_ids)
        for i, embedding in enumerate(self.geometry_embeddings):
            x = x + embedding(geometry[..., i])
        if self.use_sim:
            sim = torch.log1p(sim_feature.masked_fill(~valid_mask, 0))
            x = x + self.sim_projection(sim[..., None])
        x = x.masked_fill(~valid_mask[..., None], 0)
        if self.use_graph:
            nodes = torch.arange(batch_size*length, device=x.device).reshape(batch_size, length)
            adjacent = valid_mask[:, :-1] & valid_mask[:, 1:]
            sequence_edges = torch.stack((nodes[:, :-1][adjacent], nodes[:, 1:][adjacent]))
            sequence_edges = torch.cat((sequence_edges, sequence_edges.flip(0)), dim=1)
            pair_edges = torch.unique(torch.cat((pair_edges, pair_edges.flip(0)), dim=1), dim=1)
            x = x.reshape(-1, x.shape[-1])
            for layer in self.graph:
                x = layer(x, sequence_edges, pair_edges)
            x = x.reshape(batch_size, length, -1).masked_fill(~valid_mask[..., None], 0)
        if self.encoder_kind == "transformer":
            x = x + self._positions(length, x.shape[-1], x.device, x.dtype)
            x = self.encoder(x, src_key_padding_mask=~valid_mask)
        else:
            packed = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
            encoded, _ = self.encoder(packed)
            x, _ = pad_packed_sequence(encoded, batch_first=True, total_length=length)
        x = x.masked_fill(~valid_mask[..., None], 0)
        profile = F.softplus(self.profile_head(x).squeeze(-1)).masked_fill(~valid_mask, 0)
        pooled = x.sum(1) / lengths[:, None]
        te = self.te_head(pooled).squeeze(-1)
        return profile, te
