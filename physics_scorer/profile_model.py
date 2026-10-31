"""Three separately embedded branches for an observed codon-profile benchmark."""
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from .model import _GraphLayer, GraphSequenceScorer


class AlignedProfileModel(nn.Module):
    def __init__(self, physics=True, graph=True, context=True, encoder="transformer", width=32):
        super().__init__()
        if not any((physics, graph, context)):
            raise ValueError("At least one branch is required")
        self.enabled = (physics, graph, context)
        self.encoder_kind = encoder
        self.codons = nn.Embedding(65, width, padding_idx=64) if graph or context else None
        self.physics = nn.Sequential(nn.Linear(2, width), nn.GELU(), nn.LayerNorm(width)) if physics else None
        self.structure = nn.Linear(5, width) if graph else None
        self.graph = nn.ModuleList([_GraphLayer(width, .1) for _ in range(2)]) if graph else None
        if context:
            if encoder == "transformer":
                layer = nn.TransformerEncoderLayer(width, 4, width*2, .1, "gelu", batch_first=True, norm_first=True)
                self.context = nn.TransformerEncoder(layer, 1, norm=nn.LayerNorm(width), enable_nested_tensor=False)
            elif encoder == "bilstm":
                self.context = nn.LSTM(width, width//2, batch_first=True, bidirectional=True)
            else:
                raise ValueError("Unsupported context encoder")
        self.local_heads = nn.ModuleDict({str(i): nn.Linear(width, 1) for i, on in enumerate(self.enabled) if on})
        self.fusion = nn.Sequential(nn.Linear(width*sum(self.enabled), width), nn.GELU(), nn.Linear(width, 1))

    def forward(self, codons, valid, physics, structure, pair_edges):
        if codons.dtype != torch.long or valid.dtype != torch.bool or codons.shape != valid.shape:
            raise ValueError("Invalid codon or padding tensors")
        lengths = valid.sum(1)
        expected = torch.arange(codons.shape[1], device=codons.device)[None] < lengths[:, None]
        if not lengths.numel() or (lengths == 0).any() or not torch.equal(expected, valid):
            raise ValueError("Samples require nonempty right-padded prefixes")
        if physics.shape != (*codons.shape, 2) or structure.shape != (*codons.shape, 5):
            raise ValueError("Misaligned continuous features")
        if ((codons[valid] < 0) | (codons[valid] >= 64)).any() or (codons[~valid] != 64).any():
            raise ValueError("Invalid codon IDs")
        if not torch.isfinite(physics[valid]).all() or not torch.isfinite(structure[valid]).all():
            raise ValueError("Nonfinite valid feature")
        b, length = codons.shape
        if pair_edges.dtype != torch.long or pair_edges.ndim != 2 or pair_edges.shape[0] != 2:
            raise ValueError("Invalid edge tensor")
        if pair_edges.numel():
            if (pair_edges < 0).any() or (pair_edges >= codons.numel()).any():
                raise ValueError("Edge out of bounds")
            if not valid.flatten()[pair_edges].all() or (pair_edges[0]//length != pair_edges[1]//length).any():
                raise ValueError("Edges must stay within valid positions of one transcript")
        features, local = [], {}
        embed = self.codons(codons) if self.codons is not None else None
        if self.enabled[0]:
            features.append(self.physics(physics.masked_fill(~valid[..., None], 0)))
            local["physics"] = F.softplus(self.local_heads["0"](features[-1]).squeeze(-1)).masked_fill(~valid, 0)
        if self.enabled[1]:
            h = embed + self.structure(structure.masked_fill(~valid[..., None], 0))
            h = h.masked_fill(~valid[..., None], 0)
            nodes = torch.arange(b*length, device=codons.device).reshape(b, length)
            adjacent = valid[:, :-1] & valid[:, 1:]
            edges = torch.stack((nodes[:, :-1][adjacent], nodes[:, 1:][adjacent]))
            edges = torch.cat((edges, edges.flip(0)), 1)
            pairs = torch.unique(torch.cat((pair_edges, pair_edges.flip(0)), 1), dim=1)
            h = h.reshape(b*length, -1)
            for layer in self.graph:
                h = layer(h, edges, pairs)
            h = h.reshape(b, length, -1).masked_fill(~valid[..., None], 0)
            features.append(h)
            local["gnn"] = F.softplus(self.local_heads["1"](h).squeeze(-1)).masked_fill(~valid, 0)
        if self.enabled[2]:
            if self.encoder_kind == "transformer":
                h = embed + GraphSequenceScorer._positions(length, embed.shape[-1], embed.device, embed.dtype)
                h = self.context(h, src_key_padding_mask=~valid)
            else:
                packed = pack_padded_sequence(embed, lengths.cpu(), batch_first=True, enforce_sorted=False)
                packed, _ = self.context(packed)
                h, _ = pad_packed_sequence(packed, batch_first=True, total_length=length)
            h = h.masked_fill(~valid[..., None], 0)
            features.append(h)
            local["context"] = F.softplus(self.local_heads["2"](h).squeeze(-1)).masked_fill(~valid, 0)
        prediction = F.softplus(self.fusion(torch.cat(features, -1)).squeeze(-1)).masked_fill(~valid, 0)
        return prediction, local


def masked_transcript_mse(prediction, target, observed):
    if prediction.shape != target.shape or target.shape != observed.shape:
        raise ValueError("Prediction, label and observed mask are not aligned")
    if not observed.any(1).all():
        raise ValueError("Every transcript needs observed labels")
    residual = (prediction-target.masked_fill(~observed, 0)).square().masked_fill(~observed, 0)
    return (residual.sum(1)/observed.sum(1)).mean()
