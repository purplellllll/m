"""Controlled early fusion of aligned graph/physics features before attention."""
import torch
from torch import nn
from .fusion_encoder_model import FusionEncoderModel
from .model import GraphSequenceScorer


MODES = {
    'parallel': dict(early_physics=False, early_graph=False),
    'physics_polish': dict(early_physics=True, early_graph=False),
    'gnn_polish': dict(early_physics=False, early_graph=True),
    'physics_gnn_polish': dict(early_physics=True, early_graph=True),
    'shuffled_physics': dict(early_physics=True, early_graph=True, shuffle_physics=True),
    'self_graph': dict(early_physics=True, early_graph=True, neighbors='self'),
    'sequence_edges': dict(early_physics=True, early_graph=True, neighbors='sequence'),
    'diagonal_attention': dict(early_physics=True, early_graph=True, global_attention=False),
}


class GraphPolishModel(FusionEncoderModel):
    def __init__(self, mode='physics_gnn_polish', width=424):
        if mode not in MODES:
            raise ValueError('Unknown polish mode')
        super().__init__(encoder='transformer', width=width)
        self.mode = mode
        options = MODES[mode]
        self.early_physics = options.get('early_physics', False)
        self.early_graph = options.get('early_graph', False)
        self.shuffle_physics = options.get('shuffle_physics', False)
        self.neighbors = options.get('neighbors', 'all')
        self.global_attention = options.get('global_attention', True)
        self.polish_input = nn.Sequential(nn.Linear(3*width, width), nn.GELU(), nn.LayerNorm(width))
        self.input_logit = nn.Parameter(torch.tensor(-2.197224577))  # sigmoid = .1
        self._polish_cache = {}
        self._polish_valid = None
        self._physics_override = None
        self.codons.register_forward_hook(self._save_codons)
        self.physics.register_forward_hook(self._save_physics)
        self.graph[-1].register_forward_hook(self._save_graph)
        for layer in self.graph:
            layer.register_forward_pre_hook(self._restrict_neighbors)
        self.context.register_forward_pre_hook(self._prepare_context, with_kwargs=True)

    def _save_codons(self, module, args, value):
        self._polish_cache['codons'] = value

    def _save_physics(self, module, args, value):
        self._polish_cache['physics'] = value

    def _save_graph(self, module, args, value):
        self._polish_cache['graph'] = value

    def _restrict_neighbors(self, module, args):
        x, sequence, pair = args
        if self.neighbors == 'self':
            return x, sequence[:, :0], pair[:, :0]
        if self.neighbors == 'sequence':
            return x, sequence, pair[:, :0]
        return args

    @staticmethod
    def permute_valid(x, valid):
        output = x.clone()
        for j, length in enumerate(valid.sum(1).tolist()):
            generator = torch.Generator().manual_seed(94031+length)
            order = torch.randperm(length, generator=generator).to(x.device)
            output[j, :length] = x[j, :length][order]
        return output

    def _prepare_context(self, module, args, kwargs):
        embed = self._polish_cache['codons']
        p = self._polish_cache['physics']
        g = self._polish_cache['graph'].reshape_as(embed)
        valid = self._polish_valid
        if self.shuffle_physics or self._physics_override == 'shuffle':
            p = self.permute_valid(p, valid)
        if not self.early_physics or self._physics_override == 'zero':
            p = torch.zeros_like(p)
        if not self.early_graph:
            g = torch.zeros_like(g)
        joint = self.polish_input(torch.cat((embed, p, g), -1))
        z = embed + self.input_logit.sigmoid()*joint
        z = z.masked_fill(~valid[..., None], 0)
        positions = GraphSequenceScorer._positions(z.shape[1], z.shape[2], z.device, z.dtype)
        if not self.global_attention:
            kwargs = dict(kwargs, mask=~torch.eye(z.shape[1], device=z.device, dtype=torch.bool))
        return (z+positions, *args[1:]), kwargs

    def forward(self, codons, valid, physics, structure, pair_edges, physics_override=None):
        if physics_override not in {None, 'zero', 'shuffle'}:
            raise ValueError('Unknown early physics intervention')
        try:
            self._polish_valid = valid
            self._physics_override = physics_override
            return super().forward(codons, valid, physics, structure, pair_edges)
        finally:
            self._polish_cache.clear()
            self._polish_valid = None
            self._physics_override = None
