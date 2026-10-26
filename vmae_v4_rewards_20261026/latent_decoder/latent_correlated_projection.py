"""A compact latent with the exact projected teacher covariance and standard normal prior."""
import copy

import torch

from .model import CodonVMAE, PAD


def projected_kl(mu, source_logvar, basis):
    """KL[N(mu,U' diag(exp(lv)) U) || N(0,I)], differentiable in both heads."""
    assert source_logvar.shape[:-1] == mu.shape[:-1]
    assert basis.shape == (source_logvar.shape[-1], mu.shape[-1])
    covariance = torch.einsum('...k,ki,kj->...ij', source_logvar.exp(), basis, basis)
    factor = torch.linalg.cholesky(covariance)
    logdet = 2 * factor.diagonal(dim1=-2, dim2=-1).log().sum(-1)
    trace = covariance.diagonal(dim1=-2, dim2=-1).sum(-1)
    return .5 * (trace + mu.square().sum(-1) - mu.shape[-1] - logdet)


class ProjectedVMAE(CodonVMAE):
    def __init__(self, noise_dim, **config):
        super().__init__(**config)
        assert self.latent_dim < noise_dim
        self.logvar = torch.nn.Linear(self.mu.in_features, noise_dim)
        self.register_buffer('noise_projection', torch.zeros(noise_dim, self.latent_dim))

    def distribution(self, inputs, targets):
        return super().encode(inputs, targets)

    def encode(self, inputs, targets):
        mu, source_logvar = self.distribution(inputs, targets)
        marginal = (source_logvar.exp() @ self.noise_projection.square()).log()
        return mu, marginal * (targets < PAD)[..., None]

    def sample(self, mu, source_logvar, noise):
        assert noise.shape == source_logvar.shape
        return mu + (noise * (.5 * source_logvar).exp()) @ self.noise_projection

    def forward(self, inputs, targets, sample=True):
        mu, source_logvar = self.distribution(inputs, targets)
        z = self.sample(mu, source_logvar, torch.randn_like(source_logvar)) if sample else mu
        return self.decode(z * (targets < PAD)[..., None], targets), mu, source_logvar


@torch.no_grad()
def project_state(teacher, report):
    basis = torch.tensor(report['basis'], dtype=torch.float64)
    center = torch.tensor(report['mean'], dtype=torch.float64)
    source, dimension = basis.shape
    assert source == teacher['config']['model']['latent_dim'] and dimension < source
    torch.testing.assert_close(basis.T @ basis, torch.eye(dimension, dtype=torch.float64), rtol=1e-8, atol=1e-8)
    assert teacher['recipe']['sigma'] is None and teacher['recipe']['skip_layer'] == 0
    assert teacher['recipe']['gain'] == 1.
    result = dict(config=copy.deepcopy(teacher['config']), recipe=copy.deepcopy(teacher['recipe']),
        model=dict(teacher['model']), architecture='projected_full_covariance', noise_dim=source)
    result['config']['model']['latent_dim'] = dimension
    old, new = teacher['model'], result['model']
    dtype = old['mu.weight'].dtype
    new['mu.weight'] = (basis.T @ old['mu.weight'].double()).to(dtype)
    # Keep uncentered q mean: U' N(0,I) is exactly N(0,I), with no fitted prior translation.
    new['mu.bias'] = (basis.T @ old['mu.bias'].double()).to(dtype)
    new['from_latent.weight'] = (old['from_latent.weight'].double() @ basis).to(dtype)
    discarded_mean = center - basis @ (basis.T @ center)
    new['from_latent.bias'] = (old['from_latent.bias'].double() + old['from_latent.weight'].double() @ discarded_mean).to(dtype)
    new['noise_projection'] = basis.to(dtype)
    return result


def load_projected(state, device):
    assert state['architecture'] == 'projected_full_covariance'
    model = ProjectedVMAE(noise_dim=state['noise_dim'], **state['config']['model']).to(device)
    model.load_state_dict(state['model'], strict=True)
    return model.eval()
