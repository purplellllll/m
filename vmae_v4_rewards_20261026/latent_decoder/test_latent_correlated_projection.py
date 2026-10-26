import unittest

import torch

from .latent_correlated_projection import load_projected, project_state, projected_kl
from .latent_recovery_gpu import BASE
from .model import CodonVMAE


class CorrelatedProjectionTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(18)

    def test_kl_and_gradients_equal_multivariate_normal(self):
        basis = torch.linalg.qr(torch.randn(7, 3, dtype=torch.float64))[0]
        mu = torch.randn(2, 4, 3, dtype=torch.float64, requires_grad=True)
        lv = torch.randn(2, 4, 7, dtype=torch.float64, requires_grad=True)
        actual = projected_kl(mu, lv, basis)
        covariance = basis.T @ torch.diag_embed(lv.exp()) @ basis
        expected = torch.distributions.kl_divergence(
            torch.distributions.MultivariateNormal(mu, covariance_matrix=covariance),
            torch.distributions.MultivariateNormal(torch.zeros_like(mu), covariance_matrix=torch.eye(3, dtype=mu.dtype)))
        torch.testing.assert_close(actual, expected)
        ga = torch.autograd.grad(actual.sum(), (mu, lv), retain_graph=True)
        ge = torch.autograd.grad(expected.sum(), (mu, lv))
        for a, e in zip(ga, ge):
            torch.testing.assert_close(a, e)
        diagonal = .5 * (covariance.diagonal(dim1=-2, dim2=-1).sum(-1) + mu.square().sum(-1)
            - 3 - covariance.diagonal(dim1=-2, dim2=-1).log().sum(-1))
        self.assertGreater(float((actual - diagonal).mean().detach()), .01)

    def test_shared_source_noise_preserves_projected_teacher_distribution(self):
        config = dict(d_model=16, n_heads=2, encoder_layers=1, decoder_layers=1, ff_dim=32,
            latent_dim=8, dropout=0., max_codons=64, max_positions=64)
        teacher = CodonVMAE(**config).eval()
        basis = torch.linalg.qr(torch.randn(8, 3, dtype=torch.float64))[0]
        center = torch.randn(8, dtype=torch.float64)
        state = project_state(dict(config=dict(model=config), recipe=BASE, model=teacher.state_dict()),
            dict(basis=basis.tolist(), mean=center.tolist()))
        model = load_projected(state, 'cpu')
        h = torch.randn(2, 9, 16)
        mu, lv = teacher.mu(h), teacher.logvar(h).clamp(-8, 8)
        new_mu, new_lv = model.mu(h), model.logvar(h).clamp(-8, 8)
        noise = torch.randn_like(lv)
        z = model.sample(new_mu, new_lv, noise)
        teacher_z = mu + noise * (.5 * lv).exp()
        u = basis.float()
        torch.testing.assert_close(z, teacher_z @ u)
        expected = teacher.from_latent(center.float() + (teacher_z - center.float()) @ u @ u.T)
        torch.testing.assert_close(model.from_latent(z), expected, rtol=2e-5, atol=2e-6)
        tokens = torch.tensor([[1, 2, 4, 6]])
        logits, actual_mu, source_lv = model(tokens, tokens)
        self.assertEqual(logits.shape, (1, 4, 64))
        self.assertTrue(torch.isfinite(projected_kl(actual_mu, source_lv, model.noise_projection)).all())


if __name__ == '__main__':
    unittest.main()
