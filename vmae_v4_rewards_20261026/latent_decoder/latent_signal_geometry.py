"""Basis-invariant second-moment signal/noise diagnostics, not information estimates."""
import torch


def signal_noise_geometry(signal, noise):
    assert signal.ndim == 2 and signal.shape == noise.shape and signal.shape[0] == signal.shape[1]
    signal, noise = signal.double(), noise.double()
    assert torch.isfinite(signal).all() and torch.isfinite(noise).all()
    factor = torch.linalg.cholesky((noise+noise.T)/2)
    left = torch.linalg.solve_triangular(factor, (signal+signal.T)/2, upper=False)
    whitened = torch.linalg.solve_triangular(factor, left.T, upper=False).T
    eigen = torch.linalg.eigvalsh((whitened+whitened.T)/2).clamp_min(0).flip(0)
    return dict(generalized_eigenvalues=eigen.tolist(),
        modes_above={str(t): int((eigen>t).sum()) for t in (.001, .01, .1, 1.)},
        participation=float(eigen.sum().square()/eigen.square().sum().clamp_min(1e-30)),
        interpretation='Eigenvalues of within-AA mean covariance relative to average posterior noise; invariant to invertible linear coordinate changes. Not mutual information or a causal decoder-use count.')


def pool_signal_geometry(pool, noise):
    within = pool.cross/pool.n-(pool.aa_total.T/pool.aa_n.clamp_min(1))@pool.aa_total/pool.n
    return signal_noise_geometry(within, noise)
