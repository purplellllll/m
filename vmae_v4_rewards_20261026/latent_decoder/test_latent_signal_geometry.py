import unittest

import torch

from .latent_signal_geometry import signal_noise_geometry


class SignalGeometryTests(unittest.TestCase):
    def test_known_signal_modes(self):
        result = signal_noise_geometry(torch.diag(torch.tensor([9., 4., 0.])), torch.eye(3))
        torch.testing.assert_close(torch.tensor(result['generalized_eigenvalues']), torch.tensor([9.,4.,0.]))
        self.assertEqual(result['modes_above']['1.0'], 2)

    def test_invariant_under_rotation_and_unequal_coordinate_scaling(self):
        torch.manual_seed(8)
        x, y = torch.randn(7, 7, dtype=torch.float64), torch.randn(7, 7, dtype=torch.float64)
        signal, noise = x@x.T, y@y.T+torch.eye(7, dtype=torch.float64)
        transform = torch.linalg.qr(torch.randn_like(x))[0]@torch.diag(torch.tensor([.1,.2,.5,1.,2.,5.,10.], dtype=torch.float64))
        original = signal_noise_geometry(signal, noise)
        changed = signal_noise_geometry(transform@signal@transform.T, transform@noise@transform.T)
        torch.testing.assert_close(torch.tensor(original['generalized_eigenvalues']), torch.tensor(changed['generalized_eigenvalues']))
        self.assertEqual(original['modes_above'], changed['modes_above'])


if __name__ == '__main__':
    unittest.main()
