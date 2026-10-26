import unittest
from collections import Counter

from .latent_batch_recovery_gpu import accumulation_windows, lr_factor


class BatchRecoveryTests(unittest.TestCase):
    def setUp(self):
        self.batches = [(g, [f'{g}-{i}-0', f'{g}-{i}-1'])
                        for i in range(4) for g in ('short', 'medium', 'long')]

    def test_accumulation_one_preserves_original_order(self):
        self.assertEqual(accumulation_windows(self.batches, 1), [(g,[(i,rows)]) for i,(g,rows) in enumerate(self.batches,1)])

    def test_homogeneous_accumulation_preserves_every_exposure(self):
        windows = accumulation_windows(self.batches, 2)
        self.assertEqual(len(windows), 6)
        self.assertEqual(Counter((i,x) for i,(_,rows) in enumerate(self.batches,1) for x in rows),
                         Counter((i,x) for _,micros in windows for i,rows in micros for x in rows))
        self.assertTrue(all(len(micros)==2 and all(x.startswith(group) for _,rows in micros for x in rows)
                            for group,micros in windows))

    def test_schedule_has_same_end_rate_and_rejects_incomplete_accumulation(self):
        self.assertAlmostEqual(lr_factor(1151, 1152), .25)
        self.assertAlmostEqual(lr_factor(71, 72), .25)
        with self.assertRaises(AssertionError):accumulation_windows(self.batches, 3)


if __name__ == '__main__':
    unittest.main()
