import unittest

import torch
from torch.nn import functional as F

from .budget_comparison import native_generator, project_budget
from .model import collate_records
from .dual_space_ablation import masks
from .tokenizer import TOKEN


class BudgetTests(unittest.TestCase):
    def test_limits_synonyms_fixed_ends_and_gradient(self):
        tokens = collate_records([dict(cds_dna='ATGGCTGCTGCTTAA')])
        logits = F.one_hot(tokens, 64).float() * 2
        logits[0, 1:4, TOKEN['GCC']] = torch.tensor([3., 5., 4.])
        logits[0, -1, TOKEN['TAG']] = 10
        p = logits.softmax(-1).requires_grad_()
        for limit in (0, 1, 2, 3):
            selected = project_budget(p, tokens, limit)
            self.assertEqual(int((selected.argmax(-1) != tokens).sum()), limit)
            self.assertEqual(int(selected.argmax(-1)[0, -1]), TOKEN['TAA'])
        selected = project_budget(p, tokens, 1)
        self.assertEqual(int(selected.argmax(-1)[0, 2]), TOKEN['GCC'])
        ((selected + p - p.detach()) * torch.arange(64)).sum().backward()
        self.assertTrue(torch.isfinite(p.grad).all())
        self.assertGreater(float(p.grad.abs().sum()), 0)

    def test_shipped_generator_vocabulary_and_update(self):
        generator, order = native_generator(5, 'cpu')
        self.assertEqual(len(set(order.tolist())), 64)
        mask = torch.zeros(1, 5, 65)
        tokens = collate_records([dict(cds_dna='ATGGCTGCTGCTTAA')])
        active, support = masks(tokens)
        mask[:, :, order] = torch.where(active[..., None], support, F.one_hot(tokens, 64).bool()).float()
        p = generator.generate(mask).transpose(1, 2)[:, :, order]
        torch.testing.assert_close(p.sum(-1), torch.ones(1, 5))
        (p * torch.arange(64)).sum().backward()
        self.assertTrue(torch.isfinite(generator.seqs.grad).all())


if __name__ == '__main__':
    unittest.main()
