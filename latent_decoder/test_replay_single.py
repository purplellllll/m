"""Check the zero-LR control and mutation isolation before real checkpoint replay."""
import contextlib
import copy
import io
import unittest

import torch

from .model import CodonVMAE, collate_records, corrupt, vae_loss
from .replay_single import run_arm


class ReplayChecks(unittest.TestCase):
    def test_zero_and_positive_lr_and_checkpoint_preserved(self):
        torch.set_num_threads(2)
        torch.manual_seed(7)
        model_config = dict(d_model=16,n_heads=2,encoder_layers=1,decoder_layers=1,
                            ff_dim=32,latent_dim=4,dropout=.1,max_codons=32)
        training = dict(learning_rate=1e-3,weight_decay=.01,gradient_accumulation=2,
                        beta_kl=.001,masked_loss_weight=1.,full_loss_weight=1.,grad_clip=1.,mask_probability=.15)
        model = CodonVMAE(**model_config)
        tokens = collate_records([dict(cds_dna='ATGGCTGCCGCGTTCTTATGGAAATAG')])
        corrupted, selected = corrupt(tokens,1.)
        optimizer = torch.optim.AdamW(model.parameters(),lr=1e-3,weight_decay=.01)
        logits,mu,logvar = model(corrupted,tokens)
        vae_loss(logits,tokens,mu,logvar,selected,beta=.001)[0].backward()
        optimizer.step()
        checkpoint = copy.deepcopy(dict(config=dict(model=model_config,training=training),
            model=model.state_dict(),optimizer=optimizer.state_dict(),next_epoch=1))
        original = {k:v.clone() for k,v in checkpoint['model'].items()}
        original_step = [float(s['step']) for s in checkpoint['optimizer']['state'].values()]
        with contextlib.redirect_stdout(io.StringIO()):
            zero = run_arm(checkpoint,tokens,corrupted,selected,lr=0.,steps=2)
            positive = run_arm(checkpoint,tokens,corrupted,selected,lr=1e-3,steps=2)
            dynamic = run_arm(checkpoint,tokens,corrupted,selected,lr=1e-3,steps=2,dynamic=True)
        self.assertTrue(all(x['changed_elements']==0 for x in zero['total_parameter_changes'].values()))
        self.assertEqual(zero['initial_probe'],zero['final_probe'])
        for arm in (positive,dynamic):
            self.assertGreater(arm['total_parameter_changes']['encoder']['changed_elements'],0)
            self.assertGreater(arm['total_parameter_changes']['decoder']['changed_elements'],0)
            self.assertEqual(arm['final_optimizer_step_range'],[3.,3.])
        self.assertTrue(all(torch.equal(original[k],v) for k,v in checkpoint['model'].items()))
        self.assertEqual(original_step,[float(s['step']) for s in checkpoint['optimizer']['state'].values()])


if __name__=='__main__':
    unittest.main()
