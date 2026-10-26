import unittest
import torch
from .latent_joint_guidance import rms_direction


class RMSChecks(unittest.TestCase):
    def test_sequence_length_and_frozen_positions(self):
        g=torch.tensor([[[1.,3.],[100.,100.],[2.,4.]]]);active=torch.tensor([[True,False,True]])
        direction=rms_direction(g,active)
        self.assertTrue(torch.equal(direction[:,1],torch.zeros(1,2)))
        self.assertAlmostEqual(float(direction[active].square().mean()),1.,places=6)
        repeated=rms_direction(g.repeat(1,7,1),active.repeat(1,7))
        self.assertTrue(torch.allclose(repeated,direction.repeat(1,7,1)))

    def test_zero_and_invalid_gradients(self):
        active=torch.ones(1,2,dtype=torch.bool)
        self.assertTrue(torch.equal(rms_direction(torch.zeros(1,2,3),active),torch.zeros(1,2,3)))
        with self.assertRaises(ValueError):rms_direction(torch.full((1,2,3),float('nan')),active)
        with self.assertRaises(ValueError):rms_direction(torch.ones(1,2,3),~active)


if __name__=='__main__':unittest.main()
