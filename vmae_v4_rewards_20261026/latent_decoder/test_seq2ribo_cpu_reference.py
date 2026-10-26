import unittest
from pathlib import Path

import torch

from .seq2ribo_cpu_reference import ROOT, install


class CPUReferenceChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.backend=install();torch.set_num_threads(1)

    def compare_steps(self,model,inputs):
        model.eval()
        with torch.no_grad():
            full=model(inputs)
            conv,state=model.allocate_inference_cache(len(inputs),inputs.shape[1])
            outputs=[]
            for token in inputs.split(1,dim=1):
                value,conv,state=model.step(token,conv,state);outputs.append(value)
            steps=torch.cat(outputs,dim=1)
        self.assertTrue(torch.allclose(full,steps,atol=1e-5,rtol=1e-5),float((full-steps).abs().max()))
        return float((full-steps).abs().max())

    def test_original_forward_and_original_step_agree(self):
        torch.manual_seed(20261024)
        self.compare_steps(self.backend.Mamba(d_model=16,d_state=4),torch.randn(2,37,16))
        with self.assertRaises(NotImplementedError):self.backend.Mamba2(d_model=16)

    def test_all_four_trained_checkpoint_blocks_agree(self):
        checkpoint=ROOT/'tmp/seq2ribo_source_20261020/seq2ribo-main/weights/hek293_mamba_te_full_final_cds.pt'
        state=torch.load(checkpoint,map_location='cpu',weights_only=True)
        torch.manual_seed(20261024)
        errors=[]
        for index in range(4):
            prefix=f'base.encoder.{index}.'
            block=self.backend.Mamba(d_model=192,d_state=16,d_conv=4,expand=2)
            block.load_state_dict({key[len(prefix):]:value for key,value in state.items() if key.startswith(prefix)},strict=True)
            errors.append(self.compare_steps(block,torch.randn(1,129,192)))
        self.assertEqual(len(errors),4)
        print('Trained block forward/step maximum absolute differences:',errors)


if __name__=='__main__':unittest.main()
