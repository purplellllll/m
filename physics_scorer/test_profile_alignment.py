"""Check coordinate conversion, masking, gradients and batch isolation."""
import itertools
import json
import numpy as np
import torch
from .profile_model import AlignedProfileModel, masked_transcript_mse


def main():
    torch.set_num_threads(1)
    torch.manual_seed(301030)
    ribogl = [''.join(c) for c in itertools.product('ATCG', repeat=3)]
    seq2 = [''.join(c) for c in itertools.product('ACGU', repeat=3)]
    mapping = np.array([seq2.index(c.replace('T','U')) for c in ribogl])
    assert len(set(mapping)) == 64 and not np.array_equal(mapping, np.arange(64))
    for i in range(64):
        assert seq2[mapping[i]].replace('U','T') == ribogl[i]
    codons = torch.tensor([[14, 2, 40, 5, 48], [14, 7, 48, 64, 64]])
    valid = codons != 64
    sim, geo = torch.rand(2,5,2), torch.rand(2,5,5)
    edges = torch.tensor([[0,5],[4,7]])
    target = torch.rand(2,5)
    observed = valid.clone(); observed[0,1] = False; target[0,1] = float('nan')
    tested = 0
    for branch in itertools.product([False,True], repeat=3):
        if not any(branch): continue
        model = AlignedProfileModel(*branch).eval()
        pred, local = model(codons, valid, sim, geo, edges)
        loss = masked_transcript_mse(pred, target, observed)
        for p in local.values(): loss = loss + .1*masked_transcript_mse(p,target,observed)
        loss.backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
        with torch.no_grad():
            for i, n in enumerate([5,3]):
                own = edges[:, edges[0]//5 == i]-i*5
                p,_ = model(codons[i:i+1,:n], valid[i:i+1,:n], sim[i:i+1,:n],geo[i:i+1,:n],own)
                torch.testing.assert_close(p[0], pred[i,:n],atol=3e-6,rtol=3e-6)
            padded = [torch.cat((codons,torch.full((2,3),64)),1), torch.cat((valid,torch.zeros(2,3,dtype=torch.bool)),1),
                      torch.cat((sim,torch.full((2,3,2),float('nan'))),1),torch.cat((geo,torch.full((2,3,5),float('nan'))),1),
                      edges//5*8+edges%5]
            p,_ = model(*padded)
            torch.testing.assert_close(p[:,:5],pred,atol=3e-6,rtol=3e-6)
        try: model(codons,valid,sim,geo,torch.tensor([[0],[5]]))
        except ValueError: pass
        else: raise AssertionError('Cross-transcript edge accepted')
        assert torch.isfinite(loss)
        tested += 1
    print(json.dumps(dict(status='passed',codon_roundtrips=64,branch_configs=tested,
        checks=['masked NaN labels','all branch gradients','batch isolation','padding invariance','edge boundaries'])))


if __name__ == '__main__': main()
