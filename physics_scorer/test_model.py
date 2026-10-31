"""CPU interface/invariance checks, without sequence optimization or model fitting."""

import itertools
import json

import torch

from physics_scorer.model import GraphSequenceScorer


def inputs():
    codons = torch.tensor([[3, 7, 20, 31, 42], [4, 15, 26, 64, 64]])
    valid = codons != 64
    sim = torch.tensor([[1., 2., 3., 4., 2.], [2., 1., 0., 0., 0.]])
    geo = torch.tensor([[[1, 2, 0], [2, 1, 0], [0, 0, 1], [2, 3, 1], [1, 1, 2]],
                        [[0, 1, 0], [2, 2, 1], [2, 1, 2], [0, 0, 0], [0, 0, 0]]])
    edges = torch.tensor([[0, 1, 5], [4, 3, 7]])
    return codons, valid, sim, geo, edges


def extend_padding(args, extra):
    codons, valid, sim, geo, edges = args
    old_length = codons.shape[1]
    remapped = edges // old_length * (old_length + extra) + edges % old_length
    batch = codons.shape[0]
    return (torch.cat((codons, codons.new_full((batch, extra), 64)), dim=1),
            torch.cat((valid, valid.new_zeros((batch, extra))), dim=1),
            torch.cat((sim, sim.new_full((batch, extra), float("nan"))), dim=1),
            torch.cat((geo, geo.new_full((batch, extra, 3), -99)), dim=1), remapped)


def expect_invalid(model, args, message):
    try:
        model(*args)
    except ValueError:
        return
    raise AssertionError(message)


def main():
    torch.set_num_threads(1)
    torch.manual_seed(20261029)
    args = inputs()
    counts = {}
    for encoder, graph, sim in itertools.product(("transformer", "bilstm"), (False, True), (False, True)):
        model = GraphSequenceScorer(encoder=encoder, use_graph=graph, use_sim=sim, dropout=0)
        model.eval()
        counts[f"{encoder}_graph{int(graph)}_sim{int(sim)}"] = sum(p.numel() for p in model.parameters())
        with torch.no_grad():
            profile, te = model(*args)
            padded_profile, padded_te = model(*extend_padding(args, 4))
            torch.testing.assert_close(padded_profile[:, :5], profile, atol=2e-6, rtol=2e-6)
            torch.testing.assert_close(padded_te, te, atol=2e-6, rtol=2e-6)
            assert (profile[~args[1]] == 0).all()
            for batch in range(2):
                n = int(args[1][batch].sum())
                selected = args[4][:, args[4][0] // 5 == batch] - batch*5
                solo = tuple(x[batch:batch+1, :n] for x in args[:4]) + (selected,)
                solo_profile, solo_te = model(*solo)
                torch.testing.assert_close(solo_profile[0], profile[batch, :n], atol=2e-6, rtol=2e-6)
                torch.testing.assert_close(solo_te[0], te[batch], atol=2e-6, rtol=2e-6)
            empty_edges = args[:-1] + (args[4].new_empty((2, 0)),)
            no_edges_profile, _ = model(*empty_edges)
            assert (profile - no_edges_profile).abs().max() > 1e-5 if graph else torch.equal(profile, no_edges_profile)
            changed_sim = (*args[:2], args[2] + 2, *args[3:])
            changed_sim_profile, _ = model(*changed_sim)
            assert (profile - changed_sim_profile).abs().max() > 1e-5 if sim else torch.equal(profile, changed_sim_profile)
            duplicated = args[:-1] + (torch.cat((args[4], args[4].flip(0), args[4]), dim=1),)
            duplicate_profile, duplicate_te = model(*duplicated)
            torch.testing.assert_close(duplicate_profile, profile)
            torch.testing.assert_close(duplicate_te, te)
        profile, te = model(*args)
        loss = profile[args[1]].square().mean() + te.square().mean()
        loss.backward()
        for name, parameter in model.named_parameters():
            assert parameter.grad is not None, name
            assert torch.isfinite(parameter.grad).all(), name
        assert model.codon_embedding.weight.grad[:64].abs().sum() > 0
        expect_invalid(model, args[:-1] + (torch.tensor([[0], [5]]),), "cross-sample edge accepted")
        expect_invalid(model, args[:-1] + (torch.tensor([[5], [8]]),), "padding edge accepted")
        expect_invalid(model, args[:-1] + (torch.tensor([[0], [10]]),), "out of bounds edge accepted")
        bad_sim = args[2].clone()
        bad_sim[0, 0] = float("nan")
        expect_invalid(model, (*args[:2], bad_sim, *args[3:]), "NaN simulation count accepted")
        broken_mask = args[1].clone()
        broken_mask[0, 1] = False
        expect_invalid(model, (args[0], broken_mask, *args[2:]), "non-prefix mask accepted")
    print(json.dumps({"status": "passed", "device": "cpu", "torch": torch.__version__,
                      "configurations_tested": len(counts), "parameters": counts}, indent=2))


if __name__ == "__main__":
    main()
