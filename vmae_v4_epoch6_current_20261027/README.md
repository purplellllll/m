# Current V4: frozen epoch6

2,498,240 parameters. Frozen model SHA256: `8e95e3d3e5c62917c01a78a3add4ab45ea1afa7a1282748ce8d637f3fbea02e3`.

Full evaluation completed on 24 validation and 30 TEST CDS. Cross-environment scoring completed for 335 locked candidates in LCL, RPE-1 and iPSC, using three fixed seeds and 8 sTASEP runs per seed. No optimization was added for this comparison.

Reports and per-sequence scores are included here. Binary checkpoints, full search traces, training/resume states, fixed corpus, environment locks and source snapshot are published as release assets: https://github.com/purplellllll/m/releases/tag/v4-epoch6-results-20261028 .

Training-validation best epoch6 is not the best independent-scoring method: original V3 Hard-only search has higher mean seq2ribo scores in all four evaluated environments on the shared 30 CDS. Predictor scores are not measured expression. LinearDesign has 27 cases and full native RiboDecode has 8; do not compare unequal panels directly. RNop weights unavailable.
