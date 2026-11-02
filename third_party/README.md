# Third-party sources

No third-party network checkpoints, sequence datasets or upstream simulator/network source files are redistributed in this branch or its own-model archive.

- seq2ribo: https://github.com/Kingsford-Group/seq2ribo , original inspected commit `66c80c653e6db0d0b19a737f5d9fc8009db252b2`. Read the upstream LICENSE before obtaining or using it; academic/non-profit noncommercial research restrictions and redistribution conditions apply. Existing original simulator assets remain in the documented server projects.
- RiboDecode: https://github.com/wangfanfff/RiboDecode . The original baseline weights are referenced separately, never labelled as our model.
- Mamba: https://github.com/state-spaces/mamba , original Torch reference definitions tag v2.2.4. `seq2ribo_cpu_reference.py` is this project's loader that parses those separately obtained definitions; their MIT license and the pinned source hashes must be retained. Its expected `.runtime/mamba_v2_2_4_reference_source/` files are not bundled here.
- Human TE: Zheng et al., DOI 10.1038/s41587-025-02712-x, Supplementary Table1. Provided RiboNN folds must not be silently described as seq2ribo native splits.
- 2026-11-02 human TE update: native CDS were recovered from GENCODE v48 and matched to author truncated-MD5 split identities before native filtering; see the new protocol and handoff. No original workbook/FASTA or complete input sequence dataset is redistributed. Derived per-case prediction tables retain necessary target annotations for metric verification with this source attribution. Own fitted physics parameters and training-corpus CAI reference are included separately; upstream simulator/network source and checkpoints remain excluded.
- LIVER: https://doi.org/10.5281/zenodo.17100113 , `data/processed/LIVER.csv`; related-study data, not recovered RiboGL native train/test.
- mRFP: https://github.com/Kingsford-Group/codonmoe/blob/main/datasets/mRFP_Expression.csv ; target measurements attributed to Nieuwkoop et al.2023. Original data not copied into Git.

Historical baseline scripts retain their original local module paths. Their original workspace contains a helper under `latent_decoder/seq2ribo_cpu_reference.py`; that single backend helper is unrelated to a VMAE encoder. This branch stores its source here for provenance, and does not include or invoke the VMAE generator. Baseline scripts require path/dependency setup documented in the handoff, not just a fresh clone.
