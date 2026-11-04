# Default research baseline

The user designated **transformer_stasep_v1** as the baseline for future additions/removals on2026-11-05. Read `baselines/transformer_stasep_v1/README_中文.md` and `BASELINE.json` before changing models.

- Primary: original profile/dynamics Transformer+sTASEP ensemble(seeds30101/30102), E217 Pearson0.86753. Paired no-sTASEP control: profile/sequence, Pearson0.87198.
- Start experiments from tag `baseline-transformer-stasep-v1-20261105` or branch `baseline/transformer-stasep-v1`. Record the exact parent commit and changes. Keep the frozen baseline files/weights intact; implement additions/removals in separate experiment branches/configurations.
- Preserve identical splits, input permissions, target transforms and evaluation budget for controlled comparisons; do not tune on historical test labels.
- H is cellular mRNA half-life; S is PURE solubility. No protein-degradation head exists in this baseline.
- The later LPT, Gated, CDS-only repair and human-TE fine-tunes are research variants, not silent replacements for this user-selected baseline.
