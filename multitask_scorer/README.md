# Multi-task scorer inference

Download `multitask-own-models-20261102.tar.gz` from this release and extract it.
Install `pip install -r multitask_scorer/requirements.txt` in your environment.

```python
from multitask_scorer import Scorer
scorer = Scorer("own_models", stage="profile", arm="dynamics", device="cpu")
result = scorer.predict(dna="ATG...TAA", host="human", utr5="...", utr3="...")
```

Use real complete CDS, not the example ellipsis. All four outputs are returned,
but expression is calibrated on E.coli mRFP fluorescence, half-life on HEK293
log10(HL+1), and solubility on PURE assay percent (not clipped). Cross-host
outputs are extrapolation. No validated joint scalar or absolute-hour claim.

Stages: neutral (explicit equal-wait control) and profile (host-specific observed
RPF relative-occupancy proxy plus first150nt RNAfold structure). Arms: sequence
and dynamics. Both arms in the profile stage share the same folding statistics.
All8 exports passed strict state load and real-sequence CPU inference parity.
Training snapshots in historical/ need the documented public datasets and local
path setup; they are not claimed to be a data-free one-command training package.

Diffusion release: download the all10-model archive and use `Scorer("own_models", stage="diffusion", arm="dynamics", device="cpu")`. Fixed4-step deterministic DDIM adds inference cost; it is not a discrete-CDS generator.
