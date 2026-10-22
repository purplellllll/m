#!/usr/bin/env bash
set -Eeuo pipefail
cd /root/gpufree-data/DS_RiboDecode_VMAE_V2_5Fold_ServerBundle_20261022
export CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
exec /root/gpufree-data/envs/vmae_v2/bin/python -u -m latent_decoder.v2_followup queue \
  --data data \
  --prior /root/gpufree-data/vmae_v2_results_20261022 \
  --v1 /root/gpufree-data/vmae_v2_followup_20261022/reference/V1_epoch50.pt \
  --output /root/gpufree-data/vmae_v2_followup_20261022/results
