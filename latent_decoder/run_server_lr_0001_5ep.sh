#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
exec python -u -m latent_decoder.train --data dataset --allow-development-split \
  --config latent_decoder/lr_sweep_configs/lr_0.0001.json \
  --output runs/vmae_lr_0.0001_5ep "$@"
