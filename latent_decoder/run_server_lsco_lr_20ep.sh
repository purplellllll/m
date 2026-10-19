#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
exec python -u -m latent_decoder.train --data dataset --allow-development-split \
  --config latent_decoder/config_lsco_lr_20ep.json \
  --output runs/vmae_lsco_lr_20ep_20261018 "$@"
