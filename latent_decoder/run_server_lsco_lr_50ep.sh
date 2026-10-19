#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
mode="${1:-extend}"
if [[ "$#" -gt 1 || ( "$mode" != extend && "$mode" != resume ) ]]; then
  echo "Usage: bash latent_decoder/run_server_lsco_lr_50ep.sh [extend|resume]" >&2
  exit 2
fi
run_dir="runs/vmae_lsco_lr_20ep_20261018"
if [[ "$mode" == extend ]]; then
  config="latent_decoder/config_lsco_lr_20ep.json"
  extra=(--extend-epochs 50)
else
  config="latent_decoder/config_lsco_lr_50ep.json"
  extra=()
fi
exec python -u -m latent_decoder.train --data dataset --allow-development-split \
  --config "$config" --output "$run_dir" --resume "$run_dir/last.pt" "${extra[@]}"
