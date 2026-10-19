#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
exec python -u -m latent_decoder.lr_sweep "$@"
