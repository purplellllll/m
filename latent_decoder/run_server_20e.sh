#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
mode="${1:-start}"
if [[ "$#" -gt 1 || ( "$mode" != start && "$mode" != resume ) ]]; then
  echo "Usage: bash latent_decoder/run_server_20e.sh [start|resume]" >&2
  exit 2
fi
python - <<'PY'
import json
from pathlib import Path
config = json.loads(Path('latent_decoder/config.json').read_text(encoding='utf-8'))
assert config['training']['epochs'] == 20, 'This launcher requires exactly 20 epochs'
assert config['training']['patience'] == 0, 'Early stopping must be disabled for this run'
PY
run_dir="runs/vmae_rope_6x512_20ep"
command=(python -u -m latent_decoder.train --data dataset
  --config latent_decoder/config.json --allow-development-split --output "$run_dir")
if [[ "$mode" == resume ]]; then
  [[ -s "$run_dir/last.pt" ]] || { echo "No checkpoint to resume." >&2; exit 1; }
  command+=(--resume "$run_dir/last.pt")
else
  [[ ! -e "$run_dir" ]] || { echo "Run exists. Use resume or a new run directory." >&2; exit 1; }
fi
mkdir -p logs
log_path="logs/vmae_20ep_$(date +%Y%m%d_%H%M%S)_$$.log"
printf 'Logging to %s\n' "$log_path"
"${command[@]}" 2>&1 | tee "$log_path"
