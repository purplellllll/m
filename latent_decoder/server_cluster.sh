#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
command -v mmseqs >/dev/null
command -v python >/dev/null
if [[ -e data/latent_clusters || -e data/latent_decoder_clustered ]]; then
  echo "Output exists. Inspect it and choose a fresh location before rerunning." >&2
  exit 1
fi
mkdir -p data/latent_clusters
mmseqs version > data/latent_clusters/mmseqs_version.txt
sha256sum data/latent_decoder/proteins.fasta > data/latent_clusters/input_sha256.txt
cluster_command=(mmseqs easy-cluster data/latent_decoder/proteins.fasta
  data/latent_clusters/seq data/latent_clusters/tmp
  --min-seq-id 0.3 -c 0.8 --cov-mode 0 --cluster-mode 1
  --single-step-clustering 1 --alignment-mode 3 -s 7.5
  --threads "${SLURM_CPUS_PER_TASK:-8}")
printf '%q ' "${cluster_command[@]}" > data/latent_clusters/command.txt
printf '\n' >> data/latent_clusters/command.txt
"${cluster_command[@]}" 2>&1 | tee data/latent_clusters/mmseqs.log
sha256sum data/latent_clusters/seq_cluster.tsv > data/latent_clusters/output_sha256.txt
python -m latent_decoder.repartition --data data/latent_decoder --clusters data/latent_clusters/seq_cluster.tsv \
  --output data/latent_decoder_clustered > data/latent_clusters/prepare.log
echo "Clustering and repartition complete. Review the audit and cross-split search before formal claims."
