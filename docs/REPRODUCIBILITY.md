# Reproducibility and scope

This branch is an archival research snapshot. Core model loading and synthetic checks are portable; historical experiment drivers retain their frozen workstation/server paths. They are not a turnkey fresh-clone training pipeline.

The release carries tensor-only own-model checkpoints with a per-file SHA256 manifest and constructor specs. Use `physics_scorer.checkpoint_io.load_exported_checkpoint` after verifying the digest. Scalar protein output is standardized to train-only mean/std; invert using metadata before comparing measurements.

Source trust: resumable server checkpoints contain Python/NumPy RNG and optimizer state; only the project's own locked originals were opened during export, after hash checks where a completed lock exists. Exported checkpoints support `weights_only=True` and omit optimizer/RNG.

Do not interpret synthetic-forward checks or nonzero gradients as biological validation. No new biological test or training is authorized by this snapshot.

Current final LIVER test was not opened. mRFP219/clean217 has been opened after validation selection; do not tune recipes against that test. Old mouse profile120 results are repeated historical development diagnostics.

Do not compare profile Shape r with transcript-total r, TE r, RPF r or protein r. RiboDecode RPF includes measured RNA/environment inputs; always report the incremental result beyond an RNA/environment baseline.

Third-party sources/weights and raw datasets are not redistributed. Restore their original versions and licenses separately. Exact simulation parameters, input hashes, split-ID hashes and run logs are retained in evidence and the original server workspaces.

The original controllers have been stopped. If the user later authorizes a new experiment, use a new output directory and a bounded preregistered development protocol; do not blindly delete launch files and rerun old queues.
