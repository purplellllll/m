# Restore the formal result archive

Download all assets from https://github.com/purplellllll/m/releases/tag/v4-epoch6-results-20261028 .

`epoch6.pt` is the standalone frozen model checkpoint, SHA256 `8e95e3d3e5c62917c01a78a3add4ab45ea1afa7a1282748ce8d637f3fbea02e3`.

To reconstruct the complete archive, concatenate `research-results-20261028.tar.gz.part01`, `.part02`, `.part03` in that order as binary files. Verify the combined SHA256 is `9480db722ba6b848d98f76c0568119e23318d5cf68ae66b5021dd6621bfb8032b` before extracting with tar. Use GITHUB_VERIFIED.json for per-part sizes and SHA256. The archive uses relative paths and internal hardlinks for duplicate contents.

The archive includes 3,817 selected files: model checkpoints and optimizer/RNG resume states, full epoch6 VAL24/TEST30 searches, locked candidates, all three-environment seed-level scores and caches, method comparisons, original fixed corpus/folds, scorer assets and parameters, scientific source snapshot, configurations and logs. Archive member hashes are recorded in archive_manifest.json.

The user explicitly excluded early native RiboDecode per-proposal raw history (about 15GB). Completed method metrics, returned candidates, receipts and comparison reports remain in the formal archive. Downloadable runtime environments/external dependency caches are excluded. This release is not a full server disk image.

No server deletion or shutdown was performed by the assistant.

Also extract reproduction-inputs.tar.gz: it supplies the exact canonical RiboDecode condition tensor, scorer-subset and CUDA-parity receipts, and pinned seq2ribo source/HEK293 weights. LCL/RPE-1/iPSC scorer assets are in the main archive. File hashes are in reproduction_manifest.json.
