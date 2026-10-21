# Pretrained checkpoints

`sync_noedge.pt` contains the main SYNC model in a tensor-only checkpoint,
loadable with `weights_only=True`. It preserves the original model parameters
and uses a four-layer EGNN with 128 hidden channels, mean pooling and a
256-channel two-class MLP. Atom vocabulary IDs start at 1; bonds are included
without bond-type embeddings.

The source checkpoint is the epoch-22 model from the January 21, 2025 no-edge run.
Source SHA-256: `11d86643406d98f07b1b3b3dd168f6e7d63c52ab5cc096b3283d7f7d46748b0e`.
Its training configuration is in `docs/evidence/noedge_training_config.json`.

| File under `legacy/` | Purpose |
|---|---|
| `targetdiff/pretrained_models/pretrained_diffusion.pt` | Base TargetDiff |
| `targetdiff/pretrained_models/classifier_noedge.ckpt` | Main SYNC and TargetDiff guidance classifier |
| `targetdiff/pretrained_models/classifier_knn.ckpt` | Alternative classifier checkpoint |
| `targetdiff/pretrained_models/classifier_emb.ckpt` | Edge-aware classifier |
| `targetdiff/pretrained_models/dpo_1800.pt` | TargetDiff DPO, iteration 1800 |
| `decompdiff/pretrained_models/uni_o2_bond.pt` | Base DecompDiff |
| `decompdiff/pretrained_models/classifier_knn_8atom.ckpt` | DecompDiff eight-atom guidance classifier |
| `decompdiff/pretrained_models/dpo_3600.pt` | DecompDiff DPO, iteration 3600 |

The full research entrypoints use the original checkpoint format and environment.
The standalone scoring and two-molecule generation commands use the tensor-format
SYNC checkpoint. Source paths and file hashes are in `docs/source_manifest.json`.
