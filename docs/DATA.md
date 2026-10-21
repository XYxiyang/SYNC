# Data preparation and representations

## Labeled molecules

SYNC uses **HS = 0, ES = 1**. Create a CSV with paths relative to that CSV:

```csv
sdf,label
easy.sdf,1
hard.sdf,0
```

Each SDF may contain multiple 3D conformers. Prepare the data with:

```bash
python scripts/prepare_classifier_data.py \
  --manifest /path/to/labeled_sdfs.csv \
  --exclude-sdf /path/to/benchmark.sdf \
  --seed 530 --max-atoms 68 --output data/classifier
```

Repeat `--exclude-sdf` for every evaluation source. Preparation excludes the
specified molecular identities, groups conformers by canonical isomeric SMILES,
rejects conflicting labels, and creates a stratified 80/10/10 split. At least ten
unique molecules per class are required. Processing counts and failures are
recorded in `preparation.json`.

The upstream GASA label convention is ES = 0, HS = 1; convert those labels to the
SYNC convention when importing a labeled GASA dataset.

## Tensor-format graphs

| Key | Meaning |
|---|---|
| `atom_types` | int64 `[N]`, aromatic vocabulary IDs starting at 1 |
| `positions` | float32 `[N,3]`, coordinates in Å |
| `edge_index` | int64 `[2,E]`, both directions of each bond |
| `label` | Integer 0/1 |
| `smiles` | Canonical isomeric identity, used by training split checks |
| `source` | Optional file and record identifier |

The main classifier uses real bond connectivity without bond-type embeddings.
SDF scoring preserves the supplied conformer. During diffusion guidance,
valence-constrained KNN provides connectivity from atom types and coordinates.

## Benchmark graphs

| Dataset | Graph filename | Evaluated records |
|---|---|---:|
| TS1 | `gasa_test1_aro_1.pt` | 5,853 |
| TS2 | `gasa_test2_aro_1.pt` | 30,286 |
| TS3 | `gasa_test3_aro_1.pt` | 1,761 |
| Nonpher-Test | `nonpher_test_aro_1.pt` | 160 |
| Enamine | `enamine_aro_1.pt` | 211,197 |

To convert an original PyG benchmark graph file, install `torch-geometric==2.6.1`
and run the conversion on a trusted local file:

```bash
python scripts/export_legacy_graphs.py \
  --input /path/to/nonpher_test_aro_1.pt \
  --output data/nonpher.pt --trust-local-pickle
python scripts/evaluate_classifier.py \
  --graphs data/nonpher.pt --checkpoint checkpoints/sync_noedge.pt \
  --output outputs/nonpher.json
```

For the original Enamine graph serialization, add `--reproduce-legacy-padding`
to preserve the fixed-width representation. The converter records affected
records in its JSON sidecar. Converted graphs are evaluation inputs; use the
SDF preparation command to create training data with molecular identities.

## Training configuration and data sources

The provided classifier is the epoch-22 checkpoint from the January 21, 2025
no-edge training run. Its configuration is in
`evidence/noedge_training_config.json`; the input files are
`train_80w_tgtdiff.pt`, `val_80w_tgtdiff.pt` and `test_80w_tgtdiff.pt`.
Classifier variants and preprocessing modules are in `legacy/classifier/`.

Dataset construction and sources are described in the
[paper, Appendix B](https://proceedings.iclr.cc/paper_files/paper/2026/file/ffdb280e7c7b4c4af30e04daf5a84b98-Paper-Conference.pdf)
and the GASA study, [DOI 10.1021/acs.jcim.2c00038](https://doi.org/10.1021/acs.jcim.2c00038).
The Enamine benchmark uses the October 2023 stock snapshot.
The example SDF contains three demonstration conformers; the generation example
contains the 1WN6 pocket features from CrossDocked.
