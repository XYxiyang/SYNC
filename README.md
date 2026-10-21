# SYNC — ICLR 2026

[English](README.md) | [简体中文](README_zh-CN.md)

Official implementation of **SYNC: Measuring and Advancing Synthesizability in Structure-Based Drug Design**.

[Paper](https://proceedings.iclr.cc/paper_files/paper/2026/file/ffdb280e7c7b4c4af30e04daf5a84b98-Paper-Conference.pdf)
· [OpenReview](https://openreview.net/forum?id=y1tPw4Uuzg)
· [GitHub](https://github.com/XYxiyang/SYNC)

SYNC scores molecular synthesizability from 3D structures and provides classifier
guidance and preference optimization for structure-based molecular generation.
This release includes pretrained models, SDF scoring, classifier training,
TargetDiff/DecompDiff guidance and DPO, and benchmark utilities.

## Pretrained classifier

The included classifier reproduces the following Table 1 accuracies:

| TS1 | TS2 | TS3 | Nonpher-Test | Enamine |
|---:|---:|---:|---:|---:|
| 0.9911 | 0.8406 | 0.7564 | 0.9313 | 0.9520 |

See [validation](docs/VALIDATION.md) for evaluated sample counts and software checks.

## Score 3D molecules

The standalone scorer requires PyTorch and RDKit. It supports CPU and CUDA.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e .

sync-score --input examples/molecules.sdf \
  --checkpoint checkpoints/sync_noedge.pt \
  --output outputs/example_scores.csv
```

`--input` accepts a multi-record SDF or a directory searched recursively. With a
CUDA-enabled PyTorch installation, add `--device cuda:0`. The scorer uses the
input's 3D coordinates and bonds, removes hydrogens, and encodes aromatic atoms
with the SYNC vocabulary.

The output CSV contains `probability_es`, `prediction` (HS = 0, ES = 1), molecular
identities and per-record processing status. A JSON sidecar reports the input,
valid and failed counts, and the fraction classified ES.

## Generate molecules

SYNC guidance updates both atom-type probabilities and coordinates during
TargetDiff sampling. A tensor-format 1WN6 pocket and a two-molecule example are
included. Follow [GENERATION.md](docs/GENERATION.md) to install the GPU dependencies.

```bash
python scripts/sample_targetdiff.py \
  --pocket examples/1wn6_pocket.pt --num-samples 2 --batch-size 2 \
  --seed 2021 --output outputs/1wn6 --device cuda:0
```

The output includes generated SDF molecules, atom types, coordinates, diffusion
trajectories and a JSON summary. The same guide documents full CrossDocked
sampling, preference-pair construction and DPO for both backbones.

## Train and evaluate

```bash
python scripts/prepare_classifier_data.py \
  --manifest /path/to/labeled_sdfs.csv --output data/classifier
python scripts/train_classifier.py \
  --train data/classifier/train.pt --val data/classifier/val.pt \
  --device cuda:0 --output outputs/classifier
python scripts/evaluate_classifier.py \
  --graphs data/classifier/test.pt --checkpoint outputs/classifier/best.pt \
  --output outputs/classifier/test.json
```

See [DATA.md](docs/DATA.md) for labels, molecule grouping, input schemas and
benchmark conversion. Fingerprint, SMILES, graph and edge-feature variants are
provided in `legacy/classifier/`.

## Package contents

| Path | Contents |
|---|---|
| `sync_sbdd/` | Standalone EGNN classifier and SDF scoring API |
| `checkpoints/` | Tensor-format main SYNC checkpoint |
| `scripts/` | Data preparation, training, evaluation and sampling commands |
| `legacy/classifier/` | Classifier variants and preprocessing modules |
| `legacy/targetdiff/` | TargetDiff guidance, DPO, configurations and checkpoints |
| `legacy/decompdiff/` | DecompDiff guidance, DPO, configurations and checkpoints |
| `legacy/benchmark/` | Metric drivers and analysis utilities |
| `examples/` | Example molecules and the 1WN6 pocket |
| `docs/` | Usage, data, validation and source provenance |
| `tests/` | Numerical equivalence, invariance, gradient and interface checks |

Training datasets, CrossDocked assets and preference trajectories are configured
separately using the paths in [DATA.md](docs/DATA.md) and
[GENERATION.md](docs/GENERATION.md). Metric formats and thresholds are described
in [BENCHMARKS.md](docs/BENCHMARKS.md).

## Checks and attribution

```bash
python -m pip install -e ".[validation]"
python -m pytest -q
python scripts/verify_release.py
python scripts/check_assets.py targetdiff
python scripts/check_assets.py decompdiff
```

The asset checker reports the configured model and dataset paths. Source hashes
are recorded in `docs/source_manifest.json`; `MANIFEST.sha256` covers the release
files. See [THIRD_PARTY.md](THIRD_PARTY.md) for component attribution and licenses.

## Citation

```bibtex
@inproceedings{liu2026sync,
  title={SYNC: Measuring and Advancing Synthesizability in Structure-Based Drug Design},
  author={Liu, Yunfan and Wu, Lirong and Gao, Zhifeng and Huang, Yufei and
          Tan, Cheng and Lin, Haitao and Liu, Zicheng and Chi, Changxi and
          Yu, Chang and Li, Stan Z.},
  booktitle={The Fourteenth International Conference on Learning Representations},
  year={2026},
  url={https://openreview.net/forum?id=y1tPw4Uuzg}
}
```
