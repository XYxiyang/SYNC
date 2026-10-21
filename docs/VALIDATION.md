# Validation

## Pretrained classifier

The included SYNC checkpoint reproduces the Table 1 classification results:

| Dataset | Correct | Evaluated graph records | Accuracy | Table 1 |
|---|---:|---:|---:|---:|
| TS1 | 5,801 | 5,853 | 0.9911156672 | 0.9911 |
| TS2 | 25,459 | 30,286 | 0.8406194281 | 0.8406 |
| TS3 | 1,332 | 1,761 | 0.7563884157 | 0.7564 |
| Nonpher-Test | 149 | 160 | 0.9312500000 | 0.9313 |
| Enamine | 201,051 | 211,197 | 0.9519595449 | 0.9520 |

Evaluation uses the processed graph representations described in [DATA.md](DATA.md).
The tensor-format model preserves the original EGNN parameters and agrees with
the original classifier's reference logits.

## TargetDiff guidance

A two-molecule, 1,000-step sampling run was completed on the 1WN6 pocket with
seed 2021 on an NVIDIA L40. Both molecules reconstructed successfully as connected
molecular graphs with finite coordinates.

At steps 399, 100, 2 and 0, the release guidance function was compared with the
function saved in the January 25, 2025 training run, using the same inputs and
random state. The largest probability difference was **4.66e-10** and the largest
coordinate difference was **7.16e-7 Å**. The comparison covers both the three-step
and 200-step guidance schedules. Atom-type probabilities and coordinates are
updated by the release implementation.

Two archived output molecules were independently reconstructed and matched to
the final metric CSV by molecular identity. Their normalized SA scores were
0.55 and 0.53, exactly matching the stored values.

The standalone public command was also run end to end, generating two connected
molecules. Its output is included as `examples/1wn6_generated.sdf`, with run
metadata in `evidence/sampling_example.json`.

The sampling command and the tensor-format pocket are included in
[sampling instructions](GENERATION.md). Machine-readable validation summaries
are in `evidence/`.

## Software checks

All 14 CPU tests pass. They cover original-model numerical equivalence,
rotation/translation and atom-permutation invariance, batch consistency,
coordinate gradients, an optimizer update, checkpoint save/load, aromatic token
IDs, single-atom graphs, input validation, command-line scoring, and the original
metric threshold-equality convention.

Additional checks cover editable installation, scoring outside the source
working directory, manifest-based data preparation, one-epoch training and
checkpoint reload, and conversion/evaluation of all 160 Nonpher-Test records.

Validation environments:

- Classifier: Python 3.13, PyTorch 2.6.0+cpu, RDKit 2026.9.1, pytest 9.1.1.
- TargetDiff: Python 3.10, PyTorch 2.6.0+cu124, PyG 2.6.1, RDKit 2022.9.5,
  NumPy 1.26.4, scikit-learn 1.5.2, NVIDIA L40.

Every release file is covered by `MANIFEST.sha256`; run
`python scripts/verify_release.py` to verify an extracted package.
