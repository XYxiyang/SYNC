# Molecular generation with SYNC

## TargetDiff: GPU example

The two-molecule example uses Python 3.10, PyTorch 2.6.0 with CUDA 12.4,
PyG 2.6.1 and RDKit 2022.09.5. Create an environment separate from the CPU scorer:

```bash
python3.10 -m venv .venv-generation
source .venv-generation/bin/activate
python -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
python -m pip install -r requirements-targetdiff.txt
python -m pip install --no-index torch-scatter torch-sparse torch-cluster \
  -f https://data.pyg.org/whl/torch-2.6.0+cu124.html

python scripts/sample_targetdiff.py --pocket examples/1wn6_pocket.pt \
  --num-samples 2 --batch-size 2 --seed 2021 \
  --output outputs/1wn6 --device cuda:0
```

The example runs all 1,000 diffusion steps with joint coordinate and atom-type
guidance. The no-edge classifier is shared with the standalone SYNC scorer.
The implementation uses the guidance function saved with the January 25, 2025
DPO training run. Its source hash is in `guidance_source.json`.

Outputs are `samples.sdf`, `samples.pt` (tensor-format final states and trajectories)
and `summary.json`. To generate with the DPO checkpoint, add:

```bash
--checkpoint legacy/targetdiff/pretrained_models/dpo_1800.pt --guide origin
```

`--guide origin` performs sampling without classifier guidance. The pocket file
stores a dictionary of tensors with protein positions/features and reference
ligand fields. Its `data` keys follow `ProteinLigandData` in
`legacy/targetdiff/datasets/pl_data.py`; `examples/1wn6_pocket.pt` is a complete example.

## Full research workflows

The full classifier and generator entrypoints retain their module names and use
separate working directories. Launch them through `scripts/run_legacy.py`.
The original environment specification is:

```bash
conda env create -f environment-legacy.yml
conda activate sync-legacy
```

It specifies Python 3.8, PyTorch 1.13.1, CUDA 11.6, PyG 2.2.0 and RDKit 2022.03.2.
DecompDiff preprocessing additionally uses AlphaSpace2, mdtraj and decomposition
assets. Docking uses Vina, Meeko, PDB2PQR and receptor preparation tools. Refer to
[TargetDiff](https://github.com/guanjq/targetdiff) and
[DecompDiff](https://github.com/bytedance/DecompDiff) for dataset and tool setup.

### TargetDiff / CrossDocked

Place the CrossDocked pocket data and split under `legacy/targetdiff/data/`,
matching `configs/sampling.yml`. The base model and DPO iteration 1800 are included.

```bash
python scripts/check_assets.py targetdiff
python scripts/run_legacy.py targetdiff sample -- \
  --config configs/sampling.yml --guide_mode origin --data_id 0 \
  --num_sample 100 --result_path outputs/origin --device cuda:0
python scripts/run_legacy.py targetdiff sample -- \
  --config configs/sampling.yml --guide_mode guide --data_id 0 \
  --num_sample 100 --result_path outputs/guide --device cuda:0
python scripts/run_legacy.py targetdiff sample -- \
  --config configs/sampling_dpo.yml --guide_mode origin --data_id 0 \
  --num_sample 100 --result_path outputs/dpo --device cuda:0
```

Arguments after `--` resolve paths relative to `legacy/targetdiff/`.
The full sampling CLI uses the singular option `--num_sample`.

Generate and filter preference pairs before DPO:

```bash
python scripts/run_legacy.py targetdiff pairs -- \
  --config configs/sampling.yml --start 0 --end 10000 \
  --result_path outputs_pairwise_ruledknn --device cuda:0
python scripts/run_legacy.py targetdiff filter-pairs -- \
  --sample_path outputs_pairwise_ruledknn --config configs/sampling.yml --device cuda:0
python scripts/run_legacy.py targetdiff dpo -- \
  --config configs/dpo.yml --beta 0.5 --device cuda:0
```

The provided index list has 3,431 entries and accompanies the original preference
trajectories. For a newly generated dataset, use the index list produced by its
filtering run. Each `result_<training-subset-index>.pt` contains
`pred_ligand_pos_traj_w/l` and `pred_ligand_v_traj_w/l`.
`SYNC_TARGETDIFF_PAIRS` sets the trajectory directory. Filtering writes
`indices.npy` in the component directory; archive an existing list before replacing it.

### DecompDiff

Configure the processed CrossDocked data, `split_by_name.pt`, decomposition priors
and beta-prior assets under `legacy/decompdiff/`. Sampling modes and paths are
specified in `configs/sampling_drift.yml`. Included checkpoints are
`uni_o2_bond.pt`, `classifier_knn_8atom.ckpt` and `dpo_3600.pt`.

```bash
python scripts/check_assets.py decompdiff
python scripts/run_legacy.py decompdiff sample -- --help
python scripts/run_legacy.py decompdiff pairs -- --help
python scripts/run_legacy.py decompdiff filter-pairs -- --help
python scripts/run_legacy.py decompdiff dpo -- \
  --config configs/dpo.yml --beta 0.5 --device cuda:0
```

Preference preprocessing reads `outputs_pairwise/sdfs/<data.id>_w.sdf` and
`<data.id>_l.sdf`. Set `SYNC_DECOMPDIFF_PAIR_SDFS` to override this directory.
The supplied index list contains 1,991 pairs; regenerate indices when using new
preference molecules. The DecompDiff DPO entrypoint uses batch size 1.
Both DPO entrypoints keep reference model weights fixed and use the training
split for preference optimization.
