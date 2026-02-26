# Official Implementation of SYNC: Measuring and Advancing Synthesizability in Structure-Based Drug Design (ICLR'26)

## Installation

### Dependency

The code has been tested in the following environment:


| Package           | Version   |
|-------------------|-----------|
| Python            | 3.8       |
| PyTorch           | 1.13.1    |
| CUDA              | 11.6      |
| PyTorch Geometric | 2.2.0     |
| RDKit             | 2022.03.2 |

### Install via Conda and Pip
```bash
conda create -n targetdiff python=3.8
conda activate targetdiff
conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.6 -c pytorch -c nvidia
conda install pyg=2.2.0 -c pyg
conda install rdkit=2022.03.02 openbabel tensorboard pyyaml easydict python-lmdb -c conda-forge
/root/miniconda3/envs/diff/lib/python3.8/site-packages/vina/vina.py line 260 int64

# For Vina Docking
pip install meeko==0.1.dev3 scipy pdb2pqr vina==1.2.2 
python -m pip install git+https://github.com/Valdes-Tresanco-MS/AutoDockTools_py3
```

## Sampling Molecules as An Example
* Please see `https://github.com/guanjq/targetdiff` and  download requested data and checkpoints
* Run `python scripts/sample_diffusion` would sample 20 molecules for 0th pocket in CrossDocked testset

## File Structure
* decompdiff
* * Modified from `https://github.com/bytedance/DecompDiff.git`
* SYNC
* * Our implementation of SYNC, containing 1D, 2D, 3D versions
* targetdiff
* * Modified from `https://github.com/guanjq/targetdiff.git`, we provide a checkpoint of SYNC-3D in ./pretrained-models. You can try `TargetDiff-Guide` following the instructions in `targetdiff/README.md`.
 
