# Official Implementation of SYNC: Measuring and Advancing Synthesizability in Structure-Based Drug Design (Under Review)

## Sampling Molecules as An Example
* Please build environment follow `README.md` in targetdiff and download requested data and checkpoints
* Run `python scripts/sample_diffusion` would sample 20 molecules for 0th pocket in CrossDocked testset

## File Structure
* decompdiff
* * Modified from `https://github.com/bytedance/DecompDiff.git`
* SYNC
* * Our implementation of SYNC, containing 1D, 2D, 3D versions
* targetdiff
* * Modified from `https://github.com/guanjq/targetdiff.git`, we provide a checkpoint of SYNC-3D in ./pretrained-models. You can try `TargetDiff-Guide` following the instructions in `targetdiff/README.md`.