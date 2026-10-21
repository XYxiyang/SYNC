"""Launch recovered research entrypoints with the correct working directory."""
import argparse
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
ENTRYPOINTS = {
    "targetdiff": {
        "sample": "scripts/sample_diffusion.py",
        "pairs": "scripts/sample_pairwise.py",
        "filter-pairs": "scripts/eval_pairwise.py",
        "dpo": "scripts/dpo_diffusion.py",
        "evaluate": "scripts/evaluate_diffusion.py",
        "train": "scripts/train_diffusion.py",
    },
    "decompdiff": {
        "sample": "scripts/sample_diffusion_decomp.py",
        "pairs": "scripts/sample_pairwise.py",
        "filter-pairs": "scripts/eval_pairwise.py",
        "dpo": "scripts/dpo_diffusion.py",
        "evaluate": "scripts/evaluate_mol_from_meta_full.py",
        "train": "scripts/train_diffusion_decomp.py",
    },
    "classifier": {
        "train-1d": "train_1d.py",
        "train-2d": "train_2d.py",
        "train-graphormer": "train_2d_graphormer.py",
        "train-3d": "train_3d_sbdd.py",
        "train-edge": "train_3d_edge.py",
        "train-fp": "train_mlp.py",
    },
}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("component", choices=ENTRYPOINTS)
    p.add_argument("command")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("arguments", nargs=argparse.REMAINDER)
    args = p.parse_args()
    if args.command not in ENTRYPOINTS[args.component]:
        p.error(f"Commands: {', '.join(ENTRYPOINTS[args.component])}")
    directory = ROOT / "legacy" / args.component
    extra = args.arguments[1:] if args.arguments[:1] == ["--"] else args.arguments
    env = os.environ.copy()
    env["PYTHONPATH"] = str(directory)
    command = [sys.executable, ENTRYPOINTS[args.component][args.command], *extra]
    print(f"Working directory: {directory}", flush=True)
    print("Command:", command, flush=True)
    if not args.dry_run:
        raise SystemExit(subprocess.call(command, cwd=directory, env=env))


if __name__ == "__main__":
    main()
