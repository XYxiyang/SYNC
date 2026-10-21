"""Check release-local weights and external data before launching legacy runs."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("component", choices=["targetdiff", "decompdiff"])
    p.add_argument("--config", default=None)
    args = p.parse_args()
    try:
        import yaml
    except ImportError:
        raise SystemExit("Install pyyaml for this checker.")
    base = ROOT / "legacy" / args.component
    config_path = base / (args.config or ("configs/sampling.yml" if args.component == "targetdiff"
                                        else "configs/sampling_drift.yml"))
    config = yaml.safe_load(config_path.read_text())
    checks = []
    for group, key in [("model", "checkpoint"), ("classifier", "ckpt"), ("data", "path"),
                       ("data", "split"), ("train", "pair_indices")]:
        value = config.get(group, {}).get(key)
        if value:
            path = base / value
            checks.append(dict(field=f"{group}.{key}", path=str(path), exists=path.exists()))
    print(json.dumps(checks, indent=2))
    raise SystemExit(0 if all(c["exists"] for c in checks) else 1)


if __name__ == "__main__":
    main()
