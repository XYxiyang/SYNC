"""Convert labeled SDF inputs into portable graph files, grouping all conformers."""
import argparse
import csv
import json
import random
import sys
from pathlib import Path

import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sync_sbdd.data import molecule_to_graph, sdf_records


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", required=True, help="CSV columns sdf,label; paths relative to CSV.")
    p.add_argument("--output", required=True)
    p.add_argument("--exclude-sdf", action="append", default=[], help="Exclude benchmark molecule identities.")
    p.add_argument("--seed", type=int, default=530)
    p.add_argument("--max-atoms", type=int, default=68)
    args = p.parse_args()
    excluded = set()
    for path in args.exclude_sdf:
        for _, _, mol in sdf_records(path):
            excluded.add(molecule_to_graph(mol)["smiles"])
    groups, errors = {}, []
    manifest = Path(args.manifest)
    with manifest.open() as handle:
        for row in csv.DictReader(handle):
            label = int(row["label"])
            if label not in (0, 1):
                raise ValueError("Labels must be 0 (HS) or 1 (ES).")
            for file, index, mol in sdf_records(manifest.parent / row["sdf"]):
                try:
                    graph = molecule_to_graph(mol)
                    if len(graph["atom_types"]) > args.max_atoms:
                        raise ValueError("Exceeds max-atoms")
                except (ValueError, RuntimeError) as error:
                    errors.append(dict(file=str(file), record=index, error=str(error)))
                    continue
                key = graph["smiles"]
                if key in excluded:
                    continue
                graph["label"] = label
                graph["source"] = f"{file}:{index}"
                if key in groups and groups[key][0]["label"] != label:
                    raise ValueError(f"Conflicting labels for molecule {key}")
                groups.setdefault(key, []).append(graph)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    result = {k: [] for k in ("train", "val", "test")}
    rng = random.Random(args.seed)
    for label in (0, 1):
        keys = sorted(k for k, v in groups.items() if v[0]["label"] == label)
        if len(keys) < 10:
            raise ValueError("Need at least 10 unique molecules per class for an 80/10/10 split.")
        rng.shuffle(keys)
        ntrain, nval = int(.8 * len(keys)), int(.1 * len(keys))
        for name, subset in [("train", keys[:ntrain]), ("val", keys[ntrain:ntrain+nval]),
                             ("test", keys[ntrain+nval:])]:
            result[name].extend(g for key in subset for g in groups[key])
    for name, graphs in result.items():
        torch.save(graphs, out / f"{name}.pt")
    summary = dict(seed=args.seed, unique_molecules=len(groups), excluded_identities=len(excluded),
                   records={k: len(v) for k, v in result.items()}, failures=errors,
                   note="New deterministic grouping; does not reconstruct historical split membership.")
    (out / "preparation.json").write_text(json.dumps(summary, indent=2) + "\n")
    print({k: len(v) for k, v in result.items()})


if __name__ == "__main__":
    main()
