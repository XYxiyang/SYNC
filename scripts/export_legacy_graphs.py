"""Convert trusted local PyG graph lists to tensor-only evaluation data."""
import argparse
import json
from pathlib import Path
import torch


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--trust-local-pickle", action="store_true")
    p.add_argument("--reproduce-legacy-padding", action="store_true",
                   help="Explicitly reproduce historical zero padding for extra atom tokens.")
    args = p.parse_args()
    if not args.trust_local_pickle:
        p.error("Full-object legacy data requires --trust-local-pickle for trusted local inputs.")
    source = torch.load(args.input, map_location="cpu", weights_only=False)
    records = []
    padding = []
    for index, graph in enumerate(source):
        positions = graph.positions.float()
        difference = graph.x.numel() - len(positions)
        if difference:
            if difference < 0 or not args.reproduce_legacy_padding:
                raise ValueError(f"Record {index} has mismatched token/coordinate counts. "
                                 "See docs/PAPER_CODE_AUDIT_ZH.md before reproducing legacy padding.")
            positions = torch.nn.functional.pad(positions, (0, 0, 0, difference))
            padding.append(index)
        records.append(dict(atom_types=graph.x.flatten().long(), positions=positions,
                            edge_index=graph.edge_index.long(), label=int(graph.y.reshape(-1)[0])))
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(records, out)
    out.with_suffix(".conversion.json").write_text(json.dumps({
        "source": str(args.input), "records": len(records),
        "historical_zero_padding_indices": padding,
    }, indent=2) + "\n")
    print(f"Exported {len(records)} graphs to {out}")


if __name__ == "__main__":
    main()
