import argparse
import csv
import json
from pathlib import Path

import torch

from .data import collate_graphs, molecule_to_graph, sdf_records
from .model import load_model


def score(args):
    model, metadata = load_model(args.checkpoint, args.device)
    torch.set_num_threads(args.threads)
    rows, pending, indices = [], [], []

    def flush():
        if not pending:
            return
        with torch.inference_mode():
            logits = model(*collate_graphs(pending, args.device))
            probability = logits.softmax(-1)[:, 1].cpu().tolist()
            predictions = logits.argmax(-1).cpu().tolist()
        for row, prob, prediction in zip(indices, probability, predictions):
            rows[row].update(probability_es=prob, prediction=prediction, status="ok")
        pending.clear()
        indices.clear()

    for file, index, mol in sdf_records(args.input):
        row = dict(file=str(file), record=index, name="", smiles="",
                   probability_es="", prediction="", status="error", error="")
        rows.append(row)
        try:
            graph = molecule_to_graph(mol)
            row["smiles"] = graph["smiles"]
            row["name"] = mol.GetProp("_Name") if mol.HasProp("_Name") else ""
            pending.append(graph)
            indices.append(len(rows) - 1)
        except (ValueError, RuntimeError) as error:
            row["error"] = str(error)
        if len(pending) >= args.batch_size:
            flush()
    flush()
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    fields = ["file", "record", "name", "smiles", "probability_es", "prediction", "status", "error"]
    with out.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    valid = [r for r in rows if r["status"] == "ok"]
    summary = dict(records=len(rows), valid=len(valid), failed=len(rows) - len(valid),
                   predicted_es=sum(r["prediction"] for r in valid),
                   es_fraction_valid=sum(r["prediction"] for r in valid) / len(valid) if valid else None,
                   checkpoint=str(args.checkpoint),
                   checkpoint_source_sha256=metadata.get("source_sha256"))
    out.with_suffix(".summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    if not valid:
        raise SystemExit("No valid 3D molecules were scored; inspect the error column.")


def main():
    parser = argparse.ArgumentParser(description="SYNC: score existing 3D molecules.")
    parser.add_argument("--input", required=True, help="SDF file or directory (recursive).")
    parser.add_argument("--checkpoint", required=True, help="Converted state checkpoint.")
    parser.add_argument("--output", required=True, help="Output CSV; every SDF record is retained.")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    if args.batch_size < 1 or args.threads < 1:
        parser.error("batch-size and threads must be positive")
    score(args)


if __name__ == "__main__":
    main()
