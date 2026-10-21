"""Evaluate portable labeled graph files without changing conformations."""
import argparse
import json
import sys
from pathlib import Path

import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sync_sbdd.data import collate_graphs
from sync_sbdd.model import load_model


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--graphs", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--device", default="cpu")
    p.add_argument("--threads", type=int, default=4)
    args = p.parse_args()
    if args.batch_size < 1 or args.threads < 1:
        p.error("batch-size and threads must be positive")
    torch.set_num_threads(args.threads)
    model, _ = load_model(args.checkpoint, args.device)
    records = torch.load(args.graphs, map_location="cpu", weights_only=True)
    if not records:
        raise ValueError("Empty evaluation dataset")
    confusion = torch.zeros((2, 2), dtype=torch.long)
    with torch.inference_mode():
        for i in range(0, len(records), args.batch_size):
            batch = records[i:i+args.batch_size]
            pred = model(*collate_graphs(batch, args.device)).argmax(-1).cpu()
            labels = torch.tensor([r["label"] for r in batch])
            if not ((labels == 0) | (labels == 1)).all():
                raise ValueError("Labels must be 0/1")
            confusion += torch.bincount(2 * labels + pred, minlength=4).reshape(2, 2)
    result = dict(records=len(records), accuracy=int(confusion.trace()) / len(records),
                  confusion_true_by_predicted=confusion.tolist())
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
