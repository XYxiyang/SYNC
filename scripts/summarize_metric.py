"""Aggregate the existing metric CSV format with the final paper's thresholds."""
import argparse
import csv
import json
import math
from pathlib import Path

THRESHOLDS = {"sa": (.61, "higher"), "syba": (-18.6, "higher"), "fs": (-13., "higher"),
              "ra": (.5, "higher"), "sc": (3.1, "lower")}


def summarize(files, metric, accuracy_rule="legacy"):
    if accuracy_rule not in {"legacy", "binary"}:
        raise ValueError("accuracy_rule must be legacy or binary")
    threshold, direction = THRESHOLDS[metric]
    total = valid = es = correct = labeled = 0
    ties = {"positive": 0, "negative": 0, "unlabeled": 0}
    values = []
    for path, label in files:
        with open(path) as handle:
            reader = csv.DictReader(handle)
            if "smiles_score" not in (reader.fieldnames or []):
                raise ValueError(f"{path} needs a smiles_score column")
            for row in reader:
                total += 1
                try:
                    value = float(row["smiles_score"])
                except (ValueError, TypeError):
                    continue
                if not math.isfinite(value):
                    continue
                prediction = value > threshold if direction == "higher" else value < threshold
                if value == threshold:
                    ties["unlabeled" if label is None else "positive" if label else "negative"] += 1
                valid += 1
                es += int(prediction)
                values.append(value)
                if label is not None:
                    labeled += 1
                    if accuracy_rule == "legacy":
                        # Metrics/statis/acc.py counts ES > t and HS < t.
                        # For lower-is-better metrics it complements that count.
                        higher_correct = value > threshold if label else value < threshold
                        correct += higher_correct if direction == "higher" else not higher_correct
                    else:
                        correct += prediction == label
    if not valid:
        raise ValueError("No finite metric values")
    return dict(metric=metric, threshold=threshold, direction=direction, records=total,
                valid=valid, failed=total-valid, mean=sum(values)/valid,
                es_fraction=es/valid, accuracy=correct/labeled if labeled else None,
                accuracy_rule=accuracy_rule, threshold_ties=ties)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--metric", required=True, choices=THRESHOLDS)
    p.add_argument("--positive", help="CSV of ES molecules")
    p.add_argument("--negative", help="CSV of HS molecules")
    p.add_argument("--unlabeled", help="CSV of generated molecules")
    p.add_argument("--accuracy-rule", choices=["legacy", "binary"], default="legacy",
                   help="legacy reproduces the original label-dependent threshold equality handling.")
    p.add_argument("--output", required=True)
    args = p.parse_args()
    files = [(f, label) for f, label in [(args.positive, 1), (args.negative, 0),
                                       (args.unlabeled, None)] if f]
    if not files:
        p.error("Supply at least one CSV")
    result = summarize(files, args.metric, args.accuracy_rule)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
