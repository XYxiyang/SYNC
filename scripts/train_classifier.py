"""Train the original bond-type-free EGNN with a portable data/checkpoint format."""
import argparse
import json
import random
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sync_sbdd.data import collate_graphs
from sync_sbdd.model import SYNC, save_model


def collate(records):
    return collate_graphs(records), torch.tensor([r["label"] for r in records], dtype=torch.long)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--train", required=True)
    p.add_argument("--val", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--device", default="cpu")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--seed", type=int, default=530)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--layers", type=int, default=4)
    p.add_argument("--hidden-dim", type=int, default=128)
    p.add_argument("--threads", type=int, default=4)
    args = p.parse_args()
    if min(args.epochs, args.batch_size, args.layers, args.hidden_dim, args.threads) < 1 or args.lr <= 0:
        p.error("training dimensions, counts, and learning rate must be positive")
    torch.set_num_threads(args.threads)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    train = torch.load(args.train, weights_only=True)
    val = torch.load(args.val, weights_only=True)
    if not train or not val:
        raise ValueError("Training and validation datasets must be nonempty.")
    overlap = {r["smiles"] for r in train} & {r["smiles"] for r in val}
    if overlap:
        raise ValueError(f"{len(overlap)} molecule identities overlap between train and validation.")
    model = SYNC(hidden_dim=args.hidden_dim, num_layers=args.layers,
                 head_dim=2 * args.hidden_dim).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=.01,
                                 betas=(.9, .999))
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min",
                                                          factor=.6, patience=1, min_lr=1e-5)
    trainloader = DataLoader(train, batch_size=args.batch_size, shuffle=True, collate_fn=collate)
    valloader = DataLoader(val, batch_size=args.batch_size, collate_fn=collate)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(vars(args), indent=2) + "\n")
    history, best = [], -1.
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.
        for features, labels in trainloader:
            optimizer.zero_grad()
            logits = model(*(f.to(args.device) for f in features))
            loss = torch.nn.functional.cross_entropy(logits, labels.to(args.device))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            optimizer.step()
            train_loss += loss.item() * len(labels)
        model.eval()
        loss_sum, correct = 0., 0
        with torch.inference_mode():
            for features, labels in valloader:
                labels = labels.to(args.device)
                logits = model(*(f.to(args.device) for f in features))
                loss_sum += torch.nn.functional.cross_entropy(logits, labels, reduction="sum").item()
                correct += int((logits.argmax(-1) == labels).sum())
        accuracy, val_loss = correct / len(val), loss_sum / len(val)
        scheduler.step(val_loss)
        row = dict(epoch=epoch, train_loss=train_loss / len(train),
                   val_loss=val_loss, val_accuracy=accuracy)
        history.append(row)
        print(json.dumps(row), flush=True)
        if accuracy > best:
            best = accuracy
            save_model(out / "best.pt", model, epoch=epoch, seed=args.seed,
                       val_accuracy=accuracy, paper_table_checkpoint_verified=False)
        save_model(out / "last.pt", model, epoch=epoch, seed=args.seed,
                   paper_table_checkpoint_verified=False)
        (out / "history.json").write_text(json.dumps(history, indent=2) + "\n")


if __name__ == "__main__":
    main()
