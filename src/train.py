"""Train on ONE source surface, then evaluate on the test split of ALL surfaces.

Example:
    python -m src.train --data-root /path/to/SDNET2018 --model resnet50 --source D
"""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import balanced_accuracy_score
from torch.utils.data import DataLoader

from .data import CrackDataset, SURFACES, build_manifest, check_no_leakage, describe
from .models import MODEL_NAMES, build_model


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def predict(model, df: pd.DataFrame, img_size: int, batch_size: int, workers: int, device):
    """Return probability of the 'cracked' class for each row of df."""
    loader = DataLoader(CrackDataset(df, img_size, train=False), batch_size=batch_size,
                        shuffle=False, num_workers=workers)
    model.eval()
    probs = []
    for x, _, _ in loader:
        probs.append(torch.softmax(model(x.to(device)), dim=1)[:, 1].cpu())
    return torch.cat(probs).numpy()


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--model", default="resnet50", choices=MODEL_NAMES)
    ap.add_argument("--source", default="D", choices=SURFACES, help="surface to train on")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--patience", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--img-size", type=int, default=224)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--max-per-class", type=int, default=None, help="cap images (smoke tests only)")
    ap.add_argument("--no-pretrained", action="store_true", help="random init (smoke tests only)")
    ap.add_argument("--out", default="results")
    ap.add_argument("--ckpt-dir", default="checkpoints")
    a = ap.parse_args(argv)

    set_seed(a.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out, ckdir = Path(a.out), Path(a.ckpt_dir)
    (out / "preds").mkdir(parents=True, exist_ok=True)
    (out / "history").mkdir(parents=True, exist_ok=True)
    ckdir.mkdir(parents=True, exist_ok=True)

    # The split depends on --seed so repeated seeds also re-draw the photograph split.
    df = build_manifest(a.data_root, seed=a.seed, max_per_class=a.max_per_class)
    check_no_leakage(df)
    print(describe(df).to_string())

    src = df[df["surface"] == a.source]
    tr, va = src[src["split"] == "train"], src[src["split"] == "val"]
    tr_loader = DataLoader(CrackDataset(tr, a.img_size, train=True), batch_size=a.batch_size,
                           shuffle=True, num_workers=a.workers, drop_last=len(tr) > a.batch_size)

    counts = np.bincount(tr["label"], minlength=2).astype(float)
    weights = torch.tensor(counts.sum() / (2.0 * np.maximum(counts, 1)), dtype=torch.float32)
    print(f"class counts (uncracked, cracked) = {counts.tolist()}  loss weights = {weights.tolist()}")

    model = build_model(a.model, pretrained=not a.no_pretrained).to(device)
    loss_fn = nn.CrossEntropyLoss(weight=weights.to(device))
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.epochs)

    tag = f"{a.model}_src{a.source}_seed{a.seed}"
    best, best_epoch, history = -1.0, -1, []
    for epoch in range(a.epochs):
        model.train()
        t0, running = time.time(), 0.0
        for x, y, _ in tr_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            loss = loss_fn(model(x), y)
            loss.backward()
            opt.step()
            running += loss.item() * len(y)
        sched.step()
        p = predict(model, va, a.img_size, a.batch_size, a.workers, device)
        val_bal = balanced_accuracy_score(va["label"], (p >= 0.5).astype(int))
        history.append({"epoch": epoch + 1, "train_loss": running / len(tr), "val_bal_acc": val_bal})
        print(f"[{tag}] epoch {epoch + 1}/{a.epochs} loss={running / len(tr):.4f} "
              f"val_bal_acc={val_bal:.4f} ({time.time() - t0:.0f}s)")
        if val_bal > best:
            best, best_epoch = val_bal, epoch + 1
            torch.save({"model": a.model, "img_size": a.img_size, "state_dict": model.state_dict()},
                       ckdir / f"{tag}.pt")
        elif epoch + 1 - best_epoch >= a.patience:
            print("early stopping")
            break

    ck = torch.load(ckdir / f"{tag}.pt", map_location=device)
    model.load_state_dict(ck["state_dict"])
    test = df[df["split"] == "test"].copy()
    test["prob_cracked"] = predict(model, test, a.img_size, a.batch_size, a.workers, device)
    test["pred"] = (test["prob_cracked"] >= 0.5).astype(int)
    test["model"], test["source"], test["seed"] = a.model, a.source, a.seed
    test.to_csv(out / "preds" / f"{tag}.csv", index=False)
    (out / "history" / f"{tag}.json").write_text(json.dumps(
        {"args": vars(a), "best_epoch": best_epoch, "best_val_bal_acc": best, "history": history}, indent=2))
    print(f"saved predictions for {len(test)} test images -> results/preds/{tag}.csv")


if __name__ == "__main__":
    main()
