"""Occlusion-sensitivity maps: which image regions drive the 'cracked' prediction?

Slide a neutral patch over the image and record how much the predicted probability
of the true class drops. Large drops mark regions the model relies on.

Example:
    python -m src.explain --preds results/preds/resnet50_srcD_seed42.csv \
        --ckpt checkpoints/resnet50_srcD_seed42.pt --target P --k 4
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image

from .data import SURFACE_NAMES, make_transforms
from .models import build_model


@torch.no_grad()
def occlusion_map(model, x: torch.Tensor, target_class: int, patch: int, stride: int, device):
    """x: (3,H,W) normalised tensor. Returns (gh,gw) array of probability drops."""
    _, H, W = x.shape
    model.eval()
    base = torch.softmax(model(x[None].to(device)), 1)[0, target_class].item()
    ys = list(range(0, max(H - patch, 0) + 1, stride))
    xs = list(range(0, max(W - patch, 0) + 1, stride))
    batch, coords = [], []
    for y0 in ys:
        for x0 in xs:
            xo = x.clone()
            xo[:, y0:y0 + patch, x0:x0 + patch] = 0.0  # 0 = dataset mean after normalisation
            batch.append(xo)
            coords.append((y0, x0))
    probs = []
    for i in range(0, len(batch), 64):
        out = model(torch.stack(batch[i:i + 64]).to(device))
        probs.append(torch.softmax(out, 1)[:, target_class].cpu())
    probs = torch.cat(probs).numpy()
    return (base - probs).reshape(len(ys), len(xs)), base


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--target", required=True, choices=list(SURFACE_NAMES))
    ap.add_argument("--k", type=int, default=4, help="images per group (correct / wrong)")
    ap.add_argument("--patch", type=int, default=32)
    ap.add_argument("--stride", type=int, default=16)
    ap.add_argument("--out", default="results/occlusion")
    a = ap.parse_args(argv)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(a.ckpt, map_location=device)
    model = build_model(ck["model"], pretrained=False).to(device)
    model.load_state_dict(ck["state_dict"])
    tf = make_transforms(ck["img_size"], train=False)

    df = pd.read_csv(a.preds)
    df = df[(df["surface"] == a.target) & (df["label"] == 1)]
    groups = {"correct": df[df["pred"] == 1].head(a.k), "missed": df[df["pred"] == 0].head(a.k)}
    rows = [(name, r) for name, g in groups.items() for _, r in g.iterrows()]
    if not rows:
        raise SystemExit("No cracked images found for this target surface.")

    fig, axes = plt.subplots(2, len(rows), figsize=(2.4 * len(rows), 5.2), squeeze=False)
    for col, (name, r) in enumerate(rows):
        img = Image.open(r["path"]).convert("RGB")
        x = tf(img)
        heat, base = occlusion_map(model, x, 1, a.patch, a.stride, device)
        axes[0, col].imshow(img.resize((ck["img_size"],) * 2))
        axes[0, col].set_title(f"{name}\np(crack)={base:.2f}", fontsize=8)
        axes[1, col].imshow(heat, cmap="magma")
        axes[1, col].set_title("prob. drop", fontsize=8)
        for row in (0, 1):
            axes[row, col].axis("off")
    fig.suptitle(f"{Path(a.ckpt).stem} on {SURFACE_NAMES[a.target]} images", fontsize=10)
    fig.tight_layout()
    Path(a.out).mkdir(parents=True, exist_ok=True)
    dest = Path(a.out) / f"{Path(a.ckpt).stem}_on_{a.target}.png"
    fig.savefig(dest, dpi=150)
    print(f"saved {dest}")


if __name__ == "__main__":
    main()
