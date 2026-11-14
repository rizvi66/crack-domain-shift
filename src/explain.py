"""Occlusion-sensitivity maps: which image regions drive the model's decision?

A neutral patch slides over the image. The score is the logit margin (cracked minus uncracked),
which stays informative even when the predicted probability is ~0 or ~1 (probability saturates,
so probability-based maps look like noise for confident predictions).

    map value = margin(original) - margin(patch-occluded)
    > 0 : covering this region lowers the "cracked" evidence (region supports "cracked")
    < 0 : covering this region raises it (region argues against "cracked")

All panels in one figure share ONE symmetric colour scale, so panels are comparable and a
panel with no strong region looks flat instead of being stretched to look structured.

Example:
    python -m src.explain --preds results/preds/resnet50_srcD_seed42.csv \
        --ckpt checkpoints/resnet50_srcD_seed42.pt --target P --k 4 --select random
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
def occlusion_map(model, x: torch.Tensor, patch: int, stride: int, device):
    """x: (3,H,W) normalised tensor. Returns (map, base_margin, base_prob)."""
    _, H, W = x.shape
    model.eval()
    z = model(x[None].to(device))[0]
    base_margin = (z[1] - z[0]).item()
    base_prob = torch.softmax(z, 0)[1].item()
    ys = list(range(0, max(H - patch, 0) + 1, stride))
    xs = list(range(0, max(W - patch, 0) + 1, stride))
    batch = []
    for y0 in ys:
        for x0 in xs:
            xo = x.clone()
            xo[:, y0:y0 + patch, x0:x0 + patch] = 0.0  # 0 = dataset mean after normalisation
            batch.append(xo)
    margins = []
    for i in range(0, len(batch), 64):
        out = model(torch.stack(batch[i:i + 64]).to(device))
        margins.append((out[:, 1] - out[:, 0]).cpu())
    margins = torch.cat(margins).numpy()
    return (base_margin - margins).reshape(len(ys), len(xs)), base_margin, base_prob


def select(df: pd.DataFrame, how: str, k: int, seed: int) -> pd.DataFrame:
    if how == "first":
        return df.head(k)
    if how == "random":
        return df.sample(n=min(k, len(df)), random_state=seed)
    if how == "confident":
        return df.sort_values("prob_cracked", ascending=False).head(k)
    if how == "uncertain":
        return df.assign(d=(df["prob_cracked"] - 0.5).abs()).sort_values("d").head(k).drop(columns="d")
    raise ValueError(how)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--target", required=True, choices=list(SURFACE_NAMES))
    ap.add_argument("--k", type=int, default=4, help="images per group (detected / missed)")
    ap.add_argument("--select", default="random", choices=["first", "random", "confident", "uncertain"],
                    help="how to pick images within each group (default: random, seeded)")
    ap.add_argument("--seed", type=int, default=0)
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
    groups = {"detected": select(df[df["pred"] == 1], a.select, a.k, a.seed),
              "missed": select(df[df["pred"] == 0], a.select, a.k, a.seed)}
    rows = [(name, r) for name, g in groups.items() for _, r in g.iterrows()]
    if not rows:
        raise SystemExit("No cracked images found for this target surface.")

    maps = []
    for name, r in rows:
        img = Image.open(r["path"]).convert("RGB")
        heat, margin, prob = occlusion_map(model, tf(img), a.patch, a.stride, device)
        maps.append((name, img, heat, margin, prob))
    lim = max(float(np.abs(m[2]).max()) for m in maps) or 1.0

    n = len(maps)
    fig, axes = plt.subplots(2, n, figsize=(2.4 * n + 0.8, 5.4), squeeze=False)
    for col, (name, img, heat, margin, prob) in enumerate(maps):
        axes[0, col].imshow(img.resize((ck["img_size"],) * 2))
        axes[0, col].set_title(f"{name}\np(crack)={prob:.2f}", fontsize=8)
        im = axes[1, col].imshow(heat, cmap="RdBu_r", vmin=-lim, vmax=lim)
        for row in (0, 1):
            axes[row, col].axis("off")
    fig.suptitle(f"{Path(a.ckpt).stem} on {SURFACE_NAMES[a.target]} images "
                 f"(selection: {a.select}; shared scale +/-{lim:.1f} logit units)", fontsize=9)
    fig.tight_layout(rect=(0, 0, 0.93, 1))
    cax = fig.add_axes([0.94, 0.12, 0.012, 0.35])
    fig.colorbar(im, cax=cax, label="margin drop when covered")
    Path(a.out).mkdir(parents=True, exist_ok=True)
    dest = Path(a.out) / f"{Path(a.ckpt).stem}_on_{a.target}_{a.select}.png"
    fig.savefig(dest, dpi=150)
    print(f"saved {dest}")


if __name__ == "__main__":
    main()
