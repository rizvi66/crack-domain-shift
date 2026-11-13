"""Cross-surface evaluation: metrics, bootstrap CIs, generalisation gaps, McNemar tests.

Reads results/preds/*.csv written by train.py and writes:
    results/summary.csv     metrics (with 95% bootstrap CIs) per model x source x target surface
    results/gap.csv         in-domain minus cross-domain gap with bootstrap CI
    results/mcnemar.csv     exact McNemar test between models on identical test images
    results/heatmap_*.png   source x target heatmaps
"""
from __future__ import annotations

import argparse
import itertools
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score

from .data import SURFACES, SURFACE_NAMES

METRICS = {
    "bal_acc": lambda y, p, s: balanced_accuracy_score(y, p),
    "f1": lambda y, p, s: f1_score(y, p, zero_division=0),
}


def bootstrap_diff(y1, p1, y2, p2, fn, n_boot=1000, seed=0):
    """Bootstrap of metric(sample1) - metric(sample2) with independent resampling."""
    rng = np.random.default_rng(seed)
    diffs = np.empty(n_boot)
    for b in range(n_boot):
        i = rng.integers(0, len(y1), len(y1))
        j = rng.integers(0, len(y2), len(y2))
        diffs[b] = fn(y1[i], p1[i], None) - fn(y2[j], p2[j], None)
    return np.percentile(diffs, [2.5, 97.5])


def bootstrap_ci(y, p, fn, n_boot=1000, seed=0):
    rng = np.random.default_rng(seed)
    vals = np.empty(n_boot)
    for b in range(n_boot):
        i = rng.integers(0, len(y), len(y))
        vals[b] = fn(y[i], p[i], None)
    return np.percentile(vals, [2.5, 97.5])


def load_preds(pred_dir: Path) -> pd.DataFrame:
    files = sorted(pred_dir.glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"No prediction files in {pred_dir}")
    return pd.concat([pd.read_csv(f) for f in files], ignore_index=True)


def summarise(df: pd.DataFrame, n_boot: int) -> pd.DataFrame:
    rows = []
    for (model, source, seed, target), g in df.groupby(["model", "source", "seed", "surface"]):
        y, p = g["label"].to_numpy(), g["pred"].to_numpy()
        row = {"model": model, "source": source, "seed": seed, "target": target,
               "in_domain": source == target, "n_test": len(g), "n_cracked": int(y.sum())}
        for name, fn in METRICS.items():
            lo, hi = bootstrap_ci(y, p, fn, n_boot)
            row.update({name: fn(y, p, None), f"{name}_lo": lo, f"{name}_hi": hi})
        row["auc"] = roc_auc_score(y, g["prob_cracked"]) if 0 < y.sum() < len(y) else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def gaps(df: pd.DataFrame, n_boot: int) -> pd.DataFrame:
    rows = []
    for (model, source, seed), g in df.groupby(["model", "source", "seed"]):
        ind = g[g["surface"] == source]
        for target in SURFACES:
            if target == source:
                continue
            oth = g[g["surface"] == target]
            if oth.empty or ind.empty:
                continue
            for name, fn in METRICS.items():
                y1, p1 = ind["label"].to_numpy(), ind["pred"].to_numpy()
                y2, p2 = oth["label"].to_numpy(), oth["pred"].to_numpy()
                gap = fn(y1, p1, None) - fn(y2, p2, None)
                lo, hi = bootstrap_diff(y1, p1, y2, p2, fn, n_boot)
                rows.append({"model": model, "source": source, "seed": seed, "target": target,
                             "metric": name, "gap": gap, "gap_lo": lo, "gap_hi": hi,
                             "gap_ci_excludes_0": bool(lo > 0 or hi < 0)})
    return pd.DataFrame(rows)


def mcnemar_models(df: pd.DataFrame) -> pd.DataFrame:
    """Exact McNemar between each pair of models on the same test images."""
    rows = []
    for (source, seed, target), g in df.groupby(["source", "seed", "surface"]):
        models = sorted(g["model"].unique())
        for a, b in itertools.combinations(models, 2):
            ga = g[g["model"] == a].set_index("path")
            gb = g[g["model"] == b].set_index("path")
            common = ga.index.intersection(gb.index)
            ca = (ga.loc[common, "pred"] == ga.loc[common, "label"]).to_numpy()
            cb = (gb.loc[common, "pred"] == gb.loc[common, "label"]).to_numpy()
            n10, n01 = int((ca & ~cb).sum()), int((~ca & cb).sum())  # only A right / only B right
            p = binomtest(min(n10, n01), n10 + n01, 0.5).pvalue if n10 + n01 else 1.0
            rows.append({"source": source, "seed": seed, "target": target, "model_a": a, "model_b": b,
                         "n": len(common), "only_a_correct": n10, "only_b_correct": n01, "p_exact": p})
    return pd.DataFrame(rows)


def heatmaps(summary: pd.DataFrame, out: Path) -> None:
    for model, g in summary.groupby("model"):
        fig, axes = plt.subplots(1, len(METRICS), figsize=(4.2 * len(METRICS), 3.6))
        for ax, name in zip(np.atleast_1d(axes), METRICS):
            m = g.groupby(["source", "target"])[name].mean().unstack().reindex(
                index=SURFACES, columns=SURFACES)
            im = ax.imshow(m.to_numpy(dtype=float), vmin=0.5, vmax=1.0, cmap="viridis")
            ax.set_xticks(range(3), [SURFACE_NAMES[s] for s in SURFACES])
            ax.set_yticks(range(3), [SURFACE_NAMES[s] for s in SURFACES])
            ax.set_xlabel("test surface")
            ax.set_ylabel("train surface")
            ax.set_title(f"{model}: {name}")
            for i, j in itertools.product(range(3), range(3)):
                v = m.to_numpy(dtype=float)[i, j]
                if not np.isnan(v):
                    ax.text(j, i, f"{v:.2f}", ha="center", va="center", color="w" if v < 0.8 else "k")
            fig.colorbar(im, ax=ax, fraction=0.046)
        fig.tight_layout()
        fig.savefig(out / f"heatmap_{model}.png", dpi=150)
        plt.close(fig)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--n-boot", type=int, default=1000)
    a = ap.parse_args(argv)
    res = Path(a.results)
    df = load_preds(res / "preds")
    summary = summarise(df, a.n_boot)
    summary.to_csv(res / "summary.csv", index=False)
    gaps(df, a.n_boot).to_csv(res / "gap.csv", index=False)
    mcnemar_models(df).to_csv(res / "mcnemar.csv", index=False)
    heatmaps(summary, res)
    pd.set_option("display.width", 200)
    print(summary[["model", "source", "target", "seed", "n_test", "bal_acc", "bal_acc_lo",
                   "bal_acc_hi", "f1", "auc"]].round(3).to_string(index=False))
    print(f"\nwrote summary.csv, gap.csv, mcnemar.csv and heatmaps to {res}/")


if __name__ == "__main__":
    main()
