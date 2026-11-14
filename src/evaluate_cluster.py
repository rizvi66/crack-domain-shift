"""Photograph-level (cluster) bootstrap evaluation.

Why: crops from one photograph look alike, so test images are NOT independent. The image-level
bootstrap in evaluate.py treats them as independent and gives intervals that are too narrow.
Here the unit of resampling is the *photograph* (the `group` column written by train.py).

Writes (next to the files evaluate.py writes; nothing is overwritten):
    results/summary_cluster.csv     per model x source x seed x target: metrics + cluster-bootstrap 95% CI
    results/gap_cluster.csv         in-domain minus cross-domain gap + cluster-bootstrap 95% CI
    results/model_diff_cluster.csv  paired model difference in balanced accuracy + cluster-bootstrap 95% CI
    results/seed_agreement.csv      mean / sd of balanced accuracy across seeds (needs >= 2 seeds)

Caution: with very few photographs (fewer than ~10 per test set) percentile intervals tend to be
too narrow even at photograph level. Use several seeds (each re-draws the photograph split) and
read the spread across seeds alongside the intervals.
"""
from __future__ import annotations

import argparse
import itertools
from pathlib import Path

import numpy as np
import pandas as pd

from .data import SURFACES
from .evaluate import load_preds

# ---- metrics from confusion counts (columns: tn, fp, fn, tp) -------------------------------

def _div(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    return np.divide(a, b, out=np.zeros(np.broadcast(a, b).shape), where=b > 0)


def bal_acc_from_counts(c):
    c = np.atleast_2d(c)
    tn, fp, fn, tp = c[:, 0], c[:, 1], c[:, 2], c[:, 3]
    return 0.5 * (_div(tp, tp + fn) + _div(tn, tn + fp))


def f1_from_counts(c):
    c = np.atleast_2d(c)
    tn, fp, fn, tp = c[:, 0], c[:, 1], c[:, 2], c[:, 3]
    return _div(2 * tp, 2 * tp + fp + fn)


METRICS = {"bal_acc": bal_acc_from_counts, "f1": f1_from_counts}


def group_counts(g: pd.DataFrame) -> pd.DataFrame:
    """One row per photograph: tn, fp, fn, tp (index = group id)."""
    y, p = g["label"].to_numpy(), g["pred"].to_numpy()
    d = pd.DataFrame({"group": g["group"].to_numpy(),
                      "tn": (y == 0) & (p == 0), "fp": (y == 0) & (p == 1),
                      "fn": (y == 1) & (p == 0), "tp": (y == 1) & (p == 1)})
    return d.groupby("group")[["tn", "fp", "fn", "tp"]].sum().astype(float)


def _boot_totals(counts: np.ndarray, n_boot: int, rng) -> np.ndarray:
    """(n_boot, 4) confusion totals, resampling photographs with replacement."""
    n = len(counts)
    idx = rng.integers(0, n, size=(n_boot, n))
    weights = np.apply_along_axis(lambda r: np.bincount(r, minlength=n), 1, idx)
    return weights @ counts


def ci(x, lo=2.5, hi=97.5):
    return np.percentile(x, [lo, hi])


# ---- analyses -------------------------------------------------------------------------------

def summarise(df: pd.DataFrame, n_boot: int, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for (model, source, sd, target), g in df.groupby(["model", "source", "seed", "surface"]):
        gc = group_counts(g)
        counts = gc.to_numpy()
        row = {"model": model, "source": source, "seed": sd, "target": target,
               "in_domain": source == target, "n_images": len(g), "n_groups": len(gc)}
        boot = _boot_totals(counts, n_boot, rng)
        for name, fn in METRICS.items():
            lo, hi = ci(fn(boot))
            row.update({name: float(fn(counts.sum(0))[0]), f"{name}_lo": lo, f"{name}_hi": hi})
        rows.append(row)
    return pd.DataFrame(rows)


def gaps(df: pd.DataFrame, n_boot: int, seed: int = 1) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for (model, source, sd), g in df.groupby(["model", "source", "seed"]):
        ind = group_counts(g[g["surface"] == source]).to_numpy()
        for target in SURFACES:
            if target == source or (g["surface"] == target).sum() == 0:
                continue
            oth = group_counts(g[g["surface"] == target]).to_numpy()
            b1, b2 = _boot_totals(ind, n_boot, rng), _boot_totals(oth, n_boot, rng)
            for name, fn in METRICS.items():
                gap = float(fn(ind.sum(0))[0] - fn(oth.sum(0))[0])
                lo, hi = ci(fn(b1) - fn(b2))
                rows.append({"model": model, "source": source, "seed": sd, "target": target,
                             "metric": name, "gap": gap, "gap_lo": lo, "gap_hi": hi,
                             "n_groups_in": len(ind), "n_groups_target": len(oth),
                             "ci_excludes_0": bool(lo > 0 or hi < 0)})
    return pd.DataFrame(rows)


def model_diffs(df: pd.DataFrame, n_boot: int, seed: int = 2) -> pd.DataFrame:
    """Paired comparison of two models on the same test photographs (balanced accuracy)."""
    rng = np.random.default_rng(seed)
    rows = []
    for (source, sd, target), g in df.groupby(["source", "seed", "surface"]):
        for a, b in itertools.combinations(sorted(g["model"].unique()), 2):
            ga, gb = g[g["model"] == a], g[g["model"] == b]
            common = set(ga["path"]) & set(gb["path"])
            ga, gb = ga[ga["path"].isin(common)], gb[gb["path"].isin(common)]
            ca, cb = group_counts(ga), group_counts(gb)
            ca, cb = ca.loc[ca.index.intersection(cb.index)], cb.loc[ca.index.intersection(cb.index)]
            A, B = ca.to_numpy(), cb.to_numpy()
            n = len(A)
            idx = rng.integers(0, n, size=(n_boot, n))
            W = np.apply_along_axis(lambda r: np.bincount(r, minlength=n), 1, idx)
            d = bal_acc_from_counts(W @ A) - bal_acc_from_counts(W @ B)
            lo, hi = ci(d)
            diff = float(bal_acc_from_counts(A.sum(0))[0] - bal_acc_from_counts(B.sum(0))[0])
            rows.append({"source": source, "seed": sd, "target": target, "model_a": a, "model_b": b,
                         "n_groups": n, "bal_acc_a_minus_b": diff, "lo": lo, "hi": hi,
                         "ci_excludes_0": bool(lo > 0 or hi < 0)})
    return pd.DataFrame(rows)


def seed_agreement(summary: pd.DataFrame) -> pd.DataFrame:
    if summary["seed"].nunique() < 2:
        return pd.DataFrame()
    t = summary.groupby(["model", "source", "target"])["bal_acc"].agg(["mean", "std", "min", "max", "count"])
    return t.rename(columns={"count": "n_seeds"}).reset_index()


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--n-boot", type=int, default=2000)
    a = ap.parse_args(argv)
    res = Path(a.results)
    df = load_preds(res / "preds")
    if "group" not in df.columns:
        raise SystemExit("predictions have no 'group' column; re-run train.py from this version of the repo")
    s = summarise(df, a.n_boot)
    s.to_csv(res / "summary_cluster.csv", index=False)
    gaps(df, a.n_boot).to_csv(res / "gap_cluster.csv", index=False)
    model_diffs(df, a.n_boot).to_csv(res / "model_diff_cluster.csv", index=False)
    sa = seed_agreement(s)
    if len(sa):
        sa.to_csv(res / "seed_agreement.csv", index=False)
    pd.set_option("display.width", 200)
    print(s[["model", "source", "target", "seed", "n_groups", "bal_acc", "bal_acc_lo", "bal_acc_hi"]]
          .round(3).to_string(index=False))
    print(f"\nwrote cluster-bootstrap tables to {res}/")


if __name__ == "__main__":
    main()
