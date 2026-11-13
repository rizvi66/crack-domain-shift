"""Read-only digest of results/: reformats summary.csv, gap.csv, mcnemar.csv and history/*.json.

Computes nothing new except (a) a plain-language verdict label from the existing gap CI and
(b) across-seed mean/sd of the existing gap values when more than one seed was run.
Run:  python scripts/results_digest.py --results results
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

pd.set_option("display.width", 220)
pd.set_option("display.max_rows", 500)


def fmt_ci(v, lo, hi, nd=3):
    return f"{v:.{nd}f} [{lo:.{nd}f}, {hi:.{nd}f}]"


def training_section(res: Path) -> None:
    print("=" * 100, "\nTRAINING (from history/*.json)\n" + "=" * 100)
    rows = []
    for f in sorted((res / "history").glob("*.json")):
        h = json.loads(f.read_text())
        ran, mx = len(h["history"]), h["args"]["epochs"]
        note = ""
        if h["best_epoch"] == ran == mx:
            note = "best epoch = last epoch allowed: may be under-trained"
        elif ran < mx:
            note = "stopped early"
        rows.append({"run": f.stem, "best_epoch": h["best_epoch"], "epochs_run": ran, "epochs_max": mx,
                     "best_val_bal_acc": round(h["best_val_bal_acc"], 3), "note": note})
    print(pd.DataFrame(rows).to_string(index=False) if rows else "no history files found")


def summary_section(res: Path) -> None:
    print("\n" + "=" * 100, "\nBALANCED ACCURACY, F1, AUC per model x train surface -> test surface\n" + "=" * 100)
    s = pd.read_csv(res / "summary.csv").sort_values(["model", "seed", "source", "target"])
    out = pd.DataFrame({
        "model": s["model"], "seed": s["seed"], "train->test": s["source"] + "->" + s["target"],
        "in_domain": s["in_domain"], "n_test": s["n_test"], "n_cracked": s["n_cracked"],
        "bal_acc [95% CI]": [fmt_ci(*r) for r in zip(s["bal_acc"], s["bal_acc_lo"], s["bal_acc_hi"])],
        "f1 [95% CI]": [fmt_ci(*r) for r in zip(s["f1"], s["f1_lo"], s["f1_hi"])],
        "auc": s["auc"].round(3)})
    print(out.to_string(index=False))


def verdict(row) -> str:
    if row["gap_ci_excludes_0"]:
        return ("DROP: CI excludes 0" if row["gap"] > 0
                else "target scored HIGHER than source in-domain: CI excludes 0")
    return "CI includes 0: no clear difference (this is NOT evidence of no drop)"


def gap_section(res: Path) -> None:
    print("\n" + "=" * 100,
          "\nGAP = in-domain score minus cross-domain score (positive = worse on the other surface)\n" + "=" * 100)
    g = pd.read_csv(res / "gap.csv").sort_values(["metric", "model", "seed", "source", "target"])
    g["verdict"] = g.apply(verdict, axis=1)
    out = pd.DataFrame({
        "metric": g["metric"], "model": g["model"], "seed": g["seed"],
        "train->test": g["source"] + "->" + g["target"],
        "gap [95% CI]": [fmt_ci(*r) for r in zip(g["gap"], g["gap_lo"], g["gap_hi"])],
        "verdict": g["verdict"]})
    print(out.to_string(index=False))
    n_seeds = g["seed"].nunique()
    if n_seeds > 1:
        print(f"\nAcross {n_seeds} seeds (mean +/- sd of the per-seed gap; each seed redraws the photograph split):")
        a = (g.groupby(["metric", "model", "source", "target"])["gap"]
             .agg(["mean", "std", "min", "max", "count"]).round(3).reset_index())
        print(a.to_string(index=False))
    else:
        print("\nOnly one seed: the bootstrap CI does not capture split-to-split or training-run variability.")


def mcnemar_section(res: Path) -> None:
    print("\n" + "=" * 100, "\nMcNEMAR between models on identical test images (plain accuracy, exact test)\n" + "=" * 100)
    m = pd.read_csv(res / "mcnemar.csv").sort_values(["seed", "source", "target"])
    m["discordant"] = m["only_a_correct"] + m["only_b_correct"]
    m["more_often_right"] = [a if x > y else (b if y > x else "tie") for a, b, x, y in
                             zip(m["model_a"], m["model_b"], m["only_a_correct"], m["only_b_correct"])]
    m["p_exact"] = m["p_exact"].round(4)
    cols = ["seed", "source", "target", "model_a", "model_b", "n", "only_a_correct", "only_b_correct",
            "discordant", "more_often_right", "p_exact"]
    print(m[cols].to_string(index=False))
    print(f"\n{len(m)} tests, no multiple-comparison correction. p_exact < 0.05 in some rows is expected by chance "
          "even if the models were equivalent. A large p-value means 'cannot tell', not 'equal'.")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    res = Path(ap.parse_args(argv).results)
    training_section(res)
    summary_section(res)
    gap_section(res)
    mcnemar_section(res)
    print("\nCAVEAT: all CIs and p-values treat test images as independent. Crops from one photograph are not,\n"
          "so the true uncertainty is larger than shown. See docs/READING_RESULTS.md.")


if __name__ == "__main__":
    main()
