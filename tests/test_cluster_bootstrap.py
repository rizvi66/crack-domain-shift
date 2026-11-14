import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, f1_score

from src.evaluate_cluster import (bal_acc_from_counts, f1_from_counts, group_counts,
                                  summarise)


def _fake(n_groups=12, per=40, seed=0, p_correct=0.8):
    rng = np.random.default_rng(seed)
    rows = []
    for gi in range(n_groups):
        y = (rng.random(per) < 0.2).astype(int)
        # photograph-level difficulty makes crops within a photograph correlated
        acc = np.clip(p_correct + rng.normal(0, 0.1), 0.3, 0.99)
        p = np.where(rng.random(per) < acc, y, 1 - y)
        rows += [{"group": f"g{gi}", "label": a, "pred": b, "path": f"{gi}-{k}", "model": "m",
                  "source": "D", "seed": 0, "surface": "D", "prob_cracked": 0.5}
                 for k, (a, b) in enumerate(zip(y, p))]
    return pd.DataFrame(rows)


def test_count_metrics_match_sklearn():
    df = _fake()
    c = group_counts(df).to_numpy().sum(0)
    assert np.isclose(bal_acc_from_counts(c)[0], balanced_accuracy_score(df["label"], df["pred"]))
    assert np.isclose(f1_from_counts(c)[0], f1_score(df["label"], df["pred"]))


def test_cluster_interval_brackets_estimate_and_is_wider_than_image_level():
    from src.evaluate import bootstrap_ci
    df = _fake(n_groups=12)
    s = summarise(df, n_boot=500).iloc[0]
    assert s["bal_acc_lo"] <= s["bal_acc"] <= s["bal_acc_hi"]
    lo, hi = bootstrap_ci(df["label"].to_numpy(), df["pred"].to_numpy(),
                          lambda y, p, _: balanced_accuracy_score(y, p), n_boot=500)
    assert (s["bal_acc_hi"] - s["bal_acc_lo"]) > (hi - lo)
