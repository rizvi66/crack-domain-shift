# Change log (Claude, 2026-11-13)

Scope requested: set up on Kaggle, run `bash scripts/run_all.sh <data-root>`, fix errors without changing the method,
help read the results. Verified by `diff -r` against the uploaded `crack-domain-shift.tar.gz`.

## Changes to existing code: none

No file that was in the uploaded archive was modified (`src/*.py`, `scripts/run_all.sh`, `tests/*.py`,
`requirements.txt`, `README.md`, `LICENSE`, `.gitignore`). Nothing errored in testing, so nothing was "fixed".

## Files added (additive only, no effect on the method)

| File | Purpose |
|---|---|
| `notebooks/kaggle_run.ipynb` | Kaggle runner: GPU/Internet preflight, finds the SDNET2018 data root, gets the code in, runs the tests, prints photograph counts per split, a ~1-minute real-data smoke test (throw-away folder), then runs exactly `bash scripts/run_all.sh <root> <seed>`, checks the outputs exist, shows the digest and figures, zips `results/`. |
| `scripts/results_digest.py` | Read-only. Reformats `summary.csv`, `gap.csv`, `mcnemar.csv`, `history/*.json` and adds plain-language flags ("CI includes 0", "best epoch = last epoch"). Computes no new statistics except across-seed mean/sd of existing gap values. |
| `docs/READING_RESULTS.md` | How to read each output file honestly (small / non-significant drops, the image-level bootstrap caveat, F1 and prevalence, McNemar on accuracy, occlusion-map selection). Contains no results. |
| `CHANGES.md` | This file. |

## What was tested, and what was not

Tested (CPU only, synthetic data from `tests/make_synthetic.py`, `--no-pretrained --img-size 64 --epochs 2`, flags documented in the README):
- `pytest`: 2 passed. Also under pandas 2.2.3 (Kaggle images ship pandas 2.x) and pandas 3.x.
- The full `run_all.sh` command sequence (6 training runs, `src.evaluate`, `src.explain`) completed with no errors and no warnings from `evaluate`.
  `summary.csv`, `gap.csv`, `mcnemar.csv` were identical under pandas 2.2.3 and 3.x at the same bootstrap settings.
- The notebook, executed headlessly against a simulated `/kaggle/input` + `/kaggle/working` layout (both the tarball path; GPU/Internet checks switched off for that rehearsal only).
  The unpacked-folder path and the digest's multi-seed branch were exercised separately / on duplicated synthetic predictions (code-path test only).

NOT tested (not possible from this workspace, which cannot reach Kaggle or download.pytorch.org, and has no GPU):
- Anything on the real SDNET2018 data. **No real results exist yet; none are reported anywhere.**
- Downloading the ImageNet weights (`pretrained=True`), GPU training, real run time and memory.
- The `REPO_URL` (git clone) branch of the notebook.
- How a specific Kaggle copy of SDNET2018 is laid out. The notebook searches for `D/CD` + `D/UD` and stops with a clear message if it finds none.

## Observed in the code, deliberately left unchanged (method or cosmetic; your call)

1. `src/train.py` prints `-> results/preds/<tag>.csv` even when `--out` points elsewhere (cosmetic; files go to `--out`).
2. `src/explain.py` picks the first `k` correct and first `k` missed cracked images in file order, not by confidence or at random; the comment in `scripts/run_all.sh` ("best-case transfer examples") does not match that.
3. `scripts/run_all.sh` has no resume: a crash in run N retrains runs 1..N-1 on re-run.
4. Bootstrap CIs and McNemar treat images as independent although crops share photographs (see `docs/READING_RESULTS.md` section 1).
5. McNemar compares plain accuracy, not balanced accuracy.
6. Heatmap colour scale is clipped at 0.5; occlusion panels have per-panel colour scales.

---

# Day 1 hardening (branch `day1-hardening`)

Additive except `src/explain.py`, which is rewritten (its old behaviour is listed as item 2 and 6 above).

| File | Change |
|---|---|
| `src/evaluate_cluster.py` (new) | Photograph-level (cluster) bootstrap: CIs for each metric, in-domain-minus-cross-domain gaps, paired model differences in balanced accuracy, and across-seed mean/sd. Fixes item 4 (image-level bootstrap treats crops of one photograph as independent) and item 5 (model comparison on balanced accuracy instead of plain-accuracy McNemar). |
| `src/explain.py` (rewritten) | Uses the logit margin instead of probability (probability saturates near 0/1 and made maps of missed cracks look like noise); one shared symmetric colour scale and a colour bar; `--select random` (seeded) by default instead of the first images in file order. Output file name now ends in the selection mode. |
| `scripts/run_seeds.sh` (new) | Several seeds (each re-draws the photograph split), resumable (skips finished runs), then runs both evaluators. |
| `tests/test_cluster_bootstrap.py` (new) | Count-based metrics match scikit-learn; cluster interval brackets the estimate and is wider than the image-level one on clustered data. |

Tested on synthetic data only (CPU): `pytest` (4 passed) and a 2-seed, 2-model run through train, evaluate, evaluate_cluster and explain.
Not tested: the real data and GPU runs. With only 8 to 15 test photographs per surface, even photograph-level percentile intervals can be too narrow; read the across-seed spread (`seed_agreement.csv`) alongside them.
