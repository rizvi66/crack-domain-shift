# Reading the results honestly

This guide says how to read `results/` without overclaiming. It contains **no results**: every
number you quote must come from your own run. Facts about how the numbers are produced come from
reading `src/evaluate.py`, `src/train.py` and `src/explain.py`. `python scripts/results_digest.py`
prints the four CSVs as tables with the flags described below.

## 0. Check the run before reading any number

| Check | Where | Why it matters |
|---|---|---|
| 6 files in `results/preds/` | `ls results/preds` | 2 models x 3 source surfaces. Fewer means a run crashed. |
| `best_epoch` vs `epochs` | `results/history/*.json`, digest "TRAINING" | If best epoch is the last allowed epoch (default 10), the model may be under-trained and every gap is confounded with training length. |
| Photographs per split | notebook section 4 | See section 1: this sets how much to trust every interval. |
| Prevalence per surface | `describe()` output | F1 depends on how common cracks are (section 2). |
| Models near chance | `summary.csv`, `bal_acc` close to 0.5 | A model predicting "uncracked" for everything scores 0.5 balanced accuracy and 0 F1. Treat as a failed training run, not a result. |

## 1. The biggest caveat: the intervals are narrower than the truth

Every CI and p-value in this project resamples or pairs **images**, treating them as independent.
They are not: each image is a crop of a larger photograph, and crops from one photograph look alike.
The split respects this (whole photographs go to one split), but the statistics do not.

How few photographs there are: USU's SDNET2018 page describes the source as 230 images
(54 bridge decks, 72 walls, 104 pavements) of cracked and non-cracked concrete
([digitalcommons.usu.edu/all_datasets/48](https://digitalcommons.usu.edu/all_datasets/48)).
**If** the filename prefix maps one-to-one onto those photographs, the 70/15/15 split in
`assign_splits` leaves roughly **8 deck, 11 wall and 15 pavement photographs** in the test sets
(my arithmetic from the code's rounding; notebook section 4 prints the real counts for your run).
A test set built from ~8 photographs can swing a lot depending on which 8 were drawn.

Consequences:
- Treat every CI as a **lower bound on the real uncertainty**. "CI excludes 0" is weaker evidence than it looks.
- One seed means one draw of the test photographs. The only honest measure of split-to-split variation is to **re-run with other seeds** (`bash scripts/run_all.sh <root> 43`, etc.; each seed redraws the split and retrains). The digest prints the across-seed mean/sd of each gap when it finds several seeds. Time your first run from `run_all.log` before deciding how many seeds you can afford.

## 2. `summary.csv`: one row per model x train surface x test surface (x seed)

Columns: `bal_acc`, `f1` (each with `_lo`/`_hi` 95% bootstrap CI), `auc`, `n_test`, `n_cracked`, `in_domain`.

- **Diagonal** (`in_domain = True`) is the reference. Off-diagonal is the shifted condition.
- **Chance level** for balanced accuracy is 0.5; do not describe 0.6 as "good".
- **Look at `n_cracked`.** The positive class is a minority; a test surface with few cracked images has wide intervals on both metrics.
- **F1 is not comparable across test surfaces**, because it depends on prevalence. IEEE DataPort's listing of SDNET2018 gives cracked/non-cracked counts of 2025/11595 (D), 2608/21726 (P) and 3851/14287 (W) ([dataset page](https://ieee-dataport.org/documents/sdnet2018-concrete-crack-image-dataset-machine-learning-applications)), i.e. about 14.9%, 10.7% and 21.2% cracked. Check these against your own `describe()`. A lower F1 on pavement can be partly a prevalence effect. Use **balanced accuracy as the primary metric** (as the README does) and F1 as secondary.
- **`auc` vs `bal_acc` tells you *how* the model fails.** `bal_acc` uses a fixed 0.5 threshold that was never tuned on the target surface. If AUC stays high while balanced accuracy drops, the model still *ranks* cracks correctly but its probabilities/threshold are miscalibrated on the new surface. If both drop, the learned features themselves transfer poorly. `auc` has no CI here, so describe differences in it cautiously.

## 3. `gap.csv`: in-domain minus cross-domain

`gap = metric(source -> source) - metric(source -> target)`. **Positive = worse on the other surface.**
`gap_lo/gap_hi` is a bootstrap CI that resamples the two test sets independently.

Read each row in one of three ways, and say which:

| What you see | What you may say | What you may not say |
|---|---|---|
| CI excludes 0, gap > 0 | "On this split, balanced accuracy was lower on the other surface by about X (CI a to b; CI optimistic, section 1)." | "Models don't generalise across surfaces." |
| CI includes 0 | "This pilot could not distinguish the difference from sampling noise." Report the CI width: it is the size of effect the data could have detected. | "There is no drop." Absence of evidence is not evidence of absence, especially with ~10 test photographs. |
| CI excludes 0, gap < 0 | "The model scored *higher* on the other surface than on its own." | Anything about robustness. This is a sign the surfaces differ in intrinsic difficulty (next point). |

**The gap conflates "domain shift" with "the target surface is harder or easier".** A pavement
model scoring lower on walls than on pavements could mean the wall images are intrinsically
harder, not that the model transferred badly. To separate them, compare with the **target-trained
model's own in-domain score** (for example `P -> W` against `W -> W` for the same architecture,
both read straight from `summary.csv`). If `P -> W` is close to `W -> W`, the shift cost little.
This is a reading aid on existing numbers, not a change to the method.

**Many rows, no correction.** There are 12 gap rows per model (3 sources x 2 targets x 2 metrics). Some CIs will
exclude 0 by chance alone. Name your primary comparisons (balanced accuracy, the six train->test pairs)
before looking, and treat the rest as exploratory.

If the drop is **small**, report it as small, with its CI, and resist the temptation to call it significant
or negligible: with CIs this wide a "small" estimate is compatible with both.

## 4. `mcnemar.csv`: model A vs model B on identical test images

`model_a` is `efficientnet_b0` and `model_b` is `resnet50` (alphabetical). `only_a_correct` / `only_b_correct` are the
discordant images; `p_exact` is a two-sided exact binomial test on them.

- **It tests plain accuracy, not balanced accuracy.** With ~11-21% cracked images, accuracy is dominated by the
  uncracked class, so a model that rarely predicts "cracked" can win McNemar while being worse at finding cracks.
  In my *synthetic* smoke test (not your data) EfficientNet "won" a significant McNemar row while scoring exactly
  0.5 balanced accuracy, because it predicted "uncracked" for everything. Always read McNemar next to `bal_acc`.
- Few discordant images (`only_a_correct + only_b_correct`) means little power. A large p-value means "cannot tell", not "equivalent".
- Images from the same photograph are correlated, so p-values are optimistic (section 1). Nine tests per seed are uncorrected.
- A significant model difference is a statement about the models, not about domain shift. To say something about shift, compare how the difference
  changes between the diagonal and off-diagonal cells.

## 5. Heatmaps (`heatmap_<model>.png`)

- Rows = train surface, columns = test surface; the diagonal is in-domain.
- The colour scale is fixed at 0.5-1.0, so anything below 0.5 (possible for F1 on the rare class) is drawn the same as 0.5.
- **No intervals are drawn.** Two cells that look different may have overlapping CIs; check `summary.csv`.
- If several seeds exist, each cell is the mean over seeds, with no spread shown.
- Read **rows** to ask "what does this training surface buy me elsewhere?" and **columns** to ask "is this test surface just hard?". A uniformly low column, diagonal included, points to a hard surface rather than a shift.

## 6. Occlusion maps (`results/occlusion/*.png`)

What they are: a 32-px mean-grey patch slides at stride 16 over the 224-px image; each map cell is the drop in the model's
probability for the *cracked* class when that region is hidden (13 x 13 cells at 224 px, so blocky). Bright = hiding it hurt most.

Things to keep in mind before writing a sentence about them:
- **The examples are not sampled.** `explain.py` takes the first `k` correct and first `k` missed cracked images by file path.
  Those are likely to come from the lowest-numbered test photographs, possibly one or two. The comment in `run_all.sh` calls them
  "best-case transfer examples", but the code does not select by confidence. Do not describe them as best-case or typical.
- **Colour scales are per panel with no colour bar**, so brightness is not comparable between panels.
- A grey patch is itself unnatural input, and a drop in probability shows **sensitivity, not reasoning**. A hot region does not prove the model "uses the crack".
- The pipeline has **no pixel-level crack labels**, so it cannot measure how well the hot regions overlap the actual crack. Any statement about
  "the crack vs the background texture" rests on visual inspection of a handful of images.
- "Missed" panels show maps for the cracked-class probability of images the model scored below 0.5; the baseline probability is in each panel title.

Defensible wording: "In the N images shown (the first by file name, not sampled), the maps highlight regions that coincide with the visible crack in A of N
and with background texture or edges in B of N; this is anecdotal." Count them yourself from the figure.

## 7. Fill-in templates for the README Results section

Use only after replacing every placeholder with your own numbers.

- Transfer table: "Balanced accuracy, train surface x test surface, [model], seed [s] (95% bootstrap CI over images): ..."
- Drop, clearly present: "[model] trained on [S] lost [gap] balanced accuracy on [T] (95% CI [lo, hi], excludes 0). The interval treats images as independent and is therefore optimistic."
- Drop, small or not significant: "[model] trained on [S] scored [x] on [T] against [y] on [S] (gap [gap], 95% CI [lo, hi], includes 0). With about [n] test photographs for [T] this pilot cannot distinguish a drop of that size from sampling noise."
- Model comparison: "[A] and [B] disagreed on [d] of [n] [T] test images (exact McNemar p = [p], uncorrected, accuracy-based). Balanced accuracies were [a] and [b]."
- Occlusion: "In the [N] images shown (first by file name, not sampled) ..." as in section 6.

## 8. Claims this pilot cannot support

- That surface *material* causes the gap: SDNET2018 surfaces also differ in camera distance, lighting and labelling (README, Limitations).
- Anything beyond SDNET2018, or about "crack classifiers" in general: one dataset, two architectures, one (or a few) seeds.
- That one architecture is better than the other in general.
- That occlusion maps explain the model's decisions.
