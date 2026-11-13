# Do crack classifiers survive a change of surface?

A small pilot study of **domain shift in concrete-crack classification**. A classifier is
trained on images from one surface type (bridge decks, walls or pavements) and tested on the
others. The project measures how much performance drops, whether that drop is larger than
sampling noise, and which image regions the model relies on.

**Status:** code complete and smoke-tested on synthetic data. Results below are filled in
only after running on the real dataset (see [Results](#results)).

## Research questions

1. **Robustness under shift.** How much does accuracy fall when a crack classifier is applied to a surface it was not trained on?
2. **Is the drop real?** Are gaps and between-model differences larger than sampling noise (bootstrap confidence intervals, exact McNemar test)?
3. **What does the model look at?** Do occlusion-sensitivity maps highlight the crack itself, or background texture?

## Data

[SDNET2018](https://digitalcommons.usu.edu/all_datasets/48/) (Maguire, Dorafshan and Thomas, Utah State University): about 56,000 images of cracked and uncracked
concrete, 256x256 px, from three surfaces: **D**ecks, **W**alls, **P**avements. Download it
yourself and check its license and citation terms before use; the data is not included in this repo.

Expected layout: `D/CD, D/UD, W/CW, W/UW, P/CP, P/UP` (C = cracked, U = uncracked).

## Method

- **Leak-free splits.** Each image is a crop of a larger photograph, so crops from one
  photograph look alike. Splits (70/15/15) are made **per photograph**, per surface, using the
  number in the file name (`7001-115.jpg` belongs to photograph `7001`). `check_no_leakage` verifies this and is unit-tested.
- **Models.** ImageNet-pretrained ResNet50 and EfficientNet-B0, fully fine-tuned (AdamW, lr 1e-4, cosine schedule, early stopping on source-validation balanced accuracy).
- **Class imbalance.** Cracked images are a minority, so the loss is class-weighted, and results are reported as **balanced accuracy and F1** (plain accuracy would look good by predicting "uncracked").
- **Protocol.** Train on one surface, test on the held-out test split of all three. The in-domain test is the reference; the other two are cross-domain.
- **Statistics.** 95% bootstrap CIs for each metric; bootstrap CI for the in-domain minus cross-domain gap; exact McNemar test between models on identical test images.
- **Interpretability.** Occlusion-sensitivity maps (a neutral patch slides over the image; the drop in predicted crack probability marks important regions).

## Reproduce

```bash
pip install -r requirements.txt
pytest -q                                   # unit tests (synthetic data, no download needed)
bash scripts/run_all.sh /path/to/SDNET2018  # 2 models x 3 source surfaces, then evaluation + maps
```

Single run: `python -m src.train --data-root /path/to/SDNET2018 --model resnet50 --source D`.
For a quick check without the real data: `python tests/make_synthetic.py data/synthetic` and add
`--no-pretrained --img-size 64 --epochs 2` to the training command (accuracy is meaningless there).

On Kaggle: add SDNET2018 as a dataset, enable a GPU, clone this repo, and run the commands above
(`--data-root /kaggle/input/<dataset-name>`).

Outputs: `results/summary.csv`, `results/gap.csv`, `results/mcnemar.csv`, `results/heatmap_<model>.png`, `results/occlusion/*.png`.

## Results

_Not yet filled in. After running, add here: the train-surface x test-surface table of balanced
accuracy with CIs, the largest cross-surface gaps and whether their CIs exclude zero, the McNemar
outcome, and two or three occlusion maps with a short, honest reading of them._

## Limitations

- One dataset, one split seed per run, and two models: this is a pilot, not a benchmark. Re-run with other `--seed` values (the split is re-drawn per seed) and report the spread.
- SDNET2018 surfaces differ in more than surface type (camera distance, lighting, labelling style), so a gap cannot be attributed to the material alone.
- Crops from the same photograph share a split, but photographs of the same structure can still resemble each other across splits.
- Occlusion maps show what the model is sensitive to, not proof of causal reasoning.
- Images are 256x256 crops with binary labels; this is not crack segmentation or severity assessment.

## Development notes

Written by Md. Saiful Islam. Parts of the code were drafted with AI assistance (Claude) and then
reviewed, run and checked by the author.

## Project layout

```
src/data.py      manifest, photograph-level splits, dataset
src/models.py    model factory
src/train.py     train on one surface, predict on all
src/evaluate.py  metrics, bootstrap CIs, gaps, McNemar, heatmaps
src/explain.py   occlusion-sensitivity maps
tests/           unit tests and a synthetic-data generator
scripts/run_all.sh
```

## Citation

Dorafshan, S., Thomas, R. J., and Maguire, M. (2018). SDNET2018: An annotated image dataset for
non-contact concrete crack detection using deep convolutional neural networks. *Data in Brief*, 21, 1664-1668.
