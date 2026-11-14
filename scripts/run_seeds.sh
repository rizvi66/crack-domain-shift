#!/usr/bin/env bash
# Repeat the full experiment for several seeds (each seed also re-draws the photograph split).
# Resumable: a run whose predictions file already exists is skipped.
# Usage: bash scripts/run_seeds.sh /path/to/SDNET2018 "1 2 3"
#
# Rough GPU time on a Kaggle T4 (ResNet50 epoch on decks took ~350 s): about 4-5 hours per seed
# for both models x three surfaces. To save time, split seeds across notebook sessions of the same
# account (one seed per session) and merge the results/preds folders afterwards.
set -euo pipefail
ROOT="${1:?usage: run_seeds.sh /path/to/SDNET2018 \"1 2 3\"}"
SEEDS="${2:-1 2 3}"
for SEED in $SEEDS; do
  for MODEL in resnet50 efficientnet_b0; do
    for SRC in D W P; do
      if [ -f "results/preds/${MODEL}_src${SRC}_seed${SEED}.csv" ]; then
        echo "skip ${MODEL} ${SRC} seed ${SEED} (already done)"; continue
      fi
      python -m src.train --data-root "$ROOT" --model "$MODEL" --source "$SRC" --seed "$SEED"
    done
  done
done
python -m src.evaluate
python -m src.evaluate_cluster
