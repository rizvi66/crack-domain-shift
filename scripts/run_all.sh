#!/usr/bin/env bash
# Train every model on every source surface, then evaluate.
# Usage: bash scripts/run_all.sh /path/to/SDNET2018 [seed]
set -euo pipefail
ROOT="${1:?usage: run_all.sh /path/to/SDNET2018 [seed]}"
SEED="${2:-42}"
for MODEL in resnet50 efficientnet_b0; do
  for SRC in D W P; do
    python -m src.train --data-root "$ROOT" --model "$MODEL" --source "$SRC" --seed "$SEED"
  done
done
python -m src.evaluate
# Occlusion maps: best-case transfer examples from deck-trained ResNet50 on pavement images
python -m src.explain --preds "results/preds/resnet50_srcD_seed${SEED}.csv" \
  --ckpt "checkpoints/resnet50_srcD_seed${SEED}.pt" --target P --k 4
