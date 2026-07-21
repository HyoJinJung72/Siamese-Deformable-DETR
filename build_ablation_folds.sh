#!/usr/bin/env bash
# Build leave-one-board-out (LOBO) folds for the test-board ablation study.
#
# Source: /home/jhj/data/CustomPCB_full_class4  (num7 is its held-out test board)
# For each board below, re-split so that board becomes val/test and all other
# real boards (num9 stays in train since it is normal-only) go to train.
# num7 is intentionally skipped: CustomPCB_full_class4 already is the num7 fold.
#
# Output: /home/jhj/data/CustomPCB_lobo/fold_num{X}
#   images/{train,val}/*.jpg      (board images)
#   images/{train,val}/gerber_*.png (templates, both splits)
#   annotations/custom_{train,val}_class4.json
# Point --custom_pcb_path at each fold dir directly.
set -euo pipefail

cd /home/jhj/Deformable-DETR

SOURCE=/home/jhj/data/CustomPCB_full_class4
OUT_ROOT=/home/jhj/data/CustomPCB_lobo
BOARDS=(num1 num2 num3 num4 num5 num6 num8)

for board in "${BOARDS[@]}"; do
  out="${OUT_ROOT}/fold_${board}"
  echo "==================== building fold: ${board} -> ${out} ===================="
  python build_lobo_fold.py \
    --source "${SOURCE}" \
    --test-board "${board}" \
    --out "${out}"
done

echo "All ablation folds built under ${OUT_ROOT}"
