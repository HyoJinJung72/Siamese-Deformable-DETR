#!/usr/bin/env bash
# Phase 1 - alignment robustness sweep (evaluation only, no training).
#
# Shifts ONLY the template image by delta pixels at eval time; the test image and
# its annotations are untouched, so COCO evaluation stays valid and a reference-free
# model is exactly invariant by construction (single = flat control).
#
# Usage:
#   bash tools/run_phase1_alignment.sh deeppcb   [gpu]
#   bash tools/run_phase1_alignment.sh custompcb [gpu]

set -euo pipefail

DATASET="${1:-deeppcb}"
GPU="${2:-0}"
OUT_ROOT="${OUT_ROOT:-output/align_phase1}"

# Delta is in ORIGINAL-image pixels, so the grid is scaled per dataset to cover a
# comparable fraction of the board width:
#   DeepPCB   640x640 -> 0 / 0.31 / 0.63 / 1.25 / 2.5 % of width
#   CustomPCB 160x153 -> 0 / 0.63 / 1.25 / 2.5  % of width
case "$DATASET" in
  deeppcb)
    DELTAS="${DELTAS:-0 2 4 8 16}"
    DATA_ARGS="--dataset_file deeppcb --deep_pcb_path /home/jhj/data/DeepPCB"
    CKPT_SINGLE=output/deeppcb_avg/single_seed0/checkpoint_best.pth
    CKPT_BASIC=output/deeppcb_avg/basic_tffn_seed0/checkpoint_best.pth
    CKPT_TCDF=output/deeppcb_avg/dual_tffn_seed0/checkpoint_best.pth
    ;;
  custompcb)
    DELTAS="${DELTAS:-0 1 2 4}"
    DATA_ARGS="--dataset_file custompcb_class4_full --custom_pcb_path /home/jhj/data/CustomPCB_lobo/fold_num3"
    CKPT_SINGLE=output/lobo/lobo_num3_single_seed0/checkpoint_best.pth
    CKPT_BASIC=output/lobo/lobo_num3_basic_seed0/checkpoint_best.pth
    CKPT_TCDF=output/lobo/lobo_num3_dual_seed0/checkpoint_best.pth
    ;;
  *) echo "unknown dataset: $DATASET (use deeppcb | custompcb)"; exit 1 ;;
esac

run_one () {  # $1=method  $2=ckpt  $3=model_flags  $4=delta
  local out="${OUT_ROOT}/${DATASET}/${1}_d${4}"
  if [ -f "${out}/eval_per_class.csv" ]; then
    echo "[skip] ${DATASET}/${1} delta=${4}"
    return
  fi
  echo "[run ] ${DATASET}/${1} delta=${4}"
  mkdir -p "$out"
  CUDA_VISIBLE_DEVICES="$GPU" python main.py \
    $DATA_ARGS $3 \
    --resume "$2" --eval --num_workers 2 \
    --eval_template_perturb translate \
    --eval_template_perturb_dx "$4" --eval_template_perturb_dy "$4" \
    --output_dir "$out" > "${out}/eval.log" 2>&1
}

for d in $DELTAS; do
  run_one single "$CKPT_SINGLE" ""                          "$d"
  run_one basic  "$CKPT_BASIC"  "--use_template"            "$d"
  run_one tcdf   "$CKPT_TCDF"   "--use_template --use_tcdf" "$d"
done

echo
python tools/collect_phase1_alignment.py "${OUT_ROOT}/${DATASET}"
