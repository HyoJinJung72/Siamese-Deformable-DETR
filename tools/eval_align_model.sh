#!/usr/bin/env bash
# Evaluate one trained checkpoint across a template-misalignment sweep (eval only).
#
# Usage:
#   bash tools/eval_align_model.sh <deeppcb|custompcb> <ckpt> <tag> [gpu] [mode]
#     mode = translate (default) | rotate
#
# Override the sweep grid with VALS="...":
#   VALS="0 4 8 16" bash tools/eval_align_model.sh deeppcb <ckpt> mytag 0 translate

set -euo pipefail

DATASET="$1"; CKPT="$2"; TAG="$3"; GPU="${4:-0}"; MODE="${5:-translate}"
OUT_ROOT="${OUT_ROOT:-output/align_phase2}"

case "$DATASET" in
  deeppcb)
    DATA_ARGS="--dataset_file deeppcb --deep_pcb_path /home/jhj/data/DeepPCB"
    if [ "$MODE" = translate ]; then VALS="${VALS:-0 2 4 8 16}"; else VALS="${VALS:-0 1 2 3 5}"; fi
    ;;
  custompcb)
    DATA_ARGS="--dataset_file custompcb_class4_full --custom_pcb_path /home/jhj/data/CustomPCB_lobo/fold_num3"
    if [ "$MODE" = translate ]; then VALS="${VALS:-0 1 2 4}"; else VALS="${VALS:-0 1 2 3 5}"; fi
    ;;
  *) echo "unknown dataset: $DATASET (use deeppcb | custompcb)"; exit 1 ;;
esac

if [ ! -f "$CKPT" ]; then echo "checkpoint not found: $CKPT"; exit 1; fi

echo "=== ${DATASET} ${TAG}  mode=${MODE} ==="
for v in $VALS; do
  out="${OUT_ROOT}/${DATASET}/${TAG}_${MODE}_${v}"
  if [ -f "${out}/eval_per_class.csv" ]; then
    ap=$(grep -oP 'IoU=0\.50 .*=\s*\K[\d.]+' "${out}/eval.log" | tail -1)
    echo "  [skip] ${MODE}=${v}   AP50=${ap}"
    continue
  fi
  mkdir -p "$out"
  if [ "$MODE" = translate ]; then
    P="--eval_template_perturb translate --eval_template_perturb_dx $v --eval_template_perturb_dy $v"
  else
    P="--eval_template_perturb rotate --eval_template_perturb_angle $v"
  fi
  CUDA_VISIBLE_DEVICES="$GPU" python main.py $DATA_ARGS --use_template --use_tcdf \
    --resume "$CKPT" --eval --num_workers 2 $P --output_dir "$out" > "${out}/eval.log" 2>&1
  ap=$(grep -oP 'IoU=0\.50 .*=\s*\K[\d.]+' "${out}/eval.log" | tail -1)
  echo "  ${MODE}=${v}   AP50=${ap}"
done
