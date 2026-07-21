#!/usr/bin/env bash
# DeepPCB: single / basic_tffn / dual_tffn / dual_tffn+CB, 5 seeds 평균
#   - 4개 방법 모두 R50 pretrained 에서 출발 (공정 비교)
#   - CB: --class_balanced_sampling sqrt_inverse (단일 GPU 전용)
#   - GPU별 병렬 실행 지원: 방법 하나를 GPU 하나에 할당
#
# Usage:
#   방법 하나를 특정 GPU에서 (5 seed):
#     bash run_deeppcb_avg.sh <method> <gpu>
#     예) bash run_deeppcb_avg.sh single 0
#   전체를 4 GPU 병렬로 한 번에:
#     bash run_deeppcb_avg.sh all
#   집계만:
#     bash run_deeppcb_avg.sh summarize
#   method ∈ {single, basic_tffn, dual_tffn, dual_cb}
set -euo pipefail
cd /home/jhj/Deformable-DETR

SEEDS=(0 1 2 3 4)
DATA=/home/jhj/data/DeepPCB
R50=/home/jhj/Deformable-DETR/pretrained/r50_deformable_detr-checkpoint.pth

arch_flags() {
  case "$1" in
    single)     echo "" ;;
    basic_tffn) echo "--use_template" ;;
    dual_tffn)  echo "--use_template --use_tcdf" ;;
    dual_cb)    echo "--use_template --use_tcdf --class_balanced_sampling sqrt_inverse" ;;
    *) echo "UNKNOWN_METHOD" ;;
  esac
}

run_method() {   # method gpu  -> 해당 방법 5 seed 순차
  local method="$1" gpu="$2" arch out
  arch="$(arch_flags "$method")"
  [[ "$arch" == "UNKNOWN_METHOD" ]] && { echo "unknown method: $method"; exit 1; }
  for s in "${SEEDS[@]}"; do
    out="output/deeppcb_avg/${method}_seed${s}"
    if [[ -f "${out}_eval/eval_per_class.csv" ]]; then
      echo "[skip][gpu${gpu}] ${out}_eval already done"; continue
    fi
    echo "==== [TRAIN][gpu${gpu}] ${method} seed ${s} ===="
    CUDA_VISIBLE_DEVICES="$gpu" python main.py \
      --dataset_file deeppcb --deep_pcb_path "$DATA" \
      --pretrained "$R50" --seed "$s" $arch \
      --epochs 50 --lr_drop 40 --batch_size 2 \
      --output_dir "$out"

    echo "==== [EVAL][gpu${gpu}] ${method} seed ${s} ===="
    CUDA_VISIBLE_DEVICES="$gpu" python main.py \
      --dataset_file deeppcb --deep_pcb_path "$DATA" \
      --resume "${out}/checkpoint_best.pth" --eval $arch \
      --output_dir "${out}_eval"
  done
}

summarize() {
python3 - <<'PY'
import csv, glob, statistics as st
def means(p):
    r=list(csv.DictReader(open(p)))
    col=lambda k:[float(x[k]) for x in r]
    a,b,c=col('ap'),col('ap50'),col('ap75')
    return sum(a)/len(a), sum(b)/len(b), sum(c)/len(c)
print("\n=== DeepPCB (mean +/- std over seeds) ===")
print(f"{'method':12s} {'AP':>16s} {'AP50':>16s} {'AP75':>16s}   n")
for m in ['single','basic_tffn','dual_tffn','dual_cb']:
    cs=sorted(glob.glob(f"output/deeppcb_avg/{m}_seed*_eval/eval_per_class.csv"))
    if not cs: continue
    aps,ap50s,ap75s=zip(*[means(c) for c in cs])
    ms=lambda x:f"{st.mean(x):.3f}+/-{(st.pstdev(x) if len(x)>1 else 0):.3f}"
    print(f"{m:12s} {ms(aps):>16s} {ms(ap50s):>16s} {ms(ap75s):>16s}   {len(cs)}")
PY
}

case "${1:-}" in
  summarize) summarize; exit 0 ;;
  all)
    mkdir -p output/deeppcb_avg logs
    run_method single      0 > logs/deeppcb_single.log     2>&1 &
    run_method basic_tffn  1 > logs/deeppcb_basic.log      2>&1 &
    run_method dual_tffn   2 > logs/deeppcb_dual.log       2>&1 &
    run_method dual_cb     3 > logs/deeppcb_dual_cb.log    2>&1 &
    wait
    echo; echo "########## SUMMARY ##########"; summarize ;;
  single|basic_tffn|dual_tffn|dual_cb)
    run_method "$1" "${2:?gpu 번호를 지정하세요: bash run_deeppcb_avg.sh $1 <gpu>}" ;;
  *)
    echo "usage: bash run_deeppcb_avg.sh {all | <method> <gpu> | summarize}"; exit 1 ;;
esac
