#!/usr/bin/env python
"""B4 cost benchmark: #Params / FLOPs / Latency / FPS for SD-DETR variants.

Template 분기가 있는 방법은 test + template 를 함께 forward 하여
backbone 2회 통과 비용까지 포함해 공정하게 측정한다.

사용 예 (Single, repo 루트에서 실행):
  python tools/bench_cost.py --dataset_file custompcb_class4_full \
    --custom_pcb_path /path/to/CustomPCB

Basic TFFN:  ... --use_template
Dual TFFN :  ... --use_template --use_tcdf
"""
import os
import sys
import time
import argparse

# This script lives in tools/; make the repository root importable so that the
# main / models / datasets / util packages resolve when run as tools/bench_cost.py.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from main import get_args_parser as get_main_args_parser
from models import build_model
from datasets import build_dataset
from util.misc import nested_tensor_from_tensor_list


def get_bench_args(argv):
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument('--num_iters', type=int, default=300)
    p.add_argument('--warm_iters', type=int, default=20)
    p.add_argument('--bench_batch', type=int, default=1)
    p.add_argument('--no_flops', action='store_true')
    known, rest = p.parse_known_args(argv)
    return known, rest


@torch.no_grad()
def latency(model, test_nt, tmpl_nt, num_iters, warm):
    ts = []
    for i in range(num_iters):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        if tmpl_nt is not None:
            model(test_nt, tmpl_nt)
        else:
            model(test_nt)
        torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        if i >= warm:
            ts.append(dt)
    return sum(ts) / len(ts)


def main():
    import sys
    bench, rest = get_bench_args(sys.argv[1:])
    main_args = get_main_args_parser().parse_args(rest)
    main_args.device = 'cuda'

    dataset = build_dataset('val', main_args)
    model, _, _ = build_model(main_args)
    model.cuda().eval()

    # ---- params ----
    n_all = sum(p.numel() for p in model.parameters())
    n_tr = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # ---- inputs (real val image sizes) ----
    img = dataset.__getitem__(0)[0].cuda()
    test_list = [img for _ in range(bench.bench_batch)]
    test_nt = nested_tensor_from_tensor_list(test_list)
    use_tmpl = getattr(main_args, 'use_template', False)
    tmpl_nt = nested_tensor_from_tensor_list(test_list) if use_tmpl else None
    C, H, W = img.shape

    # ---- FLOPs (native FlopCounterMode; deformable-attn op는 미집계) ----
    gflops = None
    if not bench.no_flops:
        try:
            from torch.utils.flop_counter import FlopCounterMode
            fcm = FlopCounterMode(display=False)
            # no_grad 금지: ModuleTracker가 backward 훅으로 FLOP을 귀속하므로
            # grad_fn 이 필요하다. forward-only FLOP 만 집계된다.
            with fcm:
                if tmpl_nt is not None:
                    model(test_nt, tmpl_nt)
                else:
                    model(test_nt)
            gflops = fcm.get_total_flops() / 1e9
        except Exception as e:
            print(f"[warn] FLOPs 집계 실패: {e}")

    # ---- latency ----
    lat = latency(model, test_nt, tmpl_nt, bench.num_iters, bench.warm_iters)
    ms = lat / bench.bench_batch * 1000.0
    fps = bench.bench_batch / lat

    print("=" * 60)
    print(f"use_template        : {use_tmpl}")
    print(f"use_dual_tffn_fusion: {getattr(main_args,'use_tcdf',False)}")
    print(f"use_tffn_diff       : {getattr(main_args,'use_tffn_diff',False)}")
    print(f"input (CxHxW)       : {C}x{H}x{W}  batch={bench.bench_batch}")
    print("-" * 60)
    print(f"#Params (all)       : {n_all/1e6:.2f} M")
    print(f"#Params (trainable) : {n_tr/1e6:.2f} M")
    if gflops is not None:
        print(f"FLOPs (per image)   : {gflops/bench.bench_batch:.1f} G   "
              f"(deformable-attn 제외, 상대비교용)")
    print(f"Latency             : {ms:.2f} ms/img")
    print(f"FPS                 : {fps:.1f}")
    print("=" * 60)


if __name__ == '__main__':
    main()
