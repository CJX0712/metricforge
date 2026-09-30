# 作者: 晨星
"""cli.py —— 命令行入口。

用法：
  python -m metricforge.cli --fast --seeds 0 1 2 --out benchmark.json
  python -m metricforge.cli --datasets s1_gauss_blobs s6_hier_gauss --protocol open
"""

from __future__ import annotations

import argparse
import json
import sys
import time

from .bench import aggregate, run_benchmark
from .bench.ablation import run_ablation


def _print_summary(out: dict):
    agg = out["aggregate"]
    print("\n=== MetricForge Benchmark（主指标 MAP@R）===")
    print(f"{'dataset':<16}{'MetricForge':>14}{'best_base':>12}{'Δ':>10}")
    for name, nd in agg.items():
        mf = nd.get("metricforge", {}).get("map_mean", 0.0)
        best = 0.0
        for b, v in nd.items():
            if b == "metricforge" or b == "delta_vs_best_base":
                continue
            best = max(best, v["map_mean"])
        d = nd.get("delta_vs_best_base", 0.0)
        flag = "✅" if d >= 0.03 else ("⚠️" if d > 0 else "❌")
        print(f"{name:<16}{mf:>14.4f}{best:>12.4f}{d:>+10.4f} {flag}")
    print(f"\n总耗时 {out.get('elapsed_s', 0):.1f}s")


def main(argv=None):
    p = argparse.ArgumentParser(prog="metricforge", description="MetricForge 度量学习系统")
    p.add_argument("--fast", action="store_true", help="快速档：减 epoch / 子空间降采样")
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    p.add_argument("--datasets", nargs="+", default=None)
    p.add_argument("--protocol", choices=["open", "closed"], default="open")
    p.add_argument("--dim-out", type=int, default=12)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--kappa", type=float, default=0.25)
    p.add_argument("--out", default="benchmark.json")
    p.add_argument("--no-ablation", action="store_true")
    args = p.parse_args(argv)

    t0 = time.time()
    res = run_benchmark(
        seeds=tuple(args.seeds),
        datasets=args.datasets,
        protocol=args.protocol,
        fast=args.fast,
        dim_out=args.dim_out,
        epochs=args.epochs,
        lr=args.lr,
        kappa=args.kappa,
    )
    agg = aggregate(res)
    out = {"benchmark": res, "aggregate": agg, "elapsed_s": round(time.time() - t0, 1)}
    if not args.no_ablation:
        out["ablation"] = run_ablation()
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    _print_summary(out)
    return out


if __name__ == "__main__":
    main(sys.argv[1:])
