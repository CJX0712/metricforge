# 作者: 晨星
"""examples/run_demo.py —— 端到端演示，落盘 benchmark.json 并打印摘要。

结论先行：本脚本一键复现全部强基线对照 + 消融 + 失败案例，输出 benchmark.json。
确定性：固定 seed，同 seed 两次运行核心指标逐位一致（不变量 I10）。
"""

from __future__ import annotations

import json
import sys
import time

sys.path.insert(0, ".")

from metricforge.bench import aggregate, run_benchmark
from metricforge.bench.ablation import run_ablation


def _print(out: dict):
    agg = out["aggregate"]
    print("\n=== MetricForge Benchmark（主指标 MAP@R mean±std）===")
    print(f"{'dataset':<16}{'MetricForge':>16}{'Δ_vs_base':>12}")
    for name, nd in agg.items():
        mf = nd.get("metricforge", {})
        d = nd.get("delta_vs_best_base", 0.0)
        flag = "✅" if d >= 0.03 else ("⚠️" if d > 0 else "❌")
        print(
            f"{name:<16}{mf.get('map_mean', 0):>8.4f}±{mf.get('map_std', 0):.4f}{d:>+12.4f} {flag}"
        )
    if "ablation" in out:
        print("\n=== 消融（A 组）===")
        for name, row in out["ablation"].items():
            print(
                f"{name}: full={row['metricforge_full']}  A1={row['A1_hat_off']}  "
                f"A3={row['A3_k1']}  A5={row['A5_bidir_off']}  A7={row['A7_all_off']}  "
                f"B5={row['B5_proxy_k1']}  A7≡B5_diff={row['A7_vs_B5_diff']}"
            )
            print(f"    A8 κ-sens maps={row['A8_kappa_maps']}  range={row['A8_kappa_range']}")
    print(f"\n总耗时 {out.get('elapsed_s', 0):.1f}s")


def main():
    t0 = time.time()
    print("MetricForge 端到端演示启动（fast 档，CPU-only）...", flush=True)
    res = run_benchmark(seeds=(0, 1, 2), fast=True, dim_out=16, epochs=80, lr=3e-4)
    agg = aggregate(res)
    abl = run_ablation()
    out = {
        "benchmark": res,
        "aggregate": agg,
        "ablation": abl,
        "elapsed_s": round(time.time() - t0, 1),
    }
    with open("benchmark.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    _print(out)
    return out


if __name__ == "__main__":
    main()
