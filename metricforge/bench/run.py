# 作者: 晨星
"""bench/run.py —— 跨方法 × 数据集 × seed 的 benchmark 编排。"""

from __future__ import annotations

import time

import numpy as np

from ..data import real, synth
from ..data.split import split_closed, split_open
from ..eval.metrics import knn_accuracy, map_at_r_score, recall_at_k
from .baselines import ALL_METHODS, build_model

_DATA_SEED = {
    "s1_gauss_blobs": 11,
    "s2_concentric": 22,
    "s3_spirals": 33,
    "s4_subspace": 44,
    "s5_xor_grid": 55,
    "s6_hier_gauss": 66,
    "r1_digits": 0,
    "r2_wine": 0,
    "r3_breast_cancer": 0,
    "r4_iris": 0,
}

ALL_DATASETS = list(_DATA_SEED.keys())


def _load(name, fast=False):
    if name in synth.SYNTH:
        if fast and name == "s4_subspace":
            # fast 档缩减样本量：高维多类子空间仍有效，但把 LMNN 的 O(n²) push 循环
            # 从 2000 点压到 1000 点（约 4× 加速），使 fast 档在数分钟内完成。
            return synth.subspace(_DATA_SEED[name], n_per_class=20)
        return synth.SYNTH[name](_DATA_SEED[name])
    return real.REAL[name]()


def _scale(Xtr, Xva, Xte):
    try:
        from sklearn.preprocessing import StandardScaler

        sc = StandardScaler().fit(Xtr)
        return sc.transform(Xtr), sc.transform(Xva), sc.transform(Xte)
    except Exception:
        mu = Xtr.mean(0, keepdims=True)
        sd = Xtr.std(0, keepdims=True) + 1e-8
        return (Xtr - mu) / sd, (Xva - mu) / sd, (Xte - mu) / sd


def _rss_mb():
    try:
        import psutil

        return psutil.Process().memory_info().rss / 1e6
    except Exception:
        try:
            import tracemalloc

            if tracemalloc.is_tracing():
                _, peak = tracemalloc.get_traced_memory()
                return peak / 1e6
        except Exception:
            return -1.0
    return -1.0


def run_benchmark(
    seeds=(0, 1, 2),
    datasets=None,
    protocol="open",
    fast=False,
    dim_out=16,
    epochs=60,
    # lr=3e-4：实测 lr=1e-3 在 s5_xor 等小数据集上存在晚期训练失稳
    # （MAP@R e40=0.983 → e80=0.50 崩塌，损失仍单调下降）；3e-4 收敛稳定不崩。
    lr=3e-4,
    kappa=0.25,
    methods=None,
):
    datasets = list(datasets or ALL_DATASETS)
    methods = list(methods or ALL_METHODS)
    out = {
        "protocol": protocol,
        "seeds": list(seeds),
        "dim_out": dim_out,
        "epochs": epochs,
        "results": {},
    }
    for name in datasets:
        X, y = _load(name, fast=fast)
        n_cls = len(np.unique(y))
        out["results"][name] = {
            "n_samples": int(X.shape[0]),
            "n_features": int(X.shape[1]),
            "n_classes": int(n_cls),
            "methods": {},
        }
        for seed in seeds:
            if protocol == "open":
                tr, va, te = split_open(X, y, seed)
            else:
                tr, va, te = split_closed(X, y, seed)
            Xtr, Xva, Xte = _scale(X[tr], X[va], X[te])
            ytr, yte = y[tr], y[te]
            for m in methods:
                # MetricForge 用 K=3 多原型；其余方法 K=1
                K = 3 if m == "metricforge" else 1
                model = build_model(
                    m,
                    dim_in=X.shape[1],
                    dim_out=dim_out,
                    seed=seed,
                    K=K,
                    use_hat=(m == "metricforge"),
                    use_bidirectional=(m == "metricforge"),
                    use_queue=False,
                    use_geometry=False,
                    epochs=epochs,
                    lr=lr,
                    kappa=kappa,
                )
                t0 = time.perf_counter()
                model.fit(Xtr, ytr)
                dt = time.perf_counter() - t0
                Zte = model.transform(Xte)
                mapr = map_at_r_score(Zte, Zte, yte, yte)
                r1 = recall_at_k(Zte, Zte, yte, yte, 1)
                knn = knn_accuracy(Zte, Zte, yte, yte, 1)
                rec = {
                    "map_at_r": float(mapr),
                    "recall_at_1": float(r1),
                    "knn_acc": float(knn),
                    "time_s": round(dt, 3),
                    "rss_mb": round(_rss_mb(), 1),
                }
                out["results"][name]["methods"].setdefault(m, {})[f"seed{seed}"] = rec
    return out


def aggregate(results):
    """跨 seed 聚合 mean±std，附强基线最大值与 MetricForge 差值。"""
    agg = {}
    for name, nd in results["results"].items():
        agg[name] = {}
        method_names = sorted({m for m in nd["methods"]})
        for m in method_names:
            seeds_recs = list(nd["methods"][m].values())
            maps = [r["map_at_r"] for r in seeds_recs]
            agg[name][m] = {
                "map_mean": float(np.mean(maps)),
                "map_std": float(np.std(maps)),
            }
        # 强基线中 MetricForge 之外的最大 mean
        base = [agg[name][b]["map_mean"] for b in method_names if b != "metricforge"]
        best_base = max(base) if base else 0.0
        if "metricforge" in agg[name]:
            mf = agg[name]["metricforge"]["map_mean"]
            agg[name]["delta_vs_best_base"] = float(mf - best_base)
    return agg
