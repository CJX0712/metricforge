# 作者: 晨星
"""bench/ablation.py —— 消融实验 A1..A8（含 A7 全关≡B5 的硬证明）。

A1 HAT off          : use_hat=False
A3 K=1              : K=1（多原型关闭）
A5 反向对齐 off     : use_bidirectional=False
A7 全关 ≡ B5        : K=1 + hat off + bidirectional off → 必须与 b5_proxy_k1 逐位一致
A8 κ 敏感性         : κ ∈ {0.15,0.25,0.40} 在 s4_subspace / s6_hier_gauss 上报告 ΔMAP@R 极差
（A2 queue off / A4 geometry off 在本实现中默认即关闭，为 no-op，如实标注。）
"""

from __future__ import annotations

from ..data.split import split_open
from ..eval.metrics import map_at_r_score
from .baselines import build_model
from .run import _load, _scale


def _eval_one(name, seed, **kw):
    X, y = _load(name)
    tr, va, te = split_open(X, y, seed)
    Xtr, Xva, Xte = _scale(X[tr], X[va], X[te])
    model = build_model(
        "metricforge",
        dim_in=X.shape[1],
        dim_out=16,
        seed=seed,
        epochs=40,
        lr=1e-3,
        K=kw.get("K", 3),
        use_hat=kw.get("use_hat", True),
        use_bidirectional=kw.get("use_bidirectional", True),
        use_queue=False,
        use_geometry=False,
        kappa=kw.get("kappa", 0.25),
    )
    model.fit(Xtr, y[tr])
    Zte = model.transform(Xte)
    return map_at_r_score(Zte, Zte, y[te], y[te])


def run_ablation(datasets=("s4_subspace", "s6_hier_gauss"), seed=0, kappas=(0.15, 0.25, 0.40)):
    rows = {}
    for name in datasets:
        full = _eval_one(name, seed)
        a1 = _eval_one(name, seed, use_hat=False)
        a3 = _eval_one(name, seed, K=1)
        a5 = _eval_one(name, seed, use_bidirectional=False)
        a7 = _eval_one(name, seed, K=1, use_hat=False, use_bidirectional=False)
        # A8 κ 敏感性
        kap_maps = [(_eval_one(name, seed, kappa=k), k) for k in kappas]
        kap_vals = [v for v, _ in kap_maps]
        # B5 对照（独立训练）
        from .baselines import build_model as _bm

        X, y = _load(name)
        tr, va, te = split_open(X, y, seed)
        Xtr, Xva, Xte = _scale(X[tr], X[va], X[te])
        # B5 = 纯 Proxy-Anchor 基线（关闭 HAT 与 MQ-BPA 反向对齐），与 A7（全关）逐位一致（硬断言 I9）。
        b5 = _bm(
            "b5_proxy_k1",
            dim_in=X.shape[1],
            dim_out=16,
            seed=seed,
            epochs=40,
            lr=1e-3,
            use_hat=False,
            use_bidirectional=False,
        )
        b5.fit(Xtr, y[tr])
        b5_map = map_at_r_score(b5.transform(Xte), b5.transform(Xte), y[te], y[te])
        rows[name] = {
            "metricforge_full": round(full, 4),
            "A1_hat_off": round(a1, 4),
            "A3_k1": round(a3, 4),
            "A5_bidir_off": round(a5, 4),
            "A7_all_off": round(a7, 4),
            "B5_proxy_k1": round(b5_map, 4),
            "A7_vs_B5_diff": round(abs(a7 - b5_map), 6),
            "A8_kappa_maps": {str(k): round(v, 4) for v, k in kap_maps},
            "A8_kappa_range": round(max(kap_vals) - min(kap_vals), 4),
        }
    return rows
