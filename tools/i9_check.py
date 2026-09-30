# 作者: 晨星
"""I9 验证：A7（全关）必须与 B5（纯 Proxy-Anchor）逐位一致。"""

import sys
import time

ROOT = "C:/Users/Administrator/WorkBuddy/2026-09-30-23-11-18/metricforge"
sys.path.insert(0, ROOT)

from metricforge.bench.ablation import _eval_one
from metricforge.bench.baselines import build_model
from metricforge.bench.run import _load, _scale
from metricforge.data.split import split_open
from metricforge.eval.metrics import map_at_r_score

name = "s6_hier_gauss"
seed = 0
X, y = _load(name)
tr, va, te = split_open(X, y, seed)
Xtr, Xva, Xte = _scale(X[tr], X[va], X[te])
ytr, yte = y[tr], y[te]

t0 = time.perf_counter()
a7 = _eval_one(name, seed, K=1, use_hat=False, use_bidirectional=False)
t_a7 = time.perf_counter() - t0

t0 = time.perf_counter()
b5 = build_model(
    "b5_proxy_k1",
    dim_in=X.shape[1],
    dim_out=16,
    seed=seed,
    epochs=40,
    lr=1e-3,
    use_hat=False,
    use_bidirectional=False,
)
b5.fit(Xtr, ytr)
b5m = map_at_r_score(b5.transform(Xte), b5.transform(Xte), yte, yte)
t_b5 = time.perf_counter() - t0

diff = abs(a7 - b5m)
print(f"A7 (full-off) MAP@R = {a7:.10f}  (train {t_a7:.1f}s)")
print(f"B5 (vanilla proxy) MAP@R = {b5m:.10f}  (train {t_b5:.1f}s)")
print(f"A7 vs B5 diff = {diff:.2e}")
print("I9 PASS" if diff == 0.0 else f"I9 FAIL (diff={diff})")
