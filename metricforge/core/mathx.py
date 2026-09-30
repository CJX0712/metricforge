# 作者: 晨星
"""数学内核。

导出符号（被 tests/test_invariants_core.py 锁定）：
  logsumexp / softmax_entropy / softmax_entropy_beta / hat_temperature /
  map_at_r / map_at_r_exact / triplet_hinge / infonce_loss

其中 HAT 的二分迭代次数必须是固定常数（不变量 I10，禁止 while 收敛判据）。
所有函数支持 list / float64 ndarray 输入；下游可向量化复用。
"""

from __future__ import annotations

import math
from fractions import Fraction

import numpy as np


def logsumexp(values) -> float:
    """数值稳定的 log-sum-exp；全 -inf 时返回 -inf（不是 NaN）。"""
    arr = np.asarray(values, dtype=np.float64)
    m = float(np.max(arr))
    if not np.isfinite(m):
        return m
    return float(m + np.log(float(np.sum(np.exp(arr - m)))))


def softmax_entropy(logits) -> float:
    """p = softmax(logits) 的香农熵（单位 nat）。H = logsumexp(l) - sum(p*l)。"""
    arr = np.asarray(logits, dtype=np.float64)
    z = logsumexp(arr)
    p = np.exp(arr - z)
    return float(z - float(np.sum(p * arr)))


def softmax_entropy_beta(sims, beta) -> float:
    """以 beta = 1/tau 为逆温度时，softmax(sims / tau) 的熵。"""
    return softmax_entropy(np.asarray(sims, dtype=np.float64) * float(beta))


def hat_temperature(
    neg_sims, kappa: float = 0.25, tau_lo: float = 0.02, tau_hi: float = 2.0, iters: int = 40
) -> float:
    """I4：解 PP(tau) = exp(H(p_tau)) == kappa*N，返回 tau*。

    迭代次数 iters 为固定常数。H 关于 beta=1/tau 严格递减（引理 dH/dbeta<0），
    故在 beta 空间二分；tau 最终落在 [tau_lo, tau_hi]。
    """
    sims = np.asarray(neg_sims, dtype=np.float64)
    n = sims.shape[0]
    target = math.log(max(kappa * n, 1.0))
    lo_b = 1.0 / tau_hi
    hi_b = 1.0 / tau_lo
    for _ in range(iters):
        mid = 0.5 * (lo_b + hi_b)
        h = softmax_entropy_beta(sims, mid)
        if h < target:
            hi_b = mid
        else:
            lo_b = mid
    return 1.0 / (0.5 * (lo_b + hi_b))


def map_at_r(hits, r: int):
    """I8：返回 (MAP@R, R-Precision)，取值 [0,1]。

    口径（文档 §4.3）：r = gallery 中与该 query 同类的样本总数（固定搜索长度），
    **不是**检索列表中的命中数。hits 为长度恰为 r 的 0/1 向量，按检索排名排序。
    """
    h = np.asarray(hits).ravel().astype(int).tolist() if hasattr(hits, "__array__") else list(hits)
    if len(h) != r or r < 1:
        raise ValueError(f"hits 长度必须等于 R 且 R>=1，got len={len(h)}, R={r}")
    cum = 0
    acc = 0.0
    for i, hit in enumerate(h, start=1):
        if hit:
            cum += 1
            acc += cum / i
    return acc / r, cum / r


def map_at_r_exact(hits, r: int):
    """与 map_at_r 同义，但用有理数精确计算（供 Musgrave Table 3 逐位比对）。"""
    h = list(hits)
    if len(h) != r or r < 1:
        raise ValueError(f"hits 长度必须等于 R 且 R>=1，got len={len(h)}, R={r}")
    cum = 0
    acc = Fraction(0)
    for i, hit in enumerate(h, start=1):
        if hit:
            cum += 1
            acc += Fraction(cum, i)
    return acc / r, Fraction(cum, r)


def triplet_hinge(d_ap, d_an, margin: float) -> float:
    """I3：单条 triplet hinge 损失 max(0, d_ap - d_an + margin)。"""
    return max(0.0, float(d_ap) - float(d_an) + float(margin))


def infonce_loss(sim_pos, sim_negs, tau: float) -> float:
    """I6：单正样本 InfoNCE，稳定形式 log(1 + sum_n exp((s_n - s_p)/tau))。"""
    sp = float(sim_pos)
    vals = [0.0] + [(float(sn) - sp) / float(tau) for sn in sim_negs]
    return logsumexp(vals)
