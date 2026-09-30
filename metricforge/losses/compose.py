# 作者: 晨星
"""compose.py —— 总损失 + PCGrad-lite（多任务梯度冲突消解，ICML? NeurIPS 2020）。

PCGrad（Yu et al. NeurIPS 2020, arXiv:2001.06782）为已知方法，工程集成，
不计入本系统创新声明（见 docs/algorithm_design.md §3.3 非创新声明）。
"""

from __future__ import annotations

import numpy as np


def pcgrad_lite(grads):
    """对多个梯度做 PCGrad 投影去冲突。grads: list[np.ndarray]，形状一致。

    任意两个梯度 g_i, g_j 若 <0（方向冲突），将 g_i 投影掉 g_j 方向分量。
    单梯度或全非负则原样返回。
    """
    gs = [np.asarray(g, dtype=np.float64) for g in grads]
    if len(gs) <= 1:
        return gs[0] if gs else None
    out = gs.copy()
    for i in range(len(gs)):
        for j in range(len(gs)):
            if i == j:
                continue
            gj = gs[j].reshape(-1)
            gi = out[i].reshape(-1)
            dot = float(gi @ gj)
            if dot < 0:
                gn = float(gj @ gj)
                if gn > 1e-12:
                    proj = (dot / gn) * gj
                    out[i] = (np.asarray(out[i]).reshape(-1) - proj).reshape(out[i].shape)
    # 求和
    total = np.zeros_like(out[0])
    for g in out:
        total = total + np.asarray(g)
    return total
