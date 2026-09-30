# 作者: 晨星
"""对比/监督对比损失（B6 SupCon，及 InfoNCE 用于 Fresh-Queue）。"""

from __future__ import annotations

import numpy as np

from ..core import mathx


class SupConLoss:
    """监督对比损失（SupCon, Khosla et al. 2020 的同类正对版本）。

    无多视图增强时，正样本 = 同 batch 同类其他样本，负样本 = 异类样本。
    梯度解析推导见 module doc，逐 anchor 计算（B 较小，O(B^2) 可控）。
    """

    def __init__(self, tau: float = 0.07):
        self.tau = float(tau)

    def __call__(self, Z, y):
        Z = np.asarray(Z, dtype=np.float64)
        y = np.asarray(y)
        B = Z.shape[0]
        S = Z @ Z.T  # (B,B)，已归一化 → 余弦相似度
        dZ = np.zeros_like(Z)
        total = 0.0
        for i in range(B):
            pos = np.where((y == y[i]) & (np.arange(B) != i))[0]
            if pos.size == 0:
                continue
            logits = S[i] / self.tau
            # 分母：j≠i 的 softmax
            logits_i = logits.copy()
            logits_i[i] = -np.inf
            lse = mathx.logsumexp(logits_i)
            # 各 j≠i 的 softmax 概率 Q
            Q = np.exp(logits_i - lse)
            Q[i] = 0.0
            # 分子项：正样本 Σ Z_p
            sum_pos = Z[pos].sum(axis=0)
            # 分母期望 Σ_j Q_j Z_j（j≠i）
            sum_qz = Q @ Z
            dZi = -(1.0 / (self.tau * pos.size)) * (sum_pos - pos.size * sum_qz)
            dZ[i] += dZi
            # k≠i 的梯度
            dZ += -(1.0 / self.tau) * (Q[:, None] * Z[i])  # 对每个 i 累积
            # 标量损失
            sc_i = (S[i, pos] / self.tau).sum() / pos.size - lse
            total += -sc_i
        dZ /= B
        return total, dZ


class InfoNCELoss:
    """InfoNCE（单正 + 多负），用于 Fresh-Queue 增强对比。"""

    def __init__(self, tau: float = 0.07):
        self.tau = float(tau)

    def __call__(self, z_pos, z_anchor, z_negs):
        """z_pos: (D,) 正样本；z_anchor: (D,)；z_negs: (M, D)。返回 (loss, d_anchor)。"""
        sp = float(z_anchor @ z_pos) / self.tau
        sn = (z_negs @ z_anchor) / self.tau
        lse = mathx.logsumexp([sp] + list(sn))
        loss = lse - sp
        return loss, None  # 梯度由调用方按需实现
