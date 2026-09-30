# 作者: 晨星
"""成对/三元组损失（B4 Triplet-Batch-Hard，及 contrastive/lifted 参考实现）。"""

from __future__ import annotations

import numpy as np


def _pairwise_sq_dists(Z: np.ndarray) -> np.ndarray:
    # ||z_i - z_j||^2 = ||z_i||^2 + ||z_j||^2 - 2 z_i·z_j
    sq = np.sum(Z * Z, axis=1)
    d2 = sq[:, None] + sq[None, :] - 2.0 * (Z @ Z.T)
    return np.maximum(d2, 0.0)


class TripletBatchHard:
    """Batch-Hard 三元组损失（Hermans et al. 2017）。

    每个 anchor i 选： hardest positive（同类最远）、hardest negative（异类最近）。
    loss_i = max(0, d_ap - d_an + margin)。
    """

    def __init__(self, margin: float = 0.2):
        self.margin = float(margin)

    def __call__(self, Z, y):
        Z = np.asarray(Z, dtype=np.float64)
        y = np.asarray(y)
        B = Z.shape[0]
        d2 = _pairwise_sq_dists(Z)
        d = np.sqrt(d2 + 1e-12)
        d[np.diag_indices(B)] = np.inf  # 排除自身
        dZ = np.zeros_like(Z)
        total = 0.0
        for i in range(B):
            same = y == y[i]
            diff = y != y[i]
            if not same.sum() or not diff.sum():
                continue
            # hardest positive: 同类最远
            pj = np.where(same)[0]
            ap = pj[np.argmax(d[i, pj])]
            # hardest negative: 异类最近
            nj = np.where(diff)[0]
            an = nj[np.argmin(d[i, nj])]
            d_ap = d[i, ap]
            d_an = d[i, an]
            loss_i = max(0.0, d_ap - d_an + self.margin)
            total += loss_i
            if loss_i > 0:
                # ∂L/∂Z_i = 2( (Z_i-Z_ap)/d_ap - (Z_i-Z_an)/d_an )
                g = 2.0 * ((Z[i] - Z[ap]) / max(d_ap, 1e-12) - (Z[i] - Z[an]) / max(d_an, 1e-12))
                dZ[i] += g
                dZ[ap] += -2.0 * (Z[i] - Z[ap]) / max(d_ap, 1e-12)
                dZ[an] += 2.0 * (Z[i] - Z[an]) / max(d_an, 1e-12)
        dZ /= B
        return total, dZ
