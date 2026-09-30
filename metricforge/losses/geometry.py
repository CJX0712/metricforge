# 作者: 晨星
"""geometry.py —— SNE-KL 局部邻域保持正则（A4 可选开关）。

强制嵌入 Z 保持原始空间 X 的 k-NN 邻域结构，缓解小样本过拟合。
p_ij：基于 X 空间距离的高斯亲和（固定）；q_ij：Z 空间的 SNE 条件分布。
正则 = λ · Σ_i KL(p_i || q_i)。提供解析梯度。
"""

from __future__ import annotations

import numpy as np


class SNEKeepRegularizer:
    def __init__(self, k: int = 7, sigma: float = 1.0, weight: float = 0.1):
        self.k = int(k)
        self.sigma = float(sigma)
        self.weight = float(weight)
        self.neighbors = None  # (N, k) 索引
        self.P = None  # (N, k) 条件概率

    def fit(self, X: np.ndarray):
        X = np.asarray(X, dtype=np.float64)
        N = X.shape[0]
        D = np.sum((X[:, None, :] - X[None, :, :]) ** 2, axis=2)
        np.fill_diagonal(D, np.inf)
        knn = np.argsort(D, axis=1)[:, : self.k]
        P = np.zeros((N, self.k), dtype=np.float64)
        for i in range(N):
            di = D[i, knn[i]]
            w = np.exp(-di / (2 * self.sigma**2))
            P[i] = w / (w.sum() + 1e-12)
        self.neighbors = knn
        self.P = P
        return self

    def __call__(self, Z: np.ndarray):
        Z = np.asarray(Z, dtype=np.float64)
        N = Z.shape[0]
        dZ = np.zeros_like(Z)
        total = 0.0
        for i in range(N):
            nb = self.neighbors[i]
            diff = Z[i] - Z[nb]  # (k, D)
            d2 = np.sum(diff * diff, axis=1)
            q = np.exp(-d2 / (2 * self.sigma**2))
            q = q / (q.sum() + 1e-12)
            p = self.P[i]
            # KL(p||q) 对 Z 的梯度
            # dKL/dZ_i = -Σ_j p_j ∂/∂Z_i [log q_j] ; q_j ∝ exp(-||Z_i - Z_nbj||^2/2σ²)
            # ∂log q_j/∂Z_i = (Z_i - Z_nbj)/σ²  (约)
            grad_i = np.zeros_like(Z[i])
            for j, nbj in enumerate(nb):
                g_q = (Z[i] - Z[nbj]) / (self.sigma**2)
                # ∂KL/∂Z_i = Σ_j ( -p_j/ q_j ) ∂log q_j/∂Z_i ; 加 -p_j * ∂/∂Z_i[ -log q_j ]
                grad_i += -p[j] * (-g_q)  # = p_j * g_q
            dZ[i] += grad_i
            total += float(np.sum(p * (np.log(p + 1e-12) - np.log(q + 1e-12))))
        return self.weight * total, self.weight * dZ


# 别名（文档中亦称 SNE-KL）
SNEKL = SNEKeepRegularizer
