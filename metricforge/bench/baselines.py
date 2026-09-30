# 作者: 晨星
"""bench/baselines.py —— 基线与方法工厂（B1..B8 + MetricForge）。"""

from __future__ import annotations

import numpy as np

from ..losses import linear
from .trainer import DeepTrainer

ALL_METHODS = [
    "b1_knn",
    "b2_lmnn",
    "b3_lfda",
    "b4_triplet",
    "b5_proxy_k1",
    "b6_supcon",
    "metricforge",
]


class IdentityModel:
    def fit(self, X, y):
        return self

    def transform(self, X):
        return np.asarray(X, dtype=np.float64)


class LinearWrapper:
    def __init__(self, learner):
        self.learner = learner

    def fit(self, X, y):
        self.learner.fit(X, y)
        return self

    def transform(self, X):
        return self.learner.transform(X)


def build_model(
    method: str,
    *,
    dim_in: int,
    dim_out: int,
    seed: int,
    K: int = 1,
    use_hat: bool = True,
    use_bidirectional: bool = True,
    use_queue: bool = False,
    use_geometry: bool = False,
    epochs: int = 60,
    lr: float = 1e-3,
    batch_size: int = 128,
    kappa: float = 0.25,
):
    dim_hidden = int(min(64, max(8, dim_in * 2)))
    if method == "b1_knn":
        return IdentityModel()
    if method in ("metricforge", "b5_proxy_k1", "b4_triplet", "b6_supcon"):
        return DeepTrainer(
            method,
            dim_in=dim_in,
            dim_hidden=dim_hidden,
            dim_out=dim_out,
            seed=seed,
            lr=lr,
            epochs=epochs,
            batch_size=batch_size,
            kappa=kappa,
            K=K,
            use_hat=use_hat,
            use_bidirectional=use_bidirectional,
            use_queue=use_queue,
            use_geometry=use_geometry,
        )
    # 线性方法（LFDA/LMNN/NCA/ITML）为降维投影，要求 dim_out ≤ 输入维。
    # 超出则裁剪到 dim_in（不会增加维度——这是线性度量学习的固有边界）。
    D = int(min(dim_out, dim_in))
    if method == "b2_lmnn":
        return LinearWrapper(linear.LMNN(dim_out=D, iters=60, lr=1e-3))
    if method == "b3_lfda":
        return LinearWrapper(linear.LFDA(dim_out=D))
    if method == "b7_nca":
        return LinearWrapper(linear.NCA(dim_out=D, iters=50, lr=1e-2))
    raise ValueError(method)
