# 作者: 晨星
"""bench/trainer.py —— 深度度量学习训练器（MLP 编码器 + 可插拔损失 + Adam）。

调用单向无环：trainer → encoder / losses / mining → core。
确定性：Adam 状态、encoder 参数初始化、proxies 初始化均走 core.seeding。
"""

from __future__ import annotations

import numpy as np

from ..core import seeding
from ..core.encoder import Adam, MLPWithInput
from ..losses import contrastive, pair, proxy
from ..mining import BalancedBatchSampler, FreshQueue


class DeepTrainer:
    def __init__(
        self,
        method: str,
        dim_in: int,
        dim_hidden: int,
        dim_out: int,
        seed: int = 0,
        lr: float = 1e-3,
        epochs: int = 60,
        batch_size: int = 128,
        kappa: float = 0.25,
        K: int = 1,
        use_hat: bool = True,
        use_bidirectional: bool = True,
        use_queue: bool = False,
        queue_cap: int = 1024,
        use_geometry: bool = False,
        geom_weight: float = 0.1,
    ):
        self.method = method
        self.seed = int(seed)
        self.lr = float(lr)
        self.epochs = int(epochs)
        self.bs = int(batch_size)
        self.dim_out = int(dim_out)
        seeding.set_seed(seed)
        self.encoder = MLPWithInput(dim_in, dim_hidden, dim_out, act="relu")
        self.opt = Adam(self.encoder.parameters(), lr=self.lr)
        self.queue = FreshQueue(queue_cap) if use_queue else None
        C = None  # 在 fit 时确定
        if method in ("metricforge", "b5_proxy_k1"):
            self.loss_fn = proxy.ProxyAnchorLoss(
                num_classes=C,
                dim=dim_out,
                K=K,
                kappa=kappa,
                use_hat=use_hat,
                use_bidirectional=use_bidirectional,
                use_geometry=use_geometry,
                geom_weight=geom_weight,
            )
        elif method == "b4_triplet":
            self.loss_fn = pair.TripletBatchHard(margin=0.2)
        elif method == "b6_supcon":
            self.loss_fn = contrastive.SupConLoss(tau=0.07)
        else:
            raise ValueError(method)

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y)
        seeding.set_seed(self.seed)
        self.encoder.set_params(self.encoder.net._init_params())
        self.opt = Adam(self.encoder.parameters(), lr=self.lr)
        classes = np.unique(y)
        C = len(classes)
        if hasattr(self.loss_fn, "C"):
            self.loss_fn.C = C
            self.loss_fn.P = C * self.loss_fn.K
        # 初始化代理（用初始 embedding）
        if isinstance(self.loss_fn, proxy.ProxyAnchorLoss):
            Z0, _ = self.encoder.forward(X)
            self.loss_fn.init_proxies(Z0, y)
        sampler = BalancedBatchSampler(y, self.bs, seed=self.seed)
        for ep in range(self.epochs):
            for bidx in sampler:
                if len(bidx) < 4:
                    continue
                Z, cache = self.encoder.forward(X[bidx])
                if isinstance(self.loss_fn, proxy.ProxyAnchorLoss):
                    loss, dZ, dP, aux = self.loss_fn(Z, y[bidx])
                    grads, _ = self.encoder.backward(dZ)
                    self.opt.step(grads)
                    # 更新代理
                    self.loss_fn.proxies -= self.lr * dP
                    self.loss_fn.ema_align(Z, y[bidx])
                    if self.queue is not None:
                        self.queue.update(X[bidx])
                elif isinstance(self.loss_fn, (pair.TripletBatchHard, contrastive.SupConLoss)):
                    loss, dZ = self.loss_fn(Z, y[bidx])
                    grads, _ = self.encoder.backward(dZ)
                    self.opt.step(grads)
        return self

    def transform(self, X):
        Z, _ = self.encoder.forward(np.asarray(X, dtype=np.float64))
        return Z
