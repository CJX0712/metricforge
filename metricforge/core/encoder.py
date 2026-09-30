# 作者: 晨星
"""编码器：纯 numpy MLP（Tier-1 运行时路径）+ 可选 torch 后端（Tier-0 oracle）。

确定性：参数初始化全部走 core.seeding.rng()，禁止裸随机。
所有输出在最后一个线性层后做 L2 归一化（度量学习标准约定）。
"""

from __future__ import annotations

import numpy as np

from . import seeding


def l2_normalize(mat: np.ndarray, axis: int = 1, eps: float = 1e-12) -> np.ndarray:
    n = np.linalg.norm(mat, axis=axis, keepdims=True)
    return mat / np.maximum(n, eps)


class Adam:
    """极简 Adam 优化器（确定性，参数以 list[np.ndarray] 形态管理）。"""

    def __init__(self, params, lr: float = 1e-3, betas=(0.9, 0.999), eps: float = 1e-8):
        self.params = list(params)
        self.lr = lr
        self.b1, self.b2 = betas
        self.eps = eps
        self.t = 0
        self.m = [np.zeros_like(p) for p in self.params]
        self.v = [np.zeros_like(p) for p in self.params]

    def step(self, grads):
        self.t += 1
        for i, g in enumerate(grads):
            g = np.asarray(g, dtype=np.float64)
            self.m[i] = self.b1 * self.m[i] + (1 - self.b1) * g
            self.v[i] = self.b2 * self.v[i] + (1 - self.b2) * (g * g)
            mhat = self.m[i] / (1 - self.b1**self.t)
            vhat = self.v[i] / (1 - self.b2**self.t)
            self.params[i] -= self.lr * mhat / (np.sqrt(vhat) + self.eps)
        return self.params


class MLP:
    """两层 MLP 编码器（d -> h -> D），输出 L2 归一化。手写前向/反向。"""

    def __init__(self, dim_in: int, dim_hidden: int, dim_out: int, act: str = "relu"):
        self.d, self.h, self.D = dim_in, dim_hidden, dim_out
        self.act = act
        self.params = self._init_params()

    def _init_params(self):
        rng = seeding.rng()

        # Xavier/Glorot
        def w(inn, out):
            return (rng.standard_normal((inn, out)) * np.sqrt(2.0 / (inn + out))).astype(np.float64)

        W1 = w(self.d, self.h)
        b1 = np.zeros(self.h, dtype=np.float64)
        W2 = w(self.h, self.D)
        b2 = np.zeros(self.D, dtype=np.float64)
        return [W1, b1, W2, b2]

    def set_params(self, params):
        self.params = [np.asarray(p, dtype=np.float64) for p in params]

    def parameters(self):
        return self.params

    def _act(self, a):
        if self.act == "relu":
            return np.maximum(a, 0.0)
        if self.act == "tanh":
            return np.tanh(a)
        raise ValueError(self.act)

    def _act_prime(self, z):
        if self.act == "relu":
            return (z > 0).astype(np.float64)
        if self.act == "tanh":
            return 1.0 - z * z

    def forward(self, X):
        W1, b1, W2, b2 = self.params
        a1 = X @ W1 + b1
        z1 = self._act(a1)
        a2 = z1 @ W2 + b2
        Z = l2_normalize(a2, axis=1)
        cache = (a1, z1, a2, Z)
        self._last_cache = cache
        return Z, cache

    # 实际训练用下列封装（MLPWithInput）


class MLPWithInput:
    """封装 MLP，保存 X 以支持一步 forward+backward。"""

    def __init__(self, dim_in, dim_hidden, dim_out, act="relu"):
        self.net = MLP(dim_in, dim_hidden, dim_out, act)
        self._X = None

    def parameters(self):
        return self.net.parameters()

    def set_params(self, params):
        self.net.set_params(params)

    def forward(self, X):
        self._X = np.asarray(X, dtype=np.float64)
        return self.net.forward(self._X)

    def backward(self, dZ_logit):
        W1, b1, W2, b2 = self.net.params
        a1, z1, a2, Z = self.net._last_cache  # type: ignore[attr-defined]
        dZ = np.asarray(dZ_logit, dtype=np.float64)
        radial = np.sum(dZ * Z, axis=1, keepdims=True)
        # L2 归一化层：Z = a2 / ||a2||，∂Z/∂a2 = (1/||a2||)(I - ZZᵀ)
        # ⇒ ∂L/∂a2 = (1/||a2||)(dZ - (dZ·Z)Z)
        norm_a2 = np.sqrt(np.sum(a2 * a2, axis=1, keepdims=True)) + 1e-12
        da2 = (dZ - radial * Z) / norm_a2
        dz1 = da2 @ W2.T
        da1 = dz1 * self.net._act_prime(z1)
        dW2 = z1.T @ da2
        db2 = np.sum(da2, axis=0)
        dW1 = self._X.T @ da1
        db1 = np.sum(da1, axis=0)
        dX = da1 @ W1.T
        grads = [dW1, db1, dW2, db2]
        return grads, dX
