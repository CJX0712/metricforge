# 作者: 晨星
"""Proxy-Anchor 骨架 + 自研创新组件。

创新点 1 — HAT (Hardness-Adaptive Temperature)：
  负样本相似度用 adaptive β=1/τ，τ 由 mathx.hat_temperature 每 batch 解出，
  使"活跃负样本参与率"恒为 κ（与 batch/类数/维度无关的相对量）。
  由引理 dH/dβ=−βVar(s)<0 保证根唯一、可固定次数二分 ⇒ 逐位确定性。

创新点 2 — MQ-BPA (Memory-Queue Bi-directional Proxy Alignment)：
  * 每个类 K>1 个原型，覆盖类内多模态；
  * 反向 EMA 对齐：原型被锚定到其责任样本嵌入质心（防原型坍缩/漂移）；
  * Fresh-Queue（mining.queue）存原始输入每步重编码，提供额外负样本（零陈旧偏差）。

梯度推导（Z,P 均 L2 归一化，s=Z·P）：
  w_k = exp(-α(s_k+δ)), Lpos_i = log(1+Σ_{k∈P_c} w_k)
  v_k = exp(β(s_k-δ)), Lneg_i = log(1+Σ_{k∉P_c} v_k)
  ∂/∂Z_i = Σ_k g_k (∓α/β) P_k ;  ∂/∂P_k = Σ_i g_k (∓α/β) Z_i
  g_pos = w/(1+Σw) ; g_neg = v/(1+Σv)
"""

from __future__ import annotations

import numpy as np

from ..core import mathx, seeding
from ..core.encoder import l2_normalize


class ProxyAnchorLoss:
    def __init__(
        self,
        num_classes,
        dim,
        K: int = 1,
        alpha: float = 32.0,
        delta: float = 0.1,
        kappa: float = 0.25,
        use_hat: bool = True,
        use_bidirectional: bool = True,
        ema: float = 0.2,
        use_geometry: bool = False,
        geom_weight: float = 0.1,
    ):
        self.C = int(num_classes) if num_classes is not None else None
        self.D = int(dim)
        self.K = int(K)
        self.P = (self.C * self.K) if self.C is not None else 0
        self.alpha = float(alpha)
        self.delta = float(delta)
        self.kappa = float(kappa)
        self.use_hat = use_hat
        self.use_bidirectional = use_bidirectional
        self.ema = float(ema)
        self.use_geometry = use_geometry
        self.geom_weight = float(geom_weight)
        self.proxies = None  # (P, D)

    def init_proxies(self, Z: np.ndarray, y: np.ndarray):
        """用每类初始 embedding 均值作原型种子（确定性，取同类样本质心）。"""
        Z = np.asarray(Z, dtype=np.float64)
        y = np.asarray(y)
        self.C = len(np.unique(y))
        self.P = self.C * self.K
        P = np.zeros((self.P, self.D), dtype=np.float64)
        rng = seeding.rng()
        for c in range(self.C):
            idx = np.where(y == c)[0]
            if len(idx) == 0:
                P[c * self.K : (c + 1) * self.K] = l2_normalize(
                    rng.standard_normal((self.K, self.D))
                )
                continue
            mu = Z[idx].mean(axis=0)
            mu = l2_normalize(mu[None, :])[0]
            # K 个原型 = 均值 + 确定性扰动
            for k in range(self.K):
                if k == 0:
                    P[c * self.K + k] = mu
                else:
                    pert = l2_normalize(rng.standard_normal(self.D)[None, :])[0] * 0.1
                    P[c * self.K + k] = l2_normalize((mu + pert)[None, :])[0]
        self.proxies = P
        return self

    def _hat_beta(self, neg_sims: np.ndarray) -> float:
        """每 anchor 解一次自适应 β=1/τ。"""
        tau = mathx.hat_temperature(neg_sims, kappa=self.kappa, tau_lo=0.02, tau_hi=2.0, iters=40)
        return 1.0 / tau

    def __call__(self, Z, y):
        Z = np.asarray(Z, dtype=np.float64)
        y = np.asarray(y)
        P = self.proxies
        B = Z.shape[0]
        S = Z @ P.T  # (B, P)
        pos_mask = np.zeros((B, self.P), dtype=bool)
        for c in range(self.C):
            pos_mask[:, c * self.K : (c + 1) * self.K] = (y == c)[:, None]
        neg_mask = ~pos_mask

        # 正项
        s_pos = S * pos_mask
        w = np.exp(-self.alpha * (s_pos + self.delta)) * pos_mask
        sw = w.sum(axis=1) + 1.0
        L_pos = np.log(sw).sum()

        # 负项（HAT 自适应 β）
        s_neg = S * neg_mask
        beta = np.ones(B, dtype=np.float64) * self.alpha
        if self.use_hat:
            for i in range(B):
                ns = s_neg[i][neg_mask[i]]
                if ns.size > 0:
                    beta[i] = self._hat_beta(ns)
        v = np.exp(beta[:, None] * (s_neg - self.delta)) * neg_mask
        sv = v.sum(axis=1) + 1.0
        L_neg = np.log(sv).sum()

        loss = (L_pos + L_neg) / (2.0 * B)

        # 梯度：∂L/∂Z_i = Σ_p g_p P_p ；∂L/∂P_p = Σ_i g_p Z_i
        #   g_pos = w/(1+Σw) 为正项权重；g_neg = β·v/(1+Σv) 为负项权重
        # 用矩阵乘法避免 einsum 维度歧义（dP 的 P 维曾塌缩到 1）。
        gp = (-self.alpha) * (w / sw[:, None]) * pos_mask  # (B, P)
        gn = (beta[:, None] * (v / sv[:, None])) * neg_mask  # (B, P)
        dZ = (gp @ P) + (gn @ P)  # (B, P)@(P, D) -> (B, D)
        dP = (gp.T @ Z) + (gn.T @ Z)  # (P, B)@(B, D) -> (P, D)
        dZ /= 2.0 * B
        dP /= 2.0 * B

        aux = {
            "beta_mean": float(beta.mean()),
            "sw_mean": float(sw.mean()),
            "sv_mean": float(sv.mean()),
        }
        return loss, dZ, dP, aux

    def ema_align(self, Z, y):
        """MQ-BPA 反向对齐：原型被拉向责任样本嵌入质心（防坍缩/漂移）。"""
        if not self.use_bidirectional:
            return
        Z = np.asarray(Z, dtype=np.float64)
        y = np.asarray(y)
        P = self.proxies.copy()
        for c in range(self.C):
            idx = np.where(y == c)[0]
            if idx.size == 0:
                continue
            mu = l2_normalize(Z[idx].mean(axis=0)[None, :])[0]
            for k in range(self.K):
                pidx = c * self.K + k
                if self.K == 1:
                    tgt = mu
                else:
                    # 该原型责任 = 与该原型最近（同类的样本）
                    d = Z[idx] @ P[pidx]
                    sel = idx[d >= np.quantile(d, 1.0 - 1.0 / self.K)]
                    tgt = l2_normalize(Z[sel].mean(axis=0)[None, :])[0] if sel.size else mu
                P[pidx] = l2_normalize(((1 - self.ema) * P[pidx] + self.ema * tgt)[None, :])[0]
        self.proxies = P
