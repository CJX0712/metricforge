# 作者: 晨星
"""linear.py —— 自研纯 numpy 线性度量学习（零第三方 DML 依赖）。

  LFDA : Local Fisher Discriminant Analysis（Sugiyama 2006，闭式广义特征值，可靠）
  LMNN : Large-Margin Nearest Neighbors（Weinberger 2009，梯度法）
  NCA  : Neighbourhood Components Analysis（Goldberger 2004，梯度法）
  ITML : Information-Theoretic Metric Learning（Davis 2007，迭代约束法）

用途：B2 / B3 / B7 基线；不直接依赖 metric-learn（后者经 shim 仅作 oracle 对照）。
"""

from __future__ import annotations

import numpy as np

from ..core import seeding


def _target_neighbors(X, y, k):
    """每个样本 i 的 k 个同类最近邻（欧氏，排除自身）。"""
    N = X.shape[0]
    D = np.sum(X**2, axis=1)
    d2 = D[:, None] + D[None, :] - 2 * X @ X.T
    np.fill_diagonal(d2, np.inf)
    tn = []
    for i in range(N):
        c = y[i]
        idx = np.where(y == c)[0]
        idx = idx[idx != i]
        if idx.size == 0:
            tn.append(np.array([], dtype=int))
            continue
        di = d2[i, idx]
        order = idx[np.argsort(di)[: min(k, idx.size)]]
        tn.append(order)
    return tn


class LFDA:
    """Local Fisher Discriminant Analysis（闭式）。"""

    def __init__(self, dim_out: int = 2, k: int = 7, reg: float = 1e-3):
        self.D = int(dim_out)
        self.k = int(k)
        self.reg = float(reg)
        self.T = None
        self.mean = None

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y)
        n, d = X.shape
        classes = np.unique(y)
        self.mean = X.mean(axis=0)
        n_c = {c: int((y == c).sum()) for c in classes}
        SbL = np.zeros((d, d))  # LFDA 带权类间散度
        Sw = np.zeros((d, d))  # 标准类内散度
        for c in classes:
            idx = np.where(y == c)[0]
            m = X[idx].mean(axis=0)
            nc = n_c[c]
            SbL += nc * (1.0 - nc / n) * np.outer(m - self.mean, m - self.mean)
            Sw += (X[idx] - m).T @ (X[idx] - m)
        # 本地加权类内散度（SNE 风格亲和权重）
        SwL = np.zeros((d, d))
        for c in classes:
            idx = np.where(y == c)[0]
            if idx.size < 2:
                continue
            diff = X[idx] - X[idx].mean(axis=0)
            d2 = np.sum(diff**2, axis=1)
            maxd = max(d2.max(), 1e-12)
            for a in range(idx.size):
                for b in range(idx.size):
                    w = max(1.0 - d2[a] / (2 * maxd), 0.0)
                    SwL += w * np.outer(diff[a] - diff[b], diff[a] - diff[b])
        A = SbL + self.reg * np.eye(d)
        B = Sw + SwL + self.reg * np.eye(d)
        # 广义对称特征值 A v = λ B v（A、B 对称，B 正定）。
        # 跨 numpy 版本稳定实现：Cholesky(B)=L Lᵀ，转标准问题
        #   M = L⁻¹ A L⁻ᵀ  →  M 对称 → eigh(M)，再逆变换回 v = L⁻ᵀ u。
        try:
            L = np.linalg.cholesky(B)
            Linv = np.linalg.inv(L)
            M = Linv @ A @ Linv.T
            M = (M + M.T) / 2.0  # 消除浮点非对称
            vals, u = np.linalg.eigh(M)
            order = np.argsort(vals)[::-1]
            self.T = Linv.T @ u[:, order[: self.D]]
        except (np.linalg.LinAlgError, ValueError):
            # B 病态（如单类退化）→ 退化为 PCA 投影
            A = (A + A.T) / 2.0
            try:
                vals, vecs = np.linalg.eigh(A)
                order = np.argsort(vals)[::-1]
                self.T = vecs[:, order[: self.D]]
            except np.linalg.LinAlgError:
                self.T = np.eye(d)[:, : self.D]
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=np.float64)
        return (X - self.mean) @ self.T


class LMNN:
    """Large-Margin Nearest Neighbors（梯度法）。"""

    def __init__(
        self,
        dim_out: int = 2,
        margin: float = 0.1,
        k: int = 3,
        iters: int = 80,
        lr: float = 1e-3,
        max_grad: float = 100.0,
    ):
        self.D = int(dim_out)
        # margin 取小值：数据经 StandardScaler 标准化后距离量级偏小，
        # 大 margin 会使 push 铰链全程不激活 → pull 单独把 L 推向 0（坍缩成随机检索）。
        # margin=0.1 令 push 持续激活，既防坍缩又能在易分集上提升 MAP@R。
        self.margin = float(margin)
        self.k = int(k)
        self.iters = int(iters)
        self.lr = float(lr)
        # 梯度裁剪：LMNN 的 pull+push 梯度量级随 impostor 数/维度暴涨，
        # 不裁剪会令 L 发散（Z 溢出 → MAP@R=NaN）。裁剪到固定 Frobenius 范数上限，
        # 保留梯度方向、限制每步位移，保证 CPU-only 稳定可复现。
        self.max_grad = float(max_grad)
        self.L = None
        self.mean = None

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y)
        n, d = X.shape
        self.mean = X.mean(axis=0)
        rng = seeding.rng()
        self.L = (rng.standard_normal((d, self.D)) * 0.01).astype(np.float64)
        Xc = X - self.mean
        tn = _target_neighbors(Xc, y, self.k)
        K = max((len(t) for t in tn), default=0)
        # 预构建固定长度目标邻居索引矩阵（向量化 pull）
        T = np.full((n, K), -1, dtype=int)
        for i in range(n):
            t = tn[i]
            if t.size:
                T[i, : len(t)] = t[:K]
        for _ in range(self.iters):
            Z = Xc @ self.L
            sq = np.sum(Z**2, axis=1)
            d2 = sq[:, None] + sq[None, :] - 2.0 * (Z @ Z.T)
            np.fill_diagonal(d2, np.inf)
            grad = np.zeros_like(self.L)
            # pull：Σ_i Σ_j 2·‖Lᵀxij‖² = 2·Σ xij xijᵀ @ L（L 为 (d,D)，梯度亦 (d,D)）
            pull_sum = np.zeros((d, d), dtype=np.float64)
            for c in range(K):
                jdx = T[:, c]
                valid = jdx >= 0
                diff = Xc[valid] - Xc[jdx[valid]]
                pull_sum += diff.T @ diff
            grad += 2.0 * (pull_sum @ self.L)
            # push：逐样本向量化（impostor 批量化，去除 O(n²) Python 内层循环）
            for i in range(n):
                ti = tn[i]
                if ti.size == 0:
                    continue
                d_ap = float(d2[i, ti].max())
                thr = d_ap + self.margin
                ls = np.where((d2[i] < thr) & (y != y[i]))[0]
                if ls.size == 0:
                    continue
                for j in ti:
                    xij2 = Xc[i] - Xc[j]
                    Lxij2 = self.L.T @ xij2
                    xil = Xc[i] - Xc[ls]
                    Lxil = xil @ self.L
                    sum_xil_Lxil = xil.T @ Lxil
                    grad -= 2.0 * (np.outer(xij2, Lxij2) * ls.size - sum_xil_Lxil)
            # 梯度裁剪（Frobenius 范数上限），防止 L 发散
            gn = np.linalg.norm(grad)
            if gn > self.max_grad:
                grad = grad * (self.max_grad / gn)
            self.L -= self.lr * grad
        return self

    def transform(self, X):
        return (np.asarray(X, dtype=np.float64) - self.mean) @ self.L


class NCA:
    """Neighbourhood Components Analysis（梯度上升）。"""

    def __init__(
        self, dim_out: int = 2, iters: int = 60, lr: float = 1e-2, max_grad: float = 100.0
    ):
        self.D = int(dim_out)
        self.iters = int(iters)
        self.lr = float(lr)
        # 同 LMNN：NCA 梯度在硬数据集上易发散，裁剪 Frobenius 范数上限保证稳定。
        self.max_grad = float(max_grad)
        self.L = None
        self.mean = None

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y)
        n, d = X.shape
        self.mean = X.mean(axis=0)
        rng = seeding.rng()
        self.L = (rng.standard_normal((d, self.D)) * 0.01).astype(np.float64)
        Xc = X - self.mean
        for _ in range(self.iters):
            Z = Xc @ self.L
            sq = np.sum(Z**2, axis=1)
            d2 = sq[:, None] + sq[None, :] - 2.0 * (Z @ Z.T)
            np.fill_diagonal(d2, np.inf)
            P = np.exp(-d2)
            P = P / P.sum(axis=1, keepdims=True)
            grad = np.zeros_like(self.L)
            # 逐样本向量化：Σ_j w_j·outer(xij, Lᵀxij)，w_j=+p_j(同类)/−p_j(异类)
            for i in range(n):
                pj = P[i].copy()
                pj[i] = 0.0
                same = y == y[i]
                same[i] = False
                w = np.where(same, pj, -pj)
                Xij = Xc[i] - Xc
                LXij = Xij @ self.L
                grad += Xij.T @ (w[:, None] * LXij)
            gn = np.linalg.norm(grad)
            if gn > self.max_grad:
                grad = grad * (self.max_grad / gn)
            self.L += self.lr * grad
        return self

    def transform(self, X):
        return (np.asarray(X, dtype=np.float64) - self.mean) @ self.L


class ITML:
    """Information-Theoretic Metric Learning（迭代约束法）。"""

    def __init__(self, gamma: float = 1.0, iters: int = 30, dim_out: int = 2):
        self.gamma = float(gamma)
        self.iters = int(iters)
        self.D = int(dim_out)
        self.A = None  # 马氏精度矩阵
        self.mean = None

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y)
        n, d = X.shape
        self.mean = X.mean(axis=0)
        Xc = X - self.mean
        cov = np.cov(Xc.T) + 1e-3 * np.eye(d)
        A = np.linalg.inv(cov)
        rng = seeding.rng()
        # 约束采样
        same = [np.where(y == c)[0] for c in np.unique(y)]
        diff_pairs = [(i, j) for i in range(n) for j in range(n) if y[i] != y[j] and i < j]
        must = []
        for c in same:
            if c.size >= 2:
                for _ in range(min(20, c.size)):
                    a, b = rng.choice(c, 2, replace=False)
                    must.append((a, b, True))
        for _ in range(min(40, len(diff_pairs))):
            a, b = diff_pairs[rng.integers(len(diff_pairs))]
            must.append((a, b, False))
        for _ in range(self.iters):
            for i, j, ml in must:
                xi, xj = Xc[i], Xc[j]
                cij = np.outer(xi - xj, xi - xj)
                # slack 双写
                if ml:
                    lam = 1.0 / (self.gamma + (xi - xj) @ A @ (xi - xj))
                    A = A @ np.linalg.inv(A + lam * cij) @ A
                else:
                    violation = 1.0 - (xi - xj) @ A @ (xi - xj)
                    if violation > 0:
                        lam = 1.0 / (1.0 + self.gamma - (xi - xj) @ A @ (xi - xj))
                        A = A @ np.linalg.inv(A + lam * cij) @ A
        self.A = A
        # 投影到 dim_out：取 A 主特征向量
        vals, vecs = np.linalg.eigh(A)
        order = np.argsort(vals)[::-1]
        self.T = vecs[:, order[: self.D]]
        return self

    def transform(self, X):
        return (np.asarray(X, dtype=np.float64) - self.mean) @ self.T
