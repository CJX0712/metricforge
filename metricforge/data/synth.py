# 作者: 晨星
"""data/synth.py —— 6 个可控合成数据集（S1..S6），seed 全程可控可复现。

设计意图（对齐算法文档 §4）：
  S1 gauss_blobs : 标准高斯团（难度基线）
  S2 concentric  : 同心环（非线性可分）
  S3 spirals     : 螺旋（非线性，多臂）
  S4 subspace    : d=200 高维稀疏子空间 + 小噪声（高维多类）
  S5 xor_grid    : XOR 网格（非线性决策边界）
  S6 hier_gauss  : 大类内嵌 2~3 子类（类内多模态 —— 多原型主判别集）

所有生成器用 np.random.default_rng(seed)（带参，确定性），不污染全局种子状态。
"""

from __future__ import annotations

import numpy as np


def gauss_blobs(seed, n_per_class=80, d=2, C=6, sep=4.0, noise=0.3):
    rng = np.random.default_rng(seed)
    cen = rng.normal(scale=sep, size=(C, d))
    X = []
    y = []
    for c in range(C):
        X.append(rng.normal(loc=cen[c], scale=noise, size=(n_per_class, d)))
        y.extend([c] * n_per_class)
    return np.vstack(X), np.array(y)


def concentric(seed, n_per_class=80, C=3, r_base=1.0, noise=0.05):
    rng = np.random.default_rng(seed)
    X = []
    y = []
    for c in range(C):
        th = rng.uniform(0, 2 * np.pi, n_per_class)
        r = r_base * (c + 1) + rng.normal(0, noise, n_per_class)
        X.append(np.stack([r * np.cos(th), r * np.sin(th)], axis=1))
        y.extend([c] * n_per_class)
    return np.vstack(X), np.array(y)


def spirals(seed, n_per_class=80, C=3, turns=2.5, noise=0.1):
    rng = np.random.default_rng(seed)
    X = []
    y = []
    for c in range(C):
        t = np.linspace(0, 1, n_per_class)
        theta = t * turns * 2 * np.pi + c * 2 * np.pi / C
        rad = t + rng.normal(0, noise, n_per_class)
        X.append(np.stack([rad * np.cos(theta), rad * np.sin(theta)], axis=1))
        y.extend([c] * n_per_class)
    return np.vstack(X), np.array(y)


def subspace(seed, n_per_class=100, d=200, C=50, k=5, sep=8.0, noise=0.5):
    rng = np.random.default_rng(seed)
    # 每个类一段低维方向
    dirs = rng.normal(scale=1.0, size=(C, k))
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True) + 1e-9
    centers = rng.normal(scale=sep, size=(C, d))
    X = []
    y = []
    for c in range(C):
        coeff = rng.normal(scale=1.0, size=(n_per_class, k))
        # low 为 (n_per_class, k) 的 k 维低维坐标；dirs[c] 仅用于"每类方向"语义，
        # 在此用按轴缩放注入轻微类间差异（保持维度 k，避免 (40,5)@(5,) 退化为 1D）。
        low = coeff * dirs[c]  # (n_per_class, k) 广播
        # 随机投影到 d 维子空间
        proj = rng.normal(scale=1.0 / np.sqrt(d), size=(d, d))
        proj = proj / np.linalg.norm(proj, axis=0, keepdims=True)
        emb = low @ proj[:k, :] + centers[c]
        emb += rng.normal(scale=noise, size=(n_per_class, d))
        X.append(emb)
        y.extend([c] * n_per_class)
    return np.vstack(X), np.array(y)


def xor_grid(seed, n_per_class=80, C=4, scale=2.0, noise=0.15):
    rng = np.random.default_rng(seed)
    corners = np.array([[0, 0], [0, scale], [scale, 0], [scale, scale]])
    X = []
    y = []
    for c in range(C):
        cx, cy = corners[c]
        X.append(
            np.stack(
                [rng.normal(cx, noise, n_per_class), rng.normal(cy, noise, n_per_class)], axis=1
            )
        )
        y.extend([c] * n_per_class)
    return np.vstack(X), np.array(y)


def hier_gauss(seed, n_per_class=80, C=5, sub=3, d=2, sep=5.0, sub_sep=1.5, noise=0.3):
    rng = np.random.default_rng(seed)
    X = []
    y = []
    big = rng.normal(scale=sep, size=(C, d))
    for c in range(C):
        sub_cen = big[c] + rng.normal(scale=sub_sep, size=(sub, d))
        for s in range(sub):
            X.append(rng.normal(loc=sub_cen[s], scale=noise, size=(n_per_class, d)))
            y.extend([c] * n_per_class)
    return np.vstack(X), np.array(y)


SYNTH = {
    "s1_gauss_blobs": gauss_blobs,
    "s2_concentric": concentric,
    "s3_spirals": spirals,
    "s4_subspace": subspace,
    "s5_xor_grid": xor_grid,
    "s6_hier_gauss": hier_gauss,
}
