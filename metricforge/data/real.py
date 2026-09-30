# 作者: 晨星
"""data/real.py —— 4 个 sklearn 内置离线城市/表格集（R1..R4）。

数据许可见 docs/model_card.md（UCI 原始来源，本仓库不重新分发，仅本地按需加载）。
"""

from __future__ import annotations

import numpy as np


def _load(name):
    try:
        from sklearn.datasets import load_breast_cancer, load_digits, load_iris, load_wine
    except Exception as e:  # pragma: no cover
        raise RuntimeError("scikit-learn 不可用") from e
    if name == "r1_digits":
        d = load_digits()
    elif name == "r2_wine":
        d = load_wine()
    elif name == "r3_breast_cancer":
        d = load_breast_cancer()
    elif name == "r4_iris":
        d = load_iris()
    else:
        raise ValueError(name)
    X = np.asarray(d.data, dtype=np.float64)
    y = np.asarray(d.target)
    return X, y


def r1_digits(seed=0):
    return _load("r1_digits")


def r2_wine(seed=0):
    return _load("r2_wine")


def r3_breast_cancer(seed=0):
    return _load("r3_breast_cancer")


def r4_iris(seed=0):
    return _load("r4_iris")


REAL = {
    "r1_digits": r1_digits,
    "r2_wine": r2_wine,
    "r3_breast_cancer": r3_breast_cancer,
    "r4_iris": r4_iris,
}
