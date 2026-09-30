# 作者: 晨星
"""eval（MAP@R / Recall@k / kNN acc / 自剔除）模块单测。"""

import numpy as np

from metricforge.core.encoder import l2_normalize
from metricforge.eval.metrics import (
    knn_accuracy,
    map_at_r_score,
    nmi_ari,
    recall_at_k,
)


def _two_blobs(seed=0, n=50, d=4, sep=5.0):
    rng = np.random.default_rng(seed)
    X = np.vstack(
        [
            rng.normal(-sep, 0.3, size=(n, d)),
            rng.normal(+sep, 0.3, size=(n, d)),
        ]
    )
    y = np.array([0] * n + [1] * n)
    return X, y


def test_perfectly_separable_high_map():
    X, y = _two_blobs(seed=1)
    Z = l2_normalize(X)
    m = map_at_r_score(Z, Z, y, y)
    assert m > 0.95, f"完美可分两团 MAP@R 应接近 1，got {m}"
    r1 = recall_at_k(Z, Z, y, y, 1)
    assert r1 == 1.0
    acc = knn_accuracy(Z, Z, y, y, 1)
    assert acc == 1.0


def test_self_exclusion_not_rank1():
    # 构造：同类每个类只有 2 个样本，且两样本完全相同（必然自相似最大）。
    # query==gallery 时，自剔除必须防止自身占 rank1。
    X = np.array([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0]], dtype=np.float64)
    y = np.array([0, 0, 1, 1])
    Z = l2_normalize(X)
    # 若未剔除自身，query0 必命中自身（rank1）→ MAP@R 虚高；
    # 自剔除后，同类仅剩 1 个其它样本，每 query 召回该样本，MAP@R 应 = 1.0（同类仅1个）
    m = map_at_r_score(Z, Z, y, y)
    assert m == 1.0


def test_nmi_ari_runs():
    y_true = np.array([0, 0, 1, 1, 0, 0, 1, 1])
    y_pred = np.array([1, 1, 0, 0, 1, 1, 0, 0])
    nmi, ari = nmi_ari(y_true, y_pred)
    assert 0.0 <= nmi <= 1.0 and -0.0 <= ari <= 1.0
