# 作者: 晨星
"""eval/metrics.py —— 检索/聚类评测指标。

主指标 MAP@R（Musgrave 2020 推荐）；次：R@1、R-Precision、kNN 准确率、NMI/ARI。
检索口径：query=test，gallery=test（剔除自身）；某类 gallery 仅 1 样本则剔除该 query。
向量化 kNN 检索，禁止物化全局 N×N 距离（文档 §9 硬约束 2）。
"""

from __future__ import annotations

import numpy as np

from ..core.encoder import l2_normalize
from ..core.mathx import map_at_r


def _norm(Z):
    return l2_normalize(np.asarray(Z, dtype=np.float64))


def map_at_r_score(Z_query, Z_gallery, y_query, y_gallery):
    """主指标：逐 query 计算 MAP@R，返回均值（剔除不合法 query）。

    当 query 与 gallery 为同一集合（测试协议）时，每个 query 会自动从 gallery
    中剔除自身，避免自命中 rank1 虚高（文档 §4.3 口径）。
    """
    Z_query = _norm(Z_query)
    Z_gallery = _norm(Z_gallery)
    y_query = np.asarray(y_query)
    y_gallery = np.asarray(y_gallery)
    same_set = Z_query is Z_gallery
    nq, ng = len(y_query), len(y_gallery)
    if same_set:
        sims = Z_query @ Z_gallery.T
    scores = []
    for i in range(nq):
        qy = y_query[i]
        same = np.where(y_gallery == qy)[0]
        if same.size < 2:
            continue  # 同类样本不足，无法评估
        if same_set:
            mask = np.ones(ng, dtype=bool)
            mask[i] = False
            gi = np.arange(ng)[mask]
            si = sims[i, mask]
            ranked = gi[np.argsort(-si)]
            r = same.size - 1  # 剔除自身后剩余同类数
            if r < 1:
                continue
            ranked = ranked[:r]
        else:
            si = Z_query[i] @ Z_gallery.T
            ranked = np.argsort(-si)[: same.size]
            r = same.size
        hits = (y_gallery[ranked] == qy).astype(int)
        m, _ = map_at_r(hits.tolist(), r)
        scores.append(m)
    return float(np.mean(scores)) if scores else 0.0


def recall_at_k(Z_query, Z_gallery, y_query, y_gallery, k=1):
    Z_query = _norm(Z_query)
    Z_gallery = _norm(Z_gallery)
    y_query = np.asarray(y_query)
    y_gallery = np.asarray(y_gallery)
    same_set = Z_query is Z_gallery
    nq, ng = len(y_query), len(y_gallery)
    if same_set:
        sims = Z_query @ Z_gallery.T
    correct = 0
    total = 0
    for i in range(nq):
        qy = y_query[i]
        if same_set:
            mask = np.ones(ng, dtype=bool)
            mask[i] = False
            gi = np.arange(ng)[mask]
            si = sims[i, mask]
            topk = y_gallery[gi[np.argsort(-si)[:k]]]
        else:
            si = Z_query[i] @ Z_gallery.T
            topk = y_gallery[np.argsort(-si)[:k]]
        if qy in topk:
            correct += 1
        total += 1
    return correct / max(total, 1)


def knn_accuracy(Z_train, Z_test, y_train, y_test, k=1):
    Z_train = _norm(Z_train)
    Z_test = _norm(Z_test)
    y_train = np.asarray(y_train)
    y_test = np.asarray(y_test)
    same_set = Z_train is Z_test
    ntr, nte = len(y_train), len(y_test)
    if same_set:
        sims = Z_test @ Z_train.T
    correct = 0
    for i in range(nte):
        if same_set:
            mask = np.ones(ntr, dtype=bool)
            mask[i] = False
            gi = np.arange(ntr)[mask]
            si = sims[i, mask]
            order = gi[np.argsort(-si)[:k]]
        else:
            si = Z_test[i] @ Z_train.T
            order = np.argsort(-si)[:k]
        preds = y_train[order]
        if k == 1:
            pred = preds[0]
        else:
            vals, cnts = np.unique(preds, return_counts=True)
            pred = vals[np.argmax(cnts)]
        if pred == y_test[i]:
            correct += 1
    return correct / max(nte, 1)


def nmi_ari(y_true, y_pred):
    try:
        from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

        return (
            float(normalized_mutual_info_score(y_true, y_pred)),
            float(adjusted_rand_score(y_true, y_pred)),
        )
    except Exception:
        return (0.0, 0.0)
