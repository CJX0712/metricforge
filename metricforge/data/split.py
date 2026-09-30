# 作者: 晨星
"""data/split.py —— 划分协议（无泄漏，对齐文档 §4.2）。

P1 closed-set : 训练/验证/测试按样本随机 60/20/20（同分布）。
P2 open-set   : 类不相交，测试集由"后半类"构成（Musgrave 口径）；
                scaler / HPO 只在 train 折 fit，test 只用一次。
"""

from __future__ import annotations

import numpy as np


def split_closed(X, y, seed, train=0.6, val=0.2):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(y))
    n = len(y)
    n_tr = int(n * train)
    n_va = int(n * val)
    return idx[:n_tr], idx[n_tr : n_tr + n_va], idx[n_tr + n_va :]


def split_open(X, y, seed):
    """类不相交：按类排序，前半类入 train/val，后半类入 test。"""
    classes = np.unique(y)
    rng = np.random.default_rng(seed)
    rng.shuffle(classes)
    half = len(classes) // 2
    train_classes = classes[:half]
    test_classes = classes[half:]
    tr_idx = np.where(np.isin(y, train_classes))[0]
    te_idx = np.where(np.isin(y, test_classes))[0]
    # train 内再分 val
    sub = rng.permutation(len(tr_idx))
    n_va = max(1, len(tr_idx) // 5)
    val_idx = tr_idx[sub[:n_va]]
    tr_idx2 = tr_idx[sub[n_va:]]
    return tr_idx2, val_idx, te_idx
