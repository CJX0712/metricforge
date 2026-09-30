# 作者: 晨星
"""mining（FreshQueue / BalancedBatchSampler）模块单测。"""

import numpy as np

from metricforge.core.encoder import MLPWithInput
from metricforge.mining.queue import FreshQueue
from metricforge.mining.sampler import BalancedBatchSampler


def test_fresh_queue_capacity_and_encode():
    q = FreshQueue(capacity=5)
    X = np.random.default_rng(0).normal(size=(12, 4))
    q.update(X)
    assert q.size == 5, "超出 cap 后队列应截断为 cap"
    buf = q.get()
    assert len(buf) == 5
    enc = MLPWithInput(4, 8, 8)
    Z = q.encode(enc)
    assert Z.shape == (5, 8)


def test_fresh_queue_augment_negatives():
    q = FreshQueue(capacity=4)
    q.update(np.random.default_rng(1).normal(size=(4, 4)))
    enc = MLPWithInput(4, 8, 8)
    Zq = np.random.default_rng(2).normal(size=(3, 8))
    out = q.augment_negatives(Zq, enc)
    # 原始 3 个 query + 4 个队列负样本
    assert out.shape == (7, 8)


def test_balanced_batch_sampler_covers_all():
    y = np.array([0] * 10 + [1] * 10 + [2] * 10, dtype=int)
    seen = set()
    total = 0
    for b in BalancedBatchSampler(y, batch_size=9, seed=0):
        assert len(b) <= 9
        seen.update(b.tolist())
        total += len(b)
    assert seen == set(range(30)), "所有样本都应被采样到"
    assert total == 30
