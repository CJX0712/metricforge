# 作者: 晨星
"""mining/sampler.py —— 类均衡 batch sampler（每类 ≥ 2 样本）。"""

from __future__ import annotations

import numpy as np


class BalancedBatchSampler:
    """按类轮流采样的 mini-batch，保证 batch 内每类样本数接近均衡。"""

    def __init__(self, y, batch_size: int, seed: int = 0):
        self.y = np.asarray(y)
        self.bs = int(batch_size)
        self.rng = np.random.default_rng(seed)
        self.class_idx = {c: np.where(self.y == c)[0] for c in np.unique(self.y)}

    def __iter__(self):
        pointers = {c: 0 for c in self.class_idx}
        order = {c: self.rng.permutation(self.class_idx[c]) for c in self.class_idx}
        classes = list(self.class_idx.keys())
        n_classes = len(classes)
        per_class = max(1, self.bs // n_classes)
        total = sum(len(v) for v in self.class_idx.values())
        produced = 0
        while produced < total:
            batch = []
            for c in classes:
                if pointers[c] < len(order[c]):
                    take = order[c][pointers[c] : pointers[c] + per_class]
                    batch.extend(take.tolist())
                    pointers[c] += per_class
            if not batch:
                break
            produced += len(batch)
            yield np.array(batch, dtype=int)
