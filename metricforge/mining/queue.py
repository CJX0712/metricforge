# 作者: 晨星
"""mining/queue.py —— Fresh-Queue（存 raw X，每步重编码，零陈旧偏差）。

传统 MoCo/XBM 存历史嵌入（momentum 编码器产生，有陈旧偏差）。
MQ-BPA 增量：队列存**原始输入**，每步用当前 encoder 重编码 → 完全消除陈旧偏差，
且无需 momentum encoder。提供额外负样本池增强对比学习的负样本多样性。
"""

from __future__ import annotations

import numpy as np


class FreshQueue:
    def __init__(self, capacity: int = 1024):
        self.cap = int(capacity)
        self._buf = []

    @property
    def size(self):
        return len(self._buf)

    def update(self, X_batch: np.ndarray):
        """存入原始输入（不是嵌入）。保持队列长度 = cap。"""
        X = np.asarray(X_batch, dtype=np.float64)
        self._buf.extend(list(X))
        if len(self._buf) > self.cap:
            self._buf = self._buf[-self.cap :]

    def get(self) -> list[np.ndarray]:
        return list(self._buf)

    def encode(self, encoder):
        """用当前 encoder 重编码队列中的 raw X，返回嵌入 (M, D)。"""
        if not self._buf:
            return None
        X = np.asarray(self._buf, dtype=np.float64)
        Z, _ = encoder.forward(X)
        return Z

    def augment_negatives(self, Z_query, encoder):
        """返回与 Z_query 拼接的额外负样本嵌入（Fresh-Queue 重编码）。"""
        Zq = np.asarray(Z_query, dtype=np.float64)
        Zneg = self.encode(encoder)
        if Zneg is None:
            return Zq
        return np.vstack([Zq, Zneg])
