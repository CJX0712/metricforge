# 作者: 晨星
"""唯一确定性入口。

全局纪律（docs/algorithm_design.md §3 / §9 硬约束 3）：
  * 所有随机性必须走本模块，禁止裸 np.random.default_rng() 无参调用。
  * 唯一 seed 入口：set_seed(s) 一次设齐 Python stdlib random 与 numpy Generator。
  * 同 seed 两次运行必须逐位一致（不变量 I10）。
"""

from __future__ import annotations

import random as _random

import numpy as np

_RNG: np.random.Generator | None = None
_SEED: int | None = None


def set_seed(seed: int) -> None:
    """唯一确定性入口。重复调用以同 seed 必须产生同序列。"""
    global _RNG, _SEED
    _SEED = int(seed)
    _random.seed(_SEED)
    _RNG = np.random.default_rng(_SEED)


def get_seed() -> int | None:
    return _SEED


def rng() -> np.random.Generator:
    """返回全局确定性 Generator；未初始化时以 0 兜底（不鼓励隐式调用）。"""
    if _RNG is None:
        set_seed(0)
    return _RNG


def randn(*shape) -> np.ndarray:
    return rng().standard_normal(shape)


def uniform(*shape) -> np.ndarray:
    return rng().random(shape)


def permutation(n: int) -> np.ndarray:
    return rng().permutation(n)


def choice_indices(n: int, k: int, replace: bool = False) -> np.ndarray:
    return rng().choice(n, size=k, replace=replace)
