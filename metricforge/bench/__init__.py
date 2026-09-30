# 作者: 晨星
"""bench 子包：基线、训练器、benchmark 编排、消融。"""

from __future__ import annotations

from .ablation import run_ablation
from .baselines import ALL_METHODS, build_model
from .run import aggregate, run_benchmark

__all__ = ["ALL_METHODS", "aggregate", "build_model", "run_ablation", "run_benchmark"]
