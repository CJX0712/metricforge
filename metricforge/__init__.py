# 作者: 晨星
"""MetricForge —— 度量学习 / 对比表征学习系统。

作者署名：晨星 (CJKX0712)
随机抽中域：度量学习 Metric Learning / 对比损失（系统熵源抽签，entropy=e3cf4393）
命名风格：<Xxx>Forge

本包实现：
  * Proxy-Anchor 骨架 + 两个自研创新组件
      - HAT   : Hardness-Adaptive Temperature（硬度自适应温度）
      - MQ-BPA: Memory-Queue Bi-directional Proxy Alignment（多原型双向对齐 + Fresh-Queue）
  * 6 个强基线（欧氏kNN / 自研LMNN / LFDA / Triplet-BH / Proxy-Anchor(K=1) / SupCon）
  * 7 组消融（含"全关 ≡ Proxy-Anchor(K=1)"的硬证明）
  * 纯 numpy/scipy/sklearn 零下载必跑，torch 仅作可选梯度 oracle

核心不变量（I1~I14）见 docs/algorithm_design.md §7，单测见 tests/test_invariants_core.py。
"""

from __future__ import annotations

__version__ = "0.1.0"
__author__ = "晨星"
__license__ = "MIT"

from .core import seeding

__all__ = ["__author__", "__version__", "seeding"]
