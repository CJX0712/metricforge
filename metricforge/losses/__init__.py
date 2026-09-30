# 作者: 晨星
"""losses 子包：对比/三元组/代理/线性度量学习损失 + 几何正则 + 组合。"""

from __future__ import annotations

from . import contrastive, geometry, linear, pair, proxy

__all__ = ["contrastive", "geometry", "linear", "pair", "proxy"]
