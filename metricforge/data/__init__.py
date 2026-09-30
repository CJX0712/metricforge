# 作者: 晨星
"""data —— 数据集生成与加载（合成 S1–S6 / 真实 R1–R4 / 划分协议）。"""

from . import real, synth
from .split import split_closed, split_open

__all__ = ["real", "split_closed", "split_open", "synth"]
