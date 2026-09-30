# 作者: 晨星
"""data（合成 S1–S6 / 真实 R1–R4 / 划分协议）模块单测。"""

import numpy as np
import pytest

from metricforge.data.split import split_closed, split_open
from metricforge.data.synth import (
    SYNTH,
    concentric,
    gauss_blobs,
    hier_gauss,
    spirals,
    subspace,
    xor_grid,
)


@pytest.mark.parametrize("fn", [gauss_blobs, concentric, spirals, xor_grid, hier_gauss])
def test_synth_shapes(fn):
    X, y = fn(123, n_per_class=20)
    assert X.ndim == 2 and y.ndim == 1
    assert X.shape[0] == y.shape[0]
    assert len(np.unique(y)) >= 2


def test_subspace_highdim():
    X, y = subspace(44, n_per_class=20, d=40, C=8, k=4)
    assert X.shape == (160, 40)
    assert len(np.unique(y)) == 8


def test_synth_seeded_determinism():
    X1, y1 = gauss_blobs(11, n_per_class=30)
    X2, y2 = gauss_blobs(11, n_per_class=30)
    assert np.array_equal(X1, X2) and np.array_equal(y1, y2)


def test_split_closed_sizes():
    X, y = gauss_blobs(11, n_per_class=50, C=4)  # 共 200 样本
    tr, va, te = split_closed(X, y, 0)
    assert len(tr) == 120 and len(va) == 40 and len(te) == 40
    assert len(set(tr) | set(va) | set(te)) == 200


def test_split_open_disjoint_classes():
    X, y = gauss_blobs(22, n_per_class=30, C=6)
    tr, va, te = split_open(X, y, 0)
    tr_c = set(y[tr].tolist()) | set(y[va].tolist())
    te_c = set(y[te].tolist())
    assert tr_c.isdisjoint(te_c), "open-set：训练类与测试类必须不相交"


def test_synth_registry_keys():
    assert set(SYNTH) == {
        "s1_gauss_blobs",
        "s2_concentric",
        "s3_spirals",
        "s4_subspace",
        "s5_xor_grid",
        "s6_hier_gauss",
    }
