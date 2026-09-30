# 作者: 晨星
"""core.encoder（MLP / Adam / L2）模块单测 + 梯度数值校验。"""

import numpy as np

from metricforge.core.encoder import MLPWithInput, l2_normalize
from metricforge.core.seeding import set_seed


def test_l2_normalize_unit_norm():
    X = np.random.default_rng(0).normal(size=(10, 5))
    Z = l2_normalize(X)
    norms = np.linalg.norm(Z, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-9)


def test_encoder_forward_shape_and_cache():
    set_seed(0)
    net = MLPWithInput(4, 16, 8)
    X = np.random.default_rng(1).normal(size=(7, 4))
    Z, cache = net.forward(X)
    assert Z.shape == (7, 8)
    # 输出已归一化
    assert np.allclose(np.linalg.norm(Z, axis=1), 1.0, atol=1e-9)


def test_encoder_backward_gradient_check():
    """数值梯度校验：L = sum(Z) 对输入 X 的梯度。"""
    set_seed(0)
    net = MLPWithInput(3, 8, 4)
    X = np.random.default_rng(2).normal(size=(1, 3))

    def loss_fn(Xv):
        net2 = MLPWithInput(3, 8, 4)
        net2.set_params(net.parameters())
        Z, _ = net2.forward(Xv)
        return float(np.sum(Z))

    # 解析梯度
    Z, _ = net.forward(X)
    grads, dX = net.backward(np.ones_like(Z))
    dX_analytic = dX[0].copy()

    # 数值梯度（中心差分）
    eps = 1e-5
    dX_num = np.zeros_like(X[0])
    for j in range(X.shape[1]):
        Xp = X.copy()
        Xp[0, j] += eps
        Xm = X.copy()
        Xm[0, j] -= eps
        dX_num[j] = (loss_fn(Xp) - loss_fn(Xm)) / (2 * eps)
    rel = np.max(np.abs(dX_analytic - dX_num) / (np.abs(dX_num) + 1e-8))
    assert rel < 1e-3, f"encoder 反向梯度数值校验失败，rel_err={rel}"
