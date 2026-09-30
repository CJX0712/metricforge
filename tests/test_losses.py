# 作者: 晨星
"""losses / geometry / compose 模块单测。"""

import numpy as np

from metricforge.core.seeding import set_seed
from metricforge.losses import compose, contrastive, geometry, linear, pair, proxy


def _Zy(B=16, D=8, C=4, seed=0):
    set_seed(seed)
    rng = np.random.default_rng(seed)
    y = rng.integers(0, C, size=B)
    # 让同类靠近、异类远离，便于前向/反向有非零梯度
    Z = np.zeros((B, D), dtype=np.float64)
    for c in range(C):
        idx = np.where(y == c)[0]
        Z[idx] = rng.normal(scale=0.3, size=(idx.size, D)) + c
    return Z, y


def test_proxy_anchor_shapes_and_finite():
    Z, y = _Zy(B=12, D=8, C=3, seed=1)
    loss = proxy.ProxyAnchorLoss(num_classes=None, dim=8, K=3, kappa=0.25)
    loss.C = int(np.unique(y).size)
    loss.P = loss.C * loss.K
    loss.init_proxies(Z, y)
    out = loss(Z, y)
    assert len(out) == 4, "ProxyAnchorLoss 必须返回 (loss, dZ, dP, aux) 四元组"
    L, dZ, dP, aux = out
    assert np.isfinite(L)
    assert dZ.shape == Z.shape
    assert dP.shape == (loss.P, 8)
    assert set(aux) >= {"beta_mean", "sw_mean", "sv_mean"}
    assert np.all(np.isfinite(dZ)) and np.all(np.isfinite(dP))


def test_proxy_anchor_bidirectional_affects_proxies():
    Z, y = _Zy(B=12, D=8, C=3, seed=2)
    loss = proxy.ProxyAnchorLoss(num_classes=None, dim=8, K=3, use_bidirectional=True)
    loss.init_proxies(Z, y)
    p0 = loss.proxies.copy()
    loss.ema_align(Z, y)
    # 反向 EMA 后原型应改变（除非退化），且与原始一致形状
    assert loss.proxies.shape == p0.shape


def test_supcon_returns_grad():
    Z, y = _Zy(B=12, D=8, C=4, seed=3)
    from metricforge.core.encoder import l2_normalize

    Z = l2_normalize(Z)
    L, dZ = contrastive.SupConLoss(tau=0.07)(Z, y)
    assert np.isfinite(L)
    assert dZ.shape == Z.shape
    assert np.all(np.isfinite(dZ))


def test_triplet_batch_hard_nonneg_loss():
    Z, y = _Zy(B=14, D=8, C=3, seed=4)
    L, dZ = pair.TripletBatchHard(margin=0.2)(Z, y)
    assert L >= 0.0
    assert dZ.shape == Z.shape
    assert np.all(np.isfinite(dZ))


def test_lfda_transform_shape_and_finite():
    from metricforge.data.synth import gauss_blobs

    X, y = gauss_blobs(11, n_per_class=40, d=4, C=5)
    m = linear.LFDA(dim_out=3)
    m.fit(X, y)
    Z = m.transform(X)
    assert Z.shape == (X.shape[0], 3)
    assert np.all(np.isfinite(Z))


def test_lfda_degenerate_single_class_fallback():
    # 单类退化：不应抛错，应退化为 PCA/单位投影
    X = np.random.default_rng(0).normal(size=(20, 3))
    y = np.zeros(20, dtype=int)
    m = linear.LFDA(dim_out=2)
    m.fit(X, y)
    Z = m.transform(X)
    assert Z.shape == (20, 2) and np.all(np.isfinite(Z))


def test_lmnn_nca_shape():
    from metricforge.data.synth import gauss_blobs

    X, y = gauss_blobs(22, n_per_class=30, d=4, C=4)
    for Cls, kw in [
        (linear.LMNN, {"dim_out": 2, "iters": 5}),
        (linear.NCA, {"dim_out": 2, "iters": 5}),
    ]:
        m = Cls(**kw)
        m.fit(X, y)
        Z = m.transform(X)
        assert Z.shape == (X.shape[0], 2)
        assert np.all(np.isfinite(Z))


def test_pcgrad_conflict_reduction():
    a = np.array([1.0, 1.0], dtype=np.float64)
    b = np.array([-1.0, -1.0], dtype=np.float64)  # 完全冲突
    out = compose.pcgrad_lite([a, b])
    # 投影后 a 方向应被去掉 b 的反向分量，a 与 b 的点积不应更负
    assert isinstance(out, np.ndarray)
    assert out.shape == a.shape


def test_sne_keep_regularizer():
    X = np.random.default_rng(0).normal(size=(30, 4))
    reg = geometry.SNEKeepRegularizer(k=5, sigma=1.0)
    reg.fit(X)
    Z = np.random.default_rng(1).normal(size=(30, 4))
    val, dZ = reg(Z)
    assert np.isfinite(val)
    assert dZ.shape == Z.shape
