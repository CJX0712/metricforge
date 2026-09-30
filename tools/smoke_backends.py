#!/usr/bin/env python
"""MetricForge 后端冒烟: 真实跑一次度量学习损失/变换/检索. 作者: 晨星

用法: python tools/smoke_backends.py
前置: envs/metricforge 已按 requirements-lock.txt 装好依赖。
注意: metric-learn 0.7.0 需要 compat shim 才能在 sklearn>=1.8 上工作,
      本脚本在 import metric_learn 之前安装该 shim。
"""

import functools
import inspect
import sys

import numpy as np
import sklearn
import sklearn.utils
import sklearn.utils.validation
from sklearn.datasets import make_classification


def install_metric_learn_shim():
    """metric-learn 0.7.0 (2021, 已停止维护) 仍使用 sklearn 1.8 移除的
    `force_all_finite` 关键字。翻译成新的 `ensure_all_finite`。"""
    patched = []
    for mod, fname in (
        (sklearn.utils, "check_array"),
        (sklearn.utils.validation, "check_array"),
        (sklearn.utils.validation, "check_X_y"),
    ):
        orig = getattr(mod, fname, None)
        if orig is None:
            continue
        params = inspect.signature(orig).parameters
        if "ensure_all_finite" not in params or "force_all_finite" in params:
            continue

        @functools.wraps(orig)
        def wrapper(*args, _orig=orig, **kwargs):
            if "force_all_finite" in kwargs:
                kwargs["ensure_all_finite"] = kwargs.pop("force_all_finite")
            return _orig(*args, **kwargs)

        setattr(mod, fname, wrapper)
        patched.append(f"{mod.__name__}.{fname}")
    return patched


shimmed = install_metric_learn_shim()
print(f"shim: {len(shimmed)} 处 patch" + (f" -> {shimmed}" if shimmed else " (无需)"))

X, y = make_classification(
    n_samples=120, n_features=12, n_classes=4, n_informative=8, random_state=0
)
print(f"data: X={X.shape}, y={y.shape}, classes={np.unique(y).tolist()}")
print("-" * 68)

# ---- Tier-2: metric-learn 经典度量学习 ----
from metric_learn import (
    LFDA,
    LMNN,
    NCA,
    ITML_Supervised,
    MMC_Supervised,
)

for name, est in [
    ("LMNN", LMNN(n_neighbors=3, learn_rate=1e-6, max_iter=50, verbose=False)),
    ("NCA", NCA(max_iter=50)),
    ("ITML_Supervised", ITML_Supervised(max_iter=5)),
    ("LFDA", LFDA(n_components=6, k=2)),
    ("MMC_Supervised", MMC_Supervised(max_iter=5)),
]:
    try:
        est.fit(X, y)
        out = est.transform(X[:5])
        extra = ""
        if hasattr(est, "get_mahalanobis_matrix"):
            extra = f" M={est.get_mahalanobis_matrix().shape}"
        print(f"[metric-learn] {name:<17} OK   transform->{out.shape}{extra}")
    except Exception as e:  # noqa: BLE001
        print(f"[metric-learn] {name:<17} FAIL {type(e).__name__}: {e}")

print("-" * 68)

# ---- Tier-3: pytorch-metric-learning 深度度量学习 ----
import torch
from pytorch_metric_learning import losses, miners

print(
    f"[torch] version={torch.__version__} cuda={torch.cuda.is_available()} "
    f"threads={torch.get_num_threads()}"
)
emb = torch.randn(64, 16, requires_grad=True)
labels = torch.randint(0, 5, (64,))
for name, fn in [
    ("SupConLoss", losses.SupConLoss(temperature=0.07)),
    ("NTXentLoss", losses.NTXentLoss(temperature=0.07)),
    ("TripletMarginLoss", losses.TripletMarginLoss(margin=0.1)),
    ("ContrastiveLoss", losses.ContrastiveLoss(pos_margin=0, neg_margin=1)),
    ("ProxyAnchorLoss", losses.ProxyAnchorLoss(num_classes=5, embedding_size=16)),
    ("ProxyNCALoss", losses.ProxyNCALoss(num_classes=5, embedding_size=16)),
]:
    try:
        loss = fn(emb, labels)
        print(f"[PML] {name:<18} OK   loss={float(loss.detach()):.6f}")
    except Exception as e:  # noqa: BLE001
        print(f"[PML] {name:<18} FAIL {type(e).__name__}: {e}")

m = miners.TripletMarginMiner(margin=0.1, type_of_triplets="semihard")
a, p, n = m(emb, labels)
print(
    f"[PML] TripletMarginMiner    OK   mined a={tuple(a.shape)} "
    f"p={tuple(p.shape)} n={tuple(n.shape)}"
)

print("-" * 68)

# ---- 检索 / 聚类 / 绘图 ----
try:
    import faiss

    idx = faiss.IndexFlatL2(12)
    idx.add(np.ascontiguousarray(X, dtype="float32"))
    D, I = idx.search(np.ascontiguousarray(X[:3], dtype="float32"), 5)
    print(
        f"[faiss] version={faiss.__version__} ntotal={idx.ntotal} "
        f"search->{I.shape} d0={[round(float(v), 4) for v in D[0][:3]]}"
    )
except Exception as e:  # noqa: BLE001
    print(f"[faiss] FAIL {type(e).__name__}: {e}")

try:
    import umap

    Z = umap.UMAP(n_components=2, n_neighbors=10, random_state=0).fit_transform(X)
    print(f"[umap] version={umap.__version__} embed->{Z.shape}")
except Exception as e:  # noqa: BLE001
    print(f"[umap] FAIL {type(e).__name__}: {e}")

try:
    import importlib.metadata as _md

    import hdbscan

    cl = hdbscan.HDBSCAN(min_cluster_size=5).fit(X)
    print(
        f"[hdbscan] version={_md.version('hdbscan')} "
        f"labels->{(cl.labels_ >= 0).sum()}/{len(cl.labels_)} 已聚类"
    )
except Exception as e:  # noqa: BLE001
    print(f"[hdbscan] FAIL {type(e).__name__}: {e}")

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(2, 2))
    ax.scatter(X[:, 0], X[:, 1], s=4)
    fig.savefig("tmp_pipprobe/_render_probe.png", dpi=60)
    plt.close(fig)
    print(f"[matplotlib] version={matplotlib.__version__} Agg 渲染 OK")
except Exception as e:  # noqa: BLE001
    print(f"[matplotlib] FAIL {type(e).__name__}: {e}")

print("-" * 68)
print("DONE")
sys.exit(0)
