#!/usr/bin/env python
"""验证 metric-learn 0.7.0 与 scikit-learn>=1.8 的兼容 shim 是否可行.
metric-learn 最后一次发布是 2021 年(0.7.0), 仍在使用已被 sklearn 1.8 移除的
`force_all_finite` 关键字。本 shim 在 import metric_learn 之前把调用翻译成
新的 `ensure_all_finite`。
作者: 晨星
"""

import functools
import inspect

import sklearn
import sklearn.utils
import sklearn.utils.validation


def _patch(old_name="force_all_finite", new_name="ensure_all_finite"):
    for mod, fname in (
        (sklearn.utils, "check_array"),
        (sklearn.utils.validation, "check_X_y"),
        (sklearn.utils.validation, "check_array"),
    ):
        orig = getattr(mod, fname, None)
        if orig is None:
            continue
        params = inspect.signature(orig).parameters
        if new_name not in params or old_name in params:
            continue  # 该 sklearn 版本不需要 shim

        @functools.wraps(orig)
        def wrapper(*args, _orig=orig, **kwargs):
            if old_name in kwargs:
                kwargs[new_name] = kwargs.pop(old_name)
            return _orig(*args, **kwargs)

        setattr(mod, fname, wrapper)
        print(f"patched {mod.__name__}.{fname}: {old_name} -> {new_name}")


_patch()

# --- shim 之后必须能 import 并真跑 ---
from metric_learn import (
    LFDA,
    LMNN,
    NCA,
    ITML_Supervised,
    MMC_Supervised,
    RCA_Supervised,
)
from sklearn.datasets import make_classification

X, y = make_classification(
    n_samples=120, n_features=12, n_classes=4, n_informative=8, random_state=0
)
print(f"sklearn={sklearn.__version__}  data X={X.shape}")

for name, est in [
    ("LMNN", LMNN(n_neighbors=3, learn_rate=1e-6, max_iter=50, verbose=False)),
    ("NCA", NCA(max_iter=50)),
    ("ITML_Supervised", ITML_Supervised(max_iter=5)),
    ("LFDA", LFDA(n_components=6, k=2)),
    ("RCA_Supervised", RCA_Supervised(n_components=6)),
    ("MMC_Supervised", MMC_Supervised(max_iter=5)),
]:
    try:
        est.fit(X, y)
        out = est.transform(X[:5])
        extra = ""
        if hasattr(est, "get_mahalanobis_matrix"):
            extra = f" M={est.get_mahalanobis_matrix().shape}"
        print(f"[metric-learn] {name:<16} OK  transform->{out.shape}{extra}")
    except Exception as e:  # noqa: BLE001
        print(f"[metric-learn] {name:<16} FAIL {type(e).__name__}: {e}")

print("DONE")
