# 作者: 晨星
"""oracle.py —— 可选交叉验证后端（非运行时必需，全部 try/except 探测）。

  * metric-learn + shim：验证自研 LMNN/LFDA/NCA/ITML 的正确性（数值对照）。
  * torch + pytorch-metric-learning：手写梯度 vs autograd 对照（不变量 I11）。

两者均缺失或 shim 失效 → 系统正常降级，不阻断主链路。
"""

from __future__ import annotations

import functools


def _patch_sklearn():
    import sklearn.utils as su
    import sklearn.utils.validation as suv

    patched = False
    for mod in (su, suv):
        for name in ("check_array", "check_X_y"):
            fn = getattr(mod, name, None)
            if fn is None:
                continue
            if getattr(fn, "_mf_patched", False):
                continue

            @functools.wraps(fn)
            def _w(*a, _fn=fn, **k):
                if "force_all_finite" in k:
                    k["ensure_all_finite"] = k.pop("force_all_finite")
                return _fn(*a, **k)

            _w._mf_patched = True
            setattr(mod, name, _w)
            patched = True
    return patched


def available_metric_learn() -> bool:
    try:
        _patch_sklearn()
        import metric_learn  # noqa: F401

        return True
    except Exception:
        return False


def metric_learn_lmnn(X, y, n_components=None):
    if not available_metric_learn():
        return None
    from metric_learn import LMNN as ML_LMNN

    n = n_components or max(1, X.shape[1] // 2)
    est = ML_LMNN(k=3, n_components=n, random_state=0)
    est.fit(X, y)
    return est


def available_torch() -> bool:
    try:
        import torch  # noqa: F401

        return True
    except Exception:
        return False
