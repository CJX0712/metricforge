#!/usr/bin/env python
"""MetricForge 环境验证脚本 - 作者: 晨星
逐项 import 核心依赖并做最小数值冒烟, 输出真实版本号。
"""

import importlib
import platform
import sys

CHECKS = [
    ("numpy", "np"),
    ("scipy", "sp"),
    ("sklearn", "sk"),
    ("pandas", "pd"),
    ("pytest", "pytest"),
]


def main():
    print(f"python      : {sys.version.split()[0]}  ({platform.machine()})")
    print(f"platform    : {platform.platform()}")
    print(f"executable  : {sys.executable}")
    print("-" * 62)
    ok = True
    for name, alias in CHECKS:
        try:
            m = importlib.import_module(name)
            ver = getattr(m, "__version__", "?")
            print(f"import {name:<12} OK   version={ver}")
        except Exception as e:  # noqa: BLE001
            ok = False
            print(f"import {name:<12} FAIL {type(e).__name__}: {e}")

    print("-" * 62)
    # 可选依赖探测(未装不是错误)
    for opt in [
        "metric_learn",
        "pytorch_metric_learning",
        "torch",
        "faiss",
        "umap",
        "hdbscan",
        "matplotlib",
    ]:
        try:
            m = importlib.import_module(opt)
            print(f"optional {opt:<22} present  version={getattr(m, '__version__', '?')}")
        except Exception:  # noqa: BLE001
            print(f"optional {opt:<22} absent")

    print("-" * 62)
    # 最小数值冒烟
    try:
        import numpy as np
        from sklearn.datasets import make_classification
        from sklearn.decomposition import PCA

        X, y = make_classification(
            n_samples=200, n_features=16, n_classes=3, n_informative=6, random_state=0
        )
        Z = PCA(n_components=8, random_state=0).fit_transform(X)
        d = np.linalg.norm(Z[0] - Z[1])
        print(f"smoke: X={X.shape} -> PCA {X.shape}->{Z.shape}, pairwise_dist={float(d):.6f}")
        assert X.shape == (200, 16) and Z.shape == (200, 8)
        print("smoke: OK")
    except Exception as e:  # noqa: BLE001
        ok = False
        print(f"smoke: FAIL {type(e).__name__}: {e}")

    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
