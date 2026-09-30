#!/usr/bin/env python
"""
MetricForge Phase 0 - 轮子可用性先验探针
作者: 晨星

用途: 在 PyPI JSON API 上查询指定包是否存在 cp313 / win_amd64 预编译 wheel,
      避免"写完代码才发现装不上"。
不做任何下载, 只做元数据查询。
"""

import json
import sys
import urllib.request

PYPI = "https://pypi.org/pypi/{name}/json"
TARGET_TAG = "cp313"
TARGET_PLAT = "win_amd64"
ABIS = ("cp313", "cp313t", "abi3", "none")

PKGS = [
    "numpy",
    "scipy",
    "scikit-learn",
    "pandas",
    "metric-learn",
    "pytorch-metric-learning",
    "torch",
    "faiss-cpu",
    "umap-learn",
    "hdbscan",
    "matplotlib",
    "pytest",
    "ruff",
]


def classify(fname: str):
    """解析 wheel 文件名, 返回 (平台串, python tag, abi tag)"""
    if not fname.endswith(".whl"):
        return None
    stem = fname[:-4]
    parts = stem.split("-")
    if len(parts) < 5:
        return None
    pytag, abitag, plattag = parts[-3], parts[-2], parts[-1]
    return plattag, pytag, abitag


def plat_ok(plattag: str) -> bool:
    tags = plattag.split(".")
    return TARGET_PLAT in tags or "any" in tags


def py_ok(pytag: str, abitag: str) -> bool:
    pytags = pytag.split(".")
    if abitag == "abi3":
        # abi3 兼容, 只要 py 主版本 <= 3.13 即可
        for t in pytags:
            if t.startswith("cp3"):
                try:
                    minor = int(t[3:])
                except ValueError:
                    continue
                if minor <= 13:
                    return True
        return False
    if abitag in ("none", "none"):
        return any(t in ("py3", "py2.py3", "py3.13") for t in pytags)
    return TARGET_TAG in pytags or "cp313t" in pytags


def vkey(v: str):
    """自包含版本号排序键 (不依赖 packaging). 支持 a/b/rc/dev/post."""
    import re

    s = v.split("+")[0]
    m = re.match(r"^(\d+(?:\.\d+)*)", s)
    nums = tuple(int(x) for x in m.group(1).split(".")) if m else (0,)
    rest = s[m.end() :] if m else s
    low = rest.lower()
    if ".dev" in low or low.startswith("dev"):
        kind = -4
    else:
        mm = re.match(r"^\.?(a|b|c|rc)", low)
        if mm:
            kind = {"a": -3, "b": -2, "c": -1, "rc": -1}[mm.group(1)]
        elif low in ("", ".", "-"):
            kind = 0
        else:
            kind = 1  # post / 其它
    extra = tuple(int(x) for x in re.findall(r"\d+", rest))
    return (nums, kind, extra)


def is_stable(v: str) -> bool:
    import re

    return not re.search(r"(a|b|c|rc|dev)\d*", v.split("+")[0].lower())


def probe(name: str) -> dict:
    url = PYPI.format(name=name)
    try:
        with urllib.request.urlopen(url, timeout=30) as r:
            data = json.load(r)
    except Exception as e:  # noqa: BLE001
        return {"name": name, "error": f"{type(e).__name__}: {e}"}

    latest = data["info"]["version"]
    releases = data["releases"]
    hit = None  # 最新(含预发布)命中 cp313-win_amd64 的版本
    hit_stable = None  # 最新稳定版命中 cp313-win_amd64 的版本
    any_ver = None  # 最新命中任意 win_amd64 的版本
    pure = False  # 是否存在纯 python wheel
    sdist = False

    ordered = sorted([v for v in releases if v], key=vkey, reverse=True)

    for ver in ordered:
        files = releases.get(ver) or []
        for f in files:
            fn = f.get("filename", "")
            if fn.endswith(".tar.gz") or fn.endswith(".zip"):
                if fn.endswith(".tar.gz"):
                    sdist = True
                continue
            parsed = classify(fn)
            if not parsed:
                continue
            plattag, pytag, abitag = parsed
            if plattag == "any":
                pure = True
                continue
            if not plat_ok(plattag):
                continue
            if any_ver is None:
                any_ver = (ver, fn)
            if py_ok(pytag, abitag):
                if hit is None:
                    hit = (ver, fn)
                if hit_stable is None and is_stable(ver):
                    hit_stable = (ver, fn)
        if hit_stable and any_ver and pure:
            break
        if hit_stable and any_ver and sdist:
            break

    return {
        "name": name,
        "latest": latest,
        "cp313_win_amd64": hit[0] if hit else None,
        "cp313_file": hit[1] if hit else None,
        "cp313_stable": hit_stable[0] if hit_stable else None,
        "cp313_stable_file": hit_stable[1] if hit_stable else None,
        "any_win_amd64": any_ver[0] if any_ver else None,
        "any_win_file": any_ver[1] if any_ver else None,
        "pure_py_wheel": pure,
        "has_sdist": sdist,
    }


def main():
    print(
        f"{'package':<26}{'latest':<11}{'cp313(any)':<13}{'cp313(stable)':<15}{'win-any':<11}{'pure':<6}{'sdist':<6}"
    )
    print("-" * 90)
    results = []
    for p in PKGS:
        r = probe(p)
        results.append(r)
        if "error" in r:
            print(f"{p:<26}ERROR {r['error']}")
            continue
        print(
            f"{p:<26}{r['latest']:<11}"
            f"{r['cp313_win_amd64']!s:<13}"
            f"{r['cp313_stable']!s:<15}"
            f"{r['any_win_amd64']!s:<11}"
            f"{'yes' if r['pure_py_wheel'] else '-':<6}"
            f"{'yes' if r['has_sdist'] else '-':<6}"
        )
    with open("tools/wheel_probe_result.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    return 0


if __name__ == "__main__":
    sys.exit(main())
