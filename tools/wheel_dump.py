#!/usr/bin/env python
"""dump 指定包/版本的 wheel 文件名, 用于核对探针结论. 作者: 晨星"""

import json
import sys
import urllib.request


def dump(name, versions):
    url = f"https://pypi.org/pypi/{name}/json"
    with urllib.request.urlopen(url, timeout=30) as r:
        data = json.load(r)
    rel = data["releases"]
    print(
        f"### {name}  (latest={data['info']['version']}, "
        f"requires_python={data['info'].get('requires_python')})"
    )
    for v in versions:
        files = rel.get(v)
        if files is None:
            print(f"  [{v}] <version not found>")
            continue
        names = [f["filename"] for f in files]
        win = [n for n in names if "win" in n and n.endswith(".whl")]
        others = [n for n in names if n not in win]
        print(
            f"  [{v}] total={len(names)} sdist={len([n for n in names if n.endswith('.tar.gz')])}"
        )
        for n in win:
            print(f"      WIN: {n}")
        if not win:
            for n in others[:6]:
                print(f"      OTH: {n}")
    print()


if __name__ == "__main__":
    args = sys.argv[1:]
    for spec in args:
        name, _, vers = spec.partition(":")
        dump(name, [x for x in vers.split(",") if x])
