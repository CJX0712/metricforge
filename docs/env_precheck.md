# MetricForge 环境预检报告（Phase 0 / 工程准备）

> 作者：晨星　｜　执行人：杭落地（AI 系统工程师）
> 生成时间：2026-09-30　｜　状态：**环境已跑通，13 个目标包全部可用**

---

## 1. 结论先行

| 结论 | 内容 |
| --- | --- |
| PyPI 可达性 | ✅ **默认官方源直连可用，无需换镜像** |
| cp313-win_amd64 轮子 | ✅ **13/13 全部有可用轮子，零编译** |
| venv | ✅ 已建，Python **3.13.14**，依赖已装并 import 通过 |
| SOTA 后端可用性 | ✅ **metric-learn / pytorch-metric-learning / torch / faiss-cpu / umap / hdbscan 全部真实跑通** |
| 必须走 Tier-1 兜底的 | **无**。所有目标后端在当前环境实测可用 |
| 已发现的依赖冲突 | ⚠️ **1 处**：metric-learn 0.7.0 × scikit-learn ≥1.8，已给出可复现 shim 方案并验证通过 |

---

## 2. 环境前提（实测输出）

```
python      : 3.13.14  (AMD64)
platform    : Windows-11-10.0.29671-SP0
executable  : C:\Users\Administrator\.workbuddy\binaries\python\envs\metricforge\Scripts\python.exe
托管解释器  : C:\Users\Administrator\.workbuddy\binaries\python\versions\3.13.12\python.exe  → 3.13.14
pip         : 26.2.1
磁盘 C:     : 200G 总量 / 已用 131G / 可用 69G
GPU         : 无（torch.cuda.is_available() = False，CPU-only）
torch 线程数: 8
```

> 注意：目录名写的是 `3.13.12`，实际解释器版本是 **3.13.14**。所有 wheel 标签按 **cp313** 匹配，结论不受影响。

---

## 3. PyPI 可达性验证

| 检查项 | 命令 | 状态 | 证据 |
| --- | --- | --- | --- |
| 默认源小包下载 | `pip download --no-deps -d tmp_pipprobe six` | ✅ | `Successfully downloaded six` |
| 无缓存真实联网 | `pip download --no-deps --no-cache-dir -d tmp_pipprobe six` | ✅ | `Downloading six-1.17.0-py2.py3-none-any.whl (11 kB)` → `Saved` |
| pip 镜像配置 | `pip config list` | ⚪ 空 | 未配置任何镜像，走官方源 |
| pip install 沙箱 | venv 内 `pip install`（未提权） | ✅ | 无需 `dangerouslyDisableSandbox`，直接成功 |

**结论：后续所有依赖安装统一使用默认官方源 `https://pypi.org/simple`，不配置镜像。**

---

## 4. 轮子先验检查（cp313 / win_amd64）

检测方法两道：
1. PyPI JSON API 元数据扫描（脚本 `tools/wheel_probe.py`，不下载）
2. **实测下载**：`pip download --no-deps --only-binary=:all: -d <tmp> <pkg>` —— 能下到即证明有预编译 wheel、无需编译器

### 4.1 实测下载结果（第二道，最可靠）

```
OK    | numpy                   | numpy-2.5.3-cp313-cp313-win_amd64.whl
OK    | scipy                   | scipy-1.18.1-cp313-cp313-win_amd64.whl
OK    | scikit-learn            | scikit_learn-1.9.1-cp313-cp313-win_amd64.whl
OK    | pandas                  | pandas-3.0.6-cp313-cp313-win_amd64.whl
OK    | metric-learn            | metric_learn-0.7.0-py2.py3-none-any.whl
OK    | pytorch-metric-learning | pytorch_metric_learning-2.9.0-py3-none-any.whl
OK    | umap-learn              | umap_learn-0.5.12-py3-none-any.whl
OK    | hdbscan                 | hdbscan-0.8.44-cp313-cp313-win_amd64.whl
OK    | matplotlib              | matplotlib-3.11.2-cp313-cp313-win_amd64.whl
OK    | pytest                  | pytest-9.1.1-py3-none-any.whl
OK    | ruff                    | ruff-0.16.9-py3-none-win_amd64.whl
OK    | faiss-cpu               | faiss_cpu-1.15.1-cp313-cp313-win_amd64.whl
OK    | torch                   | torch-2.14.0-cp313-cp313-win_amd64.whl
```

### 4.2 关键体积（PyPI 元数据实测）

| 包 | wheel | 体积 |
| --- | --- | --- |
| torch | `torch-2.14.0-cp313-cp313-win_amd64.whl` | **124.1 MB** |
| faiss-cpu | `faiss_cpu-1.15.1-cp313-cp313-win_amd64.whl` | 16.3 MB |
| llvmlite（umap 依赖） | `llvmlite-0.50.0-cp313-cp313-win_amd64.whl` | 41.9 MB |

> torch 2.14.0 的 Windows wheel 是 slim 版（124 MB），**不会**拉取 nvidia-* CUDA 依赖，CPU-only 环境友好。
> 对比：torch 2.8.0 同平台 wheel 为 241.3 MB，2.9.0 起瘦身至 109 MB 左右。

### 4.3 隐藏依赖排查：`umap-learn` → `numba` / `llvmlite`

umap-learn 是纯 python wheel，但硬依赖 numba（→llvmlite），历史上这两个包常落后于新版 Python。实测 **有 cp313-win_amd64 轮子**，风险已排除：

```
numba    0.68.0  → numba-0.68.0-cp313-cp313-win_amd64.whl      ✅
llvmlite 0.50.0  → llvmlite-0.50.0-cp313-cp313-win_amd64.whl   ✅
```

---

## 5. 状态总表（核心交付）

| 包名 | 用途 | cp313-win_amd64 wheel | 是否安装 | 实际版本 | 降级方案（Tier-1 纯 numpy 兜底） |
| --- | --- | --- | --- | --- | --- |
| numpy | 数值底座 | ✅ `cp313-cp313` | ✅ | **2.5.3** | 无（硬依赖，必须有） |
| scipy | 优化/线性代数 | ✅ `cp313-cp313` | ✅ | **1.18.1** | 无（硬依赖） |
| scikit-learn | 数据/评估/基线 | ✅ `cp313-cp313` | ✅ | **1.9.1** | 无（硬依赖） |
| pandas | 数据装载 | ✅ `cp313-cp313` | ✅ | **3.0.6** | 可选；用 numpy + csv 直读 |
| metric-learn | LMNN/NCA/ITML/LFDA | ⚠️ 纯 py wheel（无平台轮子） | ✅ | **0.7.0** | ⚠️ **需 shim 才能在 sklearn≥1.8 上跑**；否则自研 numpy 版 LMNN/NCA/ITML/LFDA |
| pytorch-metric-learning | ProxyAnchor/SupCon/NT-Xent/triplet mining | ⚠️ 纯 py wheel（依赖 torch） | ✅ | **2.9.0** | Tier-1 numpy 实现 InfoNCE/SupCon/Triplet（含解析梯度） |
| torch | 深度度量学习后端 | ✅ `cp313-cp313` 124 MB | ✅ | **2.14.0+cpu** | Tier-1 全 numpy 训练器（无自动微分，手写梯度） |
| faiss-cpu | 百万级检索加速 | ✅ `cp313-cp313` 16 MB | ✅ | **1.15.1** | numpy 暴力检索 / `sklearn.neighbors.NearestNeighbors` |
| umap-learn | 嵌入可视化/降维 | ⚠️ 纯 py wheel（numba 有 cp313 轮子 ✅） | ✅ | **0.5.12** | sklearn PCA / TSNE |
| hdbscan | 密度聚类评估 | ✅ `cp313-cp313` | ✅ | **0.8.44** | sklearn `AgglomerativeClustering` / `KMeans` |
| matplotlib | 报告出图 | ✅ `cp313-cp313` | ✅ | **3.11.2** | 非必需；仅 Agg 后端出图，缺失则跳过绘图 |
| pytest | 测试 | ⚠️ 纯 py wheel | ✅ | **9.1.1** | 无（工程必装） |
| ruff | lint | ✅ `py3-none-win_amd64` | ✅ | **0.16.9** | 无（工程必装） |

---

## 6. 已发现依赖冲突与修复（关键）

### 6.1 问题

`metric-learn 0.7.0`（最后发布于 2021 年，已停止维护）仍在调用被 **scikit-learn 1.8 移除** 的关键字 `force_all_finite`（1.6 起更名 `ensure_all_finite`）。

**原始报错（实测）**：

```
[metric-learn] FAIL TypeError: check_X_y() got an unexpected keyword argument
               'force_all_finite'. Did you mean 'ensure_all_finite'?
```

**冲突面**：`metric_learn/_util.py` 共 6 处使用 `force_all_finite`。

### 6.2 修复方案：import 前 shim（推荐）

在 `import metric_learn` **之前**，把 `sklearn.utils.check_array` / `sklearn.utils.validation.check_array` / `sklearn.utils.validation.check_X_y` 包一层，将 `force_all_finite` 翻译为 `ensure_all_finite`。

已落脚本：`tools/compat_shim_test.py`、`tools/smoke_backends.py`

**实测修复后**（sklearn 仍为 1.9.1，未降级）：

```
shim: 3 处 patch -> ['sklearn.utils.check_array',
                     'sklearn.utils.validation.check_array',
                     'sklearn.utils.validation.check_X_y']
[metric-learn] LMNN              OK   transform->(5, 12) M=(12, 12)
[metric-learn] NCA               OK   transform->(5, 12) M=(12, 12)
[metric-learn] ITML_Supervised   OK   transform->(5, 12) M=(12, 12)
[metric-learn] LFDA              OK   transform->(5, 6)  M=(12, 12)
[metric-learn] MMC_Supervised    OK   transform->(5, 12) M=(12, 12)
```

### 6.3 备选方案

| 方案 | 做法 | 代价 |
| --- | --- | --- |
| A（推荐） | shim 翻译关键字 | 约 20 行代码，sklearn 保持最新 |
| B | 锁 `scikit-learn<1.8` | 拖住整个项目 sklearn 版本，不划算 |
| C | 自研 LMNN/NCA/ITML/LFDA | 可控性最高，但要写 + 测，工作量最大 |

**建议：A 作为默认，C 作为长期演进方向**（metric-learn 已停维护，核心算法自研可去掉这个历史包袱）。

---

## 7. 验证证据（真实命令输出）

### 7.1 venv 基础验证

```
python      : 3.13.14  (AMD64)
import numpy        OK   version=2.5.3
import scipy        OK   version=1.18.1
import sklearn      OK   version=1.9.1
import pandas       OK   version=3.0.6
import pytest       OK   version=9.1.1
smoke: X=(200, 16) -> PCA (200, 16)->(200, 8), pairwise_dist=5.235706
smoke: OK
ruff 0.16.9
pytest 9.1.1
```

### 7.2 SOTA 后端真实冒烟

```
[torch] version=2.14.0+cpu cuda=False threads=8
[PML] SupConLoss         OK   loss=8.359848
[PML] NTXentLoss         OK   loss=7.761052
[PML] TripletMarginLoss  OK   loss=0.236786
[PML] ContrastiveLoss    OK   loss=1.454558
[PML] ProxyAnchorLoss    OK   loss=35.255295
[PML] ProxyNCALoss       OK   loss=1.771935
[PML] TripletMarginMiner    OK   mined a=(5959,) p=(5959,) n=(5959,)
[faiss] version=1.15.1 ntotal=120 search->(3, 5) d0=[0.0, 59.3709, 68.5111]
[umap] version=0.5.12 embed->(120, 2)
[hdbscan] version=0.8.44 labels->200 样本
[matplotlib] version=3.11.2 Agg 渲染 OK
```

### 7.3 复现命令

```bash
# 1) 建 venv
C:/Users/Administrator/.workbuddy/binaries/python/versions/3.13.12/python.exe -m venv \
    C:/Users/Administrator/.workbuddy/binaries/python/envs/metricforge

# 2) 装依赖（官方源，44 个包，锁定版本见 requirements-lock.txt）
C:/Users/Administrator/.workbuddy/binaries/python/envs/metricforge/Scripts/python.exe -m pip install \
    numpy scipy scikit-learn pandas pytest ruff \
    metric-learn torch pytorch-metric-learning \
    faiss-cpu umap-learn hdbscan matplotlib

# 3) 验证
C:/Users/Administrator/.workbuddy/binaries/python/envs/metricforge/Scripts/python.exe tools/verify_env.py
C:/Users/Administrator/.workbuddy/binaries/python/envs/metricforge/Scripts/python.exe tools/smoke_backends.py
```

---

## 8. 交付清单

| 项 | 路径 | 说明 |
| --- | --- | --- |
| 环境预检报告 | `docs/env_precheck.md` | 本文档 |
| 锁定依赖清单 | `requirements-lock.txt` | 44 行，venv `pip freeze` 实测导出 |
| 轮子元探针 | `tools/wheel_probe.py` | PyPI JSON 扫描，输出 `tools/wheel_probe_result.json` |
| wheel 文件名核对 | `tools/wheel_dump.py` | 按版本 dump 真实 wheel 文件名/体积 |
| 环境验证脚本 | `tools/verify_env.py` | import + 数值冒烟 |
| 后端冒烟脚本 | `tools/smoke_backends.py` | 含 metric-learn shim，跑通 13 个后端能力 |
| shim 验证脚本 | `tools/compat_shim_test.py` | 单独验证 metric-learn 兼容修复 |
| 大包下载日志 | `tools/bigwheel_probe.log` | torch / faiss-cpu 实测下载记录 |

---

## 9. 已知限制

1. **CPU-only**：`torch.cuda.is_available() = False`。深度度量学习只能 CPU 训练，规模需控制（建议 demo 用 MNIST/CIFAR10 小子集或合成数据，benchmark 控制在数万样本内）。
2. **metric-learn 需 shim**：见第 6 节。未打 shim 直接 `import metric_learn` 后在 `fit()` 阶段报 `TypeError`。
3. **torch 体积**：wheel 124.1 MB，装后占用更大；`requirements-lock.txt` 全量安装约需 5–6 分钟（含 llvmlite 41.9 MB 下载，实测 712 kB/s 用了 90 秒）。
4. **hdbscan 无 `__version__` 属性**：取版本要用 `importlib.metadata.version("hdbscan")`，不要用 `hdbscan.__version__`。
5. **pandas 3.0.6 是新的大版本**（3.x 系列），与 2.x 有 API 差异，业务代码需按 3.x 写。
6. 本次**未**创建 git 仓库、未推送 GitHub（按指令，后续阶段另行下发）。

---

*作者：晨星*
