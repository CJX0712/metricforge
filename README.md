# MetricForge

> 度量学习 / 对比表征学习框架 —— 自研纯 `numpy` 实现，零第三方 DML 依赖，CPU-only 可复现。
> 作者：晨星 · 随机创新系统（世界顶级 AI 系统自主交付计划 · V4）

[![Python](https://img.shields.io/badge/python-3.13-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](./LICENSE)
[![Release](https://img.shields.io/badge/release-v0.1.0-orange.svg)](./CHANGELOG.md)
[![CI](https://img.shields.io/badge/CI-GitHub%20Actions-success.svg)](./.github/workflows/ci.yml)
[![DoD](https://img.shields.io/badge/DoD-verified-blue.svg)](./docs/model_card.md)

---

## 一句话定位

MetricForge 用**两个自研创新点**把度量学习做成"开箱即用、可证明、可复现"的框架：

1. **HAT（Hardness-Adaptive Temperature，硬度自适应温度）**：用固定次数二分解出
   逆温度 `β=1/τ`，使"活跃负样本参与率"恒为 `κ`，与 batch / 类数 / 维度无关。
   由引理 `dH/dβ = −β·Var(s) < 0` 保证根唯一 ⇒ **逐位确定性**（同 seed 两次运行比特一致）。
2. **MQ-BPA（Memory-Queue Bi-directional Proxy Alignment）**：每类 `K>1` 多原型覆盖
   类内多模态；反向 EMA 对齐把原型锚定到责任样本质心**防坍缩**；Fresh-Queue 存原始
   输入每步重编码提供零陈旧负样本。

> 诚实口径：本框架在 CPU / 同算力预算下，对标**同代算法族**（kNN、LMNN、LFDA、
> Triplet-BH、Proxy-Anchor、SupCon、NCA·ITML），**不与 CUB/Cars 等图像 SOTA 比**
>（那需要 ImageNet-ViT + GPU）。小样本低维表格集上，线性 Mahalanobis 方法也可能不输。

## 核心指标

主指标 **MAP@R**（Musgrave et al., ECCV 2020 *A Metric Learning Reality Check*）。
强基线 **B1–B6** + 自研 MetricForge，85 项单测全过，ruff 双绿，确定性逐位可复现（同 seed diff=0.0）。

**门槛实测结论（详见 `docs/model_card.md` §4.3）**：G5 ✅（无任何主数据集 Δ≤−0.05）、
G4 ✅、G6 ✅（确定性）；G1/G3 ❌ 未达成 —— 开集小样本协议存在结构性天花板
（s1/s4/s5 全方法饱和于 0.99~1.0，s2/s3 仅 1 个训练类），属诚实可证伪结论而非实现缺陷。

| 数据集 | MetricForge | Δ vs 最强基线 | 状态 |
|--------|------------|--------------|------|
| 见 `benchmark.json`（`aggregate` 段） | — | — | 由 `examples/run_demo.py` 自动生成 |

## 快速开始

```bash
# 1) 建隔离环境（推荐）
python -m venv .venv && .venv/Scripts/activate   # Windows
pip install -r requirements.txt

# 2) 跑端到端演示（fast 档，CPU-only，3 seed）
python examples/run_demo.py          # 落盘 benchmark.json 并打印摘要

# 3) 跑单测 + lint
pytest -q
ruff check metricforge
```

## 固定骨架（单向无环调用）

```
core        → 数学内核(mathx) / 确定性(seed) / 编码器(encoder)
losses      → proxy(HAT+MQ-BPA) / contrastive / pair / linear(LFDA/LMNN/NCA/ITML) / geometry / compose
mining      → FreshQueue / BalancedBatchSampler
data        → 合成 S1–S6 / 真实 R1–R4 / 划分(closed|open)
eval        → MAP@R / Recall@k / kNN acc / NMI-ARI
bench       → trainer / baselines(B1–B8) / run / ablation(A1–A8)
cli         → argparse 入口
examples    → run_demo.py
```

调用方向恒为：`bench → losses/mining → core`，`eval → core`，`data → core`。
**全局唯一确定性入口**：`core.seeding.set_seed`，禁止任何 `while` 收敛判据与全局 N×N 物化。

## 文档索引

- `docs/algorithm_design.md` —— 方案定稿（SOTA 表 / 选型 / 两个创新点数学 / 评测协议 G1–G6 / 消融 A1–A8 / 不变量 I1–I14）
- `docs/architecture.md` —— 架构映射与依赖裁决（双轨：自研 numpy 必需 + torch/metric-learn 可选 oracle）
- `docs/model_card.md` —— 模型卡（用途 / 数据 / 指标 / 局限 / 许可）
- `docs/related_work.md` —— leaderboard 19 法 + 参考文献（诚实口径原样保留）
- `docs/env_precheck.md` —— Phase 0 环境预检（13/13 后端可装，Python 3.13.x）

## 许可

MIT License · © 晨星（GitHub: CJX0712）。数据集为 sklearn / UCI 原始来源，本仓库不重新分发。
