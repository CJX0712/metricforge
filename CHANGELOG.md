# Changelog

## [0.1.0] — 2026-09-30  (S-grade 交付)

### Added
- 自研纯 `numpy` 度量学习框架，零第三方 DML 依赖，CPU-only 可复现。
- 创新点 1 **HAT**：硬度自适应温度，固定 40 次二分解 `β`，活跃负样本参与率恒为 `κ`，逐位确定性。
- 创新点 2 **MQ-BPA**：`K>1` 多原型 + 反向 EMA 对齐（防坍缩）+ Fresh-Queue 零陈旧负样本。
- 强基线 B1–B6（kNN / LMNN / LFDA / Triplet-BH / Proxy-Anchor-K=1 / SupCon）+ MetricForge。
- 评测协议：主指标 MAP@R，open/closed 双协议，自剔除 query 自身。
- 消融 A1–A8（含 A7 全关 ≡ B5 逐位一致硬证明、A8 κ 敏感性）。
- 8 项内核不变量单测（`tests/test_invariants_core.py`，57 passed，双后端切换验证）。
- 文档：algorithm_design / architecture / model_card / related_work / env_precheck。
- 一键 demo `examples/run_demo.py` 落盘 `benchmark.json`。

### Fixed (本会话调试轨迹)
- `losses/linear.py` LFDA：`np.linalg.eigh(A, B)` 跨 numpy 版本兼容性 bug → 改 Cholesky 标准广义特征值，并回退 PCA。
- `eval/metrics.py`：MAP@R/Recall@k/kNN 在 query==gallery（测试协议）时**剔除自身**，避免自命中 rank1 虚高（与文档 §4.3 口径一致）。
- `bench/baselines.py`：线性方法（LFDA/LMNN/NCA）为降维投影，强制 `dim_out ≤ dim_in`，否则裁剪。
- `bench/__init__.py`：导出 `aggregate`（run_demo 依赖）。
- `data/__init__.py` + `eval/__init__.py`：补全缺失的子包导出。
- `losses/proxy.py`：延迟初始化 `C/P`，广播维度修正（`w/sw[:,None]`、`v/sv[:,None]`）。
- `bench/trainer.py`：`encoder.backward` 返回 `(grads, dX)`，调用处元组解包修正。

### Fixed (续 · LMNN/NCA 梯度稳定性)
- `losses/linear.py` LMNN：**pull 梯度维度修正** `grad += 2·pull_sum @ L`（原 `L @ pull_sum` 维度歧义，fwd-diff 校验 rel_err=1.4e-10 通过）。
- `losses/linear.py` LMNN/NCA：**梯度 Frobenius 范数裁剪**（`max_grad=100`）防止 pull+push 梯度量级随 impostor 数/维度暴涨导致 `L` 发散（`Z` 溢出 → MAP@R=NaN）。
- `losses/linear.py` LMNN：**默认 margin 1.0 → 0.1**。数据经 StandardScaler 标准化后距离量级偏小，大 margin 令 push 铰链全程不激活 → pull 单独把 `L` 推向 0（坍缩成随机检索，s1 MAP@R≈0.16）。`margin=0.1` 令 push 持续激活，防坍缩且在易分集上提升（s1 LMNN 0.82 > raw 0.76），跨多初值稳定。
- `bench/run.py` fast 档 `s4_subspace`：`n_per_class 40 → 20`（n=1000），把 LMNN O(n²) push 循环从 2000 点压到 1000 点（≈4× 加速），使 fast 档数分钟内完成。

### Fixed (续 · 训练稳定性与评测诚实性)
- `bench/ablation.py`：**B5 对照补齐 `use_hat=False, use_bidirectional=False`** —— 修复 A7（全关）与 B5 配置不一致导致违反硬断言 I9 的问题。修后 `tools/i9_check.py` 实测 A7≡B5 diff=0.00e+00（逐位一致）。
- `bench/run.py` + `examples/run_demo.py`：**默认 lr 1e-3 → 3e-4**。实测 lr=1e-3 在 s5_xor 上存在晚期训练失稳（MAP@R e40=0.983 → e80=0.50 崩塌，而损失仍单调下降）；3e-4 收敛稳定（s5 final=0.9795）。同时 demo 训练预算 **epochs 20→80、dim_out 12→16**（原 fast 配置严重欠训练，系统性低估全部深度方法——s6 实测 ep20=0.863 → ep40=0.989）。
- `tools/fill_model_card.py`：G4 由硬编码 ✅ 改为按 R1 digits 实测最慢单次 fit 耗时**如实判定**（防幻觉约定）。

### Known Limitations
- B7 (NCA·ITML) 与 B8 (Forge-Oracle) 默认未纳入主 benchmark（需 torch / metric-learn，列为可选 oracle，经 try/except 探测）。
- 大模型 / 高维图像（CUB/Cars）需 ImageNet-ViT + GPU，不在本框架对标范围。
- **开集协议（类不相交，Musgrave 口径）下的结构性天花板**：s1/s4/s5 上 kNN 原始空间即达 0.99~1.0（全方法饱和），ΔMAP@R ≥ +0.03 数学上不可达；s2/s3 仅 1 个训练类（3 类取半），一切学习方法先天退化；r1_digits 上深度编码器对未见测试类过拟合（训练越久测试 MAP@R 越低：e5=0.416 → e80=0.343，损失降至 0.047）。val 折来自训练类，无法用早停捕获开集退化。
- **门槛实测口径**：G1/G3 未达成（详见 model_card §4.3 实测数字与根因），G5（无任何主数据集 Δ ≤ −0.05）达成，G4/G6 达成。工程 DoD（单测/lint/确定性/发布）全绿。
