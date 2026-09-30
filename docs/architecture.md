# MetricForge 架构文档

> 作者：晨星 · 随机创新系统交付（V4 计划）
> 固定骨架：单向无环调用 + 全局唯一确定性入口

## 1. 目录结构（固定骨架）

```
metricforge/
├── __init__.py            # 顶层导出（版本 / 子包聚合）
├── core/                  # 数学内核 + 确定性 + 编码器（不依赖任何上层）
│   ├── mathx.py           # 8 个锁定符号：logsumexp / softmax_entropy(_beta) /
│   │                       #   hat_temperature / map_at_r(_exact) / triplet_hinge / infonce_loss
│   ├── seeding.py         # 全局唯一确定性入口 set_seed / rng / randn
│   └── encoder.py         # 纯 numpy MLP + Adam + l2_normalize（Tier-1 路径）
├── losses/                # 损失层（依赖 core）
│   ├── proxy.py           # ProxyAnchorLoss + HAT + MQ-BPA（自研创新）
│   ├── contrastive.py     # SupConLoss / InfoNCELoss
│   ├── pair.py            # TripletBatchHard（B4）
│   ├── linear.py          # LFDA / LMNN / NCA / ITML（B2/B3/B7，零第三方 DML）
│   ├── geometry.py        # SNE-KL 邻域保持正则（A4 可选）
│   └── compose.py         # pcgrad_lite 梯度去冲突（NeurIPS 2020，非自研）
├── mining/                # 采样 / 队列（依赖 core）
│   ├── queue.py           # FreshQueue（存 raw X，每步重编码，零陈旧）
│   └── sampler.py         # BalancedBatchSampler（类均衡）
├── data/                  # 数据集（依赖 core）
│   ├── synth.py           # S1–S6 合成生成器（seed 可控）
│   ├── real.py            # R1–R4 sklearn 真实集
│   └── split.py          # split_closed / split_open（open-set 类不相交）
├── eval/                  # 评测指标（依赖 core）
│   └── metrics.py         # map_at_r_score / recall_at_k / knn_accuracy / nmi_ari
├── bench/                 # 编排（依赖 losses/mining/data/eval）
│   ├── trainer.py         # DeepTrainer（MLP + 可插拔损失 + Adam）
│   ├── baselines.py       # build_model / ALL_METHODS / IdentityModel / LinearWrapper
│   ├── run.py             # run_benchmark / aggregate
│   └── ablation.py        # run_ablation（A1–A8）
├── cli.py                 # argparse 入口
└── oracle.py              # 可选后端探测（metric-learn / torch，try/except 降级）

examples/
└── run_demo.py            # 端到端演示，落盘 benchmark.json
tests/                     # 57 不变量单测 + 28 模块单测（共 85，全绿）
docs/                     # algorithm_design / architecture / model_card / related_work / env_precheck
```

## 2. 调用依赖方向（单向无环）

```
bench ──▶ losses ──▶ core
  │         │
  ├──▶ mining ─▶ core
  ├──▶ data ──▶ core
  └──▶ eval ───▶ core

bench.Trainer 内部：encoder ← core.encoder；loss_fn ← losses.*；queue ← mining.FreshQueue
eval.metrics ← core.mathx（map_at_r）、core.encoder（l2_normalize）
data.split ← core.seeding（default_rng，独立 seed，不污染全局）
```

**硬约束（文档 §9）**
- 禁止物化全局 N×N 距离矩阵：kNN / MAP@R 用分块矩阵乘 `Zq @ Zg.T` 即时计算。
- `hat_temperature` 用固定 40 次二分（无 `while` 收敛判据）→ 同 seed 逐位确定性（不变量 I10）。
- 全局唯一 `set_seed`：encoder 初始化、Adam 状态、proxies 种子全部走 `core.seeding`。

## 3. 双轨依赖裁决

| 轨 | 内容 | 必要性 | 探测方式 |
|----|------|--------|----------|
| Tier-1（运行时） | numpy / scipy / scikit-learn | 必需 | 直接 import |
| Tier-0（oracle） | torch / pytorch-metric-learning / metric-learn / faiss | 可选 | `oracle.available_torch()` / `available_metric_learn()` try/except |

**为什么双轨**：自研纯 numpy DML（core + losses.linear）覆盖全部必需功能，零第三方 DML 依赖、
CPU-only 可复现；torch / metric-learn 仅作"强基线 oracle 对照"（B7 NCA·ITML、B8 Forge-Oracle），
缺失时自动降级，不阻断主链路。

**metric-learn 兼容性坑（已解决）**：metric-learn 0.7.0 在 sklearn ≥ 1.8 下 `fit` 崩，因
`force_all_finite` 参数被移除 → 自研 `oracle._patch_sklearn` 将其重写为 `ensure_all_finite`。

## 4. 创新点落点

- **HAT（hat_temperature）**：`core/mathx.hat_temperature` 每 batch 解 `PP(τ)=κN`，由引理
  `dH/dβ=−β·Var(s)<0` 保证根唯一 → 固定次数二分 ⇒ 逐位确定。调用点在 `losses/proxy.py:_hat_beta`。
- **MQ-BPA**：`losses/proxy.py` 内 `K>1` 多原型 + `ema_align`（反向 EMA 防止原型坍缩）+ `mining/queue.py`
  FreshQueue 每步重编码提供零陈旧负样本。

## 5. 评测协议（文档 §4）

- 主指标 **MAP@R**（Musgrave ECCV 2020）；次：Recall@1、kNN 准确率、NMI/ARI。
- 协议 `open`（类不相交，测试集由后半类构成）为主；`closed` 为对照。
- 指标在 `query==gallery` 时**自动剔除 query 自身**（避免自命中 rank1 虚高）。
- 跨 ≥3 seed 报告 mean±std；G6 同 seed 两次运行核心指标逐位一致。

## 6. 确定性二次校验

```bash
python examples/run_demo.py && cp benchmark.json run1.json
python examples/run_demo.py && diff <(python -c "import json;print(json.dumps(json.load(open('run1.json'))['benchmark'],sort_keys=True))") \
                              <(python -c "import json;print(json.dumps(json.load(open('benchmark.json'))['benchmark'],sort_keys=True))") \
  && echo "DETERMINISTIC: bit-identical"
```
