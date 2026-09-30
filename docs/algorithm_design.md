# MetricForge — 算法设计方案（Phase 1）

> 作者：苏推衍（AI 算法科学家） | 版本：v1.0 | 日期：2026-09-30
> 定位：`metricforge` = **CPU-only、零预训练权重、可离线复现**的度量学习 / 对比表征学习系统。
> 本文件是**方案阶段**的定稿，所有门槛与协议在进入实现前冻结；实现阶段如需改动须回到本文档修订并说明理由。

---

## 0. TL;DR（结论先行）

| 项 | 结论 |
|---|---|
| 主指标 | **MAP@R**（Musgrave et al. 2020 推荐），open-set（类不相交）协议 |
| 强基线 | 欧氏 kNN（原始空间）＋ **自研 LMNN** ＋ Triplet(batch-hard) ＋ Proxy-Anchor(K=1) ＋ SupCon |
| 胜出门槛 | ΔMAP@R ≥ **+0.03** 且 Δ > ½(σ_forge + σ_best)，≥3 seeds，在 **≥4/6** 数据集成立；并须满足 ≤60s / ≤2GB |
| 创新点 1 | **HAT**（Hardness-Adaptive Temperature）：以"负样本参与率/困惑度"为目标的单参数自整定温度，bisection 求根 |
| 创新点 2 | **MQ-BPA**（Memory-Queue **Bi-directional** Proxy Alignment）：Fresh-Queue 跨批大负样本池 ＋ 每类 K 原型 ＋ 样本↔原型双向对齐 ＋ 负载均衡 |
| 基座 | **Tier-1 纯 numpy/scipy/sklearn**（零下载必跑）；torch 仅作 **可选 Tier-0** 加速对照 |
| 关键合规发现 | `metric-learn` 0.7.0 **可安装**（`-py2.py3-none-any` 通用纯 Python wheel）但**在本栈上运行时全部崩溃**（sklearn 1.9 移除 `force_all_finite`、LFDA 移除 `dim`）→ 仍须自研 LMNN/NCA/ITML/LFDA。详见 §8.1 的**勘误** |
| 诚实对标口径 | **不对标 CUB-200/Cars196 的 CV SOTA**（那需要 ImageNet 预训练 ViT/ResNet + GPU）。MetricForge 对标的是"**向量数据的度量学习算法层**"，对标对象为 Musgrave 2020 的公平协议与经典/现代损失函数族 |

---

## 1. 域 SOTA 现状（联网检索，含来源与时点）

### 1.1 公认 benchmark 与当前数字

检索时间：2026-09-30。以下为检索到的公开结果（**均为 PyTorch + GPU + ImageNet 预训练 backbone 的条件下取得**）。

| 方法 | 年份/ venue | Backbone | CUB-200-2011 R@1 | Cars196 R@1 | SOP R@1 | 来源 |
|---|---|---|---|---|---|---|
| Proxy-Anchor | Kim et al., CVPR 2020 | R50 | 69.9–71.1 | 80.4 | — | [DVA 论文 Table 2](https://arxiv.org/html/2506.16273v3) |
| PNCA++ | Teh et al., ECCV 2020 | R50 | 70.1–72.2 | 80.8–82.0 | 89.2 | 同上 |
| HIST | Lim et al., CVPR 2022 | R50 | 71.4 | 81.1 | 88.1 | 同上 |
| MS loss + S2SD | Roth et al., CVPR 2022 表 | R50 | 67.7 | 86.5 | 77.7 | [CVPR OpenAccess PDF](https://openaccess.thecvf.com/content/CVPR2022/papers/Roth_Integrating_Language_Guidance_Into_Vision-Based_Deep_Metric_Learning_CVPR_2022_paper.pdf) |
| Hyp-ViT | Ermolov et al., CVPR 2022 | ViT | 84.0–85.6 | 90.2–91.4 | 94.8 | DVA 论文 |
| HIER | Kim et al., CVPR 2023 | ViT | 85.7 | 91.3 | 94.4 | [LaFG 论文 Table 3](https://ar5iv.arxiv.org/html/2512.06255) |
| DDML | Park et al., AAAI 2025 | ViT | 86.0 | 91.7 | 95.2 | 同上 |
| VPTSP-G | Ren et al., ICLR 2024 | ViT | 86.6 | 91.7 | 94.8 | 同上 |
| LaFG | IJCAI 2025 | ViT | 87.2 | 92.4 | 95.2 | 同上 |
| GAPan | arXiv 2025/26 | ViT | **89.1** | **93.5** | **95.3** | [arXiv 2605.09859](https://arxiv.org/html/2605.09859v1) |

**趋势判读**（苏推衍观点）：
1. 2022 之后 R@1 的主升幅来自 **backbone 从 R50 换到 ViT（+13~15 个点）**，而非损失函数。损失函数族（MS / Proxy-Anchor / PNCA++ / HIST）在**同一 backbone 下的差距只有 1~3 个点**。
2. 这与 **Musgrave, Belongie & Lim, "A Metric Learning Reality Check", ECCV 2020 (arXiv:2003.08505)** 的核心结论一致：在统一训练策略、统一超参搜索、统一划分下，所谓 SOTA 相对 contrastive / triplet 的增益"**marginal at best**"。该论文同时指出大量论文存在：超参直接对 test 调、用训练过程中最好的 checkpoint、划分不一致等复现性问题。**本系统必须吸取这个教训（见 §5 的无泄漏协议与 §6 的门槛设计）。**
3. 2025–2026 的新增增益越来越来自**外部监督信号**（LLM 生成的属性描述 LaFG、生成式外观先验 GAPan、语言引导 ELG/PLG），属于"多模态监督"而非"度量学习算法本身"。**本系统在无网络、无权重条件下不追这条线**，而是回到算法层：难度课程、原型结构、局部几何。

### 1.2 公开 leaderboard

| 资源 | 说明 | 可用性与时点 |
|---|---|---|
| [KevinMusgrave/powerful-benchmarker](https://github.com/KevinMusgrave/powerful-benchmarker) | ECCV 2020 Reality Check 的官方 benchmark：4-fold CV、在后半类上测试，含贝叶斯超参搜索日志 | 代码在 GitHub；上次实质更新 ~2021，**结果表维护已停滞**，引用需注明时效性 |
| [KevinMusgrave/pytorch-metric-learning](https://github.com/KevinMusgrave/pytorch-metric-learning) | 事实上的工业标准库，v2.9.0（2025-08-18，新增 SmoothAPLoss），含 30+ loss / miner / AccuracyCalculator | **依赖 torch**，本系统仅作"算法口径对齐参考"，不作为运行时依赖 |
| sota2.com / CUB 数据集页 | 聚合型榜单，Deep Metric Learning CUB-200-2011 Recall@1 = 73.23（时点标 2024） | 聚合源，二手数据，**不作为权威引用** |
| Papers-with-Code (DML) | 历史 leaderboard | 站点状态待验证 |

### 1.3 本系统的诚实对标口径（**禁止夸大**）

| 维度 | CV SOTA 条件 | MetricForge 条件 | 结论 |
|---|---|---|---|
| 输入 | 原始图像 224×224 | **向量特征**（表格/合成/sklearn 内置） | 不同任务层 |
| Backbone | ImageNet 预训练 ViT/R50 | **随机初始化 MLP**（无权重） | 不可比 |
| 算力 | GPU（3090/A100）× 数小时 | **CPU-only，单次 fit ≤ 60s** | 不可比 |
| 数据规模 | 5k–60k 图，100–11k 类 | 150–20000 样本，2–100 类 | 部分重叠 |

**因此本系统的对标声明固定为以下措辞**（写入 README 与报告模板，不得改写）：

> MetricForge 不与 CUB-200-2011 / Cars196 上的图像检索 SOTA 比较（那些数字依赖 ImageNet 预训练 ViT 与 GPU）。MetricForge 的对标对象是**在相同向量输入、相同 CPU 预算、相同种子协议下**的度量学习算法族：Contrastive / Triplet / N-pairs / Lifted / ProxyNCA / Proxy-Anchor / SupCon / InfoNCE / MS / Margin，以及线性方法 LMNN / ITML / NCA / LFDA。MetricForge 报告的所有数字均在此口径内。

另需诚实标注的一条**预期不利结论**（写死在文档里，防止事后美化）：

> 在小样本、低维、类别边界简单的表格数据上（wine / breast_cancer / iris），线性 Mahalanobis 方法（LMNN / LFDA / NCA）**常常持平或优于深度对比方法**。这与 Rabbani, Medri & Samad (Int. J. Data Sci. Anal., 2025, DOI 10.1007/s41060-024-00681-z) 在 28 个表格数据集上的数据中心基准结论一致：_"traditional methods are frequently superior on easy data sets with presumably simpler decision boundaries"_。MetricForge 的预期优势域是：**类别多、类内多模态（层次化子类）、维度高、batch 相对类数偏小** 的场景。**这是可证伪预测，见 §7。**

---

## 2. 算法族清单与选型建议

### 2.1 对比矩阵

记号：$B$ = batch size，$D$ = 嵌入维，$C$ = batch 内类数，$C_{all}$ = 全量类数，$N$ = 样本数，$d$ = 原始维，$k$ = 目标邻居数。

| # | 方法 | 类别 | 单步复杂度 | 需 mining | 关键超参 | CPU 友好 | 纯 numpy 可行性 | 适用场景 | 参考 |
|---|---|---|---|---|---|---|---|---|---|
| L1 | **Contrastive** | pair | $O(B^2 D)$ | 建议 | margin $m$ | ★★★★★ | ✅ 完全 | 二分类/验证、小数据 | Hadsell et al. 2006 |
| L2 | **Triplet (random)** | triplet | $O(B^3)$ 全枚举 | — | $m$ | ★★ | ✅ | 教学基线 | Schroff et al. 2015 |
| L3 | **Triplet + semi-hard** | triplet | $O(B^2)$ | 是 | $m$ | ★★★★★ | ✅ | 通用首选 | Schroff et al. 2015 |
| L4 | **Triplet + batch-hard** | triplet | $O(B^2)$ | 是 | $m$ | ★★★★★ | ✅ | 类内紧、类间难 | Hermans et al. 2017 |
| L5 | **Triplet + DWS** | triplet | $O(B^2\log B)$ | 采样 | $m,\lambda$ | ★★★★ | ✅（需逆 CDF 表） | 防 collapse、负样本多样性 | Wu et al., ICCV 2017 |
| L6 | **N-pairs / Multi-class N-pair** | tuple | $O(B^2)$ | 否 | — | ★★★★★ | ✅ | 每类 ≥2 样本的 batch | Sohn, NeurIPS 2016 |
| L7 | **Lifted-Structure** | pair(结构化) | $O(B^2)$ | 否（内建加权） | $\alpha$ | ★★★★ | ✅ | batch 内类多 | Oh Song et al., CVPR 2016 |
| L8 | **ProxyNCA** | proxy | $O(BC_{all})$ | 否 | $\sigma$ | ★★★★★ | ✅ | 类数多、batch 小 | Movshovitz-Attias et al. 2017 |
| L9 | **Proxy-Anchor** | proxy | $O(BC_{all})$ | 否（内建） | $\alpha,\delta$ | ★★★★★ | ✅ | **CPU 小 batch 首选** | Kim et al., CVPR 2020 |
| L10 | **SupCon** | 对比（多正） | $O(B^2 D)$ | 否 | $\tau$ | ★★★★ | ✅ | 有增强/同类多样本 | Khosla et al., NeurIPS 2020 |
| L11 | **InfoNCE / NT-Xent** | 对比（1正） | $O(B^2 D)$ | 否 | $\tau$ | ★★★★ | ✅ | 自监督/实例判别 | Oord et al. 2018; Chen et al. 2020 |
| L12 | **MS loss** | pair(加权) | $O(B^2)$ | 内建软加权 | $\alpha,\beta,\epsilon,\lambda$ | ★★★ | ✅ | 综合强、超参多 | Wang et al., CVPR 2019 |
| L13 | **Margin / Distance-weighted** | pair | $O(B^2)$ | 是 | $\beta$(可学习), $\gamma$ | ★★★★ | ✅ | 稳定、Reality Check 强基线 | Wu et al., ICCV 2017 |
| L14 | **SoftTriple / Multi-Proxy** | proxy(K) | $O(BKC_{all})$ | 否 | $K,\gamma,\delta,\tau$ | ★★★★ | ✅ | **类内多模态** | Qian et al., ICCV 2019 |
| M1 | **LMNN** | 线性 Mahalanobis | $O(B^2 d)$ mini-batch | 内建 impostor | $\mu, k$ | ★★★★ | ✅（自研） | 小样本表格、可解释 | Weinberger & Saul, JMLR 2009 |
| M2 | **ITML** | 线性 Mahalanobis | $O(B^2 d)$ | — | $\gamma, u/l$ | ★★★★ | ✅（自研） | 有相似/不相似约束对 | Davis et al., ICML 2007 |
| M3 | **NCA** | 线性/深度 | $O(B^2 d)$ | 否 | — | ★★★★ | ✅（自研） | kNN 直接优化 | Goldberger et al., 2005 |
| M4 | **LFDA** | 线性（闭式） | $O(Nd^2 + d^3)$ | 否 | $k, \beta$ | ★★★★★ | ✅（scipy.linalg.eigh） | 超快基线、高维 | Sugiyama, JMLR 2007 |

### 2.2 选型建议（本系统采用）

| 角色 | 选型 | 理由 |
|---|---|---|
| **主损失的骨架** | **L9 Proxy-Anchor** | $O(BC)$ 而非 $O(B^2)$；无需 miner（miner 是 CPU 小 batch 上最大的不稳定源）；Reality Check 中表现稳健 |
| **类内结构增强** | **L14 Multi-Proxy（K>1）** | 覆盖层次化/多模态类；是本系统创新点 2 的载体 |
| **负样本池** | **Fresh-Queue（跨批，重新前向）** | 突破 CPU 小 batch 的负样本稀缺 |
| **难度课程** | **HAT（自研自适应温度）** | 替代手工 $\tau$ schedule 与启发式 mining |
| **局部几何** | **SNE-style KL 邻域保持（可开关）** | 防小样本过拟合 |
| **必跑基线** | M1 LMNN、M4 LFDA、L4 Triplet-BH、L9 Proxy-Anchor(K=1)、L10 SupCon、B1 欧氏 kNN | 覆盖线性/深度/代理/对比四族 |
| **不采用** | L2（全枚举 $O(B^3)$）、L12 MS（4 个超参，在 ≤60s 预算下无法公平调参）、L5 DWS（需额外分布假设，且被 HAT 部分覆盖） | 复杂度/调参预算 |

---

## 3. 自研创新点（核心）

整体损失：

$$\boxed{\mathcal{L}_{\text{Forge}} \;=\; \mathcal{L}_{\text{MQ-BPA}}^{\;\tau_{\text{HAT}}} \;+\; \lambda_{\text{geom}}(t)\,\mathcal{L}_{\text{geom}} \;+\; \lambda_{\text{lb}}\,\mathcal{L}_{\text{lb}}}$$

其中梯度合成用 **PCGrad-lite**（见 §3.3，属已知方法的工程集成，不计入创新声明）。

---

### 3.1 创新点 1：**HAT — Hardness-Adaptive Temperature**（以参与率为目标的自整定温度）

#### 一句话直觉
对比损失的梯度只对"**中等难度**"的负样本敏感：太简单的（softmax 概率≈0）推不动，太难的（概率≈1，往往是标注噪声/伪负样本）反而会把模型拽偏。固定温度 $\tau$ 时，随着训练推进负样本集体变简单，梯度会**静默消失**。HAT 不手工排 schedule，而是每步对每个 anchor **解一个方程**：把温度调到"恰好有固定比例的负样本在干活"。

#### 数学核心

设 anchor $i$ 的负样本相似度向量 $s_i \in \mathbb{R}^{N_i}$（$N_i = |\mathcal{N}_i|$，含 batch 内负样本 + Fresh-Queue 负样本）。Softmax 分布与温度 $\tau$（记 $\beta = 1/\tau$）：

$$p_{ij}(\beta) = \frac{e^{\beta s_{ij}}}{\sum_{k \in \mathcal{N}_i} e^{\beta s_{ik}}}$$

定义**参与率**（participation ratio / perplexity）：

$$\mathrm{PP}_i(\beta) \;=\; \exp\!\big(H(p_i(\beta))\big), \qquad H(p) = -\sum_j p_j \log p_j$$

**关键引理（可证，且不变量 #4 的基础）**：

$$H(\beta) = \log Z(\beta) - \beta\,\mathbb{E}_{p_\beta}[s] \;\Longrightarrow\;
\frac{dH}{d\beta} = \frac{Z'}{Z} - \mathbb{E}[s] - \beta\frac{d\mathbb{E}[s]}{d\beta}
= \mathbb{E}[s] - \mathbb{E}[s] - \beta\,\mathrm{Var}_{p_\beta}(s)
= -\beta\,\mathrm{Var}_{p_\beta}(s) \;<\; 0$$

即 **$H$（从而 $\mathrm{PP}$）关于 $\beta=1/\tau$ 严格单调递减、关于 $\tau$ 严格单调递增**（除非所有 $s$ 相等，此时 $\mathrm{Var}=0$，退化为常数 $\log N_i$）。
⇒ **方程 $\mathrm{PP}_i(\tau) = \kappa N_i$ 在 $\mathrm{Var}(s)>0$ 时有唯一根，且可用二分法在固定迭代数内以求到机器精度。**

**HAT 温度**：$\tau_i^{\star}$ = 上述方程的解，目标 $\kappa \in (0,1)$（默认 $\kappa = 0.25$），$\tau$ 被 clamp 到 $[\tau_{\min}, \tau_{\max}] = [0.02, 2.0]$。

**损失**（把 $\tau_i^{\star}$ 当作**常数**参与反传，即 detached / straight-through，与 mining 操作同类处理）：

$$\mathcal{L}_{\text{HAT-NCE}} = \frac{1}{B}\sum_{i=1}^{B} -\log
\frac{\sum_{j \in \mathcal{P}_i} \exp\!\big(s_{ij}/\tau_i^{\star}\big)}
{\sum_{j \in \mathcal{P}_i} \exp\!\big(s_{ij}/\tau_i^{\star}\big) \;+\; \sum_{k \in \mathcal{N}_i} \exp\!\big(s_{ik}/\tau_i^{\star}\big)}$$

在 MQ-BPA 中，同一 $\tau_i^{\star}$ 也用于 proxy 侧的 softmin/softmax 加权（见 §3.2）。

**自动退火性质**：训练后期负样本变简单 → 同一 $\tau$ 下 $\mathrm{PP}$ 会**下降**（分布更不均）→ HAT 自动**减小** $\tau$ 来锐化分布、重新暴露难例。反之初期自动**增大** $\tau$ 避免被离群点带偏。这正好复现了"easy→hard 课程"，但由数据驱动，无需任何 schedule 超参。

#### 伪代码

```python
def hat_temperature(S_neg, kappa=0.25, tau_lo=0.02, tau_hi=2.0, iters=40, eps=1e-12):
    """S_neg: (B, Nneg) cosine similarities for each anchor's negative pool.
    Returns tau: (B,) solved so that perplexity(p_i) == kappa * Nneg."""
    B, N = S_neg.shape
    target = np.log(max(kappa * N, 1.0))  # target entropy H*
    lo = np.full(B, 1.0 / tau_hi)  # beta_lo (large tau)
    hi = np.full(B, 1.0 / tau_lo)  # beta_hi (small tau)
    for _ in range(iters):  # FIXED iteration count -> bitwise deterministic
        mid = 0.5 * (lo + hi)
        H = logsumexp(mid[:, None] * S_neg, axis=1) - mid * (
            softmax(mid[:, None] * S_neg, axis=1) * S_neg
        ).sum(1)
        too_sharp = H < target  # entropy too low -> need larger tau -> smaller beta
        lo = np.where(too_sharp, lo, mid)
        hi = np.where(too_sharp, mid, hi)
    beta = 0.5 * (lo + hi)
    return 1.0 / beta
```

> **边界条件（必须实现，否则会出现 NaN）**：
> 1. 若某 anchor 的负样本数 $N_i = 0$ → 该 anchor 跳过（class-balanced sampler 保证 batch 内 ≥2 类，因此实际只在极端退化时发生，但仍需显式 guard）。
> 2. 若 $\mathrm{Var}_{p_\beta}(s) < 10^{-12}$（所有相似度相等，$H \equiv \log N_i$ 与 $\beta$ 无关）→ 直接返回 $\tau_{\min}$（若 $\kappa N_i < N_i$）或 $\tau_{\max}$（若 $\kappa N_i \ge N_i$），**禁止**进入二分（否则 bracket 不成立）。
> 3. 掩码掉的负样本用 $-\infty$ 填充时，需先减去行最大值再 exp（数值稳定），且 `logsumexp` 必须对全 $-\infty$ 行返回 $-\infty$ 而非 NaN。
>
> 复杂度：每步 $O(B \cdot T \cdot N_i)$，$T=40$ (fixed)，$N_i = B + M$。取 $B=128$、$M=1024$、$T=40$：$128 \times 40 \times 1152 \approx 5.9\times10^6$ flops/step。1000 步共 $5.9\times10^9$ flops ≈ **CPU 上 3–6 s**，可接受。可设 `T=24` 做 fast profile（精度 $\Delta\beta/\beta_{\text{range}} = 2^{-24}$，远超需要）。
> **瓶颈**：不是 HAT，而是 Fresh-Queue 重新前向 $O(M d h)$（$M=1024, d=64, h=128 \Rightarrow 8.4\times10^6$ MACs/step）。见 §4.4 预算表。

#### 为什么预期更强

1. **消除梯度饱和**：固定 $\tau$ 在训练后期使绝大多数负样本落入"太简单"区，梯度范数塌缩。HAT 维持恒定有效样本数 → 梯度信噪比近似恒定。
2. **超参可迁移**：$\tau$ 的合适值随 $B$、$C$、$D$、数据尺度剧烈变化（这是 DML 复现性灾难的主因之一）；$\kappa$ 是**相对量**（"我要 25% 的负样本活跃"），跨设置稳定 → 只需一个默认值，显著减少调参面（也减少 test 泄漏风险）。
3. **对伪负样本鲁棒**：argmax 式 batch-hard mining 会把单个"标注噪声导致的假负样本"放大成全部梯度；HAT 是**软加权**，$\kappa N$ 个负样本共同承担，方差更低。
4. **与 Fresh-Queue 天然耦合**：$\mathrm{PP}$ 是分布层面的统计量，负样本越多估计越稳；$M$ 大 → HAT 的 $\tau^\star$ 更平滑。

#### 如何被消融证伪

- **A1 开关**：`--hat off` → 退化为固定 $\tau$，该 $\tau$ 由 **val 集网格搜索**（$\tau \in \{0.05,0.07,0.1,0.15,0.2,0.3,0.5\}$）选出。
- **证伪条件**：若在所有 6 个数据集上 HAT 的 MAP@R ≤ 固定 $\tau$（在 ±1σ 内），或 HAT 的优势在任何一个小 batch（$B\le 32$）设置下都不出现 → **创新点 1 的"小 batch 下优势更大"假设被证伪**，需在报告中如实写明并把 HAT 降级为"等价但更少调参的便利组件"。
- **额外判别实验**（可证伪的细分预测）：画 $\tau^\star$ 随 step 的曲线，预测其**单调下降**并趋于 $\tau_{\min}$ 附近的平台。若曲线非单调或上升，则"自动退火"叙事不成立。

---

### 3.2 创新点 2：**MQ-BPA — Memory-Queue Bi-directional Proxy Alignment**

三个子件：**Fresh-Queue**（大负样本池）＋ **K 原型** ＋ **双向对齐 + 负载均衡**。

#### (a) Fresh-Queue：无陈旧性的跨批记忆队列

MoCo 队列（He et al., CVPR 2020）与 XBM（Wang et al., CVPR 2020）都复用**历史嵌入**，随 encoder 更新而"陈旧"（staleness），通常靠 momentum encoder 缓解。
**MetricForge 的做法**：队列里存的是 **原始输入 $x$**（不是 $z$），每步用**当前的** $f_\theta$ 重算一遍 $\tilde{z} = f_\theta(x_{\text{queue}})$。

- 这在图像域不可行（重跑 ViT 太贵），但本系统 encoder 是**小 MLP 且数据小**，每步重跑 $M$ 个样本只花 $O(M d h)$ —— 约等于一次 batch 前向。
- ⇒ **零陈旧偏差**，且不需要 momentum encoder（少一个超参 $m$）。

$$\mathcal{N}_i = \{\,j \in \text{batch} : y_j \ne y_i\,\} \;\cup\; \{\,q \in \mathcal{Q} : y_q \ne y_i\,\},\qquad |\mathcal{N}_i| \approx \tfrac{C-1}{C}(B+M)$$

- 队列更新：FIFO，每步 enqueue 当前 batch 的 $(x, y)$，dequeue 最老的 $B$ 条；$|Q| = M$ 恒定。
- **诚实声明**："跨批记忆队列"本身是已知机制（MoCo / XBM）；本系统的增量是 **Fresh re-encode（去陈旧）+ 与 HAT 的耦合 + 与多原型的统一形式**，不是凭空发明队列。

#### (b) K 原型与双向对齐

每类 $c$ 维护 $K$ 个原型 $\{p_{c,k}\}_{k=1}^{K} \subset \mathbb{R}^D$（L2 归一化，可学习参数）。
分配（hard，但带负载均衡）：

$$a_i \;=\; \arg\max_{k}\; s\big(z_i,\; p_{y_i,k}\big), \qquad
\mathcal{P}_{c,k} = \{i : y_i = c,\; a_i = k\},\quad
\mathcal{N}_{c,k} = \{i : y_i \ne c\}$$

**方向 1（样本 → 原型）**：标准 Proxy-Anchor，但以原型为 anchor：

$$\mathcal{L}_{\text{PA}}^{(K)} =
\frac{1}{|C^{+}|}\sum_{(c,k)\,\text{active}} \log\!\Big(1 + \sum_{i \in \mathcal{P}_{c,k}} e^{-\alpha\,(s_{i,c,k} - \delta)}\Big)
\;+\;
\frac{1}{K C}\sum_{c,k} \log\!\Big(1 + \sum_{i \in \mathcal{N}_{c,k}^{\text{hard}}} e^{+\alpha\,(s_{i,c,k} + \delta)}\Big)$$

其中 $\mathcal{N}_{c,k}^{\text{hard}}$ 由 **HAT 选出的活跃负样本**构成（取 $s/\tau^\star$ 后 softmax 概率最高的 $\lceil\kappa N\rceil$ 个），而非全部 $N$ → 把 HAT 与 ProxyAnchor 统一起来，且把负项复杂度从 $O(KC \cdot N)$ 降到 $O(KC \cdot \kappa N)$。

**方向 2（原型 → 样本质心）**：每个原型被锚定到"它当前负责的那些样本"的嵌入质心的 EMA：

$$\mu_{c,k}^{(t)} = \beta_{\mu}\,\mu_{c,k}^{(t-1)} + (1 - \beta_{\mu})\,\frac{1}{|\mathcal{P}_{c,k}|}\sum_{i \in \mathcal{P}_{c,k}} z_i \quad (\text{detached}),\qquad
\mathcal{L}_{\text{anch}} = \sum_{c,k} \big\|\,p_{c,k} - \tfrac{\mu_{c,k}}{\|\mu_{c,k}\|}\,\big\|_2^2$$

**作用**：K 原型方法（SoftTriple / MPA）最常见的失败模式是**原型坍缩**（$K$ 个原型收敛到同一点）或**原型漂移**（原型跑到没有样本支撑的区域）。方向 2 用一个**统计量驱动**、不参与反传的目标把原型拉回其责任区质心 —— 这是"反向对齐"，与方向 1 的"样本被拉向原型"构成闭环。

**负载均衡**（防退化到单原型）：

$$\mathcal{L}_{\text{lb}} = \sum_{c} \mathrm{KL}\!\left(\pi_c \,\Big\|\, \tfrac{1}{K}\mathbf{1}_K\right),\qquad
\pi_{c,k} = \frac{|\mathcal{P}_{c,k}|}{\sum_{k'} |\mathcal{P}_{c,k'}|}$$

#### (c) 局部几何正则（可开关，第三项）

在**输入空间**上预计算每个 train 样本的 top-$k$ 近邻（$k=10$，一次性，$O(N^2 d)$ 或 `sklearn NearestNeighbors`），构造稀疏的条件分布 $p_{\cdot|i}$；嵌入空间侧用同一邻域集合上的 softmax $q_{\cdot|i}$：

$$q_{j|i} = \frac{e^{-\|z_i - z_j\|^2}}{\sum_{l \in \mathcal{K}_i} e^{-\|z_i - z_l\|^2}},\qquad
\mathcal{L}_{\text{geom}} = \frac{1}{B}\sum_{i} \sum_{j \in \mathcal{K}_i} p_{j|i}\log\frac{p_{j|i}}{q_{j|i}}$$

作用：保留局部拓扑，抑制小样本下的"类内撕裂"。$p$ 只依赖 train 输入且**不含标签**（因此不构成标签泄漏），但仍需断言"只在 train 上预计算"。

#### 伪代码（合并）

```python
def forge_step(X_b, y_b, theta, P, mu, queue, t):
    Z_b = l2norm(mlp_forward(theta, X_b))  # (B, D)
    Z_q = l2norm(mlp_forward(theta, queue.X))  # (M, D)  Fresh re-encode
    Z = concat([Z_b, Z_q])
    Y = concat([y_b, queue.y])  # (B+M, D)

    S = Z_b @ Z.T  # (B, B+M)
    negmask = Y[None, :] != y_b[:, None]  # (B, B+M)
    tau = hat_temperature(where(negmask, S, -inf))  # (B,)     <- HAT, detached

    # --- proxy side ---
    Sproxy = Z_b @ P.T  # (B, K*C)
    a_i = argmax_over_own_class(Sproxy, y_b)  # (B,)
    L_pa = proxy_anchor_K(Sproxy, y_b, a_i, alpha, delta, tau_hard=median(tau))
    L_anch = sum((P - l2norm(mu)) ** 2)
    L_lb = load_balance(a_i, y_b, K)

    # --- geometry ---
    L_geom = sne_kl(Z_b, knn_idx_of_X_b)

    # --- gradient composition (PCGrad-lite) ---
    g1 = grad(L_pa + lam_lb * L_lb, theta)
    g2 = grad(L_geom, theta)
    if dot(g1, g2) < 0:
        g2 = g2 - dot(g2, g1) / dot(g1, g1) * g1  # Yu et al., NeurIPS 2020
    theta -= lr * (g1 + lam_geom(t) * g2)
    P -= lr_P * grad(L_pa + lam_anch * L_anch + lam_lb * L_lb, P)
    mu = ema_update(mu, Z_b, a_i, y_b)
    queue.push(X_b, y_b)  # FIFO, |Q| = M
```

#### 复杂度

| 环节 | 复杂度 | $B{=}128, M{=}1024, D{=}32, d{=}64, h{=}128, C{=}10, K{=}3$ 下的量级 |
|---|---|---|
| Fresh re-encode | $O(M d h)$ | $8.4\times10^6$ MAC |
| Batch forward | $O(B d h)$ | $1.0\times10^6$ MAC |
| 相似度矩阵 | $O(B M D)$ | $4.2\times10^6$ MAC |
| **HAT 求根** | $O(B T (B+M))$ | $5.9\times10^6$ flops（$T{=}40$） |
| Proxy-Anchor (K) | $O(BKC + KC\kappa(B+M))$ | $\sim10^5$ |
| SNE-KL | $O(B k D)$ | $4\times10^4$ |
| **瓶颈** | **Fresh re-encode + 相似度矩阵，二者同阶** | $\sim 2\times10^7$ MAC/step ⟹ 1000 步 ≈ $2\times10^{10}$ ⟹ **CPU 约 20–40 s** |

> 这个预算**逼近 60s 上限**，因此 fast profile 默认 $M=512, T=24, \text{epochs}=60$。见 §4.4。

#### 为什么预期更强 & 如何被证伪

| 预测 | 判别数据集 | 证伪条件 |
|---|---|---|
| $K>1$ 显著优于 $K=1$ | `hier_gauss`（大类内嵌 2–3 个子类） | 若 `hier_gauss` 上 $K=3$ 的 MAP@R ≤ $K=1$（±1σ），则"多原型处理类内多模态"的假设被证伪 |
| $K>1$ **不**显著帮助单模态数据 | `gauss_blobs`（各类单一高斯团） | 若 `gauss_blobs` 上 $K=3$ 也显著更好，说明收益来自"更多参数"而非"多模态建模" → 需改写叙事 |
| Fresh-Queue 在类数多时收益大 | `subspace`（$C{=}50$）、digits（$C{=}10$） | 若 $M=1024$ 与 $M=0$ 差异 < 1σ，队列机制在本规模下无意义 |
| $\mathcal{L}_{\text{geom}}$ 在小样本上防过拟合 | wine / breast_cancer | 若在 $N<600$ 的数据上开几何正则反而掉点，则该项应默认关闭 |

---

### 3.3 非创新声明（工程集成，须如实标注）

| 组件 | 来源 | 声明 |
|---|---|---|
| PCGrad 梯度投影 | Yu et al., "Gradient Surgery for Multi-Task Learning", **NeurIPS 2020**（arXiv:2001.06782） | **已知方法**，本系统做轻量集成并命名为 "PCGrad-lite"，**不计入自研创新** |
| 跨批记忆队列 | He et al. (MoCo) CVPR 2020；Wang et al. (XBM) CVPR 2020 | **已知机制**，增量 = Fresh re-encode + 与 HAT/多原型的统一 |
| Proxy-Anchor 主体 | Kim et al., CVPR 2020 | **已知损失**，增量 = $K$ 原型 + 反向对齐 + 负载均衡 + HAT 加权 |
| SNE 邻域 KL | Hinton & Roweis 2002 / van der Maaten 2008 | **已知目标**，此处作为正则项使用 |

---

## 4. 评测协议

### 4.1 数据集清单

**合规前提**：HF 默认不可达、需"零下载可跑 demo"。`sklearn.datasets.fetch_*`（20newsgroups / openml / covtype）**都需要联网** ⇒ **一律不得进入必跑集**，只能在 `--online` 显式开启时作为可选扩展。以下 A 组全部为**离线可生成/随 wheel 内置**。

#### A 组：合成可控数据（有 ground-truth 生成参数，可精确控制难度）

| ID | 名称 | 生成方式 | 规模 / 维数 | 设计意图 | 关键可调参数 |
|---|---|---|---|---|---|
| S1 | `gauss_blobs` | $K$ 个各向异性高斯团 | $N{=}4000,\ d{=}32,\ C{=}16$ | 标准单模态；欧氏 kNN 已很强 → 检验"不打不过就别吹" | 团间 Mahalanobis 间距 $\Delta$（默认 2.0） |
| S2 | `concentric` | 2–4 个同心环 + 角度扰动 | $N{=}3000,\ d{=}2$ | **欧氏距离结构性失败**的教科书案例 | 环间距 / 噪声 |
| S3 | `spirals` | 三螺旋 | $N{=}3000,\ d{=}2$ | 需要高度非线性的度量 | 圈数、噪声 |
| S4 | `subspace` | 每类在一个随机 $d_{sub}$ 维线性子空间上 + 噪声 | $N{=}5000,\ d{=}200,\ C{=}50,\ d_{sub}{=}8$ | 高维稀疏、类数多 → 检验 queue 与 proxy 扩展性 | $d, d_{sub}, C$, SNR |
| S5 | `xor_grid` | 棋盘/XOR 块 | $N{=}2000,\ d{=}2$ | 线性 Mahalanobis **必然失败** → 给深度方法正名 | 块数 |
| S6 | `hier_gauss` | $C$ 个大类，每类含 2–3 个分离子类 | $N{=}4500,\ d{=}32,\ C{=}15$（45 子团） | **创新点 2（K>1）的主判别集** | 子类间距 vs 大类间距之比 $\rho$ |

#### B 组：真实小规模（sklearn 内置，随 wheel 打包，**离线可用**）

| ID | 数据集 | $N \times d$ | $C$ | 备注 |
|---|---|---|---|---|
| R1 | `load_digits` | 1797 × 64 | 10 | 主战场：类数适中、有手写体多模态（"1" 有带底座/不带底座） |
| R2 | `load_wine` | 178 × 13 | 3 | 小样本 easy —— 预期线性方法占优（诚实标注） |
| R3 | `load_breast_cancer` | 569 × 30 | 2 | 二分类；MAP@R 会很高，作为饱和检查 |
| R4 | `load_iris` | 150 × 4 | 3 | 极小；仅作冒烟/回归测试，不进主结果表（标 "smoke only"） |

#### C 组：可选（需显式 `--online`）

`fetch_openml`（如 `segment`, `glass`, `vehicle`, `vowel`, `letter`）、UCI 小型集、**20newsgroups + TF-IDF**(需下载 ~14 MB)。
**规则**：C 组结果只能出现在附录，且必须在表格中显式标注 `network=required`；CI 与 demo 不得依赖 C 组。

### 4.2 划分与无泄漏协议（**方案阶段冻结**）

| 协议 | 定义 |
|---|---|
| **P1 Closed-set** | stratified 划分 train/val/test = **60 / 20 / 20**（按样本），类集合相同 |
| **P2 Open-set（主协议）** | 类按稳定顺序（label 排序 + 固定 seed 的整体 shuffle）分成两半：**前半类** → train+val（其中 val = 前半类样本的 20%），**后半类** → test。类集合**完全不相交**。这是 Musgrave et al. 2020 的 "test on 2nd-half of classes" 口径，也是度量学习唯一有意义的泛化检验。 |
| **预处理** | `StandardScaler`（或 `MinMaxScaler`）**只在 train 上 fit**，`transform` val/test。合成数据的 S4 额外提供可选 PCA-64（同样只在 train fit）。 |
| **超参选择** | 所有损失超参（$\alpha, \delta, \kappa, K, \lambda_{\text{geom}}, \lambda_{\text{anch}}, \lambda_{\text{lb}}, \text{lr}, M, \tau$）**只在 val 上**选，固定搜索预算（默认 24 次随机搜索 / 方法 / 数据集）。 |
| **Test 使用** | **每个方法在每个数据集上只允许用 test 一次**（选定配置后）。禁止"看 test 再调"。 |
| **Seeds** | $\{0, 1, 2\}$（≥3）用于划分、初始化、数据顺序；报告 mean ± std。 |
| **确定性** | 唯一 seed 入口 `metricforge.set_seed(s)`；禁止任何未受控的 `np.random` 全局调用；`PYTHONHASHSEED` 固定。 |
| **时间/序列** | **不适用** —— 全部数据集无时间维度，禁止滑窗、禁止任何未来信息 |

**无泄漏的硬断言（进单测，见 §8）**：
- `assert scaler.n_samples_seen_ == len(X_train)`
- `assert set(y_test).isdisjoint(set(y_train))`（P2 协议）
- `assert knn_input_graph` 只在 `X_train` 上构建
- 超参搜索记录必须落盘 `runs/<id>/hparams.json`，且 test 指标文件的 mtime 不得早于 hparams 选择完成时刻（用单次 pipeline 保证）

### 4.3 指标

| 指标 | 定义 | 角色 |
|---|---|---|
| **MAP@R** | $\frac{1}{R}\sum_{i=1}^{R} P(i)\,\mathbb{1}[\text{第 }i\text{ 个检索正确}]$，$P(i)=\frac{\#\text{前 }i\text{ 个中同类}}{i}$ | **主指标**（Musgrave 2020：比 R@1 信息量更大、更稳定，lag-1 自相关 0.81 vs R@1 0.73） |
| Recall@1 / Precision@1 | top-1 邻居同类比例 | 次（对外可比性） |
| R-Precision | 前 $R$ 个中同类的比例 $r/R$ | 次 |
| kNN accuracy | $k{=}1,5$ 的分类准确率 | 次 |
| NMI / ARI | 在 test 嵌入上跑 KMeans（$K{=}C_{test}$）后与真值比较 | 聚类侧 |
| triplet-satisfaction | 满足 $d_{ap} < d_{an}$ 的三元组占比 | 诊断（不进主表） |
| wall-clock / peak RSS | `time.perf_counter` + `psutil` | **门槛项** |

**检索口径**：query = test，reference(gallery) = test（**剔除 query 自身**）。若某类在 gallery 中仅 1 个样本，该 query 的 MAP@R 记为 NaN 并剔除（MAP@R 依赖 $R\ge 1$ 且需排除自身）。

> ⚠️ **$R$ 的口径陷阱（实现阶段最高频 bug）**：$R$ 是 **gallery 中与该 query 同类的样本总数**（固定搜索长度），**不是**"检索列表中命中的数量"。检索列表长度恒为 $R$，其中命中数 $r \le R$。因此：
> - `R_precision = r / R`
> - `MAP@R = (1/R) · Σ_{i: 第 i 个命中} P(i)`，其中 $P(i) = c(i)/i$，$c(i)$ = 前 $i$ 个中的命中数。
> - 若误把 $R$ 取成 $r$，Musgrave Table 3 的四个用例会全部算成 100/100/100/100，测试"看似通过"实则错误 —— 这正是 I8 必须用**逐位比对**而非"范围检查"的原因。

### 4.4 时间/内存预算

| 项 | 上限 | fast profile 默认 |
|---|---|---|
| demo 端到端 | **≤ 60 s** | $M{=}512,\ T{=}24,\ \text{epochs}{=}60,\ B{=}128,\ h{=}[128],\ D{=}32$ |
| 单次 `fit` (benchmark) | ≤ 60 s（R1 digits） | 同上 |
| peak RSS | **≤ 2 GB** | 相似度矩阵 $(B{+}M)^2$ 若全算 = $1152^2\times8$ B ≈ 10 MB（安全）；**禁止**物化 $N\times N$ 全局矩阵（$N{=}5000 \Rightarrow 200$ MB，尚可但需 chunk） |
| 全量 benchmark（6 数据 × 6 方法 × 3 seeds） | ≤ 25 min | 通过 `--jobs` 顺序执行，可分 shard |

---

## 5. 强基线定义 + 性能门槛（**方案阶段定死**）

### 5.1 基线清单

| ID | 方法 | 实现来源 | 说明 |
|---|---|---|---|
| **B1** | 欧氏 kNN（原始空间，标准化后） | sklearn | 零学习下界；**若某数据集 B1 ≥ 0.97 MAP@R，该集标记"饱和"，不计入门槛统计** |
| **B2** | **LMNN**（mini-batch, 自研纯 numpy） | 自研 | `metric-learn` 0.7.0 在本栈运行时崩溃（见 §8.1 勘误），必须自研 |
| **B3** | **LFDA**（闭式，scipy.linalg.eigh） | 自研 | 超快线性基线 |
| **B4** | **Triplet + batch-hard** + 同一 MLP encoder | 自研 | 深度 pair 族代表 |
| **B5** | **Proxy-Anchor ($K{=}1$)** + 同一 MLP encoder | 自研 | 深度 proxy 族代表（= 本系统关掉全部创新的退化版） |
| **B6** | **SupCon** + 同一 MLP encoder | 自研 | 深度对比族代表 |
| B7 | NCA / ITML | 自研 | 附加经典对照，进附录 |
| **B8** | **Forge-Oracle**：Forge 超参在 **test 上**调 | 自研 | **复现性对照**：用来量化"正确协议 vs 泄漏协议"的差距，写入报告以示警戒（不参与胜负判定） |

> **所有深度基线（B4/B5/B6）与 MetricForge 使用完全相同的 encoder 架构、相同的优化器、相同的 epoch 数、相同的超参搜索预算**。这是吸取 Musgrave 2020 "Reality Check" 教训的核心措施：只有训练策略一致，比较才有意义。

### 5.2 门槛（冻结，不得事后修改）

**主指标**：MAP@R（P2 open-set 协议，test 集，3 seeds）。

定义 $\Delta_d = \overline{\text{MAP@R}}^{\text{Forge}}_{d} - \max_{b \in \{B2..B6\}} \overline{\text{MAP@R}}^{b}_{d}$。

**胜出（Win）判定，须同时满足：**

| # | 条件 |
|---|---|
| **G1** | 在 **≥ 4 / 6** 个主数据集上（S2, S3, S4, S5, S6, R1；S1 与 R3 为饱和检查集，可替换）满足 $\Delta_d \ge +0.03$ |
| **G2** | 同一批数据集上满足误差棒判据 $\Delta_d > \tfrac{1}{2}\big(\sigma^{\text{Forge}}_d + \sigma^{\text{best}}_d\big)$（3 seeds 的样本标准差） |
| **G3** | 6 个数据集的**宏平均** $\bar\Delta \ge +0.02$ |
| **G4** | 效率：R1 digits 单次 fit **≤ 60 s** 且 peak RSS **≤ 2 GB** |
| **G5** | 不劣化：不存在任何数据集上 $\Delta_d \le -0.05$（显著劣化即判负，防止靠 2 个数据集刷分） |
| **G6** | 确定性：同 seed 两次运行 MAP@R 差 $= 0$（逐位级，非浮点容差） |

**补充统计（诚实标注强度）**：
- 3 seeds **不足以**给出 $p<0.05$ 的参数检验。因此除 G2 外，附加 **paired bootstrap**（对 query 级 MAP@R 重采样 10 000 次）给出 95% CI，要求 CI 不跨 0。
- 报告措辞固定为："**方向性证据（3 seeds + bootstrap CI）**"，**禁止**写"统计显著（$p<0.05$）"。

**分层报告**：结果表必须**分开**报告 S 组合成集与 R 组真实集，禁止只报宏平均（防止用合成集刷分掩盖真实集无增益）。

---

## 6. 消融设计

全部消融在 **同一 seed 集 {0,1,2}、同一 val 选出的超参**下进行，只切换一个开关。

| ID | 关掉的组件 | 退化到 | 预期结论 | 反证/证伪判据 |
|---|---|---|---|---|
| **A1** | HAT（改固定 $\tau$，val 网格搜最优） | 标准 Proxy-Anchor / SupCon | MAP@R 下降 **0.01–0.04**；且**下降幅度随 batch 减小而增大**（$B=32$ 时最大） | 若任何设置下都不降 → HAT 无价值（§3.1 证伪） |
| **A2** | Fresh-Queue（$M=0$） | 纯 batch 内负样本 | 在 $C$ 大的 S4/R1 上下降 0.01–0.03；在 $C$ 小的 S2上几乎无变化 | 若 $C$ 大时也无变化 → 队列无价值 |
| **A3** | 多原型（$K=1$） | 标准 Proxy-Anchor | **S6 hier_gauss 上下降最明显（预期 ≥0.03）**；S1 gauss_blobs 上 ≤0.01 | 见 §3.2 双条件证伪 |
| **A4** | $\mathcal{L}_{\text{geom}}$（$\lambda_{\text{geom}}{=}0$） | 无几何正则 | 小样本 R2/R3 上下降 0.01–0.02；大样本 S4 上无差异 | 若小样本也无差异 → 删该项 |
| **A5** | 反向对齐 $\mathcal{L}_{\text{anch}}$（$\lambda_{\text{anch}}{=}0$） | 多原型无锚定 | 原型坍缩度量（$K$ 原型两两余弦相似度均值）上升；S6 上掉点 | 若坍缩未发生 → 说明负载均衡已足够，可合并两项 |
| **A6** | PCGrad-lite（改直接相加） | 朴素多任务加权 | 在 $\lambda_{\text{geom}}$ 较大时梯度冲突导致掉点 | 若掉点 < 1σ → PCGrad-lite 是过度设计，简化 |
| **A7** | 全部创新关（$K{=}1$, $M{=}0$, 固定 $\tau$, 无 geom） | **= B5 Proxy-Anchor** | 用作 sanity check：A7 的数值必须与 B5 **逐位一致** | 若不一致 → 实现有 bug（这是硬不变量，见 §7-I9） |
| **A8** | **κ 敏感性**（主理人拍板 1 指定，必做） | 同完整版，仅改 $\kappa$ | $\kappa \in \{0.15, 0.25, 0.40\} \times \{$S1 gauss_blobs, S6 hier_gauss, S4 subspace$\}$，报 ΔMAP@R 的**极差** | **若极差 > 0.02 → 判定"κ 敏感"**：必须在 README / model_card 的局限章节写明，并列入 v1.1 roadmap 的 adaptive-κ。**本轮不做自适应实现**（会破坏逐位确定性的论证成本过高，性价比不足） |

**要求**：每组消融给出 $\Delta$MAP@R（vs 完整版）+ 效率变化（s / MB）。至少 A1、A3 必须出图（$\tau^\star$ 随 step 曲线；$K$ 原型在 S6 上的 2D 投影散点）。

**拍板 2（hier_gauss 证伪，主理人裁定，不得规避）**：A3 在 S6 hier_gauss 上若实测 $\Delta(K{=}3 \text{ vs } K{=}1) < 0.01$，须**如实写入报告的失败/证伪章节**，叙事降级为"多原型在本协议下未复现预期增益"。**严禁**通过换数据集、改指标、加 seed 数来救。这是本交付可信度的来源——宁可 A 级也不许刷成 S 级。

---

## 7. 可验证不变量（硬金标准，逐条可单测）

记号：$Z$ = L2 归一化后的嵌入矩阵，$S = ZZ^\top$，$D$ = 欧氏距离矩阵，$B$ = batch，$N$ = 负样本数。

| # | 不变量 | 断言形式 | 容差 / 条件 |
|---|---|---|---|
| **I1** | 嵌入 L2 归一化 | $\big|\|z_i\|_2 - 1\big| \le \epsilon$ 对所有 $i$ | $\epsilon = 10^{-6}$（float64）；$\epsilon=10^{-5}$（float32） |
| **I2** | 距离/相似度矩阵结构 | $D = D^\top$；$\mathrm{diag}(D) = 0$；$D_{ij} \ge 0$；$S \in [-1,1]^{B\times B}$；$S_{ii} = 1$ | `np.allclose(D, D.T, atol=1e-12)`；`abs(D.max()) <= 2+1e-12` |
| **I3** | Triplet / Contrastive 损失非负且可达零 | $\mathcal{L}_{\text{tri}} \ge 0$；在构造的"完美分离"样本（$\max d_{ap} < \min d_{an} - m$）上 $\mathcal{L}_{\text{tri}} = 0$ **精确成立** | `L >= 0`；`L == 0.0` |
| **I4** | **HAT 单调性与求根正确性** | (a) 对随机 $s$，$H(1/\tau)$ 关于 $\tau$ 严格递增（网格 20 点上逐点 $> $ 前一点）；(b) bisection 求出的 $\tau^\star$ 满足 $\big|\log \mathrm{PP}(\tau^\star) - \log(\kappa N)\big| \le 10^{-6}$；(c) 与 `scipy.optimize.brentq` 独立求根结果差 $< 10^{-6}$ | 独立交叉验证（b/c），**必须过** |
| **I5** | **HAT 引理**：$\frac{dH}{d\beta} = -\beta\,\mathrm{Var}_{p_\beta}(s)$ | 中心差分 vs 解析式，maxRelErr $< 10^{-6}$ | float64，随机 $s$，20 个 $\beta$ 点 |
| **I6** | InfoNCE 损失上下界（紧） | $\mathcal{L}_i \in \big[\log(1+(B{-}1)e^{-2/\tau}),\ \log(1+(B{-}1)e^{2/\tau})\big]$；且 $\tau \to \infty$ 时 $\mathcal{L}_i \to \log B$（$\tau{=}10^{6}$ 时差 $<10^{-3}$） | 由 $s\in[-1,1]$ 推出，两侧均可构造达到 |
| **I7** | SNE 几何项 | $p_{\cdot|i}$ 与 $q_{\cdot|i}$ 行和 $= 1$；$\mathcal{L}_{\text{geom}} \ge 0$；当 $q \equiv p$ 时 $\mathcal{L}_{\text{geom}} = 0$（$<10^{-12}$） | KL 非负性 |
| **I8** | **MAP@R ≤ R-Precision**，等号 iff 所有同类排在异类之前 | 随机压力测试 50 000 次（violations 必须为 0）+ 人工构造 4 个 Musgrave Table 3 用例（$R{=}10$ 固定：第1个中→RP10/MAP10；第1与第10个中→RP20/MAP12；第1与第2个中→RP20/MAP20；全中→100/100） | **必须逐位复现 Musgrave Table 3** |
| **I9** | **退化一致性**：全创新关闭 ≡ B5 Proxy-Anchor | 同一 seed 下两者的损失序列与最终嵌入 `np.array_equal`（或 maxAbsDiff $<10^{-12}$） | 这是 A7 的硬断言，也是"创新组件确实被正确开关"的证明 |
| **I10** | **确定性 / 逐位复现** | 同 seed 两次 `fit` → `np.array_equal(Z1, Z2)` 为 True；MAP@R 差 $= 0.0$ | **不允许浮点容差**（这是本系统最硬的卖点） |
| **I11** | 梯度检验（analytic vs central difference） | $\max_i \frac{|g^{ana}_i - g^{num}_i|}{\max(1, |g^{ana}_i|, |g^{num}_i|)} < 10^{-6}$ | 对 $\mathcal{L}_{\text{PA}}^{(K)}$、$\mathcal{L}_{\text{geom}}$、$\mathcal{L}_{\text{anch}}$（HAT 的 $\tau$ 与 assignment $a_i$ 视为 detached 常数） |
| **I12** | 训练单调性（确定性设置下） | 全批量、固定 lr（足够小）、关掉 mining/queue 时，每步损失**非增**：$\mathcal{L}^{(t+1)} \le \mathcal{L}^{(t)} + 10^{-10}$ | 用于抓 optimizer/梯度符号类 bug |
| **I13** | 队列不变式 | $|Q| = M$ 恒定；FIFO 顺序正确；同一步内 $\mathcal{N}_i$ 不含 $i$ 自身 | 任意 push 序列后成立 |
| **I14** | 收敛质量 | 收敛后 train 上 triplet-satisfaction $\ge 0.95$（S1/S6）；同类均值距离 < 异类均值距离 | 诊断，进 CI 但阈值为软断言（warn） |

> **强制要求**：I4 / I8 / I9 / I10 / I11 为 **Release-blocking**，任一失败不得打 tag。

### 7.1 方案阶段已完成的数值自检（纯 stdlib，无需 numpy）

以下三条在**写本文档时已实跑验证**，不是纸面断言。实现阶段直接把它们抄成单测：

| 自检 | 结果 |
|---|---|
| **I5 引理** $\frac{dH}{d\beta} = -\beta\,\mathrm{Var}_{p_\beta}(s)$：中心差分 vs 解析式，$\beta \in \{0.05, 0.2, 0.5, 1, 2, 5\}$ | **maxRelErr = 7.2e-10** ✅ |
| **I4(a) 单调性**：$H$ 在 $\tau \in [0.02, 6]$ 的 20 点几何网格上严格递增（$H(0.02){=}0.2553 \to H(5.99){=}2.4762$，满量程 $\log 12{=}2.4849$） | **True** ✅ |
| **I4(b) 求根精度**：$N \in \{12, 128, 1152, 64\}$、$\kappa \in \{0.25, 0.10\}$、相似度尺度 $\in \{0.05, 0.18, 0.3, 1.0\}$ 共 5 组 | **$\lvert\log\mathrm{PP}(\tau^\star) - \log(\kappa N)\rvert \le 1.0\text{e-}11$** ✅（40 次固定二分） |
| **I8 Musgrave Table 3**：$R{=}10$ 四个用例 | **MAP@R = 10 / 12 / 20 / 100，RP = 10 / 20 / 20 / 100 —— 与论文逐位一致** ✅ |
| **I8 不等式压力测试**：50 000 次随机排序 | **violations = 0；"等号但排序不完美" = 0** ✅ |

> 注意：自检中 $\tau^\star$ 随相似度尺度缩小而**自动减小**（尺度 0.18 → $\tau^\star{=}0.112$；尺度 0.05 → $\tau^\star{=}0.031$），这正好演示了 HAT 的"分布越集中、温度越锐"的自适应行为，与 §3.1 的退火叙事一致。

**已落盘为单测**：上述 I3 / I4 / I5 / I6 / I8 五条已写成 pytest 单测，见 `tests/test_invariants_core.py`（57 个用例，三种环境下均全绿）。该文件**优先从 `metricforge` 包导入被测内核，导入失败时回退到文件内参考实现**，因此今天即可跑，等真实实现落地后同一批测试自动切换去测真实代码。

> 补充一条**实测发现的数值边界**（写测试时踩到，值得记录）：当 $\tau$ 极小（$N$ 小、相似度尺度大）时 softmax 完全 one-hot，$H$ 在 float64 下**下溢为精确 `0.0`**，相邻网格点出现平台段。这不是引理失效，是浮点表示极限。因此 I4(a) 的单测断言拆成三层：全局"单调不减"（带相对容差）＋ 在 $H$ 高于下溢地板的区间内"严格递增" ＋ 首尾跨度严格为正。

---

## 8. Tier-0 / Tier-1 选型结论表

**核验方式**：直接查询 PyPI JSON API，按 **PEP 427 wheel tag**（python tag × abi tag × platform tag）判定可安装性。核验时间 **2026-09-30**，Python **3.13.14 (MSC v.1944 64-bit AMD64)**，`pip 26.2.1`，PyPI 可达。

> ⚠️ **本节含一处已订正的错误，见 §8.1。简言之：v1.0 版曾判 `metric-learn` / `umap-learn` "无 win wheel → 禁选"，该判断是错的（判据只匹配了文件名含 `win_amd64` 的包，系统性漏掉了 `-none-any` 通用纯 Python wheel）。正确结论是两者**都能装**；`metric-learn` 最终仍不可用，但原因是**运行时与 sklearn 1.9 API 不兼容**（实测），不是 wheel 缺失。**结论方向未变，理由已换。**

| Tier | 包 | 用途 | 最新版本 | cp313+win_amd64 wheel | 体积 | 失败降级路径 |
|---|---|---|---|---|---|---|
| **T1（必需）** | `numpy` | 全部数值计算 | 2.5.3 | ✅ `numpy-2.5.3-cp313-cp313-win_amd64.whl` | 12.6 MB | 无（硬依赖） |
| **T1（必需）** | `scipy` | `logsumexp`, `linalg.eigh`(LFDA), `optimize.brentq`(I4 交叉验证), 稀疏 kNN | 1.18.1 | ✅ | 36.6 MB | `logsumexp` 自实现；`eigh` 退 `numpy.linalg.eigh`；`brentq` 退二分 |
| **T1（必需）** | `scikit-learn` | 内置数据集(A组离线)、`StandardScaler`、`NearestNeighbors`、KMeans、NMI/ARI、`train_test_split` | 1.9.1 | ✅ | 8.2 MB | 数据集改合成；指标自实现（NMI/ARI 有闭式） |
| **T1（可选）** | `pandas` | 结果表落盘/汇总 | 3.0.6 | ✅ | 9.6 MB | 改 `csv` 模块 |
| **T1（可选）** | `matplotlib` | 消融图（$\tau^\star$ 曲线、原型投影） | 3.11.2 | ✅ | 9.3 MB | 不出图，仅写 csv；`--no-plot` |
| **T1（可选）** | `psutil` | peak RSS 监控（G4 门槛） | 7.2.2 | ✅ (cp313t) | 0.1 MB | 用 `tracemalloc` 估计，标注"估算值" |
| **T1（可选）** | `pytest` | 单测（I1–I14） | 9.1.1 | ⚪ py3-none-any（无需编译） | — | 无 |
| **T1（可选）** | `pyyaml` | 配置文件 | 6.0.3 | ✅ | 0.2 MB | 改 `json` |
| **T0（可选加速）** | `torch` | 可选 MLP encoder / 自动微分，用于与 Tier-1 结果交叉验证梯度 | 2.14.0 | ✅ | **124.1 MB** | **try/except 降级到 numpy 手写反向**；不得作为唯一路径 |
| **T0（可选）** | `faiss-cpu` | 大规模检索加速（$N > 50$k 时） | 1.15.1 | ✅ | 16.3 MB | 改 `sklearn NearestNeighbors` / numpy 分块精确检索（本规模下足够） |
| **T0（可选）** | `numba` | HAT 二分/相似度热点 JIT | 0.68.0 | ✅ | — | 直接 numpy 向量化 |
| **T0（可选）** | `hdbscan` | 聚类侧补充指标 | 0.8.44 | ✅ | 1.9 MB | 改 KMeans |
| ⚠️ **可装但不可用** | **`metric-learn`** | LMNN/ITML/NCA/LFDA | 0.7.0 | ✅ **可装**：`metric_learn-0.7.0-py2.py3-none-any.whl`（纯 Python 通用 wheel，无需编译器） | 0.07 MB | **实测运行时崩溃** → 自研纯 numpy（§5 B2/B7）。见 §8.1 |
| ✅ **可装（未使用）** | **`umap-learn`** | 可视化 / 非线性基线 | 0.5.12 | ✅ **可装**：`umap_learn-0.5.12-py3-none-any.whl`（纯 Python） | 0.09 MB | 本方案不用它做基线；可视化改用 PCA。运行时兼容性**待验证**（未实测） |
| ❌ **禁选** | `pytorch-metric-learning` | 参考实现 | — | 依赖 torch（124 MB）+ 需 GPU 才有意义 | — | **仅作算法口径参考**，禁止运行时依赖 |
| ❌ **禁选** | `openml` / 20newsgroups | 数据集 | — | 需联网下载 | — | 走 `--online` 显式开关，不进 CI |

**关键决策**：
1. **Tier-1 完全自足**：`numpy + scipy + scikit-learn` 三个 wheel 合计 **57.4 MB**，覆盖全部必需能力，零编译、零下载数据。
2. **LMNN / LFDA / NCA / ITML 自研**：原因不是"装不上"，而是 §8.1 实测的 **sklearn 1.9 API 不兼容**。副作用是正向的——MetricForge 具备"无第三方 DML 库依赖"的完整性，且自研实现可保证 I10 的逐位确定性。
3. **torch 仅作交叉验证**：用 torch 的 autograd 独立算一遍 $\mathcal{L}_{\text{PA}}^{(K)}$ 与 $\mathcal{L}_{\text{geom}}$ 的梯度，与 Tier-1 手写反向比对（I11 的第二重验证）。torch 缺失时该测试 skip，**不阻断**。

### 8.1 勘误：`metric-learn` / `umap-learn` 的可用性（v1.1 订正）

**v1.0 的错误**：v1.0 用"文件名是否包含 `win_amd64`"来判定可安装性。这个判据**系统性漏掉了所有 `-none-any` 通用纯 Python wheel**，因此把 `metric-learn` 与 `umap-learn` 误判为"无 wheel → 禁选"。正确判据是解析 PEP 427 的三段 tag。

**按正确判据重做的全表核验结果**：上表 25 个包**全部可安装**，没有任何一个因缺 wheel 而被禁。

**`metric-learn` 的实测（Python 3.13.14 + numpy 2.5.3 + scikit-learn 1.9.1）**：

| 步骤 | 结果 |
|---|---|
| `pip install metric-learn` | ✅ 成功（0.07 MB 纯 Python wheel，无需 MSVC） |
| `import metric_learn` | ✅ 成功，`__version__ = 0.7.0` |
| `LMNN().fit(X, y)` | ❌ `TypeError: check_X_y() got an unexpected keyword argument 'force_all_finite'` |
| `NCA().fit(X, y)` | ❌ 同上 |
| `ITML().fit(X, y)` | ❌ 同上 |
| `LFDA(k=3, dim=8)` | ❌ `TypeError: LFDA.__init__() got an unexpected keyword argument 'dim'` |

根因：sklearn 已将 `force_all_finite` 重命名为 `ensure_all_finite`（1.6 起弃用、1.8 起移除），并移除了 LFDA 的 `dim` 参数；`metric-learn` 最新版仍停留在 **0.7.0**，未跟进。

**因此**：
- **结论方向不变**：LMNN / NCA / ITML / LFDA 仍必须自研纯 numpy（§5 B2 / B3 / B7）。
- **理由更换**：从"wheel 缺失"（错误的元数据推断）改为"**与 sklearn 1.9 的 API 不兼容**"（可复现的实测证据）。后者更硬，也更可执行。
- **不做的事**：不为迁就 metric-learn 而下压 sklearn 版本（会拖累整个栈，且与"当前环境"约束冲突）。

#### 8.1.1 二次订正（v1.2）：打 shim 后 oracle **可行**

上文"连带影响"一条曾判定"用 metric-learn 作交叉验证 oracle **不可行**"。**该判定也需修正**：

- 上述崩溃是**未打兼容 shim** 时的裸跑结果。
- 主理人已裁决走**双轨**：自研纯 numpy 实现作为**运行时必经路径**（Tier-1），`metric-learn` + compat shim 降级为**可选交叉验证 oracle**。`pytorch-metric-learning` + torch 作手写梯度的 autograd oracle。三者均 `try/except` 探测，缺失即 skip，不阻断。
- 据杭落地实测：**打 shim 后 5 个 estimator 均能跑通**，故 oracle 路径成立（shim 的具体实现以其 compat 模块为准，本方案不规定实现细节）。

**最终的三层验证结构（双保险，不互相依赖）**：

| 层 | 机制 | 作用 | 缺失时 |
|---|---|---|---|
| 1 | **内在自证（不需任何第三方）** | LFDA：广义特征值问题的 Rayleigh 商单调性；NCA：I11 梯度检验；LMNN：目标邻居距离不增 + impostor 约束违反数下降的自洽性断言 | —（始终生效） |
| 2 | **metric-learn + shim oracle** | 与自研 LMNN/NCA/ITML/LFDA 在同样小数据上比对学到的 $M = L^\top L$ 与下游 kNN/MAP@R | skip，不阻断 |
| 3 | **torch autograd oracle** | 独立复算 $\mathcal{L}_{\text{PA}}^{(K)}$、$\mathcal{L}_{\text{geom}}$ 的梯度，与 Tier-1 手写反向比对 | skip，不阻断 |

> 第 1 层是**主防线**（不依赖 oracle，也不需要任何额外依赖）；第 2、3 层是**增量证据**。即便两个 oracle 都不可用，自研实现的正确性仍有内在自证兜底 —— 这正是当初设计第 1 层的原因，该设计**予以保留**。

**方法论教训（写入团队纪律）**：wheel 可安装性 ≠ 运行时可用性。元数据（wheel tag）只能回答"能不能装"，必须实跑一次 `import` + 一次最小 `fit` 才能回答"能不能用"。今后所有第三方 SOTA 后端选型都须走完这两步再下结论。

---

## 9. 架构落地映射（给系统工程师的接口约定）

```
metricforge/
  core/
    seeding.py        # set_seed(s) 唯一入口；确定性工具
    linalg.py         # l2norm / pairwise distance / similarity / logsumexp（含 scipy 降级）
    encoder.py        # numpy MLP (Tier-1)；MLPTorch (Tier-0, try/except)
  losses/
    pair.py           # contrastive / triplet(+semi-hard, batch-hard) / n_pairs / lifted
    proxy.py          # proxy_nca / proxy_anchor / multi_proxy_anchor (MQ-BPA)
    contrastive.py    # supcon / infonce
    hat.py            # HAT 温度求解（核心创新 1）
    geometry.py       # SNE-KL 邻域保持正则
    linear.py         # 自研 LMNN / NCA / ITML / LFDA
    compose.py        # 总损失 + PCGrad-lite
  mining/
    queue.py          # Fresh-Queue（存 raw X，每步重编码）
    sampler.py        # 类均衡 batch sampler（保证每类 >=2 样本）
  data/
    synth.py          # S1..S6 合成生成器（全部 seed 可控）
    real.py           # R1..R4 sklearn 内置；C 组 --online
    split.py          # P1 closed-set / P2 open-set；无泄漏封装
  eval/
    metrics.py        # MAP@R（必须复现 Musgrave Table 3）/ R@1 / RP / kNN / NMI / ARI
    protocol.py       # train/val/test 一次流转，防 test 二次使用
  bench/
    baselines.py      # B1..B8
    run.py            # 3 seeds × 方法 × 数据集，落 csv/json
    ablation.py       # A1..A7
  tests/
    invariants.py     # I1..I14
```

**给 ai-system-engineer 的三条硬约束**：
1. `hat_temperature` 的二分**迭代次数必须是固定常数**（不得用 while 收敛判据），否则 I10 逐位确定性无法保证。
2. 相似度矩阵**禁止物化全局 $N\times N$**（S4 有 $N{=}5000$），必须分块；peak RSS 断言进单测。
3. 所有随机性必须走 `core.seeding`，任何 `np.random.default_rng()` 的无参调用在 lint 中报错。

---

## 10. 参考（含年份，便于报告引用）

| 主题 | 引用 |
|---|---|
| Reality Check / MAP@R | Musgrave, Belongie, Lim. *A Metric Learning Reality Check*. ECCV 2020 (arXiv:2003.08505) |
| 库参考 | Musgrave et al. *PyTorch Metric Learning* (arXiv:2008.09164), v2.9.0 2025-08 |
| Sampling matters / Margin loss | Wu, Manmatha, Smola, Krahenbuhl. ICCV 2017 |
| Triplet / semi-hard | Schroff, Kalenichenko, Philbin. *FaceNet*. CVPR 2015 |
| Batch-hard | Hermans, Beyer, Leibe. *In Defense of the Triplet Loss*. 2017 |
| N-pairs | Sohn. NeurIPS 2016 |
| Lifted Structure | Oh Song et al. CVPR 2016 |
| ProxyNCA | Movshovitz-Attias et al. ICCV 2017 |
| Proxy-Anchor | Kim, Kim, Kim, Kim. CVPR 2020 |
| SoftTriple（多原型） | Qian et al. ICCV 2019 |
| SupCon | Khosla et al. NeurIPS 2020 |
| InfoNCE/CPC | Oord, Li, Vinyals. 2018 |
| MS loss / GPW | Wang et al. CVPR 2019 |
| LMNN | Weinberger & Saul. JMLR 2009 |
| ITML | Davis, Kulis, Jain, Sra, Dhillon. ICML 2007 |
| NCA | Goldberger, Roweis, Hinton, Salakhutdinov. NIPS 2005 |
| LFDA | Sugiyama. JMLR 2007 |
| MoCo 队列 | He, Fan, Wu, Xie, Girshick. CVPR 2020 |
| XBM | Wang et al. CVPR 2020 |
| PCGrad | Yu et al. NeurIPS 2020 |
| SNE | Hinton & Roweis 2002 |
| 表格数据上的 DL vs ML | Rabbani, Medri, Samad. *Int. J. Data Sci. Anal.* 20(4):3069–3091, 2025 |

---

## 附：待验证 / 不确定项（明确标注）

| 项 | 状态 |
|---|---|
| SOTA 表中 GAPan / LaFG 的 venue 与年份 | 检索自 arXiv 预印本聚合页，**venue 为二手标注，待验证** |
| Papers-with-Code DML leaderboard 当前可用性 | **待验证**（未直接访问） |
| `psutil` cp313 wheel 检索到的是 `cp313t`（free-threaded）变体 | 需实测 `pip install` 在 3.13.14 默认（GIL）解释器上是否解析成功；失败则降 `tracemalloc` |
| torch 2.14.0 在 AMD Ryzen CPU 上的实际 fit 加速比 | **待实测**；若无加速则 Tier-0 仅保留"梯度交叉验证"用途 |
| S6 `hier_gauss` 上 $K{>}1$ 的优势幅度（预期 ≥0.03） | **预测值，未实测**；若实测 < 0.01 则按 §3.2 证伪流程处理 |
| HAT 的 $\kappa=0.25$ 默认值 | 基于"约 1/4 负样本活跃"的启发式，**需在 val 上确认跨数据集稳定性**；若波动大则改自适应 $\kappa$（写入 v1.1） |

### 附-补：v1.3 工程侧实测订正（杭落地，2026-09-30）

上表两项"待验证"已实测关闭，脚本与结果如下（均可复现）：

| 待验证项 | 实测结论 | 证据 / 脚本 |
|---|---|---|
| `psutil` cp313 wheel 是否可用 | ✅ **可用，无需降级 `tracemalloc`**。解析到 `psutil-7.2.2-cp37-abi3-win_amd64.whl`（abi3 轮子，GIL 解释器直接吃），`pip install psutil` 成功，`Process().memory_info().rss` 实测可读出 45.6 MB | `tools/wheel_dump.py`；env_precheck §4 |
| torch 2.14.0 在 AMD CPU 上的加速比 | ⚠️ **有加速，但强依赖线程数**：最佳 `torch.set_num_threads(2)` → **2.49×**（0.740 vs numpy 1.843 ms/step）；默认 8 线程 → 2.14×；**16 线程反而劣化到 0.34×（5.397 ms）** | `tools/bench_torch_vs_numpy.py` |

torch 线程数实测扫描（$B{=}256,\ 64\to128\to32$，前+反传播，5×40 次中位数）：

```
numpy      :    1.843 ms/step
torch t= 1 :    0.993 ms/step  (1.85x)
torch t= 2 :    0.740 ms/step  (2.49x)   ← 最佳
torch t= 4 :    0.880 ms/step  (2.09x)
torch t= 8 :    0.860 ms/step  (2.14x)
torch t=16 :    5.397 ms/step  (0.34x)   ← 严重过订阅
```

**落地要求（工程侧）**：
1. Tier-0 初始化时**显式** `torch.set_num_threads(min(4, os.cpu_count()))`，**禁止**放任默认（本机默认 8 尚可，但 `cpu_count=16` 一旦被取用会劣化 3 倍）。
2. torch **不只是**梯度 oracle —— 在 CPU 小 batch 上它比 numpy 快约 2.1–2.5×，因此 §4.4 的 ≤60 s 预算下 Tier-0 有实际价值；但因其为可选依赖，Tier-1 仍是运行时必经路径。
3. torch 作 autograd oracle 的数值可信度已验证：与手写 numpy 梯度 **maxAbsDiff = 8.674e-18 (dW1) / 2.776e-17 (dW2)**，远优于 I11 的 $10^{-6}$ 门槛。

> 方法学备注：首轮基准曾测得 torch 8.282 ms/step（劣于 numpy），系计时循环内混入 `loss.item()` 同步与 `.clone()` 分配所致。已改为"循环内不做同步/拷贝、取 5 次中位数"后复测，以上述数字为准。
