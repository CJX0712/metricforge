# MetricForge Related Work —— SOTA 对标与参考文献

# 作者：晨星

> **版本**：v1.0 | **日期**：2026-09-30 | **检索时点**：2026-09-30
> **配套文档**：`docs/algorithm_design.md`（方案，§1 为其摘要来源）、`docs/model_card.md`（模型卡）

---

## ⚠️ 文首声明：本节逐字复制，一字未改

以下内容**逐字复制自 `docs/algorithm_design.md` §1.3，未作任何修改**（包括标题编号、措辞与标点）。其中的"§7"指**源文档** `algorithm_design.md` 的 §7（可验证不变量），非本文的章节。

---

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

（以上声明结束，以下为本文正文。）

---

## 1. 领域背景：度量学习的两条演进主线

**主线一：损失函数族**（2015–2020）。从 Contrastive / Triplet（Schroff et al., CVPR 2015）出发，先后出现 N-pairs（Sohn, NeurIPS 2016）、Lifted-Structure（Oh Song et al., CVPR 2016）、Margin / Distance-Weighted Sampling（Wu et al., ICCV 2017）、ProxyNCA（Movshovitz-Attias et al., ICCV 2017）、Multi-Similarity（Wang et al., CVPR 2019）、SoftTriple（Qian et al., ICCV 2019）、Proxy-Anchor（Kim et al., CVPR 2020）。

**主线二：外部监督与 backbone**（2020–至今）。2022 年之后，CUB-200 / Cars196 上 Recall@1 的大幅提升**主要来自 backbone 从 ResNet-50 换到 ViT，以及引入语言/生成式监督**，而非损失函数本身的改进。

**关键的复现性警示**：Musgrave, Belongie & Lim 的 *A Metric Learning Reality Check*（ECCV 2020）在统一训练策略、统一超参搜索、统一划分后发现，所谓 SOTA 相对 contrastive / triplet 的增益"**marginal at best**"，并指出大量论文存在超参直接对 test 调、用训练过程中最优 checkpoint、划分不一致等问题。**本系统的评测协议（`algorithm_design.md` §4.2 / §5.2）正是为规避这些问题而设计的。**

---

## 2. 公开 Benchmark 结果（CUB-200-2011 / Cars196 / SOP，Recall@1）

### 2.1 数据性质声明（**读表前必看**）

| 项 | 说明 |
|---|---|
| 数值来源 | 下表数字**均为二手聚合数据**，抄录自下述论文中的对比表（而非官方 leaderboard 直接抓取）。**未经本仓库复现。** |
| Backbone 条件 | 全部为 **PyTorch + GPU + ImageNet 预训练** backbone（R50 或 ViT）。 |
| 可比性 | 同一 backbone 行内可比；**跨 backbone 组不可比**。 |
| 时效标注 | 表中"年份"部分为来源论文的二手标注，少数条目（GAPan、DVA）为 arXiv 预印本，**正式 venue 待验证**。 |
| SOP 列 | 来源表表头解析不唯一（部分表头被压缩为 `1 2 4 8 1`），此处按上下文取 **R@1** 列，**标注待验证**。 |

### 2.2 结果表

| 方法 | Backbone | CUB-200-2011 R@1 | Cars196 R@1 | SOP R@1 | 年份 / venue | 来源 |
|---|---|---|---|---|---|---|
| CBML | R50 | 69.9 | 80.4 | 87.2 | TPAMI 2023 | [GAPan 表 2](https://arxiv.org/html/2605.09859v1) |
| NIR | R50 | 70.5 | 80.6 | — | CVPR 2022 | 同上 |
| HSE | R50 | 70.6 | 80.1 | 87.1 | ICCV 2023 | 同上 |
| IDML | R50 | 70.7 | 80.2 | — | TPAMI 2024 | 同上 |
| Proxy-Anchor | R50 | 71.1 | 80.4 | — | CVPR 2020 | [DVA 表 2](https://arxiv.org/html/2506.16273v3) |
| HIST | R50 | 71.4 | 81.1 | 88.1 | CVPR 2022 | 同上 |
| PNCA++ | R50 | 72.2 | 82.0 | 89.2 | ECCV 2020 | 同上 |
| MS loss + S2SD | R50 | 67.7 | 86.5 | 77.7 | CVPR 2022（Roth et al. 表 2） | [CVPR OpenAccess PDF](https://openaccess.thecvf.com/content/CVPR2022/papers/Roth_Integrating_Language_Guidance_Into_Vision-Based_Deep_Metric_Learning_CVPR_2022_paper.pdf) |
| DIML | ViT | 76.7 | — | — | TPAMI 2024 | [LaFG 表 3](https://ar5iv.arxiv.org/html/2512.06255) |
| LM-Metric | ViT | 77.1 | — | — | PR 2024 | 同上 |
| DFML-PA | ViT | 79.1 | 86.8 | — | CVPR 2023 | 同上 |
| DVA | ViT | 84.9 | 90.6 | 94.5 | arXiv 2025（**venue 待验证**） | [arXiv 2506.16273](https://arxiv.org/html/2506.16273v3) |
| DPHM | ViT | 85.5 | 91.3 | 94.6 | PR 2025 | LaFG 表 3 |
| Hyp-ViT | ViT | 85.6 | 91.4 | 94.8 | CVPR 2022 | 同上 |
| HIER | ViT | 85.7 | 91.3 | 94.4 | CVPR 2023 | 同上 |
| SEE | ViT | 85.8 | 91.4 | 94.6 | IJCAI 2025 | 同上 |
| DDML | ViT | 86.0 | 91.7 | 95.2 | AAAI 2025 | 同上 |
| VPTSP-G | ViT | 86.6 | 91.7 | 94.8 | ICLR 2024 | 同上 |
| LaFG | ViT | 87.2 | 92.4 | 95.2 | IJCAI 2025 | [LaFG](https://ar5iv.arxiv.org/html/2512.06255) |
| GAPan | ViT | **89.1** | **93.5** | **95.3** | arXiv 预印本（**venue 待验证**） | [GAPan](https://arxiv.org/html/2605.09859v1) |

### 2.3 读表结论（三条）

1. **backbone 换代的贡献远大于损失函数**：R50 组的 R@1 集中在 69.9–72.2（跨度 2.3 点），ViT 组的跨度从 76.7 到 89.1。R50 → ViT 的跃迁贡献约 **+13～15 点**，而同期各损失函数在同一 backbone 下的差距仅 **1～3 点**。
2. **2025 年后的新增增益多数来自外部监督**：LLM 生成的属性描述（LaFG）、生成式外观先验（GAPan）、语言引导（ELG/PLG）属于"多模态监督"，不是度量学习算法层的改进。**本系统在无网络、无预训练权重的条件下不追这条线。**
3. **这与 Reality Check 的结论一致**：在公平协议下，算法层的边际收益很小。这既是对领域的警示，也说明本系统把创新点放在**难度课程（HAT）**与**类内结构建模（MQ-BPA）**上是合理的定位——它们是"训练动态"与"类内几何"层面的改进，不与 backbone 竞赛正面冲突。

### 2.4 Leaderboard 资源

| 资源 | 说明 | 时效 / 状态 |
|---|---|---|
| [KevinMusgrave/powerful-benchmarker](https://github.com/KevinMusgrave/powerful-benchmarker) | ECCV 2020 Reality Check 的官方 benchmark：4-fold CV、在后半类上测试，含贝叶斯超参搜索日志 | 实质更新停在 ~2021，**结果表维护已停滞**，引用须注明时效 |
| [KevinMusgrave/pytorch-metric-learning](https://github.com/KevinMusgrave/pytorch-metric-learning) | 事实上的工业标准库（v2.9.0，2025-08），30+ loss / miner / AccuracyCalculator | 依赖 torch；本系统**仅作算法口径参考**，非运行时依赖 |
| sota2.com 等聚合站 | 二手聚合（如 CUB-200-2011 DML R@1 = 73.23，时点标 2024） | **不作权威引用** |
| Papers-with-Code (DML) | 历史 leaderboard | 站点当前状态**待验证** |

---

## 3. 参考文献（完整列表）

格式：**作者 / 年份 / 会议或期刊 / arXiv 或 DOI**。标注 `[*]` 者为本系统的**直接算法组件来源**；标注 `[~]` 者为**已知方法的工程集成**（不计入自研创新）。

### 3.1 评测与复现性

| 文献 | 作者 | 年份 | 会议/期刊 | arXiv / DOI |
|---|---|---|---|---|
| A Metric Learning Reality Check | Musgrave, Belongie, Lim | 2020 | **ECCV 2020** | arXiv:2003.08505 · DOI 10.48550/arXiv.2003.08505 |
| PyTorch Metric Learning | Musgrave, Belongie, Lim | 2020 | arXiv（软件库） | arXiv:2008.09164 |
| Powerful Benchmarker | Musgrave, Lim, Belongie | 2019 | GitHub（工具） | https://github.com/KevinMusgrave/powerful-benchmarker |

### 3.2 深度度量学习：pair / triplet 族 `[*]`

| 文献 | 作者 | 年份 | 会议/期刊 | arXiv / DOI |
|---|---|---|---|---|
| FaceNet（triplet / semi-hard mining） | Schroff, Kalenichenko, Philbin | 2015 | **CVPR 2015** | arXiv:1503.03832 |
| In Defense of the Triplet Loss（batch-hard） | Hermans, Beyer, Leibe | 2017 | arXiv（后发表于 ICPR 2018 相关venue） | arXiv:1703.07737 |
| Sampling Matters in Deep Embedding Learning（DWS / Margin loss） | Wu, Manmatha, Smola, Krähenbühl | 2017 | **ICCV 2017** | DOI 10.1109/ICCV.2017.309 |
| Improved Deep Metric Learning with Multi-class N-pair Loss | Sohn | 2016 | **NeurIPS 2016** | arXiv:1609.07257 |
| Deep Metric Learning via Lifted Structured Feature Embedding | Oh Song, Xiang, Jegelka, Sra | 2016 | **CVPR 2016** | arXiv:1511.06452 |
| Multi-Similarity Loss with General Pair Weighting | Wang, Han, Huang, et al. | 2019 | **CVPR 2019** | arXiv:1904.06627 |
| Integrating Language Guidance into Vision-Based DML（ELG / PLG） | Roth, Roth, Chitta, et al. | 2022 | **CVPR 2022** | CVF OpenAccess（见 §2.2 来源列） |

### 3.3 深度度量学习：proxy 族 `[*]`

| 文献 | 作者 | 年份 | 会议/期刊 | arXiv / DOI |
|---|---|---|---|---|
| No Fuss Distance Metric Learning using Proxies（**ProxyNCA**） | Movshovitz-Attias, Toshev, LeCun, Najibi, Kannan, Ioffe, Singh | 2017 | **ICCV 2017** | arXiv:1703.07464 · DOI 10.1109/ICCV.2017.47 |
| Proxy Anchor Loss for Deep Metric Learning（**Proxy-Anchor**） | Kim, Kim, Kim, Kim | 2020 | **CVPR 2020** | arXiv:2003.13911 |
| SoftTriple Loss: Deep Metric Learning Without Triplet Sampling（**多原型**） | Qian, Shang, Sun, Li, Jin | 2019 | **ICCV 2019** | arXiv:1902.00957 |
| Multi Proxy Anchor Loss（MPA） | — | 2021 | Neurocomputing | arXiv:2110.03997 |

### 3.4 对比表征学习 `[*]`

| 文献 | 作者 | 年份 | 会议/期刊 | arXiv / DOI |
|---|---|---|---|---|
| Representation Learning with Contrastive Predictive Coding（**InfoNCE / CPC**） | Oord, Li, Vinyals | 2018 | arXiv | arXiv:1807.03748 |
| A Simple Framework for Contrastive Learning of Visual Representations（**SimCLR / NT-Xent**） | Chen, Kornblith, Norouzi, Hinton | 2020 | **ICML 2020** | arXiv:2002.05709 |
| Supervised Contrastive Learning（**SupCon**） | Khosla, Teterwak, Wang, et al. | 2020 | **NeurIPS 2020** | arXiv:2004.11362 |
| Momentum Contrast for Unsupervised Visual Representation Learning（**MoCo / 记忆队列**） | He, Fan, Wu, Xie, Girshick | 2020 | **CVPR 2020** | arXiv:1911.05722 |
| Cross-Batch Memory for Embedding Learning（**XBM**） | Wang, Zhang, Huang, Scott | 2020 | **CVPR 2020（Oral）** | arXiv:1912.06798 · DOI 10.1109/CVPR42600.2020.00642 |

> **Fresh-Queue 的定位说明**：本系统的跨批记忆队列与 MoCo 队列、XBM 同属"跨批负样本池"这一**已知机制**。本系统的增量在于：① 队列存**原始输入**而非历史嵌入，每步用当前 encoder **重新前向**（消除 staleness，因而不需要 momentum encoder）；② 与 HAT 的参与率目标耦合；③ 与多原型形式统一。这三点之外的队列机制本身**不是本系统首创**，见 `algorithm_design.md` §3.3 的非创新声明。

### 3.5 线性 / 经典度量学习 `[*]`

| 文献 | 作者 | 年份 | 会议/期刊 | 标识 |
|---|---|---|---|---|
| Distance Metric Learning for Large Margin Nearest Neighbor Classification（**LMNN**） | Weinberger, Saul | 2009 | **JMLR 10:207–244** | JMLR |
| Information-Theoretic Metric Learning（**ITML**） | Davis, Kulis, Jain, Sra, Dhillon | 2007 | **ICML 2007** | DOI 10.1145/1273496.1273518 |
| Neighbourhood Components Analysis（**NCA**） | Goldberger, Roweis, Hinton, Salakhutdinov | 2005 | **NIPS 2005** | NIPS Proceedings |
| Local Fisher Discriminant Analysis（**LFDA**） | Sugiyama | 2007 | **JMLR 8:1027–1061** | JMLR |
| Neighbourhood Components Analysis 的深度学习扩展 | — | — | — | 见 3.2/3.3 |

### 3.6 多任务梯度与几何正则 `[~]`

| 文献 | 作者 | 年份 | 会议/期刊 | arXiv / DOI |
|---|---|---|---|---|
| Gradient Surgery for Multi-Task Learning（**PCGrad**） | Yu, Kumar, Gupta, Levine, Hausman, Finn | 2020 | **NeurIPS 2020** | arXiv:2001.06782 · [NeurIPS Proceedings](https://proceedings.neurips.cc/paper/2020/hash/3fe78a8acf5fda99de95303940a2420c-Abstract.html) |
| Stochastic Neighbor Embedding（**SNE**） | Hinton, Roweis | 2002 | **NIPS 2002** | NIPS Proceedings |
| Visualizing Data using t-SNE | van der Maaten, Hinton | 2008 | **JMLR 9:2579–2605** | JMLR |

> **勘误记录**：本仓库 `algorithm_design.md` 的早期版本曾将 PCGrad 标注为 "ICML 2020"，**经核实应为 NeurIPS 2020**（arXiv:2001.06782，NeurIPS 2020 Proceedings）。已于 v1.2 修正。

### 3.7 表格 / 向量数据上的学习（本系统的真实数据语境）

| 文献 | 作者 | 年份 | 会议/期刊 | arXiv / DOI |
|---|---|---|---|---|
| Attention versus contrastive learning of tabular data: a data-centric benchmarking | Rabbani, Medri, Samad | 2025 | **Int. J. Data Sci. Anal. 20(4):3069–3091** | DOI 10.1007/s41060-024-00681-z |
| ICE-T: Interactions-aware Cross-column Contrastive Embedding for Heterogeneous Tabular Datasets | Tokar, Sanner | 2025 | **AAAI 2025** | DOI 10.1609/aaai.v39i20.35385 |

### 3.8 评测指标

| 文献 | 作者 | 年份 | 说明 |
|---|---|---|---|
| Product Quantization for Nearest Neighbor Search（Recall@k 的检索评测用法） | Jegou, Douze, Schmid | 2011 | TPAMI 2011 |
| Introduction to Information Retrieval（NMI / 聚类外部指标） | Manning, Raghavan, Schütze | 2008 | Cambridge University Press |

---

## 4. 本系统为何不与上述 leaderboard 直接比较

这是本文最关键的一节。**下表逐条给出不可比的技术理由，任何对外材料不得跳过或淡化。**

### 4.1 四条硬性的不可比理由

| # | 维度 | leaderboard 侧 | MetricForge 侧 | 后果 |
|---|---|---|---|---|
| **R1** | **输入模态** | 原始图像（224×224 裁剪），backbone 直接从像素学习 | **已特征化的向量**（表格特征 / 合成向量 / sklearn 内置） | 任务层不同：前者是"表示学习 + 度量"，后者只有"度量"这一层 |
| **R2** | **预训练权重** | ImageNet 预训练的 **ViT / ResNet-50** | **随机初始化小 MLP**，无任何预训练权重 | 起点完全不同。leaderboard 上 R50→ViT 的 +13～15 点增益本系统**根本无法获取** |
| **R3** | **算力与预算** | GPU（3090 / A100）× 数小时，batch 可达 1800 | **CPU-only**（AMD Ryzen），单次 `fit` ≤ 60 s，batch 128 | 训练步数与负样本规模差 1–2 个数量级。XBM 已证明负样本池规模对 DML 影响巨大，此项直接决定性能上限 |
| **R4** | **数据规模与类数** | 5k–60k 图像，100–11,318 类 | 150–20,000 样本，2–50 类 | 部分重叠但重心不同：leaderboard 偏向"极多类、每类极少样本"（SOP 11k 类），本系统偏向"中等类数、类内多模态" |

### 4.2 三条方法论理由

| # | 理由 | 说明 |
|---|---|---|
| **R5** | **评测协议不同** | leaderboard 多用 closed-set 或特定划分；本系统主协议是 **P2 open-set（类不相交，后半类测试）**，这是 Musgrave et al. ECCV 2020 推荐的、也是度量学习唯一有意义的泛化检验。协议不同则数字不可直接并列。 |
| **R6** | **指标口径不同** | leaderboard 惯用 **Recall@1**；本系统主指标是 **MAP@R**。Musgrave et al. 指出 MAP@R 信息量更大且更稳定（lag-1 自相关 0.81 vs R@1 的 0.73），但两者**数值不可互换**。 |
| **R7** | **复现性风险** | Reality Check 已证明该领域存在大量超参对 test 调、最优 checkpoint、划分不一致的问题。直接引用他人数字而不复现，等于继承这些偏差。本系统所有数字必须**自测**，这也是 `model_card.md` §0 强制 `{{TODO}}` 占位的原因。 |

### 4.3 结论：本系统的正确对标对象

**MetricForge 的对标对象是算法族，不是 leaderboard 排名。**

在**相同向量输入、相同 CPU 预算、相同种子协议、相同调参预算**下，与下列方法逐一比较：

- **pair / triplet 族**：Contrastive、Triplet（含 semi-hard / batch-hard）、N-pairs、Lifted-Structure、Margin
- **proxy 族**：ProxyNCA、Proxy-Anchor、SoftTriple / Multi-Proxy
- **对比族**：SupCon、InfoNCE / NT-Xent
- **线性族**：LMNN、ITML、NCA、LFDA
- **零学习下界**：原始空间欧氏 kNN

并且所有深度基线（Triplet-BH / Proxy-Anchor / SupCon）**必须与 MetricForge 共用同一 encoder 架构、同一优化器、同一 epoch 数、同一超参搜索预算**——这是吸取 Reality Check 教训的核心措施，否则比较无意义。

### 4.4 一句话总结（可用于对外材料）

> MetricForge 不在 CUB-200-2011 / Cars196 / SOP 的图像检索 leaderboard 上竞争，也不声称达到这些榜单上的任何数字。MetricForge 的贡献是**在 CPU-only、无预训练权重、零下载的约束下，提供一套可逐位复现、带自证不变量的度量学习算法实现，并在公平协议下与该算法族比较**。所有性能声明均限于此口径。
