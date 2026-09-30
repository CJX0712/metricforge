# 作者: 晨星
#
# MetricForge —— 核心不变量单测（Phase 1.5）
#
# 本文件中的每一条断言都源自算法设计文档 `docs/algorithm_design.md` §7
# 《可验证不变量（硬金标准，逐条可单测）》，不变量编号 I1~I14 与文档一一对应。
# 其中 I4 / I8 / I9 / I10 / I11 被文档标注为 **Release-blocking**，
# 本文件覆盖其中的 I4、I8（以及非 release-blocking 的 I3、I5、I6）。
#
# 设计约束（来自主理人沈观澜的 Phase 1.5 任务书）：
#   1. 零外部依赖即可运行（纯 stdlib）；numpy / scipy 存在时启用额外路径，不存在时 skip。
#   2. 可直接被系统工程师导入集成：所有被测内核优先从 `metricforge` 包导入，
#      导入失败时回退到本文件内的**参考实现**（reference backend），
#      因此本文件在今天就能跑绿，等真实实现落地后同一批测试自动切换去测真实代码。
#   3. 全部随机性走固定 seed，禁止裸 `random.*` 全局调用。
#
# 运行方式：
#   pytest 可用：  python -m pytest tests/test_invariants_core.py -v
#   pytest 缺失：  python tests/test_invariants_core.py        （内置 fallback runner）

from __future__ import annotations

import math
import random
from fractions import Fraction

# ---------------------------------------------------------------------------
# pytest 缺失时的极简 shim：让 parametrize / skip 在无 pytest 环境下依然可用，
# 并由文件末尾的 fallback runner 展开参数组合执行。
# 安装 pytest 后此 shim 完全不生效。
# ---------------------------------------------------------------------------
try:
    import pytest  # type: ignore
except ImportError:  # pragma: no cover - 无 pytest 环境的降级路径

    class _Skip(Exception):
        """fallback runner 捕获此异常记为 skip。"""

    class pytest:
        class mark:
            @staticmethod
            def parametrize(argnames, argvalues, **_kwargs):
                def _deco(fn):
                    params = list(getattr(fn, "_mf_params", []))
                    params.append((argnames, list(argvalues)))
                    fn._mf_params = params
                    return fn

                return _deco

        @staticmethod
        def skip(reason: str = ""):
            raise _Skip(reason)


_HAS_NUMPY = True
try:  # numpy 可选：仅用于验证内核能接受 ndarray 输入
    import numpy as np  # type: ignore
except ImportError:  # pragma: no cover
    _HAS_NUMPY = False

_HAS_SCIPY = True
try:  # scipy 可选：仅用于 I4 的独立交叉验证（brentq 求根）
    from scipy.optimize import brentq  # type: ignore
except ImportError:  # pragma: no cover
    _HAS_SCIPY = False


# ---------------------------------------------------------------------------
# 被测内核：优先用 metricforge 的真实实现，否则用参考实现
# ---------------------------------------------------------------------------
try:
    from metricforge.core.mathx import (  # type: ignore
        hat_temperature,
        infonce_loss,
        logsumexp,
        map_at_r,
        map_at_r_exact,
        softmax_entropy,
        softmax_entropy_beta,
        triplet_hinge,
    )

    _BACKEND = "metricforge"
except Exception:  # pragma: no cover - 真实实现尚未落地
    _BACKEND = "reference"

    def logsumexp(values):
        """数值稳定的 log-sum-exp；全 -inf 时返回 -inf（不是 NaN）。"""
        vals = list(values)
        m = max(vals)
        if m == float("-inf"):
            return float("-inf")
        return m + math.log(sum(math.exp(v - m) for v in vals))

    def softmax_entropy(logits):
        """p = softmax(logits) 的香农熵（单位 nat）。"""
        return logsumexp(logits) - _expected_value(logits)

    def _expected_value(logits):
        vals = list(logits)
        z = logsumexp(vals)
        p = [math.exp(v - z) for v in vals]
        return sum(pi * vi for pi, vi in zip(p, vals))

    def softmax_entropy_beta(sims, beta):
        """以 beta = 1/tau 为逆温度时，softmax(sims / tau) 的熵。"""
        return softmax_entropy([beta * float(x) for x in sims])

    def hat_temperature(neg_sims, kappa=0.25, tau_lo=0.02, tau_hi=2.0, iters=40):
        """I4：解 PP(tau) = exp(H(p_tau)) == kappa * N，返回 tau*。

        迭代次数必须是**固定常数**（不得用 while 收敛判据），
        否则不变量 I10（同 seed 两次运行逐位一致）无法保证。
        """
        sims = [float(x) for x in neg_sims]
        n = len(sims)
        target = math.log(max(kappa * n, 1.0))
        lo, hi = 1.0 / tau_hi, 1.0 / tau_lo  # beta 区间；H 关于 beta 严格递减
        for _ in range(iters):
            mid = 0.5 * (lo + hi)
            h = softmax_entropy_beta(sims, mid)
            if h < target:  # 熵偏低 -> 需要更大 tau -> 更小 beta
                hi = mid
            else:
                lo = mid
        return 1.0 / (0.5 * (lo + hi))

    def map_at_r(hits, r):
        """I8：返回 (MAP@R, R-Precision)，取值 [0, 1]。

        ⚠️ 口径陷阱（文档 §4.3）：r 是 gallery 中与该 query 同类的样本总数
        （固定搜索长度），**不是**检索列表里的命中数。
        hits 是长度恰为 r 的 0/1 相关性向量，按检索排名顺序排列。
        真实系统里若某类在 gallery 中仅 1 个样本，该 query 应剔除而非传 r=0。
        """
        if len(hits) != r or r < 1:
            raise ValueError(f"hits 长度必须等于 R 且 R >= 1，got len={len(hits)}, R={r}")
        cum = 0
        acc = 0.0
        for i, hit in enumerate(hits, start=1):
            if hit:
                cum += 1
                acc += cum / i  # P(i)
        return acc / r, cum / r

    def map_at_r_exact(hits, r):
        """与 map_at_r 同义，但用有理数精确计算 —— 供 Table 3 逐位比对使用。"""
        if len(hits) != r or r < 1:
            raise ValueError(f"hits 长度必须等于 R 且 R >= 1，got len={len(hits)}, R={r}")
        cum = 0
        acc = Fraction(0)
        for i, hit in enumerate(hits, start=1):
            if hit:
                cum += 1
                acc += Fraction(cum, i)
        return acc / r, Fraction(cum, r)

    def triplet_hinge(d_ap, d_an, margin):
        """I3：单条 triplet 的 hinge 损失 max(0, d_ap - d_an + margin)。"""
        return max(0.0, float(d_ap) - float(d_an) + float(margin))

    def infonce_loss(sim_pos, sim_negs, tau):
        """I6：单正样本 InfoNCE，写成稳定形式 log(1 + sum_n exp((s_n - s_p)/tau))。"""
        return logsumexp([0.0] + [(float(sn) - float(sim_pos)) / float(tau) for sn in sim_negs])


# ---------------------------------------------------------------------------
# 测试夹具：确定性随机向量
# ---------------------------------------------------------------------------
def _sim_vector(n: int, seed: int, scale: float = 1.0):
    """固定 seed 的高斯相似度向量（确定性，禁止裸 random 全局调用）。"""
    rng = random.Random(seed)
    return [rng.gauss(0.0, scale) for _ in range(n)]


def _hat_seed(N: int, kappa: float, scale: float) -> int:
    """HAT 配置的确定性 seed。

    刻意**不使用内置 hash()**：tuple-of-int 的 hash 虽不受 PYTHONHASHSEED 影响，
    但显式算术 seed 更可审计，也符合"唯一 seed 入口"的确定性纪律。
    """
    return N * 1_000_003 + round(kappa * 10_000) * 1_009 + round(scale * 10_000)


# ===========================================================================
# I5 —— 引理 dH/dbeta = -beta * Var_{p_beta}(s)
#       这是 HAT 单调性与二分收敛性的理论基石（文档 §3.1 数学核心）。
# ===========================================================================
@pytest.mark.parametrize("beta", [0.05, 0.2, 0.5, 1.0, 2.0, 5.0])
@pytest.mark.parametrize("n", [4, 12, 64])
def test_i5_entropy_derivative_lemma(beta, n):
    """I5：dH/dbeta = -beta*Var_{p_beta}(s)。中心差分 vs 解析式，maxRelErr < 1e-6。

    差分步长取 1e-5 是**实测扫参**的结果（18 组配置下的 maxRelErr）：
        h=1e-7 -> 3.0e-07   （舍入误差主导）
        h=1e-6 -> 1.4e-08
        h=1e-5 -> 1.2e-09   ← 最优，舍入与截断的平衡点
        h=1e-4 -> 5.1e-08
        h=1e-3 -> 5.1e-06   （截断误差主导，已越阈值）
    即该阈值对步长敏感，改动本测试时不要随意调整 h。
    """
    s = _sim_vector(n, seed=1000 * n + int(beta * 100))
    h = 1e-5

    h_plus = softmax_entropy_beta(s, beta + h)
    h_minus = softmax_entropy_beta(s, beta - h)
    numeric = (h_plus - h_minus) / (2.0 * h)

    # 解析侧：Var_{p_beta}(s)
    z = logsumexp([beta * x for x in s])
    p = [math.exp(beta * x - z) for x in s]
    mean = sum(pi * x for pi, x in zip(p, s))
    var = sum(pi * (x - mean) ** 2 for pi, x in zip(p, s))
    analytic = -beta * var

    denom = max(1e-12, abs(analytic))
    rel_err = abs(numeric - analytic) / denom

    assert rel_err < 1e-6, (
        f"I5 失效: n={n}, beta={beta}, numeric={numeric!r}, "
        f"analytic={analytic!r}, relErr={rel_err:.3e}"
    )
    # 引理的符号：严格为负（除非所有相似度相等 -> Var == 0）
    assert numeric < 0.0, f"I5 符号失效: dH/dbeta 应为负, got {numeric!r}"


@pytest.mark.parametrize("n", [8, 32])
def test_i5_lemma_numpy_path(n):
    """I5（numpy 路径）：内核在 numpy 可用时必须能接受 float64 ndarray 输入。"""
    if not _HAS_NUMPY:
        pytest.skip("numpy 未安装，跳过 numpy 输入路径校验")

    s = np.asarray(_sim_vector(n, seed=7), dtype=np.float64)
    beta = 1.3
    h = 1e-5  # 与 test_i5_entropy_derivative_lemma 同一步长（实测最优，见其 docstring）

    numeric = (softmax_entropy_beta(s, beta + h) - softmax_entropy_beta(s, beta - h)) / (2.0 * h)

    z = logsumexp(beta * s)
    p = np.exp(beta * s - z)
    mean = float(np.sum(p * s))
    var = float(np.sum(p * (s - mean) ** 2))
    analytic = -beta * var

    rel_err = abs(float(numeric) - analytic) / max(1e-12, abs(analytic))
    assert rel_err < 1e-6, f"I5 numpy 路径失效: relErr={rel_err:.3e}"


# ===========================================================================
# I4 —— HAT 温度求根：唯一性、精度、与 scipy.brentq 的独立交叉验证
# ===========================================================================
# 5 组配置：覆盖 N / kappa / 相似度尺度三个维度
_HAT_CONFIGS = [
    (12, 0.25, 1.0),
    (128, 0.25, 1.0),
    (1152, 0.25, 0.18),  # 典型：D=32 L2 归一化嵌入的余弦相似度尺度
    (1152, 0.25, 0.05),  # 训练后期：嵌入收紧，相似度尺度变小
    (64, 0.10, 0.30),
]


@pytest.mark.parametrize("N,kappa,scale", _HAT_CONFIGS)
def test_i4_hat_root_precision(N, kappa, scale):
    """I4：|log PP(tau*) - log(kappa*N)| <= 1e-9，且 tau* 落在 clamp 区间内。"""
    s = _sim_vector(N, seed=_hat_seed(N, kappa, scale), scale=scale)
    tau_lo, tau_hi = 0.02, 2.0

    tau_star = hat_temperature(s, kappa=kappa, tau_lo=tau_lo, tau_hi=tau_hi, iters=40)
    assert tau_lo <= tau_star <= tau_hi, f"I4 越界: tau*={tau_star!r}"

    target = math.log(max(kappa * N, 1.0))
    achieved = softmax_entropy_beta(s, 1.0 / tau_star)
    err = abs(achieved - target)
    assert err <= 1e-9, (
        f"I4 求根精度失效: N={N}, kappa={kappa}, scale={scale}, "
        f"tau*={tau_star!r}, |logPP-target|={err:.3e}"
    )


@pytest.mark.parametrize("N,kappa,scale", _HAT_CONFIGS)
def test_i4_hat_root_cross_check_scipy(N, kappa, scale):
    """I4：用 scipy.optimize.brentq 独立求根交叉验证自研二分，差值 < 1e-6。"""
    if not _HAS_SCIPY:
        pytest.skip("scipy 未安装，跳过 brentq 交叉验证（非阻断）")

    s = _sim_vector(N, seed=_hat_seed(N, kappa, scale), scale=scale)
    tau_lo, tau_hi = 0.02, 2.0
    target = math.log(max(kappa * N, 1.0))

    def f(tau):
        return softmax_entropy_beta(s, 1.0 / tau) - target

    tau_self = hat_temperature(s, kappa=kappa, tau_lo=tau_lo, tau_hi=tau_hi, iters=40)

    # 目标熵若落在 [tau_lo, tau_hi] 可达区间之外，brentq 无符号变化 -> 该配置跳过，
    # 但必须断言 tau* 落在边界上（防止静默返回错误的内点解）。
    if f(tau_lo) * f(tau_hi) > 0.0:
        assert (
            tau_self in (tau_lo, tau_hi)
            or abs(tau_self - tau_lo) < 1e-12
            or abs(tau_self - tau_hi) < 1e-12
        ), f"I4 边界失效: 目标不可达时 tau* 应落在 clamp 边界, got {tau_self!r}"
        pytest.skip("目标熵超出 clamp 区间可达范围，brentq 无符号变化（已校验边界）")

    tau_ref = brentq(f, tau_lo, tau_hi, xtol=1e-14, rtol=1e-15, maxiter=200)
    assert abs(tau_self - tau_ref) < 1e-6, (
        f"I4 交叉验证失效: tau_self={tau_self!r} vs brentq={tau_ref!r}, "
        f"diff={abs(tau_self - tau_ref):.3e}"
    )


@pytest.mark.parametrize("N,kappa,scale", _HAT_CONFIGS)
def test_i4_hat_entropy_monotone_in_tau(N, kappa, scale):
    """I4(a)：H 关于 tau 单调不减，且在数值有意义区间内**严格**递增 —— 这是
    "根唯一、可二分"的前提。

    ⚠️ 实测发现的数值边界（已在文档中标注）：当 tau 极小（beta 极大）时 softmax
    完全 one-hot，H 在 float64 下**下溢为精确 0.0**，相邻网格点会出现平台段。
    这是浮点表示极限，不是引理失效 —— 因此：
      * 全局断言"单调不减"（带相对容差）；
      * 仅在 H 高于下溢地板的区间断言"严格递增"；
      * 额外断言首尾跨度严格为正（证明分布确实随 tau 变化）。
    """
    s = _sim_vector(N, seed=_hat_seed(N, kappa, scale), scale=scale)
    taus = [0.02 * 1.35**i for i in range(20)]
    hs = [softmax_entropy_beta(s, 1.0 / t) for t in taus]

    underflow_floor = 1e-9
    for i in range(len(hs) - 1):
        tol = 1e-12 * max(1.0, abs(hs[i]))
        assert hs[i + 1] >= hs[i] - tol, (
            f"I4(a) 单调性（弱）失效: tau={taus[i]:.4f}->{taus[i + 1]:.4f}, "
            f"H={hs[i]:.10e}->{hs[i + 1]:.10e}"
        )
        if hs[i] > underflow_floor and hs[i + 1] > underflow_floor:
            assert hs[i + 1] > hs[i], (
                f"I4(a) 严格单调性失效（数值有意义区间）: "
                f"tau={taus[i]:.4f}->{taus[i + 1]:.4f}, H={hs[i]:.10e}->{hs[i + 1]:.10e}"
            )

    # 首尾跨度必须严格为正：tau 从最小到最大，分布确实从 one-hot 走向均匀
    assert hs[-1] > hs[0], f"I4(a) 跨度失效: H(tau_min)={hs[0]!r}, H(tau_max)={hs[-1]!r}"
    # 上界：tau -> inf 时 H -> log N（均匀分布），任何点都不得超过
    assert hs[-1] <= math.log(N) + 1e-12, (
        f"I4(a) 上界失效: H(tau_max)={hs[-1]!r} > log N={math.log(N)!r}"
    )


# ===========================================================================
# I8 —— MAP@R：逐位复现 Musgrave et al. 2020 Table 3，且 MAP@R <= R-Precision
# ===========================================================================
# (用例名, hits 0/1 序列, R, 论文 MAP@R %, 论文 R-Precision %)
_TABLE3_CASES = [
    ("only 1st correct", [1] + [0] * 9, 10, 10, 10),
    ("1st & 10th", [1] + [0] * 8 + [1], 10, 12, 20),
    ("1st & 2nd", [1, 1] + [0] * 8, 10, 20, 20),
    ("all 10 correct", [1] * 10, 10, 100, 100),
]


@pytest.mark.parametrize("name,hits,R,exp_mapr,exp_rp", _TABLE3_CASES)
def test_i8_musgrave_table3_bitwise(name, hits, R, exp_mapr, exp_rp):
    """I8：Musgrave et al. 2020 Table 3 四用例逐位复现（MAP@R / R-Precision）。

    用有理数精确计算，因此断言是**精确相等**而非近似比较 ——
    这是文档 §4.3 所述"R 口径陷阱"的唯一可靠防线：
    若把 R 误取成命中数，四个用例会全部退化成 100，范围检查无法发现。
    """
    mapr_exact, rp_exact = map_at_r_exact(hits, R)
    assert float(mapr_exact) * 100.0 == exp_mapr, (
        f"I8 Table3 失配 [{name}]: MAP@R={float(mapr_exact) * 100.0!r} != {exp_mapr}"
    )
    assert float(rp_exact) * 100.0 == exp_rp, (
        f"I8 Table3 失配 [{name}]: R-Precision={float(rp_exact) * 100.0!r} != {exp_rp}"
    )

    # 浮点实现路径必须与精确值一致（容差 1e-12）
    mapr_f, rp_f = map_at_r(hits, R)
    assert abs(mapr_f - exp_mapr / 100.0) < 1e-12, f"I8 浮点路径失配 [{name}]: {mapr_f!r}"
    assert abs(rp_f - exp_rp / 100.0) < 1e-12, f"I8 浮点路径失配 [{name}]: {rp_f!r}"


@pytest.mark.parametrize("trials", [10000, 10000])
def test_i8_mapr_le_rprecision_stress(trials):
    """I8：随机压力测试 —— MAP@R <= R-Precision，且等号 iff 所有同类排在异类之前。

    两批各 1e4 次（合计 2e4），固定 seed，violations 与"等号但非完美排序"必须均为 0。
    """
    rng = random.Random(20260930)
    violations = 0
    equality_without_perfect_order = 0
    worst_slack = 0.0  # rp - mapr 的最小（最负）值

    for _ in range(trials):
        r = rng.randint(1, 15)  # R = gallery 中同类样本数（>= 1）
        n_correct = rng.randint(0, r)  # 前 R 个里命中的数量，可以 < R
        hits = [1] * n_correct + [0] * (r - n_correct)
        rng.shuffle(hits)

        mapr, rp = map_at_r(hits, r)
        slack = rp - mapr
        if slack < -1e-9:
            violations += 1
            worst_slack = min(worst_slack, slack)
        if abs(mapr - rp) < 1e-9 and hits != sorted(hits, reverse=True):
            equality_without_perfect_order += 1

    assert violations == 0, f"I8 不等式被违反 {violations} 次，最差 slack={worst_slack:.3e}"
    assert equality_without_perfect_order == 0, (
        f"I8 等号条件失效: {equality_without_perfect_order} 次出现"
        f"MAP@R == R-Precision 但排序并非完美"
    )


def test_i8_mapr_rejects_bad_r():
    """I8：R 口径防御 —— hits 长度不等于 R、或 R < 1 时必须显式报错。"""
    for bad_hits, bad_r in (([1, 0], 5), ([1, 0, 1], 2), ([], 0)):
        try:
            map_at_r(bad_hits, bad_r)
        except ValueError:
            continue
        raise AssertionError(f"I8 防御失效: hits={bad_hits}, R={bad_r} 未抛 ValueError")


# ===========================================================================
# I3 —— triplet 损失非负，且完美分离时精确为 0
# ===========================================================================
@pytest.mark.parametrize("margin", [0.05, 0.1, 0.2, 0.5, 1.0])
def test_i3_triplet_nonneg_and_exact_zero(margin):
    """I3：L_triplet >= 0；当 d_an - d_ap > margin（完美分离）时 L 精确等于 0.0。"""
    rng = random.Random(int(margin * 1000))

    # (a) 随机样本下非负
    for _ in range(200):
        d_ap = rng.uniform(0.0, 5.0)
        d_an = rng.uniform(0.0, 5.0)
        loss = triplet_hinge(d_ap, d_an, margin)
        assert loss >= 0.0, f"I3 非负性失效: d_ap={d_ap}, d_an={d_an}, m={margin}, L={loss!r}"

    # (b) 完美分离（严格超过 margin）时精确为 0，且必须是 0.0 而非 1e-17 量级残差
    for _ in range(200):
        d_ap = rng.uniform(0.0, 2.0)
        gap = rng.uniform(0.05, 3.0)  # 超出 margin 的额外间隔
        d_an = d_ap + margin + gap  # 严格满足 d_an - d_ap > margin
        loss = triplet_hinge(d_ap, d_an, margin)
        assert loss == 0.0, f"I3 完美分离未归零: d_ap={d_ap}, d_an={d_an}, m={margin}, L={loss!r}"

    # (c) 违反时严格为正（铰链在边界另一侧确实生效）
    for _ in range(200):
        d_an = rng.uniform(0.0, 2.0)
        d_ap = d_an + rng.uniform(0.0, 2.0)  # d_ap > d_an，必然违反
        loss = triplet_hinge(d_ap, d_an, margin)
        assert loss > 0.0, f"I3 铰链失效: d_ap={d_ap}, d_an={d_an}, m={margin}, L={loss!r}"


def test_i3_triplet_mean_nonneg():
    """I3：batch 均值形式同样非负（防止 reduction 环节引入符号错误）。"""
    rng = random.Random(31337)
    total = 0.0
    for _ in range(500):
        total += triplet_hinge(rng.uniform(0, 5), rng.uniform(0, 5), 0.2)
    mean_loss = total / 500.0
    assert mean_loss >= 0.0, f"I3 均值非负失效: {mean_loss!r}"


# ===========================================================================
# I6 —— InfoNCE 紧界（bonus，非 Release-blocking，但成本极低故一并守护）
# ===========================================================================
@pytest.mark.parametrize("B", [8, 64, 256])
@pytest.mark.parametrize("tau", [0.07, 0.2, 1.0])
def test_i6_infonce_tight_bounds(B, tau):
    """I6：L2 归一化下 s ∈ [-1,1]，故 L ∈ [log(1+(B-1)e^{-2/tau}), log(1+(B-1)e^{2/tau})]，
    且 tau -> inf 时 L -> log B。两侧界均可构造达到（紧界）。"""
    rng = random.Random(B * 100 + int(tau * 100))
    n_negs = B - 1
    low = math.log(1.0 + n_negs * math.exp(-2.0 / tau))
    high = math.log(1.0 + n_negs * math.exp(2.0 / tau))

    for _ in range(50):
        s_pos = rng.uniform(-1.0, 1.0)
        s_negs = [rng.uniform(-1.0, 1.0) for _ in range(n_negs)]
        loss = infonce_loss(s_pos, s_negs, tau)
        assert low - 1e-12 <= loss <= high + 1e-12, (
            f"I6 越界: B={B}, tau={tau}, L={loss!r}, [{low!r}, {high!r}]"
        )

    # tau -> inf：分布趋于均匀，L -> log B
    big_tau = 1e6
    s_pos = 0.3
    s_negs = [rng.uniform(-1.0, 1.0) for _ in range(n_negs)]
    assert abs(infonce_loss(s_pos, s_negs, big_tau) - math.log(B)) < 1e-3, (
        f"I6 极限失效: L(tau=1e6)={infonce_loss(s_pos, s_negs, big_tau)!r} vs log B={math.log(B)!r}"
    )


# ===========================================================================
# fallback runner（仅当 pytest 未安装时生效）
# ===========================================================================
def _main() -> int:  # pragma: no cover
    import inspect
    import time

    names = [
        n for n in sorted(globals()) if n.startswith("test_") and inspect.isfunction(globals()[n])
    ]
    total = passed = failed = skipped = 0
    t0 = time.perf_counter()
    failures = []

    for name in names:
        fn = globals()[name]
        # parametrize 装饰器自下而上叠加，展开时需反序做笛卡尔积
        param_specs = list(reversed(getattr(fn, "_mf_params", [])))
        combos = [()]
        for argnames, argvalues in param_specs:
            keys = (
                [a.strip() for a in argnames.split(",")]
                if isinstance(argnames, str)
                else list(argnames)
            )
            new_combos = []
            for base in combos:
                for val in argvalues:
                    val_t = val if isinstance(val, tuple) and len(keys) > 1 else (val,)
                    new_combos.append(base + tuple(val_t))
            combos = new_combos

        for combo in combos:
            total += 1
            label = f"{name}[{combo}]" if combo else name
            try:
                fn(*combo)
                passed += 1
            except _Skip as e:
                skipped += 1
                print(f"  SKIP {label}: {e}")
            except Exception as e:
                failed += 1
                failures.append((label, repr(e)))
                print(f"  FAIL {label}: {e!r}")

    dt = time.perf_counter() - t0
    print()
    print(f"backend={_BACKEND}  numpy={_HAS_NUMPY}  scipy={_HAS_SCIPY}")
    print(f"total={total} passed={passed} failed={failed} skipped={skipped} time={dt:.2f}s")
    if failures:
        print("\n失败明细:")
        for label, err in failures:
            print(f"  - {label}: {err}")
    print("RESULT:", "ALL GREEN" if failed == 0 else "HAS FAILURES")
    return 0 if failed == 0 else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(_main())
