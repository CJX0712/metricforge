#!/usr/bin/env python
"""torch(CPU) vs numpy 的 MLP 前反传播实测加速比. 作者: 晨星

回答 algorithm_design.md 附表中的待验证项:
  "torch 2.14.0 在 AMD Ryzen CPU 上的实际 fit 加速比 —— 待实测"

方法学要求:
  - 两侧执行**完全相同**的运算集(同一 loss、同一组可求导参数)
  - 计时循环内**不做** .item() / .clone() / numpy 往返(避免把同步开销算进去)
  - 取 5 次重复的**中位数**, 而非单次
  - torch 侧扫描线程数, 找最优点
"""

import statistics
import time

import numpy as np

D_IN, H, D_OUT, B, ITERS, REPEATS = 64, 128, 32, 256, 40, 5

rng = np.random.default_rng(0)
X = rng.standard_normal((B, D_IN))
W1_0 = rng.standard_normal((D_IN, H)) * 0.05
W2_0 = rng.standard_normal((H, D_OUT)) * 0.05


def bench_numpy():
    Xn = X
    W1, W2 = W1_0.copy(), W2_0.copy()
    b1, b2 = np.zeros(H), np.zeros(D_OUT)
    gW1 = np.zeros_like(W1)
    gW2 = np.zeros_like(W2)

    def step():
        nonlocal gW1, gW2
        Z1 = Xn @ W1 + b1
        A1 = np.maximum(Z1, 0.0)
        Z2 = A1 @ W2 + b2
        dZ2 = Z2 / B
        gW2 = A1.T @ dZ2
        dZ1 = (dZ2 @ W2.T) * (Z1 > 0)
        gW1 = Xn.T @ dZ1
        return 0.5 * np.sum(Z2**2) / B

    step()
    ts = []
    for _ in range(REPEATS):
        t0 = time.perf_counter()
        for _ in range(ITERS):
            step()
        ts.append((time.perf_counter() - t0) / ITERS)
    return statistics.median(ts), gW1, gW2


def bench_torch(nt):
    import torch

    torch.set_num_threads(nt)
    Xt = torch.from_numpy(X)
    W1 = torch.from_numpy(W1_0.copy()).requires_grad_(True)
    W2 = torch.from_numpy(W2_0.copy()).requires_grad_(True)
    b1 = torch.zeros(H, requires_grad=True)
    b2 = torch.zeros(D_OUT, requires_grad=True)

    def step():
        Z1 = Xt @ W1 + b1
        A1 = torch.relu(Z1)
        Z2 = A1 @ W2 + b2
        (0.5 * torch.sum(Z2**2) / B).backward()
        W1.grad = W2.grad = b1.grad = b2.grad = None

    step()
    ts = []
    for _ in range(REPEATS):
        t0 = time.perf_counter()
        for _ in range(ITERS):
            step()
        ts.append((time.perf_counter() - t0) / ITERS)
    return statistics.median(ts)


t_np, gW1_np, gW2_np = bench_numpy()
print(
    f"numpy      : {t_np * 1e3:8.3f} ms/step   (B={B}, {D_IN}->{H}->{D_OUT}, "
    f"median of {REPEATS}x{ITERS})"
)

best = None
try:
    import torch

    for nt in (1, 2, 4, 8, 16):
        t = bench_torch(nt)
        flag = ""
        if best is None or t < best[1]:
            best = (nt, t)
        print(f"torch t={nt:>2}  : {t * 1e3:8.3f} ms/step  ({t_np / t:.2f}x vs numpy)")
    print(
        f"\n最佳 torch 配置: threads={best[0]}  {best[1] * 1e3:.3f} ms/step  "
        f"= {t_np / best[1]:.2f}x vs numpy"
    )

    # 梯度一致性(torch 作 autograd oracle 的可信度)
    torch.set_num_threads(best[0])
    W1 = torch.from_numpy(W1_0.copy()).requires_grad_(True)
    W2 = torch.from_numpy(W2_0.copy()).requires_grad_(True)
    Z2 = torch.relu(torch.from_numpy(X) @ W1) @ W2
    (0.5 * torch.sum(Z2**2) / B).backward()
    d1 = float(np.abs(gW1_np - W1.grad.numpy()).max())
    d2 = float(np.abs(gW2_np - W2.grad.numpy()).max())
    print(f"梯度一致性 maxAbsDiff: dW1={d1:.3e}  dW2={d2:.3e}")
except Exception as e:  # noqa: BLE001
    print(f"torch 不可用: {type(e).__name__}: {e}")

print("DONE")
