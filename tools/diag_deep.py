# 作者: 晨星
"""诊断脚本：深度训练欠拟合 vs 崩溃定位。
- 逐 epoch 记录 MAP@R（判断欠训练曲线）
- s5_xor 专项：损失曲线 / Z 方差 / NaN / 原型范数（判断坍缩或爆炸）
- 对比 lr∈{1e-3, 3e-4} 与 epochs=80
"""

import sys

import numpy as np

sys.path.insert(0, ".")
from metricforge.bench.baselines import IdentityModel, build_model
from metricforge.bench.run import _load, _scale
from metricforge.core import seeding
from metricforge.core.encoder import Adam
from metricforge.data.split import split_open
from metricforge.eval.metrics import map_at_r_score
from metricforge.mining import BalancedBatchSampler

DATASETS = ["s1_gauss_blobs", "s3_spirals", "s5_xor_grid", "s6_hier_gauss", "r1_digits", "r2_wine"]
EPOCHS = 80
PROBE_EPOCHS = [1, 5, 10, 20, 40, 60, 80]


def train_with_probe(method, Xtr, ytr, Xte, yte, seed, epochs, lr, dim_out=16, probe=True):
    """复刻 DeepTrainer.fit 循环，附加逐探针点评测。"""
    model = build_model(
        method,
        dim_in=Xtr.shape[1],
        dim_out=dim_out,
        seed=seed,
        K=3 if method == "metricforge" else 1,
        use_hat=(method == "metricforge"),
        use_bidirectional=(method == "metricforge"),
        epochs=epochs,
        lr=lr,
    )
    enc = model.encoder
    loss_fn = model.loss_fn
    seeding.set_seed(seed)
    enc.set_params(enc.net._init_params())
    opt = Adam(enc.parameters(), lr=lr)
    C = len(np.unique(ytr))
    if hasattr(loss_fn, "C"):
        loss_fn.C = C
        loss_fn.P = C * loss_fn.K
    if hasattr(loss_fn, "init_proxies"):
        Z0, _ = enc.forward(Xtr)
        loss_fn.init_proxies(Z0, ytr)
    sampler = BalancedBatchSampler(ytr, model.bs, seed=seed)
    curve = {}
    losses = []
    for ep in range(1, epochs + 1):
        ep_loss = []
        for bidx in sampler:
            if len(bidx) < 4:
                continue
            Z, _ = enc.forward(Xtr[bidx])
            if hasattr(loss_fn, "init_proxies"):
                loss, dZ, dP, aux = loss_fn(Z, ytr[bidx])
                grads, _ = enc.backward(dZ)
                opt.step(grads)
                loss_fn.proxies -= lr * dP
                loss_fn.ema_align(Z, ytr[bidx])
            else:
                loss, dZ = loss_fn(Z, ytr[bidx])
                grads, _ = enc.backward(dZ)
                opt.step(grads)
            ep_loss.append(float(loss))
        losses.append(float(np.mean(ep_loss)))
        if probe and ep in PROBE_EPOCHS:
            Zte = model.transform(Xte)
            zvar = float(np.mean(np.var(model.transform(Xtr), axis=0)))
            nan = int(np.sum(~np.isfinite(Zte)))
            pn = (
                float(np.mean(np.linalg.norm(loss_fn.proxies, axis=1)))
                if hasattr(loss_fn, "proxies") and loss_fn.proxies is not None
                else -1.0
            )
            curve[ep] = (
                round(map_at_r_score(Zte, Zte, yte, yte), 4),
                round(zvar, 4),
                nan,
                round(pn, 3),
            )
    Zte = model.transform(Xte)
    final = map_at_r_score(Zte, Zte, yte, yte)
    return final, curve, losses


def main():
    seed = 0
    print(f"{'dataset':<15} {'knn':>6} | mf@lr1e-3 curve (ep: MAP, zvar, NaN, |P|)")
    print("-" * 100)
    for name in DATASETS:
        X, y = _load(name)
        tr, va, te = split_open(X, y, seed)
        Xtr, Xva, Xte = _scale(X[tr], X[va], X[te])
        ytr, yte = y[tr], y[te]
        knn_model = IdentityModel()
        Zte_raw = knn_model.transform(Xte)
        knn_map = map_at_r_score(Zte_raw, Zte_raw, yte, yte)

        mf_final, mf_curve, mf_losses = train_with_probe(
            "metricforge", Xtr, ytr, Xte, yte, seed, EPOCHS, 1e-3
        )
        curve_str = "  ".join(f"e{k}:{v[0]:.3f}" for k, v in mf_curve.items())
        detail = "  ".join(
            f"e{k}:[map={v[0]},zvar={v[1]},nan={v[2]},|P|={v[3]}]" for k, v in mf_curve.items()
        )
        print(f"{name:<15} {knn_map:>6.4f} | mf_final={mf_final:.4f}  {curve_str}")
        print(f"{'':<15} {'':>6} | {detail}")
        print(
            f"{'':<15} {'':>6} | loss e1={mf_losses[0]:.4f} e10={mf_losses[9]:.4f} "
            f"e40={mf_losses[39]:.4f} e80={mf_losses[-1]:.4f} "
            f"max={max(mf_losses):.4f} nan={int(sum(not np.isfinite(l) for l in mf_losses))}"
        )
        # 低学习率对照
        mf2, _, _ = train_with_probe(
            "metricforge", Xtr, ytr, Xte, yte, seed, EPOCHS, 3e-4, probe=False
        )
        print(f"{'':<15} {'':>6} | mf@lr3e-4 final={mf2:.4f}")
        sys.stdout.flush()
    print("DONE")


if __name__ == "__main__":
    main()
