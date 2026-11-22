"""
Extra experiments for the PINN remedy comparison (stiff damped oscillator).

Adds, under the SAME protocol as the paper (4x50 tanh MLP, Xavier, float32,
5,000 random collocation points, Adam lr 3e-4 for 5,000 steps, then 500 L-BFGS
iterations with frozen weights, seeds 1000-1009, relative L2 error on 2,501 times):

  vanilla     : all weights 1 (paper baseline)
  ntk_shared  : NTK weighting, u(0) and u'(0) share ONE trace term (paper version)
  ntk_sep     : NTK weighting with SEPARATE trace terms for u(0) and u'(0)   <-- new
  hard_ic     : hard-constrained initial conditions u = 1 + a*t + t^2*N(t),
                residual-only loss (a = -(1+S)/2)                            <-- new

IMPORTANT: this is a clean re-implementation. If your own code differs in any detail,
port the two new methods (ntk_sep, hard_ic) into YOUR code so that every number in the
paper comes from one implementation. Do not mix numbers from two code bases.

Usage
  python extra_oscillator_experiments.py tune                  # pick alpha for ntk_sep (S=200,400; seeds 100-102)
  python extra_oscillator_experiments.py final --alpha-sep 0.9 # final runs (S in 120 250 300 350 600; seeds 1000-1009)
  python extra_oscillator_experiments.py report                # tables, Wilcoxon tests, 300-DPI figure
  Add --smoke for a tiny 200-step test of the whole pipeline.
Results are appended to results.csv, so an interrupted run can be resumed.
"""
import argparse, os, time, math
import numpy as np, pandas as pd, torch, torch.nn as nn
from torch.func import functional_call, vmap, grad

torch.set_default_dtype(torch.float32)
torch.set_num_threads(2)
LR = 3e-4
CSV = "results.csv"
T_END = 5.0


def make_net():
    layers = [nn.Linear(1, 50), nn.Tanh()]
    for _ in range(3):
        layers += [nn.Linear(50, 50), nn.Tanh()]
    layers += [nn.Linear(50, 1)]
    net = nn.Sequential(*layers)
    for m in net:
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight); nn.init.zeros_(m.bias)
    return net


def exact(t, S):
    return 0.5 * (np.exp(-t) + np.exp(-S * t))


class Problem:
    def __init__(self, S, method, net):
        self.S, self.method, self.net = S, method, net
        self.a = -(1.0 + S) / 2.0

    def u(self, t):                       # network solution (with hard ansatz if requested)
        n = self.net(t)
        if self.method == "hard_ic":
            return 1.0 + self.a * t + t ** 2 * n
        return n

    def derivs(self, t):
        t = t.clone().requires_grad_(True)
        u = self.u(t)
        ut = torch.autograd.grad(u.sum(), t, create_graph=True)[0]
        utt = torch.autograd.grad(ut.sum(), t, create_graph=True)[0]
        return u, ut, utt

    def losses(self, t_col, t0):
        _, _, _ = None, None, None
        u, ut, utt = self.derivs(t_col)
        Lr = ((utt + (1 + self.S) * ut + self.S * u) ** 2).mean()
        if self.method == "hard_ic":
            return Lr, torch.zeros(()), torch.zeros(())
        u0, u0t, _ = self.derivs(t0)
        return Lr, ((u0 - 1.0) ** 2).mean(), ((u0t - self.a) ** 2).mean()

    # ---- NTK trace terms via per-sample gradients w.r.t. parameters
    def traces(self, t_col):
        net, S = self.net, self.S
        p = {k: v.detach() for k, v in net.named_parameters()}

        def u_fn(p_, t):
            return functional_call(net, p_, (t.reshape(1, 1),)).reshape(())
        ut_fn = grad(u_fn, argnums=1)
        utt_fn = grad(ut_fn, argnums=1)
        res_fn = lambda p_, t: utt_fn(p_, t) + (1 + S) * ut_fn(p_, t) + S * u_fn(p_, t)

        def tr(fn, ts):
            g = vmap(grad(fn, argnums=0), in_dims=(None, 0))(p, ts)
            return sum((v.reshape(v.shape[0], -1) ** 2).sum(1) for v in g.values()).mean().item()
        t0 = torch.zeros(1)
        return tr(res_fn, t_col.reshape(-1)), tr(u_fn, t0), tr(ut_fn, t0)


def run(method, S, seed, alpha, adam_steps=5000, lbfgs_iters=500, n_col=5000):
    torch.manual_seed(seed); np.random.seed(seed)
    net = make_net()
    P = Problem(S, method, net)
    g = torch.Generator().manual_seed(seed)
    t_col = (torch.rand(n_col, 1, generator=g) * T_END)
    t0 = torch.zeros(1, 1)
    lam = torch.ones(3)                                   # [residual, u(0), u'(0)]
    opt = torch.optim.Adam(net.parameters(), lr=LR)

    def total():
        Lr, L0, L1 = P.losses(t_col, t0)
        return lam[0] * Lr + lam[1] * L0 + lam[2] * L1

    for step in range(adam_steps):
        if method in ("ntk_shared", "ntk_sep") and step % 100 == 0:
            trr, tr0, tr1 = P.traces(t_col)
            if method == "ntk_sep":
                tr_all = trr + tr0 + tr1
                new = torch.tensor([tr_all / trr, tr_all / tr0, tr_all / tr1])
            else:                                          # shared: one trace for both ICs
                tr_ic = tr0 + tr1
                tr_all = trr + tr_ic
                w_ic = tr_all / tr_ic
                new = torch.tensor([tr_all / trr, w_ic, w_ic])
            lam = (1 - alpha) * lam + alpha * new
        opt.zero_grad(); loss = total(); loss.backward(); opt.step()

    lam = lam.detach()                                    # freeze weights for L-BFGS
    lb = torch.optim.LBFGS(net.parameters(), lr=1.0, max_iter=lbfgs_iters, line_search_fn="strong_wolfe",
                           tolerance_grad=1e-12, tolerance_change=1e-14)

    def closure():
        lb.zero_grad(); l = total(); l.backward(); return l
    try:
        lb.step(closure)
    except Exception:
        pass

    tt = np.linspace(0, T_END, 2501)
    with torch.no_grad():
        pred = P.u(torch.tensor(tt, dtype=torch.float32).reshape(-1, 1)).numpy().ravel()
    ex = exact(tt, S)
    err = float(np.linalg.norm(pred - ex) / np.linalg.norm(ex))
    return err


def append(row):
    pd.DataFrame([row]).to_csv(CSV, mode="a", header=not os.path.exists(CSV), index=False)


def done_set():
    if not os.path.exists(CSV):
        return set()
    d = pd.read_csv(CSV)
    return set(zip(d.phase, d.method, d.S, d.seed, d.alpha))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["tune", "final", "report"])
    ap.add_argument("--alpha-sep", type=float, default=0.9)
    ap.add_argument("--alpha-shared", type=float, default=0.9)      # paper's tuned value for oscillator
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    kw = dict(adam_steps=200, lbfgs_iters=20, n_col=500) if a.smoke else {}
    global CSV
    if a.smoke:
        CSV = "results_smoke.csv"

    if a.mode == "tune":
        done = done_set()
        for alpha in (0.1, 0.5, 0.9):
            for S in (200, 400):
                for seed in (100, 101, 102):
                    key = ("tune", "ntk_sep", S, seed, alpha)
                    if key in done: continue
                    t = time.time(); e = run("ntk_sep", S, seed, alpha, **kw)
                    append(dict(phase="tune", method="ntk_sep", S=S, seed=seed, alpha=alpha, err=e))
                    print(key, round(e, 4), f"{time.time()-t:.0f}s", flush=True)
        d = pd.read_csv(CSV); d = d[(d.phase == "tune") & (d.method == "ntk_sep")]
        print("\nMedian error by alpha (pick the lowest):\n", d.groupby("alpha").err.median())
        # also tune nothing for hard_ic / vanilla: they have no remedy hyperparameter
    elif a.mode == "final":
        done = done_set()
        grid = [120, 250, 300, 350, 600]              # 300 is a NEW in-range value (never used for tuning)
        methods = [("vanilla", 0.0), ("ntk_shared", a.alpha_shared), ("ntk_sep", a.alpha_sep), ("hard_ic", 0.0)]
        for S in grid:
            for seed in range(1000, 1010):
                for m, al in methods:
                    key = ("final", m, S, seed, al)
                    if key in done: continue
                    t = time.time(); e = run(m, S, seed, al, **kw)
                    append(dict(phase="final", method=m, S=S, seed=seed, alpha=al, err=e))
                    print(key, round(e, 4), f"{time.time()-t:.0f}s", flush=True)
    else:
        report()


def report():
    from scipy.stats import friedmanchisquare, wilcoxon
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    d = pd.read_csv(CSV); d = d[d.phase == "final"]
    order = ["vanilla", "ntk_shared", "ntk_sep", "hard_ic"]
    lab = {"vanilla": "Vanilla", "ntk_shared": "NTK (shared)", "ntk_sep": "NTK (separate)", "hard_ic": "Hard IC"}
    rows = []
    for S, g in d.groupby("S"):
        piv = g.pivot_table(index="seed", columns="method", values="err")[order].dropna()
        row = {"S": S}
        for m in order:
            e = piv[m].values
            row[lab[m]] = f"{(e<0.1).sum()} solved / {((e>=0.1)&(e<=0.5)).sum()} partial / {(e>0.5).sum()} collapsed, median {np.median(e):.3g}"
        if len(piv) >= 5:
            try:
                row["Friedman p"] = f"{friedmanchisquare(*[piv[m] for m in order]).pvalue:.3g}"
                raw = {m: wilcoxon(piv[m], piv["vanilla"]).pvalue for m in order[1:]}
                srt = sorted(raw.items(), key=lambda x: x[1]); adj, prev = {}, 0
                for i, (m, p) in enumerate(srt):
                    prev = max(prev, min(1, p * (len(srt) - i))); adj[m] = prev     # Holm
                row["Holm p vs vanilla"] = ", ".join(f"{lab[m]}: {adj[m]:.3g}" for m in order[1:])
            except ValueError:
                row["Holm p vs vanilla"] = "n/a (identical values)"
        rows.append(row)
    out = pd.DataFrame(rows); print(out.to_string(index=False)); out.to_csv("summary.csv", index=False)

    Ss = sorted(d.S.unique()); fig, axs = plt.subplots(1, len(Ss), figsize=(3.2 * len(Ss), 3.4), sharey=True)
    cols = ["#555555", "#1f77b4", "#d95f02", "#1b9e77"]
    rng = np.random.default_rng(0)
    for ax, S in zip(np.atleast_1d(axs), Ss):
        for i, m in enumerate(order):
            e = d[(d.S == S) & (d.method == m)].err.values
            ax.scatter(i + rng.uniform(-.15, .15, len(e)), e, s=12, color=cols[i], alpha=.8)
            ax.hlines(np.median(e), i - .3, i + .3, color="k", lw=2)
        ax.axhline(0.1, color="g", ls="--", lw=1); ax.axhline(0.5, color="r", ls=":", lw=1)
        ax.set_yscale("log"); ax.set_title(f"S={S}", fontsize=10)
        ax.set_xticks(range(4)); ax.set_xticklabels([lab[m] for m in order], rotation=60, ha="right", fontsize=8)
    np.atleast_1d(axs)[0].set_ylabel("Relative $L_2$ error")
    plt.tight_layout(); plt.savefig("figure_extra_oscillator.png", dpi=300); print("saved figure_extra_oscillator.png (300 DPI)")


if __name__ == "__main__":
    main()