"""
Remedies for the matched-budget comparison. Each remedy only changes the loss
weights (or training distribution) -- never the network or the optimizer.
"""
import torch
from torch.func import functional_call, jacrev, vmap


def _sq_jac_trace(fn, params, *inputs, chunk=1000):
    """mean over samples of ||d fn / d theta||^2  (a per-sample NTK diagonal, averaged)."""
    n = inputs[0].shape[0]
    total = 0.0
    for s in range(0, n, chunk):
        sl = [a[s:s + chunk] for a in inputs]
        jac = vmap(jacrev(fn, argnums=0), in_dims=(None,) + (0,) * len(sl))(params, *sl)
        total += sum((j ** 2).sum() for j in jac.values())
    return (total / n).item()


def make_ntk_hook(alpha: float = 0.9):
    """
    NTK-trace loss balancing, following Wang, Yu & Perdikaris [4]:

        lam_i_hat = (sum_j Tr K_jj) / Tr K_ii,   i, j in the equation's terms

    Tr K_ii is the (sample-averaged) squared norm of the gradient of each
    constraint's network output w.r.t. the weights. A term whose NTK trace is
    small (slow to learn) gets a large weight. Weights are smoothed with a
    moving average:  lam <- (1 - alpha) lam + alpha lam_hat.
    The equation supplies its own pointwise terms via eq.ntk_terms (terms an
    equation does not have, e.g. BC for an ODE, are simply skipped).
    """
    def hook(pinn, step):
        model = pinn.model
        params = {k: v.detach() for k, v in model.named_parameters()}

        def u(p, x, t):
            return functional_call(model, p, (x.reshape(1, 1), t.reshape(1, 1))).squeeze()

        terms = pinn.eq.ntk_terms(u, pinn)
        tr = {k: _sq_jac_trace(fn, params, *inp) for k, (fn, inp) in terms.items()}
        total = sum(tr.values())
        for k in tr:
            lam_hat = total / (tr[k] + 1e-12)
            pinn.lam[k] = (1 - alpha) * pinn.lam[k] + alpha * lam_hat
    return hook


def make_causal_hook(n_slices: int = 32, eps_schedule=(1e-2, 1e-1, 1.0, 10.0, 100.0),
                     delta: float = 0.99):
    """
    Causal weighting, following Wang, Sankaran & Perdikaris [5].

    Time is cut into n_slices. The residual loss of slice i gets the weight
        w_i = exp(-eps * sum_{k<i} L_k)
    so a later slice only counts once all earlier slices are already fitted.
    Weights are treated as constants (no gradient flows through them).
    eps is raised step by step (annealing) each time every slice is "open"
    (min w_i > delta), which gradually turns the causal pressure up.

    Only the residual term is changed; bc/ic weights stay at 1, so this is
    causal weighting in isolation. Requires a time axis (true for our suite).
    The hook wraps pinn.loss_terms on its first call so pinn.py stays untouched.
    """
    state = {"wrapped": False, "eps_idx": 0, "w": None}

    def hook(pinn, step):
        if not state["wrapped"]:
            def loss_terms():
                r2 = pinn.eq.residual(pinn.model, pinn.x_r, pinn.t_r).squeeze(1) ** 2
                t0, t1 = pinn.eq.t_range
                idx = ((pinn.t_r.squeeze(1) - t0) / (t1 - t0) * n_slices).long()
                idx = idx.clamp(0, n_slices - 1)
                sums = torch.zeros(n_slices).index_add(0, idx, r2)
                cnt = torch.zeros(n_slices).index_add(0, idx, torch.ones_like(r2)).clamp(min=1)
                L_slice = sums / cnt
                with torch.no_grad():
                    eps = eps_schedule[state["eps_idx"]]
                    w = torch.exp(-eps * (torch.cumsum(L_slice, 0) - L_slice))
                state["w"] = w
                L_r = (w * L_slice).mean()
                L_ic = pinn.eq.ic_loss(pinn.model, pinn.x_ic)
                L_bc = pinn.eq.bc_loss(pinn.model, pinn.t_bc)
                return L_r, L_bc, L_ic
            pinn.loss_terms = loss_terms
            state["wrapped"] = True
        # anneal eps when the causal weights are all (almost) fully open
        if state["w"] is not None and state["w"].min() > delta \
                and state["eps_idx"] < len(eps_schedule) - 1:
            state["eps_idx"] += 1
        pinn.eps_now = eps_schedule[state["eps_idx"]]
    return hook


def make_continuation_hook(n_stages: int = 6, hook_every: int = 100):
    """
    Stiffness continuation (curriculum over beta). NOT the STL-PINN of [7].

    The Adam budget is split into n_stages equal parts. Stage k trains the SAME
    network on beta_k = beta_target * k / n_stages (k = 1..n_stages), so the last
    stage is the real target problem. The network is carried over between
    stages (warm start). The total number of Adam steps, the optimizer, and the
    L-BFGS polish (run at the target beta) are identical to every other remedy,
    so the budget stays matched.

    n_stages is fixed in advance (pre-specified); do not tune it on test betas.
    """
    state = {"target": None}

    def hook(pinn, step):
        if state["target"] is None:
            state["target"] = pinn.eq.beta
        total = pinn.cfg.adam_steps
        stage = min(n_stages - 1, int(step * n_stages / total))
        if step + hook_every >= total:      # last hook call: always end on the TARGET beta
            stage = n_stages - 1            # (L-BFGS and evaluation must see the real problem)
        pinn.eq.beta = state["target"] * (stage + 1) / n_stages
        pinn.beta_now = pinn.eq.beta
    return hook