"""
Core PINN framework for the matched-budget remedy comparison.

Design rules (these are what make the comparison "matched budget"):
  * Architecture, optimizer schedule, collocation counts, and step counts live in
    one Config and are identical across remedies for a given equation.
  * Remedies plug in ONLY through the `lam` weights (Eq. 1) or the training
    distribution -- never through the network or optimizer.
  * Boundary conditions are soft-penalty (periodic BC for convection).
"""
import math
import time
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn
from torch.func import grad

torch.set_default_dtype(torch.float32)


# --------------------------------------------------------------------------- #
# Config (held fixed across all remedies)
# --------------------------------------------------------------------------- #
@dataclass
class Config:
    width: int = 50
    depth: int = 4               # number of hidden layers
    n_res: int = 5000            # residual collocation points
    n_ic: int = 256
    n_bc: int = 256
    adam_steps: int = 2000
    adam_lr: float = 1e-3
    lbfgs_steps: int = 500
    n_eval_x: int = 256
    n_eval_t: int = 101
    device: str = "cpu"


# --------------------------------------------------------------------------- #
# Network
# --------------------------------------------------------------------------- #
class MLP(nn.Module):
    def __init__(self, cfg: Config, in_dim=2, out_dim=1):
        super().__init__()
        self.in_dim = in_dim
        layers, d = [], in_dim
        for _ in range(cfg.depth):
            layers += [nn.Linear(d, cfg.width), nn.Tanh()]
            d = cfg.width
        layers.append(nn.Linear(d, out_dim))
        self.net = nn.Sequential(*layers)
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x, t):
        inp = t if self.in_dim == 1 else torch.cat([x, t], dim=1)   # ODEs ignore x
        return self.net(inp)


# --------------------------------------------------------------------------- #
# Equations
# --------------------------------------------------------------------------- #
class Convection:
    """u_t + beta u_x = 0, x in [0,2pi], t in [0,1], u(x,0)=sin(x), periodic in x.
    Exact solution: u = sin(x - beta t) (method of characteristics)."""
    name = "convection"
    x_range = (0.0, 2 * math.pi)
    t_range = (0.0, 1.0)

    def __init__(self, beta: float):
        self.beta = beta

    def exact(self, x, t):
        return torch.sin(x - self.beta * t)

    def ic(self, x):
        return torch.sin(x)

    def residual(self, model, x, t):
        x = x.clone().requires_grad_(True)
        t = t.clone().requires_grad_(True)
        u = model(x, t)
        ones = torch.ones_like(u)
        u_t = torch.autograd.grad(u, t, ones, create_graph=True)[0]
        u_x = torch.autograd.grad(u, x, ones, create_graph=True)[0]
        return u_t + self.beta * u_x

    in_dim = 2

    def bc_loss(self, model, t):
        x0 = torch.full_like(t, self.x_range[0])
        x1 = torch.full_like(t, self.x_range[1])
        return torch.mean((model(x0, t) - model(x1, t)) ** 2)

    def ic_loss(self, model, x_ic):
        return torch.mean((model(x_ic, torch.zeros_like(x_ic)) - self.ic(x_ic)) ** 2)

    def eval_points(self, cfg):
        x = torch.linspace(*self.x_range, cfg.n_eval_x)
        t = torch.linspace(*self.t_range, cfg.n_eval_t)
        X, T = torch.meshgrid(x, t, indexing="ij")
        return X.reshape(-1, 1), T.reshape(-1, 1)

    def ntk_terms(self, u, pinn):
        """Pointwise functions + their sample points, used by NTK weighting.
        u(params, x, t) -> scalar network output."""
        x0, x1 = self.x_range
        def res(p, x, t):
            return grad(u, argnums=2)(p, x, t) + self.beta * grad(u, argnums=1)(p, x, t)
        def ic(p, x):
            return u(p, x, torch.zeros(()))
        def bc(p, t):
            return u(p, torch.tensor(x0), t) - u(p, torch.tensor(x1), t)
        return {"r": (res, (pinn.x_r, pinn.t_r)),
                "ic": (ic, (pinn.x_ic,)),
                "bc": (bc, (pinn.t_bc,))}


class Oscillator:
    """Stiff damped oscillator  u'' + (1+S) u' + S u = 0,  t in [0,5],
    u(0)=1, u'(0) = -(1+S)/2.   (omega0^2 = S, 2*zeta*omega0 = 1+S, i.e.
    zeta = (1+S)/(2 sqrt(S)).)  Decay rates are 1 and S, so S is the stiffness
    ratio. Exact solution: u = 0.5 (exp(-t) + exp(-S t)): a fast initial drop of
    width ~1/S followed by slow decay. Stored as `beta` so remedies are generic."""
    name = "oscillator"
    in_dim = 1
    x_range = (0.0, 1.0)        # dummy: the network ignores x
    t_range = (0.0, 5.0)

    def __init__(self, beta: float):
        self.beta = beta

    def exact(self, x, t):
        return 0.5 * (torch.exp(-t) + torch.exp(-self.beta * t))

    def residual(self, model, x, t):
        t = t.clone().requires_grad_(True)
        u = model(x, t)
        ones = torch.ones_like(u)
        u_t = torch.autograd.grad(u, t, ones, create_graph=True)[0]
        u_tt = torch.autograd.grad(u_t, t, torch.ones_like(u_t), create_graph=True)[0]
        return u_tt + (1 + self.beta) * u_t + self.beta * u

    def ic_loss(self, model, x_ic):
        t0 = torch.zeros_like(x_ic).requires_grad_(True)
        u0 = model(x_ic, t0)
        u_t0 = torch.autograd.grad(u0, t0, torch.ones_like(u0), create_graph=True)[0]
        return torch.mean((u0 - 1.0) ** 2 + (u_t0 + (1 + self.beta) / 2) ** 2)

    def bc_loss(self, model, t):
        return torch.zeros(())          # an ODE initial-value problem has no BC term

    def eval_points(self, cfg):
        t = torch.linspace(*self.t_range, 2501).reshape(-1, 1)   # fine grid: resolves the transient
        return torch.zeros_like(t), t

    def ntk_terms(self, u, pinn):
        S = self.beta
        def res(p, x, t):
            u_t = lambda p_, x_, t_: grad(u, argnums=2)(p_, x_, t_).reshape(())   # scalar
            u_tt = grad(u_t, argnums=2)(p, x, t).reshape(())
            return u_tt + (1 + S) * u_t(p, x, t) + S * u(p, x, t)
        def ic(p, x):
            z = torch.zeros(())
            return torch.stack([u(p, x, z) - 1.0,
                                grad(u, argnums=2)(p, x, z) + (1 + S) / 2])
        return {"r": (res, (pinn.x_r, pinn.t_r)), "ic": (ic, (pinn.x_ic,))}


# --------------------------------------------------------------------------- #
# PINN trainer
# --------------------------------------------------------------------------- #
class PINN:
    def __init__(self, eq, cfg: Config, seed: int = 0):
        self.eq, self.cfg, self.seed = eq, cfg, seed
        torch.manual_seed(seed)
        np.random.seed(seed)
        self.model = MLP(cfg, in_dim=eq.in_dim).to(cfg.device)
        self.lam = {"r": 1.0, "bc": 1.0, "ic": 1.0}   # Eq. (1) weights
        self.history = []
        self._sample()

    def _sample(self):
        c, e = self.cfg, self.eq
        g = torch.Generator().manual_seed(self.seed)
        (x0, x1), (t0, t1) = e.x_range, e.t_range
        u = lambda n: torch.rand(n, 1, generator=g)
        self.x_r = (x0 + (x1 - x0) * u(c.n_res)).to(c.device)
        self.t_r = (t0 + (t1 - t0) * u(c.n_res)).to(c.device)
        self.x_ic = (x0 + (x1 - x0) * u(c.n_ic)).to(c.device)
        self.t_bc = (t0 + (t1 - t0) * u(c.n_bc)).to(c.device)

    # ---- loss terms (the quantities every remedy reasons about) ----
    def loss_terms(self):
        r = self.eq.residual(self.model, self.x_r, self.t_r)
        L_r = torch.mean(r ** 2)
        L_ic = self.eq.ic_loss(self.model, self.x_ic)
        L_bc = self.eq.bc_loss(self.model, self.t_bc)
        return L_r, L_bc, L_ic

    def total_loss(self):
        L_r, L_bc, L_ic = self.loss_terms()
        return self.lam["r"] * L_r + self.lam["bc"] * L_bc + self.lam["ic"] * L_ic

    # ---- remedy hook: called every `hook_every` Adam steps ----
    def train(self, weight_hook=None, hook_every=100, log_every=250, verbose=True):
        c = self.cfg
        t_start = time.time()
        opt = torch.optim.Adam(self.model.parameters(), lr=c.adam_lr)
        for step in range(c.adam_steps):
            if weight_hook is not None and step % hook_every == 0:
                weight_hook(self, step)
            opt.zero_grad()
            loss = self.total_loss()
            loss.backward()
            opt.step()
            if step % log_every == 0:
                self._log("adam", step, loss.item(), verbose)

        # L-BFGS polish with weights frozen at their final Adam values
        lb = torch.optim.LBFGS(self.model.parameters(), lr=1.0, max_iter=c.lbfgs_steps,
                               max_eval=int(c.lbfgs_steps * 1.25),
                               history_size=50, tolerance_grad=1e-9,
                               tolerance_change=1e-12, line_search_fn="strong_wolfe")

        def closure():
            lb.zero_grad()
            l = self.total_loss()
            l.backward()
            return l

        lb.step(closure)
        self._log("lbfgs", c.adam_steps, self.total_loss().item(), verbose)
        self.train_time = time.time() - t_start

    def _log(self, phase, step, loss, verbose):
        self.history.append((phase, step, loss))
        if verbose:
            print(f"[{phase:5s}] step {step:5d}  loss {loss:.3e}")

    # ---- evaluation ----
    @torch.no_grad()
    def predict_grid(self, eq=None):
        eq = eq or self.eq
        c = self.cfg
        xf, tf = eq.eval_points(c)
        xf, tf = xf.to(c.device), tf.to(c.device)
        return xf, tf, self.model(xf, tf).cpu()

    def evaluate(self, eq=None):
        """Relative L2, max pointwise error, final residual norm."""
        eq = eq or self.eq
        xf, tf, u_pred = self.predict_grid(eq)
        u_true = eq.exact(xf, tf).cpu()
        rel_l2 = (torch.norm(u_pred - u_true) / torch.norm(u_true)).item()
        max_err = (u_pred - u_true).abs().max().item()
        idx = torch.randperm(xf.shape[0])[:5000]
        res = eq.residual(self.model, xf[idx], tf[idx]).detach()
        return {"rel_l2": rel_l2, "max_err": max_err,
                "res_norm": torch.sqrt(torch.mean(res ** 2)).item(),
                "train_time": getattr(self, "train_time", float("nan"))}


if __name__ == "__main__":
    # Smoke test: vanilla PINN on easy convection (beta=1)
    cfg = Config()
    p = PINN(Convection(beta=1.0), cfg, seed=0)
    p.train()
    print(p.evaluate())