"""
Pilot: every remedy x several seeds at one beta, identical budget.
Resumable: finished runs are saved to pilot_results.json and skipped on restart.

Usage:  python pilot.py [beta] [adam_steps] [n_seeds] [convection|oscillator]
Run overnight-safe on a Mac with:  caffeinate -i python pilot.py 30 10000 5
"""
import json, os, sys
import numpy as np
from pinn import PINN, Config, Convection, Oscillator
from remedies import make_ntk_hook, make_causal_hook, make_continuation_hook

beta = float(sys.argv[1]) if len(sys.argv) > 1 else 30.0
steps = int(sys.argv[2]) if len(sys.argv) > 2 else 10000
n_seeds = int(sys.argv[3]) if len(sys.argv) > 3 else 5
eqname = sys.argv[4] if len(sys.argv) > 4 else "convection"
EQS = {"convection": Convection, "oscillator": Oscillator}
tag = "" if eqname == "convection" else eqname + ":"   # keeps old convection keys valid
FILE = "pilot_results.json"

methods = {"vanilla": None, "ntk": make_ntk_hook, "causal": make_causal_hook,
           "continuation": make_continuation_hook}
cfg = Config(adam_steps=steps, lbfgs_steps=500)
res = json.load(open(FILE)) if os.path.exists(FILE) else {}

for seed in range(n_seeds):                 # seed-major order: every method gets equal coverage early
    for name, mk in methods.items():
        key = f"{tag}{name}|beta={beta:g}|adam={steps}|seed={seed}"
        if key in res:
            continue
        p = PINN(EQS[eqname](beta), cfg, seed=seed)
        p.train(weight_hook=mk() if mk else None, verbose=False)
        r = p.evaluate()
        res[key] = {k: r[k] for k in ("rel_l2", "max_err", "res_norm", "train_time")}
        json.dump(res, open(FILE, "w"), indent=1)
        print(f"{key:55s} relL2={r['rel_l2']:.3e}  t={r['train_time']:.0f}s", flush=True)

print("\n=== summary (relative L2 error) ===")
print(f"{'method':14s} {'n':>2s} {'mean':>9s} {'std':>9s} {'median':>9s} {'min':>9s} {'solved(<0.1)':>13s}")
for name in methods:
    v = np.array([r["rel_l2"] for k, r in res.items()
                  if k.startswith(f"{tag}{name}|beta={beta:g}|adam={steps}|")])
    if len(v):
        print(f"{name:14s} {len(v):2d} {v.mean():9.3e} {v.std():9.3e} {np.median(v):9.3e} "
              f"{v.min():9.3e} {int((v < 0.1).sum()):>7d}/{len(v)}")