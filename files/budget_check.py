"""Budget check: do the remedies work if we simply train longer?
Usage: python budget_check.py <vanilla|ntk|causal|continuation> [adam_steps] [beta] [seed] [convection|oscillator]"""
import json, sys
from pinn import PINN, Config, Convection, Oscillator
from remedies import make_ntk_hook, make_causal_hook, make_continuation_hook

name = sys.argv[1]
steps = int(sys.argv[2]) if len(sys.argv) > 2 else 10000
beta = float(sys.argv[3]) if len(sys.argv) > 3 else 20.0
seed = int(sys.argv[4]) if len(sys.argv) > 4 else 0
eqname = sys.argv[5] if len(sys.argv) > 5 else "convection"
EQS = {"convection": Convection, "oscillator": Oscillator}
hooks = {"vanilla": None, "ntk": make_ntk_hook, "causal": make_causal_hook, "continuation": make_continuation_hook}
cfg = Config(adam_steps=steps, lbfgs_steps=500)
p = PINN(EQS[eqname](beta), cfg, seed=seed)
p.train(weight_hook=hooks[name]() if hooks[name] else None, verbose=False, log_every=1000)
r = p.evaluate()
print(f"{eqname[:4]} {name:8s} seed={seed} beta={beta:4.0f} adam={steps}  relL2={r['rel_l2']:.3e}  maxerr={r['max_err']:.3f}  "
      f"eps_final={getattr(p,'eps_now','-')}  beta_end={p.eq.beta}  t={r['train_time']:.0f}s", flush=True)
json.dump(r, open(f"budget_{eqname}_{name}_{int(beta)}_{steps}_s{seed}.json", "w"), indent=1)