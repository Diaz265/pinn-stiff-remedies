"""Usage: python test_remedy.py <ntk|causal> beta1 beta2 ...   (same budget for every remedy)"""
import json, sys
from pinn import PINN, Config, Convection
from remedies import make_ntk_hook, make_causal_hook

name = sys.argv[1]
betas = [float(b) for b in sys.argv[2:]] or [20, 30]
hooks = {"ntk": make_ntk_hook, "causal": make_causal_hook}
cfg = Config(adam_steps=1500, lbfgs_steps=300)
out = {}
for b in betas:
    p = PINN(Convection(b), cfg, seed=0)
    p.train(weight_hook=hooks[name](), verbose=False)
    out[b] = p.evaluate()
    print(f"{name}  beta={b:5.1f}  relL2={out[b]['rel_l2']:.3e}  maxerr={out[b]['max_err']:.3f}  "
          f"eps_final={getattr(p,'eps_now',float('nan'))}  t={out[b]['train_time']:.0f}s", flush=True)
json.dump(out, open(f"test_{name}.json", "w"), indent=1)