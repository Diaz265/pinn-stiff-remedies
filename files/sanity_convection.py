"""Step 2: reproduce vanilla-PINN failure on convection as beta increases."""
import json, sys
from pinn import PINN, Config, Convection

betas = [float(b) for b in sys.argv[1:]] or [1, 10, 20, 30]
cfg = Config(adam_steps=1500, lbfgs_steps=300)
out = {}
for b in betas:
    p = PINN(Convection(b), cfg, seed=0)
    p.train(verbose=False)
    out[b] = p.evaluate()
    print(f"beta={b:5.1f}  relL2={out[b]['rel_l2']:.3e}  maxerr={out[b]['max_err']:.3f}  "
          f"res={out[b]['res_norm']:.2e}  t={out[b]['train_time']:.0f}s", flush=True)
json.dump(out, open("sanity_convection.json", "w"), indent=1)
