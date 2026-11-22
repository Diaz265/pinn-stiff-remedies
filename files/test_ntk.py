"""Step 3a: does NTK-style weighting rescue the cases where vanilla failed?"""
import json, sys
from pinn import PINN, Config, Convection
from remedies import make_ntk_hook

betas = [float(b) for b in sys.argv[1:]] or [20, 30]
cfg = Config(adam_steps=1500, lbfgs_steps=300)   # SAME budget as the vanilla sanity check
out = {}
for b in betas:
    p = PINN(Convection(b), cfg, seed=0)
    p.train(weight_hook=make_ntk_hook(), verbose=False)
    out[b] = {**p.evaluate(), "final_lam": dict(p.lam)}
    print(f"beta={b:5.1f}  relL2={out[b]['rel_l2']:.3e}  maxerr={out[b]['max_err']:.3f}  "
          f"lam_bc={p.lam['bc']:.1f}  lam_ic={p.lam['ic']:.1f}  t={out[b]['train_time']:.0f}s", flush=True)
json.dump(out, open("test_ntk.json", "w"), indent=1)
