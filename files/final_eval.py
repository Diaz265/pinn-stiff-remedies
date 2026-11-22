"""
Final evaluation (see PROTOCOL.md). Settings are FROZEN below; do not edit after starting.
Resumable; one results file per equation so both equations can run at the same time.

Usage:  python final_eval.py <convection|oscillator> [seed_lo seed_hi]     (seeds inclusive, default 1000-1009)
Parallel example (3 terminals):  NTHREADS=2 python final_eval.py oscillator 1000 1003  /  1004 1006  /  1007 1009
Run on a Mac with:  caffeinate -i python final_eval.py convection
"""
import json, os, sys
import torch
if os.environ.get("NTHREADS"):                      # lets several processes share the CPU cores
    torch.set_num_threads(int(os.environ["NTHREADS"]))
from pinn import PINN, Config, Convection, Oscillator
from tuning import build, SHARED_LR            # same hook builders as the tuning stage

EQS = {"convection": Convection, "oscillator": Oscillator}
STEPS = {"convection": 10000, "oscillator": 5000}
SEEDS = list(range(1000, 1010))

# frozen settings (selected on validation values, protocol v2)
FIXED = {
    "convection": {"vanilla": 3e-4, "ntk": 0.1,  "causal": 1.0,  "continuation": 6},
    "oscillator": {"vanilla": 3e-4, "ntk": 0.9,  "causal": 1e-4, "continuation": 6},
}
# evaluation grid: regime -> stiffness values (none used for tuning/piloting)
GRID = {
    "convection": {"interp": [24.0, 26.0], "extrap_low": [18.0], "extrap_high": [34.0]},
    "oscillator": {"interp": [250.0, 350.0], "extrap_low": [120.0], "extrap_high": [600.0]},
}

def check_against_json(eqname):
    f = f"best_settings_v2_{eqname}.json"
    if not os.path.exists(f):
        return
    best = json.load(open(f))
    for m, knob in (("ntk", "alpha"), ("causal", "eps_scale"), ("continuation", "n_stages")):
        if m in best and abs(best[m][knob] - FIXED[eqname][m]) > 1e-12:
            print(f"WARNING: frozen {m} setting {FIXED[eqname][m]} differs from {f} ({best[m][knob]})")

def run(eqname, seeds=SEEDS, steps=None, lbfgs_steps=500, file=None):
    steps = steps or STEPS[eqname]
    if file is None:
        file = (f"final_results_{eqname}.json" if list(seeds) == SEEDS
                else f"final_results_{eqname}_s{min(seeds)}-{max(seeds)}.json")
    check_against_json(eqname)
    res = json.load(open(file)) if os.path.exists(file) else {}
    print(f"Final evaluation: {eqname}, adam_steps={steps}, shared lr={SHARED_LR:g}, settings={FIXED[eqname]}", flush=True)
    todo = [(s, reg, v, m) for s in seeds for reg, vals in GRID[eqname].items() for v in vals for m in FIXED[eqname]]
    for n, (seed, regime, val, method) in enumerate(todo, 1):      # seed-major: complete seeds first
        key = f"{eqname}|{regime}|val={val:g}|{method}|seed={seed}"
        if key in res:
            continue
        cfg = Config(adam_steps=steps, lbfgs_steps=lbfgs_steps, adam_lr=SHARED_LR)
        p = PINN(EQS[eqname](val), cfg, seed=seed)
        p.train(weight_hook=None if method == "vanilla" else build(method, FIXED[eqname][method]), verbose=False)
        r = p.evaluate()
        res[key] = {k: r[k] for k in ("rel_l2", "max_err", "res_norm", "train_time")}
        json.dump(res, open(file, "w"), indent=1)
        print(f"[{n}/{len(todo)}] {key:70s} relL2={r['rel_l2']:.3e}", flush=True)
    print(f"done: {len(res)} runs in {file}")

if __name__ == "__main__":
    if len(sys.argv) >= 4:
        run(sys.argv[1], seeds=list(range(int(sys.argv[2]), int(sys.argv[3]) + 1)))
    else:
        run(sys.argv[1])