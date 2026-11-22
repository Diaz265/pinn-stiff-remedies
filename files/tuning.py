"""
Validation-grid tuning of remedy hyperparameters (equal effort for every method).

PROTOCOL v2  (amended after the first oscillator round -- see note below)
  * ONE shared optimizer setting for ALL methods: Adam lr = 3e-4, same network, same
    Adam/L-BFGS budget. Only the remedy's own hyperparameter differs between runs.
  * Each remedy has ONE hyperparameter with 3 candidate values (default included).
    Vanilla has no remedy hyperparameter: it is run once at the shared lr.
  * Runs per equation: 3 remedies x 3 settings x 2 validation values x 3 tuning seeds
    (100,101,102) = 54 new runs; vanilla's 6 runs are reused from round 1 (identical
    conditions: lr = 3e-4, no hook, same seeds and validation values).
  * Validation values are NEVER used in the final evaluation grid.
  * Selection rule (fixed in advance): lowest MEDIAN relative L2 over the 6 runs.

AMENDMENT NOTE (report this in the paper): round 1 tuned vanilla's learning rate while
the remedies stayed at lr = 1e-3, which broke the "identical optimizer" rule and, since lr
changed vanilla's error by >100x, would have been unfair to the remedies. Round-1 results
are kept (keys without the 'v2|' prefix) and belong in an appendix.

Usage:  python tuning.py <convection|oscillator> [adam_steps]            (resumable)
        python tuning.py <convection|oscillator> [adam_steps] summary
"""
import json, os, sys
import numpy as np
from pinn import PINN, Config, Convection, Oscillator
from remedies import make_ntk_hook, make_causal_hook, make_continuation_hook

EQS = {"convection": Convection, "oscillator": Oscillator}
VAL_VALUES = {"convection": [22.0, 28.0], "oscillator": [200.0, 400.0]}
DEFAULT_STEPS = {"convection": 10000, "oscillator": 5000}
SEEDS = [100, 101, 102]
BASE_EPS = (1e-2, 1e-1, 1.0, 10.0, 100.0)
SHARED_LR = 3e-4

# method -> (name of knob, candidate values, default value)
GRID = {
    "vanilla":      ("lr",        [SHARED_LR],        SHARED_LR),   # no remedy knob
    "ntk":          ("alpha",     [0.1, 0.5, 0.9],    0.9),
    "causal":       ("eps_scale", [1.0, 1e-2, 1e-4],  1.0),
    "continuation": ("n_stages",  [3, 6, 12],         6),
}

def build(method, knob_value):
    """-> weight_hook or None   (the optimizer lr is the shared SHARED_LR for everyone)"""
    if method == "vanilla":
        return None
    if method == "ntk":
        return make_ntk_hook(alpha=knob_value)
    if method == "causal":
        return make_causal_hook(eps_schedule=tuple(e * knob_value for e in BASE_EPS))
    if method == "continuation":
        return make_continuation_hook(n_stages=int(knob_value))

def key_v2(eqname, steps, method, knob, c, val, seed):
    return f"v2|{eqname}|{steps}|{method}|{knob}={c:g}|val={val:g}|seed={seed}"

def key_v1_vanilla(eqname, steps, val, seed):          # round-1 key, same conditions as v2 vanilla
    return f"{eqname}|{steps}|vanilla|lr={SHARED_LR:g}|val={val:g}|seed={seed}"

def summarize(eqname, steps, res, file_out=True):
    n_full = len(VAL_VALUES[eqname]) * len(SEEDS)
    print(f"\n=== tuning summary (protocol v2, shared lr={SHARED_LR:g}): {eqname}, adam_steps={steps} "
          f"(median rel. L2 over {n_full} runs) ===")
    best = {}
    for method, (knob, cands, default) in GRID.items():
        rows = []
        for c in cands:
            v = [r["rel_l2"] for k, r in res.items()
                 if k.startswith(f"v2|{eqname}|{steps}|{method}|{knob}={c:g}|")]
            if v:
                rows.append((c, float(np.median(v)), len(v)))
                solved = sum(x < 0.1 for x in v)
                print(f"{method:13s} {knob}={c:<8g} median={np.median(v):9.3e}  min={min(v):9.3e}  "
                      f"max={max(v):9.3e}  solved(<0.1)={solved}/{len(v)}"
                      + ("   (default)" if c == default and len(cands) > 1 else ""))
        full = [r for r in rows if r[2] == n_full]
        if len(full) == len(cands):
            b = min(full, key=lambda r: (r[1], r[0] != default))
            best[method] = {knob: b[0], "median_rel_l2": b[1]}
            if len(cands) > 1:
                print(f"  -> selected {knob}={b[0]:g}\n")
    if file_out and best:
        json.dump(best, open(f"best_settings_v2_{eqname}.json", "w"), indent=1)
        print(f"saved best_settings_v2_{eqname}.json")

def run(eqname, steps, summary_only=False, lbfgs_steps=500, file="tuning_results.json"):
    res = json.load(open(file)) if os.path.exists(file) else {}
    if not summary_only:
        for seed in SEEDS:                                   # seed-major: all methods progress evenly
            for val in VAL_VALUES[eqname]:
                for method, (knob, cands, default) in GRID.items():
                    for c in cands:
                        key = key_v2(eqname, steps, method, knob, c, val, seed)
                        if key in res:
                            continue
                        if method == "vanilla":              # reuse identical round-1 run if present
                            old = key_v1_vanilla(eqname, steps, val, seed)
                            if old in res:
                                res[key] = res[old]
                                json.dump(res, open(file, "w"), indent=1)
                                print(f"{key:82s} (reused from round 1) relL2={res[key]['rel_l2']:.3e}", flush=True)
                                continue
                        cfg = Config(adam_steps=steps, lbfgs_steps=lbfgs_steps, adam_lr=SHARED_LR)
                        p = PINN(EQS[eqname](val), cfg, seed=seed)
                        p.train(weight_hook=build(method, c), verbose=False)
                        r = p.evaluate()
                        res[key] = {k: r[k] for k in ("rel_l2", "max_err", "res_norm", "train_time")}
                        json.dump(res, open(file, "w"), indent=1)
                        print(f"{key:82s} relL2={r['rel_l2']:.3e}  t={r['train_time']:.0f}s", flush=True)
    summarize(eqname, steps, res)


if __name__ == "__main__":
    eqname = sys.argv[1]
    steps = int(sys.argv[2]) if len(sys.argv) > 2 else DEFAULT_STEPS[eqname]
    run(eqname, steps, summary_only=(len(sys.argv) > 3 and sys.argv[3] == "summary"))