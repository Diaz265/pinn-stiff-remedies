"""
Results figure for the paper (strip plot of relative L2 error, 8 cells x 4 methods).
Reads final_results_convection*.json and final_results_oscillator*.json (parallel chunks are merged).
Needs matplotlib:  pip install matplotlib     Saves fig_results.pdf and fig_results.png
"""
import glob, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

M = ["vanilla", "ntk", "causal", "continuation"]
lab = {"vanilla": "Vanilla", "ntk": "NTK", "causal": "Causal", "continuation": "Contin."}
col = {"vanilla": "#555555", "ntk": "#0072B2", "causal": "#D55E00", "continuation": "#009E73"}
cells = {"convection": [("interp", 24, r"$\beta$=24 (interp.)"), ("interp", 26, r"$\beta$=26 (interp.)"),
                        ("extrap_low", 18, r"$\beta$=18 (extrap. below)"), ("extrap_high", 34, r"$\beta$=34 (extrap. above)")],
         "oscillator": [("interp", 250, "S=250 (interp.)"), ("interp", 350, "S=350 (interp.)"),
                        ("extrap_low", 120, "S=120 (extrap. below)"), ("extrap_high", 600, "S=600 (extrap. above)")]}

def load(eq):
    res = {}
    for f in sorted(glob.glob(f"final_results_{eq}*.json")):
        res.update(json.load(open(f)))
    return res

fig, axes = plt.subplots(2, 4, figsize=(7.2, 4.4), sharey="row")
rng = np.random.default_rng(1)
for r, eq in enumerate(("convection", "oscillator")):
    res = load(eq)
    for c, (reg, val, title) in enumerate(cells[eq]):
        ax = axes[r, c]
        for i, m in enumerate(M):
            v = np.array([res[f"{eq}|{reg}|val={val}|{m}|seed={s}"]["rel_l2"] for s in range(1000, 1010)])
            ax.scatter(i + rng.uniform(-0.16, 0.16, len(v)), v, s=11, color=col[m], alpha=0.8, linewidths=0)
            ax.hlines(np.median(v), i - 0.3, i + 0.3, color="black", lw=1.4)
        ax.axhline(0.1, color="green", ls="--", lw=0.7)     # solved threshold
        ax.axhline(0.5, color="red", ls=":", lw=0.7)        # collapsed threshold
        ax.set_yscale("log"); ax.set_xticks(range(4)); ax.set_xticklabels([lab[m] for m in M], rotation=60, fontsize=6.5)
        ax.set_title(title, fontsize=6.8); ax.tick_params(axis="y", labelsize=6.5)
        if c == 0:
            ax.set_ylabel(("1D convection" if r == 0 else "Stiff oscillator") + "\nrelative $L_2$ error", fontsize=7)
fig.tight_layout()
fig.savefig("fig_results.pdf"); fig.savefig("fig_results.png", dpi=200)
print("saved fig_results.pdf and fig_results.png")