"""
Analysis of the final evaluation, exactly as pre-specified in PROTOCOL.md.
Reads final_results_convection.json / final_results_oscillator.json (whichever exist),
prints the report and saves it to analysis_report.md.

Needs scipy for the tests:  pip install scipy   (tables are printed even without it)
"""
import glob, json, math, os, sys
import numpy as np
try:
    from scipy import stats
except ImportError:
    stats = None
if os.environ.get("ANALYZE_NO_SCIPY"):
    stats = None

METHODS = ["vanilla", "ntk", "causal", "continuation"]
REGIMES = ["interp", "extrap_low", "extrap_high"]
SOLVED, COLLAPSED = 0.1, 0.5
out = []
def say(s=""):
    print(s); out.append(s)

def load(eq):
    files = sorted(glob.glob(f"final_results_{eq}*.json"))      # merges parallel chunks
    if not files:
        return None
    raw = {}
    for f in files:
        raw.update(json.load(open(f)))
    cells = {}
    for key, r in raw.items():
        e, regime, val, method, seed = key.split("|")
        cells.setdefault((regime, float(val.split("=")[1])), {}).setdefault(method, {})[int(seed.split("=")[1])] = r["rel_l2"]
    return cells

def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))

def holm(ps):
    order = np.argsort(ps); m = len(ps); adj = np.empty(m); run = 0.0
    for rank, i in enumerate(order):
        run = max(run, (m - rank) * ps[i]); adj[i] = min(1.0, run)
    return adj

def counts(v):
    v = np.asarray(v)
    return int((v < SOLVED).sum()), int(((v >= SOLVED) & (v <= COLLAPSED)).sum()), int((v > COLLAPSED).sum())

say("# Final evaluation report (pre-specified analysis, PROTOCOL.md)\n")
if stats is None:
    say("NOTE: scipy not available -> statistical tests are skipped (run `pip install scipy`).\n")

winners = {}                       # (eq, regime, val) -> method with lowest median
pooled = {m: {"interp": [0, 0], "extrap_high": [0, 0], "extrap_low": [0, 0]} for m in METHODS}
per_eq_h2 = {}

for eq in ("convection", "oscillator"):
    cells = load(eq)
    if cells is None:
        say(f"## {eq}: no results file yet\n"); continue
    say(f"## {eq}\n")
    for regime in REGIMES:
        for (rg, val) in sorted(k for k in cells if k[0] == regime):
            data = cells[(rg, val)]
            seeds_all = sorted(set.intersection(*[set(data.get(m, {})) for m in METHODS])) if all(m in data for m in METHODS) else []
            say(f"### {regime}, value {val:g}  (complete paired seeds: {len(seeds_all)})")
            say("| method | n | median | IQR | solved | partial | collapsed | solved 95% CI |")
            say("|---|---|---|---|---|---|---|---|")
            meds = {}
            for m in METHODS:
                v = np.array(list(data.get(m, {}).values()))
                if len(v) == 0:
                    continue
                s, p, c = counts(v); lo, hi = wilson(s, len(v)); q1, q3 = np.percentile(v, [25, 75])
                meds[m] = float(np.median(v))
                say(f"| {m} | {len(v)} | {np.median(v):.3g} | {q1:.3g}-{q3:.3g} | {s} | {p} | {c} | {lo:.2f}-{hi:.2f} |")
                pooled[m][regime][0] += s; pooled[m][regime][1] += len(v)
                per_eq_h2.setdefault((eq, m, regime), [0, 0]); per_eq_h2[(eq, m, regime)][0] += s; per_eq_h2[(eq, m, regime)][1] += len(v)
            if meds:
                winners[(eq, regime, val)] = min(meds, key=meds.get)
                say(f"\nLowest median: **{winners[(eq, regime, val)]}**")
            if stats is not None and len(seeds_all) >= 5:
                M = np.array([[data[m][s] for m in METHODS] for s in seeds_all])
                fr = stats.friedmanchisquare(*[M[:, j] for j in range(4)])
                say(f"Friedman across methods: chi2={fr.statistic:.2f}, p={fr.pvalue:.3g}")
                if fr.pvalue < 0.05:
                    ps = []
                    for j in (1, 2, 3):
                        try: ps.append(stats.wilcoxon(M[:, j], M[:, 0]).pvalue)
                        except ValueError: ps.append(1.0)
                    adj = holm(np.array(ps))
                    for j, pr, pa in zip((1, 2, 3), ps, adj):
                        d = "better" if np.median(M[:, j]) < np.median(M[:, 0]) else "worse/equal"
                        say(f"  {METHODS[j]} vs vanilla: Wilcoxon p={pr:.3g}, Holm-adjusted p={pa:.3g} ({d} median)")
                order = np.argsort(np.median(M, axis=0))
                b, n2 = order[0], order[1]
                try: pw = stats.wilcoxon(M[:, b], M[:, n2]).pvalue
                except ValueError: pw = 1.0
                say(f"Best vs next-best ({METHODS[b]} vs {METHODS[n2]}): Wilcoxon p={pw:.3g}")
            say("")

say("## H1: no single remedy has the best median in every cell")
if winners:
    names = sorted(set(winners.values()))
    for k, w in winners.items():
        say(f"- {k[0]} / {k[1]} / {k[2]:g}: {w}")
    if len(winners) < 8:
        say(f"\n(only {len(winners)} of 8 cells available so far; verdict provisional)")
    verdict = "FALSIFIED (one method wins every cell)" if len(names) == 1 else f"NOT falsified ({len(names)} different winners: {', '.join(names)})"
    say(f"\nH1 verdict: {verdict}\n")

say("## H2: solved fraction lower in extrapolation-above than in interpolation (per method)")
say("| method | interp solved | extrap_high solved | extrap_low solved (control) | Fisher one-sided p (interp > extrap_high) | verdict |")
say("|---|---|---|---|---|---|")
for m in METHODS:
    (si, ni), (se, ne), (sl, nl) = pooled[m]["interp"], pooled[m]["extrap_high"], pooled[m]["extrap_low"]
    if ni == 0 or ne == 0:
        continue
    fi, fe = si / ni, se / ne
    pval = stats.fisher_exact([[si, ni - si], [se, ne - se]], alternative="greater")[1] if stats is not None else float("nan")
    if fe >= fi:
        verdict = "FALSIFIED (extrap >= interp)"
    elif stats is None:
        verdict = "needs scipy"
    else:
        verdict = "supported" if pval < 0.05 else "inconclusive"
    say(f"| {m} | {si}/{ni} | {se}/{ne} | {sl}/{nl} | {pval:.3g} | {verdict} |")

open("analysis_report.md", "w").write("\n".join(out))
print("\nsaved analysis_report.md")