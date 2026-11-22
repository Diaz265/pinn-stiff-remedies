# Final evaluation report (pre-specified analysis, PROTOCOL.md)

## convection

### interp, value 24  (complete paired seeds: 10)
| method | n | median | IQR | solved | partial | collapsed | solved 95% CI |
|---|---|---|---|---|---|---|---|
| vanilla | 10 | 0.233 | 0.0638-0.504 | 4 | 3 | 3 | 0.17-0.69 |
| ntk | 10 | 0.0709 | 0.0446-0.0933 | 8 | 2 | 0 | 0.49-0.94 |
| causal | 10 | 0.411 | 0.0484-0.59 | 4 | 3 | 3 | 0.17-0.69 |
| continuation | 10 | 0.512 | 0.192-0.567 | 1 | 3 | 6 | 0.02-0.40 |

Lowest median: **ntk**
Friedman across methods: chi2=12.36, p=0.00625
  ntk vs vanilla: Wilcoxon p=0.00977, Holm-adjusted p=0.0293 (better median)
  causal vs vanilla: Wilcoxon p=0.375, Holm-adjusted p=0.75 (worse/equal median)
  continuation vs vanilla: Wilcoxon p=0.492, Holm-adjusted p=0.75 (worse/equal median)
Best vs next-best (ntk vs vanilla): Wilcoxon p=0.00977

### interp, value 26  (complete paired seeds: 10)
| method | n | median | IQR | solved | partial | collapsed | solved 95% CI |
|---|---|---|---|---|---|---|---|
| vanilla | 10 | 0.678 | 0.613-0.697 | 0 | 1 | 9 | 0.00-0.28 |
| ntk | 10 | 0.183 | 0.0638-0.392 | 4 | 4 | 2 | 0.17-0.69 |
| causal | 10 | 0.65 | 0.554-0.701 | 0 | 2 | 8 | 0.00-0.28 |
| continuation | 10 | 0.689 | 0.644-0.716 | 0 | 1 | 9 | 0.00-0.28 |

Lowest median: **ntk**
Friedman across methods: chi2=13.08, p=0.00447
  ntk vs vanilla: Wilcoxon p=0.00391, Holm-adjusted p=0.0117 (better median)
  causal vs vanilla: Wilcoxon p=0.375, Holm-adjusted p=0.75 (better median)
  continuation vs vanilla: Wilcoxon p=0.695, Holm-adjusted p=0.75 (worse/equal median)
Best vs next-best (ntk vs causal): Wilcoxon p=0.0371

### extrap_low, value 18  (complete paired seeds: 10)
| method | n | median | IQR | solved | partial | collapsed | solved 95% CI |
|---|---|---|---|---|---|---|---|
| vanilla | 10 | 0.0417 | 0.0229-0.0895 | 8 | 2 | 0 | 0.49-0.94 |
| ntk | 10 | 0.0236 | 0.018-0.0277 | 10 | 0 | 0 | 0.72-1.00 |
| causal | 10 | 0.0207 | 0.0154-0.0286 | 9 | 1 | 0 | 0.60-0.98 |
| continuation | 10 | 0.0406 | 0.0382-0.0524 | 10 | 0 | 0 | 0.72-1.00 |

Lowest median: **causal**
Friedman across methods: chi2=14.04, p=0.00285
  ntk vs vanilla: Wilcoxon p=0.0273, Holm-adjusted p=0.0547 (better median)
  causal vs vanilla: Wilcoxon p=0.0137, Holm-adjusted p=0.041 (better median)
  continuation vs vanilla: Wilcoxon p=0.625, Holm-adjusted p=0.625 (better median)
Best vs next-best (causal vs ntk): Wilcoxon p=1

### extrap_high, value 34  (complete paired seeds: 10)
| method | n | median | IQR | solved | partial | collapsed | solved 95% CI |
|---|---|---|---|---|---|---|---|
| vanilla | 10 | 0.821 | 0.773-0.843 | 0 | 0 | 10 | 0.00-0.28 |
| ntk | 10 | 0.716 | 0.703-0.733 | 0 | 0 | 10 | 0.00-0.28 |
| causal | 10 | 0.798 | 0.778-0.814 | 0 | 0 | 10 | 0.00-0.28 |
| continuation | 10 | 0.849 | 0.825-0.87 | 0 | 0 | 10 | 0.00-0.28 |

Lowest median: **ntk**
Friedman across methods: chi2=26.04, p=9.36e-06
  ntk vs vanilla: Wilcoxon p=0.00195, Holm-adjusted p=0.00586 (better median)
  causal vs vanilla: Wilcoxon p=0.557, Holm-adjusted p=0.557 (better median)
  continuation vs vanilla: Wilcoxon p=0.00391, Holm-adjusted p=0.00781 (worse/equal median)
Best vs next-best (ntk vs causal): Wilcoxon p=0.00195

## oscillator

### interp, value 250  (complete paired seeds: 10)
| method | n | median | IQR | solved | partial | collapsed | solved 95% CI |
|---|---|---|---|---|---|---|---|
| vanilla | 10 | 0.15 | 0.027-0.329 | 4 | 4 | 2 | 0.17-0.69 |
| ntk | 10 | 0.986 | 0.985-0.988 | 0 | 0 | 10 | 0.00-0.28 |
| causal | 10 | 12.4 | 0.0375-28.1 | 4 | 1 | 5 | 0.17-0.69 |
| continuation | 10 | 0.0683 | 0.0352-0.278 | 6 | 4 | 0 | 0.31-0.83 |

Lowest median: **continuation**
Friedman across methods: chi2=12.24, p=0.0066
  ntk vs vanilla: Wilcoxon p=0.084, Holm-adjusted p=0.168 (worse/equal median)
  causal vs vanilla: Wilcoxon p=0.0273, Holm-adjusted p=0.082 (worse/equal median)
  continuation vs vanilla: Wilcoxon p=0.492, Holm-adjusted p=0.492 (better median)
Best vs next-best (continuation vs vanilla): Wilcoxon p=0.492

### interp, value 350  (complete paired seeds: 10)
| method | n | median | IQR | solved | partial | collapsed | solved 95% CI |
|---|---|---|---|---|---|---|---|
| vanilla | 10 | 5.9 | 2.17-118 | 1 | 0 | 9 | 0.02-0.40 |
| ntk | 10 | 0.988 | 0.987-0.989 | 0 | 0 | 10 | 0.00-0.28 |
| causal | 10 | 48.5 | 42.4-53.7 | 0 | 0 | 10 | 0.00-0.28 |
| continuation | 10 | 0.122 | 0.0477-0.249 | 3 | 6 | 1 | 0.11-0.60 |

Lowest median: **continuation**
Friedman across methods: chi2=19.56, p=0.000209
  ntk vs vanilla: Wilcoxon p=0.00977, Holm-adjusted p=0.0195 (better median)
  causal vs vanilla: Wilcoxon p=1, Holm-adjusted p=1 (worse/equal median)
  continuation vs vanilla: Wilcoxon p=0.00586, Holm-adjusted p=0.0176 (better median)
Best vs next-best (continuation vs ntk): Wilcoxon p=0.084

### extrap_low, value 120  (complete paired seeds: 10)
| method | n | median | IQR | solved | partial | collapsed | solved 95% CI |
|---|---|---|---|---|---|---|---|
| vanilla | 10 | 0.00592 | 0.00216-0.0145 | 10 | 0 | 0 | 0.72-1.00 |
| ntk | 10 | 0.975 | 0.974-0.976 | 1 | 0 | 9 | 0.02-0.40 |
| causal | 10 | 0.0152 | 0.00491-0.0288 | 9 | 0 | 1 | 0.60-0.98 |
| continuation | 10 | 0.0314 | 0.0221-0.0386 | 9 | 1 | 0 | 0.60-0.98 |

Lowest median: **vanilla**
Friedman across methods: chi2=20.76, p=0.000118
  ntk vs vanilla: Wilcoxon p=0.00195, Holm-adjusted p=0.00586 (worse/equal median)
  causal vs vanilla: Wilcoxon p=0.193, Holm-adjusted p=0.193 (worse/equal median)
  continuation vs vanilla: Wilcoxon p=0.00586, Holm-adjusted p=0.0117 (worse/equal median)
Best vs next-best (vanilla vs causal): Wilcoxon p=0.193

### extrap_high, value 600  (complete paired seeds: 10)
| method | n | median | IQR | solved | partial | collapsed | solved 95% CI |
|---|---|---|---|---|---|---|---|
| vanilla | 10 | 76.3 | 24.4-294 | 0 | 0 | 10 | 0.00-0.28 |
| ntk | 10 | 0.991 | 0.991-0.992 | 0 | 0 | 10 | 0.00-0.28 |
| causal | 10 | 118 | 86.3-208 | 0 | 0 | 10 | 0.00-0.28 |
| continuation | 10 | 71.1 | 6.64-294 | 0 | 1 | 9 | 0.00-0.28 |

Lowest median: **ntk**
Friedman across methods: chi2=16.68, p=0.000822
  ntk vs vanilla: Wilcoxon p=0.00195, Holm-adjusted p=0.00586 (better median)
  causal vs vanilla: Wilcoxon p=0.322, Holm-adjusted p=0.645 (worse/equal median)
  continuation vs vanilla: Wilcoxon p=0.695, Holm-adjusted p=0.695 (better median)
Best vs next-best (ntk vs continuation): Wilcoxon p=0.00391

## H1: no single remedy has the best median in every cell
- convection / interp / 24: ntk
- convection / interp / 26: ntk
- convection / extrap_low / 18: causal
- convection / extrap_high / 34: ntk
- oscillator / interp / 250: continuation
- oscillator / interp / 350: continuation
- oscillator / extrap_low / 120: vanilla
- oscillator / extrap_high / 600: ntk

H1 verdict: NOT falsified (4 different winners: causal, continuation, ntk, vanilla)

## H2: solved fraction lower in extrapolation-above than in interpolation (per method)
| method | interp solved | extrap_high solved | extrap_low solved (control) | Fisher one-sided p (interp > extrap_high) | verdict |
|---|---|---|---|---|---|
| vanilla | 9/40 | 0/20 | 18/20 | 0.0185 | supported |
| ntk | 12/40 | 0/20 | 11/20 | 0.00399 | supported |
| causal | 8/40 | 0/20 | 18/20 | 0.0301 | supported |
| continuation | 10/40 | 0/20 | 19/20 | 0.0112 | supported |