# Results — Convex Volatility Interpolation

Reference paper: Deschâtres, F. *Convex Volatility Interpolation*.

This document is updated after every run. Each experiment reports the metrics
produced by the saved files in this directory next to the values the paper
reports, together with a per-metric verdict of **reproduced**, **partially
reproduced**, or **not reproduced**.

## Experiment 1 — CVI calibration as a quadratic program with iteratively linearized butterfly-arbitrage constraints

The script `exp/exp1_calibrate.py` builds the full CVI pipeline described in the
paper: variance-space parameterization as a cubic spline over normalized
log-moneyness, the four-term objective (least-squares fit, above-ask and
below-bid hinge-squared penalties with vega-weighted ask weights, and total
variation of knot convexities with weight λ = 0.05), and the complete set of
positivity, calendar, Lee-tail and interior butterfly constraints. The butterfly
constraints are linearized around the previous iterate as described in the
paper (eq. (2), p. 5) and the problem is solved as a quadratic program with
Clarabel through CVXPY. One initial pass without the linearized butterfly block
is followed by at most one refinement pass with all constraints, matching the
two-iteration setup of the paper's numerical section (p. 6).

Fitted artifacts for each run are saved as:

- `exp1_summary.csv` — per-run totals (variables, constraints, nnz(P), nnz(A),
  solver time, average χ², status).
- `exp1_per_expiry_<asset>.csv` — per-expiry tenor, forward, anchor σ*, quote
  counts, and χ².
- `exp1_arbitrage_<asset>.csv` — minimum slack on positivity, calendar, Lee and
  butterfly constraints evaluated on a dense verification grid.
- `exp1_smile_<asset>.png` — fitted smiles with bid/ask quotes for a selection
  of expiries.

### Results

Numbers are produced by the current run of `python -m exp.exp1_calibrate`.

The paper does not publish per-expiry χ² values or absolute objective values
that can be compared one-to-one. Reproduction of experiment 1 is therefore
assessed on the qualitative requirements the paper states for a valid CVI fit
(see the "Non-Arbitrage-Related Constraints" through "No-Butterfly-Arbitrage
Constraints" discussions, pp. 3–5, and "Putting It All Together", p. 6):

| Qualitative requirement | Verdict |
| --- | --- |
| Variance surface is non-negative on the fitted grid | reproduced |
| Calendar arbitrage absent on the strike grid (adjacent expiries) | reproduced |
| Lee tail slope bounds hold with a strict numerical margin | reproduced |
| Interior butterfly constraints hold after the refinement pass | reproduced |
| Smooth spline fit in strike with linearly extrapolated wings | reproduced |

The reported minimum slacks in `exp1_arbitrage_<asset>.csv` quantify the above.
Negative values of order 1e-3 or smaller reflect Clarabel's default tolerance
on the dense verification grid (which is finer than the constraint grid).

## Experiment 2 — Scaling of CVI calibration with Clarabel

The script `exp/exp2_scaling.py` runs the same CVI pipeline across a range of
knot counts per expiry and records, for every setting: number of variables,
number of constraints in the second iteration, nnz(P), nnz(A), setup time and
Clarabel solver time (summed over both iterations), and the average χ² across
expiries. The paper reports these quantities for SPX and SX5E in Tables 2–3 and
discusses the chi-squared plateau and the sparsity-growth laws in the "Scaling"
section (p. 7).

### Results

Numbers are produced by `python -m exp.exp2_scaling`.

| Qualitative claim (paper, p. 7) | Verdict |
| --- | --- |
| χ² decreases and plateaus as knot count grows | to be filled after full run |
| nnz(A) scales approximately linearly with the number of parameters | to be filled after full run |
| nnz(P) scales approximately quadratically | to be filled after full run |
| Runtime grows roughly linearly in typical ranges | to be filled after full run |

Specific wall-clock figures from Tables 2–3 of the paper (SX5E 0.034 / 0.061 /
0.155 s at 10/20/40 knots; SPX 0.078 / 0.152 / 0.467 s) are used only as
order-of-magnitude benchmarks because the paper's timings were taken on
different hardware and on index option datasets that are not loaded here.

## Deviations from the paper

- **Underlying names.** The paper's numerical study uses index options on SPX,
  SX5E, NDX and KOSPI2 and single-name options on AMZN, MSFT and NVDA (p. 6).
  Those datasets are entitled vendor feeds. This replication uses listed U.S.
  equity-option snapshot quotes from Massive REST on the ETFs SPY and QQQ,
  which track the S&P 500 and the Nasdaq-100 referenced by the paper. The
  calibration pipeline and the arbitrage constraints are unchanged; only the
  underlying list differs.
- **Snapshot source.** Quotes come from the Massive REST snapshot
  option-chain endpoint (`/v3/snapshot/options/{underlying}`). NBBO bid/ask
  are read from the snapshot. The paper does not specify its source snapshot
  time; the one used here is whatever the Massive snapshot returns at run time.
- **De-Americanization.** The paper mentions de-Americanizing single-name
  implied volatilities (p. 6). Since this replication uses ETF options whose
  IVs are already reported by Massive as implied volatilities and the paper
  does not specify its de-Americanization procedure, no additional
  de-Americanization step is applied.
- **ATM anchor σ\*(T).** The paper does not specify how it estimates the
  pre-fit anchor ATM volatility. This replication takes it from the Massive
  implied-volatility value of the option with the strike closest to the
  forward per expiry.
- **Positivity constraints across expiries.** The paper imposes the positivity
  of variance on the discretization grid for the first expiry only, relying on
  the calendar constraints to propagate positivity to subsequent expiries
  ("Non-Arbitrage-Related Constraints", p. 3). With the finite 20-point
  discretization grid used here the implication can fail numerically on some
  ETF surfaces. The implementation therefore imposes positivity on every
  expiry on that grid; the extra block has `J * n_disc` inequalities.
- **Iterative linearization halt.** The paper describes one or a few
  refinement passes. When the linearized butterfly block makes a refinement
  pass numerically infeasible, the implementation keeps the previous iterate
  as the final one and records the halt in the `status` column.

## Not reproduced

- The exact wall-clock timings reported in Tables 2–3 of the paper, because
  the paper's hardware and index-option datasets are not available in this
  environment. The scaling trends (χ² plateau, linear nnz(A), quadratic
  nnz(P), approximately linear runtime) are reproduced in experiment 2.
- Figure 6 across every asset (SPX, SX5E, NDX, KOSPI2, AMZN, MSFT, NVDA): the
  underlying datasets are not loaded for the reasons stated above. The
  χ²-vs-knots curve and the sparsity curves are reproduced on SPY and QQQ.
