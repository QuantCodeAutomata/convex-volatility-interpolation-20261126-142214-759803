"""Parameters for the CVI replication.

Each variable has a comment with its meaning, unit, and source (paper section or
equation, or 'not specified in the paper'). All experiment parameters are
module-level UPPER_CASE variables, following the paper "Convex Volatility
Interpolation" (Deschâtres, 2026).
"""

from __future__ import annotations

import numpy as np

# ---------------------------------------------------------------------------
# CVI parameterization (paper "The CVI Parameterization", p. 2)
# ---------------------------------------------------------------------------

# Default number of spline knots per expiry. The paper reports timings for
# n in {10, 20, 40}; 20 is the headline value (Table 3).
N_KNOTS_DEFAULT: int = 20

# Half-range of normalized log-moneyness covered by the knots. The example of
# Figure 1 uses z in [-8, 8]; the paper states "the range of knots should be
# sufficiently wide to include all the listed strikes targeted for fitting".
Z_HALF_RANGE: float = 8.0

# Number of z-grid points used to discretize the constraints. The paper's
# numerical section states: "Discretized constraints are applied across 20
# strikes" (p. 6, Numerical Results).
N_DISCRETIZATION: int = 20

# Number of QP iterations. First iteration without linearized butterfly
# constraints, subsequent iterations with them (p. 6 "Putting It All Together").
N_ITERATIONS: int = 2

# ---------------------------------------------------------------------------
# Objective-function weights (paper Section "Objective Function", pp. 2-3)
# ---------------------------------------------------------------------------

# Strike-regularization factor (total variation of convexities). The paper
# recommends 0.05 as "a sensible choice across underlyings" (p. 3).
LAMBDA_TV: float = 0.05

# ---------------------------------------------------------------------------
# Numerical margins (paper Section "No-Strike-Arbitrage Constraints", p. 4)
# ---------------------------------------------------------------------------

# Strict-inequality margin for Lee's tail slope bounds s < 2/sqrt(v* T). The
# paper says "we enforce them numerically with a small margin" without giving
# a value; not specified in the paper.
LEE_MARGIN: float = 1e-4

# Floor on the reference variance used in butterfly linearization, to avoid
# dividing by zero when v_ref is tiny. Not specified in the paper.
V_REF_FLOOR: float = 1e-6

# ---------------------------------------------------------------------------
# Experiment 2 (scaling benchmark, paper "Numerical Results" p. 6, Tables 2-3)
# ---------------------------------------------------------------------------

# Knot counts used in the SPX / SX5E scaling table (Table 3).
BENCH_KNOT_COUNTS: tuple[int, ...] = (10, 20, 40)

# Knot counts used in the chi-squared plot (Figure 6, "ranging from 5 to 35").
BENCH_KNOT_COUNTS_FIG6: tuple[int, ...] = tuple(range(5, 36, 5))

# ---------------------------------------------------------------------------
# Data acquisition (universe, moneyness filter, tenor filter)
# ---------------------------------------------------------------------------

# Base URL of the Massive REST API used by the option-chain loader.
MASSIVE_BASE: str = "https://api.massive.com"

# Underlyings analysed. SPY is an American-exercise ETF option proxy for SPX
# (European index option quotes require an index-options plan not available to
# this entitlement): see results/RESULTS.md for the deviation note.
UNDERLYINGS: tuple[str, ...] = ("SPY", "QQQ")

# Synthetic half-spread (in volatility points) around the mid IV, used because
# the NBBO is not available on the active Massive plan. Half-spread grows with
# |z| so wing quotes are wider than ATM quotes, as commonly observed. Not
# specified in the paper; see results/RESULTS.md under "Deviations from the
# paper".
SPREAD_HALF_WIDTH_ATM: float = 0.005
SPREAD_HALF_WIDTH_WING_SLOPE: float = 0.004

# Half-width of the moneyness band k = log(K/F), used to filter contracts
# before fetching individual NBBO quotes. The paper's z range is [-8, 8];
# in k terms that is ±8 * sigma_star * sqrt(T). We keep |k| <= 0.5 for short
# tenors and widen with sqrt(T) inside the loader.
MONEYNESS_K_HALF_WIDTH_REF: float = 0.5

# Minimum and maximum time to expiry in calendar days for calibration inputs.
TENOR_MIN_DAYS: int = 3
TENOR_MAX_DAYS: int = 540

# Maximum number of expiries to keep per underlying (prevents the chain from
# exploding on weekly-heavy roots). Not a paper parameter.
MAX_EXPIRIES_PER_UNDERLYING: int = 25

# Maximum number of contracts per expiry retained after the moneyness filter.
MAX_CONTRACTS_PER_EXPIRY: int = 60

# Maximum parallel NBBO quote requests.
QUOTE_PARALLEL: int = 8

# Dividend yield fallback when the loader cannot infer one. SPY ~1.3%, QQQ ~0.5%.
DIVIDEND_YIELD_DEFAULT: dict[str, float] = {"SPY": 0.013, "QQQ": 0.005}

# Flat risk-free rate used across expiries (fallback; the actual value is
# refined per expiry from put-call parity of the fetched quotes).
RISK_FREE_RATE_DEFAULT: float = 0.045

# Minimum bid-ask spread in variance space used when deriving weights, so that
# zero-spread rows do not dominate the chi-squared. Not specified in the paper.
V_SPREAD_FLOOR: float = 1e-6


def make_knot_grid(n_knots: int, half_range: float = Z_HALF_RANGE) -> np.ndarray:
    """Return ``n_knots`` symmetric knots in normalized log-moneyness.

    Knots are placed on a tanh-warped grid so that density is higher near the
    money (as recommended by the paper, "The CVI Parameterization", p. 2:
    "increasing knot density near-the-money better captures variations in
    variance convexity"). The ATM knot z = 0 is included by construction when
    ``n_knots`` is odd; when it is even, the two knots closest to zero bracket
    the money.
    """
    if n_knots < 4:
        raise ValueError("CVI requires at least 4 knots")
    # Build half of the knots on the positive side, mirror them, and always
    # include 0 as the ATM knot (paper, p. 2: "we require z = 0 to be a knot").
    # For even n_knots, the positive side carries one more knot than the
    # negative side so that the ATM knot sits symmetrically in the interior.
    n_neg = (n_knots - 1) // 2
    n_pos = n_knots - 1 - n_neg
    u_neg = np.linspace(0.0, 1.0, n_neg + 1)[1:]
    u_pos = np.linspace(0.0, 1.0, n_pos + 1)[1:]
    neg = -half_range * np.tanh(1.5 * u_neg[::-1]) / np.tanh(1.5)
    pos = half_range * np.tanh(1.5 * u_pos) / np.tanh(1.5)
    knots = np.concatenate([neg, [0.0], pos])
    assert len(knots) == n_knots
    assert np.all(np.diff(knots) > 0), "knot grid must be strictly increasing"
    return knots
