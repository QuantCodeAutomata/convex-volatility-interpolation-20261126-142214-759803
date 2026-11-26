"""CVI calibration as a quadratic program.

Assembles the four-term objective and the complete set of arbitrage/positivity
constraints from Sections "Objective Function", "Non-Arbitrage-Related
Constraints", "No-Calendar-Spread-Arbitrage Constraints",
"No-Strike-Arbitrage Constraints" and "No-Butterfly-Arbitrage Constraints"
(Deschâtres, 2026, pp. 2-6), and solves with Clarabel via CVXPY.

The nonlinear no-butterfly inequality (paper eq. (2), p. 5) is handled by the
iterative linearization of Section "Putting It All Together" (p. 6):
``N_ITERATIONS`` passes, the first one dropping the linearized butterfly block
and the subsequent ones adding it from the previous iterate.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import cvxpy as cp
import numpy as np
from scipy.stats import norm

from .config import (
    LAMBDA_TV,
    LEE_MARGIN,
    N_ITERATIONS,
    V_REF_FLOOR,
)
from .preprocess import ExpiryData
from .spline import CVISplineBasis


@dataclass
class CalibrationResult:
    """Output of :func:`calibrate`."""

    knots: np.ndarray
    params: list[np.ndarray]                   # one (n+2,)-vector per expiry
    expiries: list[ExpiryData]
    solver_time_s: float                       # sum across iterations
    setup_time_s: float
    objective_value: float
    n_variables: int
    n_constraints_final: int
    nnz_P: int
    nnz_A: int
    iterations_done: int
    chi2_per_expiry: list[float] = field(default_factory=list)
    status: str = ""

    @property
    def avg_chi2(self) -> float:
        if not self.chi2_per_expiry:
            return float("nan")
        return float(np.mean(self.chi2_per_expiry))

    def fitted_variance(self, j: int, z_eval: np.ndarray) -> np.ndarray:
        """Return fitted variance at ``z_eval`` for expiry ``j``."""
        basis = CVISplineBasis(self.knots)
        v_eval, _, _ = basis.evaluate(self.params[j], z_eval)
        return v_eval


# ---------------------------------------------------------------------------
# Butterfly linearization (paper p. 5, eq. (2) and surrounding text)
# ---------------------------------------------------------------------------


def _butterfly_rhs_and_grad(z: np.ndarray, v_ref: np.ndarray, s_ref: np.ndarray,
                            v_star: float, T: float
                            ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Right-hand side of the paper's butterfly inequality (2) and its gradient.

    Returns ``(f_ref, df_dv, df_ds)`` at ``(v_ref, s_ref)`` evaluated at the
    moneyness points ``z``. The constraint added to the QP is

        c_eval  >=  f_ref + df_dv (v_eval - v_ref) + df_ds (s_eval - s_ref)

    which is a linear inequality in the parameters once ``c_eval``, ``v_eval``
    and ``s_eval`` are expressed through the spline basis.
    """
    v_ref = np.maximum(np.asarray(v_ref, dtype=float), V_REF_FLOOR)
    s_ref = np.asarray(s_ref, dtype=float)
    z = np.asarray(z, dtype=float)
    sqrtT = np.sqrt(T)
    sqrtvT = np.sqrt(v_ref * T)
    vstar_over_v = v_star / v_ref
    sqrt_vstar_over_v = np.sqrt(vstar_over_v)

    # d_1 at reference: log(K/F) = z sigma_* sqrt(T); divide by v_ref sqrt(T).
    # d_1 = log(K/F) / (sigma_ref sqrt(T)) - 0.5 sigma_ref sqrt(T)
    sigma_ref = np.sqrt(v_ref)
    d1 = z * np.sqrt(v_star) / sigma_ref - 0.5 * sigma_ref * sqrtT
    d2 = d1 - sigma_ref * sqrtT

    # f(v, s) as a function of (v, s) only (z, v_star, T fixed):
    # f = (v*/(2v)) s^2 + sqrt(v* T) s
    #     - 2 ( 1 + d_1(v) sqrt(v*/v) s + d_1(v) d_2(v) (v*/(4v)) s^2 )
    term_quad = (v_star / (2.0 * v_ref)) * s_ref ** 2
    term_lin = np.sqrt(v_star * T) * s_ref
    term_bs1 = d1 * sqrt_vstar_over_v * s_ref
    term_bs2 = d1 * d2 * (v_star / (4.0 * v_ref)) * s_ref ** 2
    f_ref = term_quad + term_lin - 2.0 * (1.0 + term_bs1 + term_bs2)

    # Partial derivatives. We use closed-form derivatives in v and s treating
    # d_1, d_2 as functions of v (through sigma_ref).
    # df/ds (at fixed v):
    df_ds = (v_star / v_ref) * s_ref + np.sqrt(v_star * T) \
        - 2.0 * (d1 * sqrt_vstar_over_v + d1 * d2 * (v_star / (2.0 * v_ref)) * s_ref)

    # df/dv: use dd1/dv and dd2/dv.
    # sigma_ref = v^{1/2} => dsigma/dv = 1/(2 sigma_ref)
    # d1 = z sqrt(v*)/sigma - 0.5 sigma sqrt(T)
    # dd1/dv = -z sqrt(v*)/sigma^2 * dsigma/dv - 0.5 sqrt(T) dsigma/dv
    #        = (-z sqrt(v*)/v - 0.5 sqrt(T)) / (2 sigma_ref)
    dsig_dv = 1.0 / (2.0 * sigma_ref)
    dd1_dv = (-z * np.sqrt(v_star) / v_ref) * dsig_dv - 0.5 * sqrtT * dsig_dv
    dd2_dv = dd1_dv - sqrtT * dsig_dv   # d2 = d1 - sigma sqrt(T)

    # Derivatives of each term wrt v:
    # d/dv [ (v*/(2v)) s^2 ] = -(v*/(2 v^2)) s^2
    d_quad_dv = -(v_star / (2.0 * v_ref ** 2)) * s_ref ** 2
    # d/dv [ sqrt(v* T) s ] = 0
    # d/dv [ d1 sqrt(v*/v) s ] = [dd1/dv * sqrt(v*/v) + d1 * d/dv(sqrt(v*/v))] s
    # d/dv sqrt(v*/v) = -0.5 sqrt(v*) / v^{3/2} = -0.5 sqrt(v*/v) / v
    dsqrtvsv_dv = -0.5 * sqrt_vstar_over_v / v_ref
    d_bs1_dv = (dd1_dv * sqrt_vstar_over_v + d1 * dsqrtvsv_dv) * s_ref
    # d/dv [ d1 d2 (v*/(4v)) s^2 ]
    #   = [ (dd1/dv)*d2 + d1*(dd2/dv) ] (v*/(4v)) s^2
    #     + d1 d2 * (-v*/(4 v^2)) s^2
    d_bs2_dv = ((dd1_dv * d2 + d1 * dd2_dv) * (v_star / (4.0 * v_ref))
                - d1 * d2 * (v_star / (4.0 * v_ref ** 2))) * s_ref ** 2
    df_dv = d_quad_dv - 2.0 * (d_bs1_dv + d_bs2_dv)
    return f_ref, df_dv, df_ds


# ---------------------------------------------------------------------------
# Main calibration driver
# ---------------------------------------------------------------------------


def calibrate(
    expiries: Sequence[ExpiryData],
    knots: np.ndarray,
    *,
    lambda_tv: float = LAMBDA_TV,
    n_iterations: int = N_ITERATIONS,
    n_discretization: int | None = None,
    solver: str = "CLARABEL",
    verbose: bool = False,
    warm_start_params: list[np.ndarray] | None = None,
) -> CalibrationResult:
    """Calibrate the CVI surface across ``expiries``.

    Parameters
    ----------
    expiries : sequence of :class:`ExpiryData`
        Market data per expiry, sorted by tenor.
    knots : np.ndarray
        Common knot grid in normalized log-moneyness ``z``; must contain 0.
    lambda_tv : float
        Strike-regularization weight ``lambda`` of paper p. 3.
    n_iterations : int
        Number of QP passes; the first one drops the linearized butterfly
        constraints, subsequent ones add them from the previous iterate
        (paper "Putting It All Together", p. 6).
    n_discretization : int
        Number of z-points used to discretize positivity / calendar / interior
        butterfly constraints (paper "Numerical Results", p. 6: 20).
    solver : str
        CVXPY solver name; "CLARABEL" is the paper's choice (p. 6).
    """
    import time

    expiries = list(expiries)
    if not expiries:
        raise ValueError("at least one expiry is required")
    # Enforce non-decreasing tenor.
    tenors = [ex.tenor_years for ex in expiries]
    if any(tenors[j] >= tenors[j + 1] for j in range(len(tenors) - 1)):
        order = np.argsort(tenors)
        expiries = [expiries[i] for i in order]
    J = len(expiries)
    n_disc = int(n_discretization) if n_discretization is not None else 20

    basis = CVISplineBasis(np.asarray(knots, dtype=float))
    n_params_per_expiry = basis.n_params
    z0, zn = float(basis.knots[0]), float(basis.knots[-1])
    z_disc = np.linspace(z0, zn, n_disc)

    # Precompute operator rows we'll need per expiry at each type of evaluation
    # point. All of these are independent of the parameters.
    t_setup_start = time.perf_counter()
    op_quotes = []   # (B_v,) at the market quote z-points of the expiry
    op_disc = []     # (B_v, B_vp, B_vpp) at the discretization grid
    op_edges = []    # (B_v, B_vp, B_vpp) at [z_0, z_{n-1}]
    op_cal = []      # for calendar: each expiry at its own z-grid sampled at
                     # the same k as the next expiry's grid (set up below)

    # Clip market quote z to the knot range for the least-squares / below-bid
    # penalties (paper "Below Bid Penalty", p. 3: "Options with normalized
    # log-moneyness outside [z_0, z_{n-1}] are excluded").
    in_range_idx: list[np.ndarray] = []
    for ex in expiries:
        inside = (ex.z >= z0) & (ex.z <= zn)
        in_range_idx.append(inside)
        z_q = ex.z[inside]
        B_v_q, _, _ = basis.operators(z_q)
        op_quotes.append(B_v_q)

        B_v_d, B_vp_d, B_vpp_d = basis.operators(z_disc)
        op_disc.append((B_v_d, B_vp_d, B_vpp_d))

        B_v_e, B_vp_e, B_vpp_e = basis.operators(np.array([z0, zn]))
        op_edges.append((B_v_e, B_vp_e, B_vpp_e))

    # Calendar grid: take the next expiry at log-moneyness k_i = z_disc *
    # sigma_*_j * sqrt(T_j). Convert to the next expiry's z:
    # z_next = k_i / (sigma_*_{j+1} sqrt(T_{j+1})).
    op_cal_pairs = []
    for j in range(J - 1):
        ex_j, ex_k = expiries[j], expiries[j + 1]
        k_grid = z_disc * ex_j.sigma_star * np.sqrt(ex_j.tenor_years)
        z_in_next = k_grid / (ex_k.sigma_star * np.sqrt(ex_k.tenor_years))
        B_v_j, _, _ = basis.operators(z_disc)
        B_v_k, _, _ = basis.operators(z_in_next)
        op_cal_pairs.append((B_v_j, B_v_k))

    # Build CVXPY variables: one parameter block per expiry.
    p_vars = [cp.Variable(n_params_per_expiry, name=f"p_{j}") for j in range(J)]

    # Objective pieces.
    obj_terms = []
    q_j_list: list[float] = []  # for ask/bid weights (eq. for q_j, paper p. 3)

    for j, ex in enumerate(expiries):
        inside = in_range_idx[j]
        has_mid = ex.has_mid & inside
        has_ask = ex.has_ask & inside
        has_bid = ex.has_bid & inside

        # Least-squares penalty (paper eq. for Least-Squares Penalty, p. 2-3).
        if has_mid.any():
            v_mid = ex.v_mid[inside][has_mid[inside]]
            v_bid = ex.v_bid[inside][has_mid[inside]]
            v_ask = ex.v_ask[inside][has_mid[inside]]
            spread = np.maximum(v_ask - v_bid, 1e-8)
            w_ls = 1.0 / (spread ** 2)
            rows = np.flatnonzero(has_mid[inside])
            B = op_quotes[j][rows, :]
            resid = B @ p_vars[j] - v_mid
            weight = w_ls / len(rows)
            obj_terms.append(cp.sum(cp.multiply(weight, cp.square(resid))))
            q_j = float(np.sum(w_ls))
        else:
            q_j = 0.0
        q_j_list.append(q_j)

        # Above-ask penalty (paper "Above Ask Penalty", p. 3).
        # Weight w_{i,j,ask} = q_j * Vega_ask(K_i) / sum(Vega_ask).
        if q_j > 0 and has_ask.any():
            rows_ask = np.flatnonzero(has_ask[inside])
            vega = ex.vega_ask[inside][has_ask[inside]]
            vega_sum = float(vega.sum())
            if vega_sum > 0:
                w_ask = q_j * vega / vega_sum
                v_ask_vals = ex.v_ask[inside][has_ask[inside]]
                B_a = op_quotes[j][rows_ask, :]
                excess = B_a @ p_vars[j] - v_ask_vals
                obj_terms.append(
                    cp.sum(cp.multiply(w_ask, cp.square(cp.pos(excess)))) / len(rows_ask)
                )

        # Below-bid penalty, mirroring the above-ask one (paper p. 3).
        if q_j > 0 and has_bid.any():
            rows_bid = np.flatnonzero(has_bid[inside])
            # Use vega at bid IV; approximate with vega_ask when missing.
            sigma_bid = ex.sigma_bid[inside][has_bid[inside]]
            strikes = ex.strikes[inside][has_bid[inside]]
            vega_b = _vega(ex.forward, strikes, ex.tenor_years, sigma_bid)
            vega_b_sum = float(vega_b.sum())
            if vega_b_sum > 0:
                w_bid = q_j * vega_b / vega_b_sum
                v_bid_vals = ex.v_bid[inside][has_bid[inside]]
                B_b = op_quotes[j][rows_bid, :]
                shortfall = v_bid_vals - B_b @ p_vars[j]
                obj_terms.append(
                    cp.sum(cp.multiply(w_bid, cp.square(cp.pos(shortfall)))) / len(rows_bid)
                )

        # Strike regularization: lambda * sum_j sum_i |c_{i+1} - c_i|
        # c is the sub-vector p_vars[j][2:], length n.
        c_vec = p_vars[j][2:]
        n_c = n_params_per_expiry - 2
        D = np.zeros((n_c - 1, n_c))
        for i in range(n_c - 1):
            D[i, i] = -1.0
            D[i, i + 1] = 1.0
        obj_terms.append(lambda_tv * cp.norm1(D @ c_vec))

    # ------------------------------------------------------------------
    # Build the fixed (non-butterfly) constraint list used by every pass.
    # ------------------------------------------------------------------
    fixed_constraints = []
    # Edge convexities c_0 = c_{n-1} = 0.
    for j in range(J):
        fixed_constraints.append(p_vars[j][2] == 0.0)
        fixed_constraints.append(p_vars[j][-1] == 0.0)

    # Positivity of variance on the discretization grid. The paper imposes
    # this on the first expiry only, arguing that the calendar constraints
    # propagate positivity to subsequent ones ("Non-Arbitrage-Related
    # Constraints", p. 3). With the finite discretization grid used here
    # the implication can fail numerically, so positivity is enforced on
    # every expiry; the extra constraints are J * n_disc inequalities.
    for j in range(J):
        B_v_j, _, _ = op_disc[j]
        fixed_constraints.append(B_v_j @ p_vars[j] >= 0.0)
    # Positivity in the tails (dv/dz <= 0 at z_0, >= 0 at z_{n-1}) for all j.
    for j in range(J):
        _, _, _ = op_edges[j]
        B_v_e, B_vp_e, _ = op_edges[j]
        fixed_constraints.append(B_vp_e[0, :] @ p_vars[j] <= 0.0)   # left tail
        fixed_constraints.append(B_vp_e[1, :] @ p_vars[j] >= 0.0)   # right tail

    # No-calendar-spread-arbitrage on the discrete strike grid (paper p. 4).
    for j in range(J - 1):
        ex_j, ex_k = expiries[j], expiries[j + 1]
        B_v_j, B_v_k = op_cal_pairs[j]
        fixed_constraints.append(
            cp.multiply(ex_j.tenor_years, B_v_j @ p_vars[j])
            <= cp.multiply(ex_k.tenor_years, B_v_k @ p_vars[j + 1])
        )
    # Tail calendar inequalities on skew (paper p. 4).
    for j in range(J - 1):
        ex_j, ex_k = expiries[j], expiries[j + 1]
        _, B_vp_e_j, _ = op_edges[j]
        _, B_vp_e_k, _ = op_edges[j + 1]
        # dv/dz at z_0: left-tail inequality
        #   s(z_0, j) sqrt(v*_j T_j) >= s(z_0, j+1) sqrt(v*_{j+1} T_{j+1})
        # with s = dv/dz / v* => s sqrt(v* T) = dv/dz * sqrt(T/v*).
        coef_j_left = np.sqrt(ex_j.tenor_years / ex_j.v_star)
        coef_k_left = np.sqrt(ex_k.tenor_years / ex_k.v_star)
        fixed_constraints.append(
            coef_j_left * (B_vp_e_j[0, :] @ p_vars[j])
            >= coef_k_left * (B_vp_e_k[0, :] @ p_vars[j + 1])
        )
        coef_j_right = np.sqrt(ex_j.tenor_years / ex_j.v_star)
        coef_k_right = np.sqrt(ex_k.tenor_years / ex_k.v_star)
        fixed_constraints.append(
            coef_j_right * (B_vp_e_j[1, :] @ p_vars[j])
            <= coef_k_right * (B_vp_e_k[1, :] @ p_vars[j + 1])
        )

    # Lee tail slope bounds (paper "No-Strike-Arbitrage Constraints", p. 4).
    # Right edge: s(z_{n-1}, T) < 2 / sqrt(v* T) => dv/dz(z_{n-1}) < 2 v* / sqrt(v* T)
    # Left edge:  s(z_0,     T) > -2 / sqrt(v* T) => dv/dz(z_0)     > -2 v* / sqrt(v* T)
    for j, ex in enumerate(expiries):
        _, B_vp_e, _ = op_edges[j]
        bound = 2.0 * ex.v_star / np.sqrt(ex.v_star * ex.tenor_years) - LEE_MARGIN
        fixed_constraints.append(B_vp_e[1, :] @ p_vars[j] <= bound)
        fixed_constraints.append(B_vp_e[0, :] @ p_vars[j] >= -bound)

    # ------------------------------------------------------------------
    # Iterative solve
    # ------------------------------------------------------------------
    setup_time = time.perf_counter() - t_setup_start

    objective = cp.Minimize(cp.sum(obj_terms))
    solver_time_total = 0.0
    last_status = ""
    last_params: list[np.ndarray] | None = warm_start_params
    nnz_P, nnz_A, n_constraints_final = 0, 0, 0
    objective_value = float("nan")
    iterations_done = 0

    for it in range(max(1, n_iterations)):
        extra = []
        if it > 0 and last_params is not None:
            # Build linearized interior butterfly constraints from the current
            # iterate (paper eq. (2) + linearization, p. 5).
            for j, ex in enumerate(expiries):
                B_v_d, B_vp_d, B_vpp_d = op_disc[j]
                v_ref = B_v_d @ last_params[j]
                s_ref = (B_vp_d @ last_params[j]) / ex.v_star
                # Points where the current iterate is at or below the positivity
                # floor are dominated by the positivity constraint and would give
                # the butterfly linearization ill-conditioned gradients; skip.
                keep = v_ref > 2.0 * V_REF_FLOOR
                if not keep.any():
                    continue
                z_sub = z_disc[keep]
                f_ref, df_dv, df_ds = _butterfly_rhs_and_grad(
                    z_sub, v_ref[keep], s_ref[keep], ex.v_star, ex.tenor_years
                )
                B_v_s = B_v_d[keep]
                B_vp_s = B_vp_d[keep]
                B_vpp_s = B_vpp_d[keep]
                # c_eval >= f_ref + df_dv (v_eval - v_ref) + df_ds (s_eval - s_ref)
                # with c_eval = (1/v*) B_vpp @ p, v_eval = B_v @ p, s_eval = (1/v*) B_vp @ p.
                c_expr = (1.0 / ex.v_star) * (B_vpp_s @ p_vars[j])
                v_expr = B_v_s @ p_vars[j]
                s_expr = (1.0 / ex.v_star) * (B_vp_s @ p_vars[j])
                rhs = (
                    f_ref
                    + cp.multiply(df_dv, v_expr - v_ref[keep])
                    + cp.multiply(df_ds, s_expr - s_ref[keep])
                )
                extra.append(c_expr >= rhs)

        problem = cp.Problem(objective, fixed_constraints + extra)
        t0 = time.perf_counter()
        try:
            problem.solve(solver=solver, verbose=verbose)
            iter_status = problem.status
        except cp.error.SolverError as exc:
            iter_status = f"solver_error:{exc}"
        solver_time_total += time.perf_counter() - t0
        ok = iter_status in ("optimal", "optimal_inaccurate")
        # A linearized butterfly block can occasionally make the QP numerically
        # infeasible for an iterate it was built from. The paper says "one or
        # a few iterations" suffice (p. 6); when a refinement pass fails we
        # keep the previous iterate as the final one rather than aborting.
        if not ok:
            if it == 0 or last_params is None:
                raise RuntimeError(
                    f"CVXPY/{solver} returned status {iter_status} on the "
                    f"initial pass; no usable iterate available"
                )
            iterations_done = it
            last_status = f"{last_status} (refinement halted: {iter_status})"
            break
        last_status = iter_status
        last_params = [np.asarray(p.value, dtype=float) for p in p_vars]
        objective_value = float(problem.value)
        iterations_done = it + 1

        # Sparsity stats from the last solve.
        try:
            data = problem.get_problem_data(solver)[0]
        except Exception:
            data = {}
        P = data.get("P")
        A = data.get("A")
        if P is not None:
            nnz_P = int(P.nnz)
        if A is not None:
            nnz_A = int(A.nnz)
        n_constraints_final = len(fixed_constraints) + len(extra)

    # Chi-squared per expiry: average squared normalized residual over mids.
    chi2_list: list[float] = []
    for j, ex in enumerate(expiries):
        inside = in_range_idx[j]
        has_mid = ex.has_mid & inside
        if not has_mid.any():
            chi2_list.append(float("nan"))
            continue
        rows = np.flatnonzero(has_mid[inside])
        v_mid = ex.v_mid[inside][has_mid[inside]]
        v_bid = ex.v_bid[inside][has_mid[inside]]
        v_ask = ex.v_ask[inside][has_mid[inside]]
        spread = np.maximum(v_ask - v_bid, 1e-8)
        B = op_quotes[j][rows, :]
        v_fit = B @ last_params[j]
        chi2 = float(np.mean(((v_fit - v_mid) / spread) ** 2))
        chi2_list.append(chi2)

    n_vars_total = sum(v.size for v in p_vars)
    return CalibrationResult(
        knots=basis.knots,
        params=last_params,
        expiries=expiries,
        solver_time_s=solver_time_total,
        setup_time_s=setup_time,
        objective_value=objective_value,
        n_variables=int(n_vars_total),
        n_constraints_final=int(n_constraints_final),
        nnz_P=int(nnz_P),
        nnz_A=int(nnz_A),
        iterations_done=int(iterations_done),
        chi2_per_expiry=chi2_list,
        status=str(last_status),
    )


def _vega(F: float, K: np.ndarray, T: float, sigma: np.ndarray) -> np.ndarray:
    """Black-76 vega without discount factor; see :mod:`.preprocess`."""
    K = np.asarray(K, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        sqrtT = np.sqrt(T)
        d1 = (np.log(F / K) + 0.5 * sigma ** 2 * T) / (sigma * sqrtT)
        phi = np.exp(-0.5 * d1 ** 2) / np.sqrt(2 * np.pi)
        vega = F * sqrtT * phi
    return np.where(np.isfinite(vega) & (vega > 0), vega, 0.0)
