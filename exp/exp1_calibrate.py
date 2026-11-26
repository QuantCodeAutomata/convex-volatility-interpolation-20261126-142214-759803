"""Experiment 1: fit the CVI surface on real option-chain data.

Reproduces the calibration pipeline of "Convex Volatility Interpolation"
(Deschâtres, 2026, Sections "Analysis of the Problem" to "Putting It All
Together", pp. 1-6): assemble the four-term QP objective, the full set of
no-arbitrage constraints, solve twice with Clarabel (the second iteration
adds the linearized interior butterfly block), and verify the resulting
surface on the paper's discretization grids.

Outputs, written to ``results/``:
    * ``exp1_summary.csv``   -- per-underlying summary metrics.
    * ``exp1_per_expiry_<U>.csv`` -- per-expiry fit statistics.
    * ``exp1_arbitrage_<U>.csv``  -- min values of positivity / calendar /
      Lee / butterfly residuals on the discretization grids (positive means
      the constraint holds).
    * ``exp1_smile_<U>.png``  -- fitted smile overlaid on mid IVs for a
      small selection of expiries.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from data.data_loader import load_chain
from src.calibrate import _butterfly_rhs_and_grad, calibrate
from src.config import (
    LAMBDA_TV,
    N_DISCRETIZATION,
    N_DISCRETIZATION_EXP1,
    N_ITERATIONS,
    N_KNOTS_DEFAULT,
    UNDERLYINGS,
    V_REF_FLOOR,
    make_knot_grid,
)
from src.preprocess import ExpiryData, preprocess_expiry
from src.spline import CVISplineBasis

RESULTS = Path(__file__).resolve().parent.parent / "results"


def build_expiries(snapshot) -> list[ExpiryData]:
    """Run :func:`preprocess_expiry` for every expiry in ``snapshot``."""
    out: list[ExpiryData] = []
    for exp_date in sorted(snapshot.quotes_by_expiry):
        T = snapshot.tenors_years[exp_date]
        F = snapshot.forwards[exp_date]
        try:
            ex = preprocess_expiry(snapshot.quotes_by_expiry[exp_date],
                                   tenor_years=T, forward=F)
        except ValueError:
            continue
        out.append(ex)
    return out


def arbitrage_report(result, n_disc: int = N_DISCRETIZATION) -> dict[str, float]:
    """Verify the paper's static-arbitrage constraints on the fitted surface.

    Returns the worst-case (most negative) slack across the discretization
    grids for each constraint family (paper, Sections "No-Calendar-Spread-
    Arbitrage Constraints" and "No-Butterfly-Arbitrage Constraints", p. 4).
    A non-negative value means the constraint holds.
    """
    basis = CVISplineBasis(result.knots)
    z_disc = np.linspace(float(basis.knots[0]), float(basis.knots[-1]), n_disc)
    J = len(result.expiries)

    pos_v_min = []
    cal_slack_min = []
    lee_slack_min = []
    bf_slack_min = []
    for j, ex in enumerate(result.expiries):
        p = result.params[j]
        v, s_raw, c_raw = basis.evaluate(p, z_disc)
        pos_v_min.append(float(np.min(v)))

        # Lee tail bounds: |s(z)| * sqrt(v* T) < 2 at edges (paper p. 4).
        s = s_raw / ex.v_star
        lee_left = 2.0 - (-s[0]) * np.sqrt(ex.v_star * ex.tenor_years)
        lee_right = 2.0 - s[-1] * np.sqrt(ex.v_star * ex.tenor_years)
        lee_slack_min.append(float(min(lee_left, lee_right)))

        # Butterfly inequality (paper eq. (2), p. 5):
        #   c >= RHS(v, s). We evaluate the exact nonlinear RHS, not the
        #   linearized one, so the check is tighter than what the QP sees.
        v_safe = np.maximum(v, V_REF_FLOOR)
        c = c_raw / ex.v_star
        f_rhs, _, _ = _butterfly_rhs_and_grad(z_disc, v_safe, s,
                                              ex.v_star, ex.tenor_years)
        bf_slack_min.append(float(np.min(c - f_rhs)))

        if j + 1 < J:
            ex_k = result.expiries[j + 1]
            v_k, _, _ = basis.evaluate(result.params[j + 1], z_disc)
            cal = ex_k.tenor_years * v_k - ex.tenor_years * v
            cal_slack_min.append(float(np.min(cal)))

    return {
        "min_variance": float(np.min(pos_v_min)),
        "min_calendar_slack": float(np.min(cal_slack_min)) if cal_slack_min else float("nan"),
        "min_lee_slack": float(np.min(lee_slack_min)),
        "min_butterfly_slack": float(np.min(bf_slack_min)),
    }


def per_expiry_table(result) -> pd.DataFrame:
    """One-row-per-expiry quality table."""
    rows = []
    for j, ex in enumerate(result.expiries):
        rows.append({
            "expiry_index": j,
            "tenor_years": ex.tenor_years,
            "forward": ex.forward,
            "sigma_star": ex.sigma_star,
            "n_quotes_mid": int(ex.has_mid.sum()),
            "n_quotes_ask": int(ex.has_ask.sum()),
            "n_quotes_bid": int(ex.has_bid.sum()),
            "chi2": result.chi2_per_expiry[j],
        })
    return pd.DataFrame(rows)


def plot_smiles(result, underlying: str, out_path: Path) -> None:
    """Overlay fitted smile and mid IVs for a small set of expiries."""
    basis = CVISplineBasis(result.knots)
    J = len(result.expiries)
    pick = np.unique(np.linspace(0, J - 1, min(J, 6)).astype(int))
    fig, axes = plt.subplots(2, 3, figsize=(13, 7), sharey=False)
    axes = axes.ravel()
    for ax in axes[len(pick):]:
        ax.axis("off")
    for k, j in enumerate(pick):
        ax = axes[k]
        ex = result.expiries[j]
        z_grid = np.linspace(float(basis.knots[0]),
                             float(basis.knots[-1]), 400)
        v, _, _ = basis.evaluate(result.params[j], z_grid)
        sigma_fit = np.sqrt(np.maximum(v, 0.0))
        has_mid = ex.has_mid
        ax.scatter(ex.z[has_mid], ex.sigma_mid[has_mid],
                   s=12, c="tab:blue", label="mid IV", alpha=0.7)
        ax.plot(z_grid, sigma_fit, c="tab:red", lw=1.2, label="CVI fit")
        ax.set_title(f"T={ex.tenor_years:.3f}y  F={ex.forward:.2f}")
        ax.set_xlabel("normalized log-moneyness z")
        ax.set_ylabel("implied volatility")
        ax.grid(alpha=0.3)
        if k == 0:
            ax.legend(loc="best", fontsize=8)
    fig.suptitle(
        f"CVI fitted smile vs market mid IV — {underlying}  "
        f"(n_knots={len(result.knots)}, iterations={N_ITERATIONS})"
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def run_one(underlying: str, n_knots: int, n_discretization: int) -> dict:
    snap = load_chain(underlying, use_cache=True)
    expiries = build_expiries(snap)
    if not expiries:
        raise RuntimeError(f"{underlying}: no usable expiry after preprocessing")
    knots = make_knot_grid(n_knots)
    result = calibrate(expiries, knots,
                       lambda_tv=LAMBDA_TV,
                       n_iterations=N_ITERATIONS,
                       n_discretization=n_discretization)

    # Per-expiry table.
    df_exp = per_expiry_table(result)
    df_exp.to_csv(RESULTS / f"exp1_per_expiry_{underlying}.csv", index=False)

    # Arbitrage verification.
    arb = arbitrage_report(result, n_disc=max(60, n_discretization))
    pd.DataFrame([arb]).to_csv(
        RESULTS / f"exp1_arbitrage_{underlying}.csv", index=False)

    # Smile plot.
    plot_smiles(result, underlying, RESULTS / f"exp1_smile_{underlying}.png")

    return {
        "underlying": underlying,
        "n_expiries": len(expiries),
        "n_quotes": snap.n_quotes,
        "n_knots": int(n_knots),
        "n_variables": int(result.n_variables),
        "n_constraints_final": int(result.n_constraints_final),
        "nnz_P": int(result.nnz_P),
        "nnz_A": int(result.nnz_A),
        "iterations": int(result.iterations_done),
        "setup_time_s": result.setup_time_s,
        "solver_time_s": result.solver_time_s,
        "objective_value": result.objective_value,
        "avg_chi2": result.avg_chi2,
        "status": result.status,
        **{f"arb_{k}": v for k, v in arb.items()},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--underlyings", nargs="+", default=list(UNDERLYINGS))
    parser.add_argument("--n-knots", type=int, default=N_KNOTS_DEFAULT)
    parser.add_argument("--n-discretization", type=int,
                        default=N_DISCRETIZATION_EXP1)
    args = parser.parse_args()
    RESULTS.mkdir(exist_ok=True)

    rows = []
    for u in args.underlyings:
        try:
            rows.append(run_one(u, args.n_knots, args.n_discretization))
            print(f"[{u}] OK avg_chi2={rows[-1]['avg_chi2']:.4f} "
                  f"solve={rows[-1]['solver_time_s']:.3f}s")
        except Exception as exc:                                    # noqa: BLE001
            rows.append({"underlying": u, "status": f"error: {exc}"})
            print(f"[{u}] ERROR: {exc}")
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS / "exp1_summary.csv", index=False)
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
