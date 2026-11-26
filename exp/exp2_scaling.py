"""Experiment 2: Clarabel calibration scaling study.

Reproduces the benchmarks of "Convex Volatility Interpolation"
(Deschâtres, 2026, Section "Numerical Results" and "Scaling", pp. 6-7).

For each underlying and each knot count ``n`` in ``BENCH_KNOT_COUNTS``:
    * calibrate the full chain in two Clarabel iterations
      (the first without the linearized butterfly block, the second with
      it, as in "Putting It All Together", p. 6);
    * record number of variables, constraints, non-zero coefficients in
      the sparse ``P`` and ``A`` matrices, Clarabel solver time, CVXPY
      setup time, and the chi-squared quality metric.

A second sweep over ``BENCH_KNOT_COUNTS_FIG6`` traces the
chi-squared-versus-knot-count curve (Figure 6 of the paper), to see the
expected improvement-then-plateau behaviour.

Outputs, written to ``results/``:
    * ``exp2_scaling.csv``   -- the Table-3-style timing table.
    * ``exp2_fig6.csv``      -- chi-squared and timings over the wider
      knot range used in the paper's Figure 6.
    * ``exp2_runtime_vs_knots.png`` / ``exp2_chi2_vs_knots.png`` /
      ``exp2_sparsity_vs_knots.png`` -- the three scaling plots.
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
from src.calibrate import calibrate
from src.config import (
    BENCH_KNOT_COUNTS,
    BENCH_KNOT_COUNTS_FIG6,
    N_DISCRETIZATION,
    N_ITERATIONS,
    UNDERLYINGS,
    make_knot_grid,
)
from src.preprocess import ExpiryData, preprocess_expiry

RESULTS = Path(__file__).resolve().parent.parent / "results"


def build_expiries(snapshot) -> list[ExpiryData]:
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


def run_sweep(underlying: str, knot_counts, n_discretization: int,
              expiries: list[ExpiryData]) -> list[dict]:
    """Calibrate at each knot count and return the full stats row."""
    n_exp = len(expiries)
    n_quotes = sum(int(ex.has_bid.sum() + ex.has_ask.sum()) for ex in expiries)
    out = []
    for n in knot_counts:
        knots = make_knot_grid(int(n))
        res = calibrate(expiries, knots,
                        n_iterations=N_ITERATIONS,
                        n_discretization=n_discretization)
        out.append({
            "underlying": underlying,
            "n_expiries": n_exp,
            "n_vol_quotes": n_quotes,
            "n_knots_per_expiry": int(n),
            "n_variables": int(res.n_variables),
            "n_constraints_final": int(res.n_constraints_final),
            "nnz_P": int(res.nnz_P),
            "nnz_A": int(res.nnz_A),
            "setup_time_s": res.setup_time_s,
            "solver_time_s": res.solver_time_s,
            "avg_chi2": res.avg_chi2,
            "iterations": int(res.iterations_done),
            "status": res.status,
        })
        print(f"  {underlying}  n_knots={n:>3}  "
              f"solve={res.solver_time_s:.3f}s  "
              f"vars={res.n_variables}  cons={res.n_constraints_final}  "
              f"nnzP={res.nnz_P}  nnzA={res.nnz_A}  "
              f"chi2={res.avg_chi2:.4f}")
    return out


def plot_runtime(df: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    for u, grp in df.groupby("underlying"):
        g = grp.sort_values("n_knots_per_expiry")
        ax.plot(g["n_knots_per_expiry"], g["solver_time_s"],
                "-o", label=f"{u} solver")
        ax.plot(g["n_knots_per_expiry"], g["setup_time_s"],
                "--s", label=f"{u} setup", alpha=0.6)
    ax.set_xlabel("knots per expiry")
    ax.set_ylabel("time (s)")
    ax.set_title("CVI calibration time vs. number of knots (Clarabel)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_chi2(df: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    for u, grp in df.groupby("underlying"):
        g = grp.sort_values("n_knots_per_expiry")
        ax.plot(g["n_knots_per_expiry"], g["avg_chi2"], "-o", label=u)
    ax.set_xlabel("knots per expiry")
    ax.set_ylabel(r"average $\chi^2$ across expiries")
    ax.set_title("CVI fit quality vs. number of knots")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_sparsity(df: pd.DataFrame, path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharex=True)
    for u, grp in df.groupby("underlying"):
        g = grp.sort_values("n_knots_per_expiry")
        axes[0].plot(g["n_knots_per_expiry"], g["nnz_P"], "-o", label=u)
        axes[1].plot(g["n_knots_per_expiry"], g["nnz_A"], "-o", label=u)
    for ax, title in zip(axes, ["nnz(P)", "nnz(A)"]):
        ax.set_xlabel("knots per expiry")
        ax.set_ylabel(title)
        ax.set_title(title + " vs. knots")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=9)
    fig.suptitle("Sparsity of the CVI QP matrices")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--underlyings", nargs="+", default=list(UNDERLYINGS))
    parser.add_argument("--knot-counts", type=int, nargs="+",
                        default=list(BENCH_KNOT_COUNTS))
    parser.add_argument("--fig6-knot-counts", type=int, nargs="+",
                        default=list(BENCH_KNOT_COUNTS_FIG6))
    parser.add_argument("--n-discretization", type=int, default=N_DISCRETIZATION)
    parser.add_argument("--skip-fig6", action="store_true",
                        help="Skip the wider sweep used for Figure 6.")
    args = parser.parse_args()
    RESULTS.mkdir(exist_ok=True)

    scaling_rows: list[dict] = []
    fig6_rows: list[dict] = []
    for u in args.underlyings:
        print(f"[{u}] loading chain...")
        snap = load_chain(u, use_cache=True)
        expiries = build_expiries(snap)
        if not expiries:
            print(f"[{u}] no usable expiry, skipping")
            continue
        print(f"[{u}] {len(expiries)} expiries, "
              f"{sum(int(ex.has_bid.sum()+ex.has_ask.sum()) for ex in expiries)} quotes")

        print(f"[{u}] Table-3 sweep over knot counts {args.knot_counts}")
        scaling_rows.extend(run_sweep(u, args.knot_counts,
                                      args.n_discretization, expiries))

        if not args.skip_fig6:
            print(f"[{u}] Figure-6 sweep over knot counts {args.fig6_knot_counts}")
            fig6_rows.extend(run_sweep(u, args.fig6_knot_counts,
                                       args.n_discretization, expiries))

    df_scaling = pd.DataFrame(scaling_rows)
    df_scaling.to_csv(RESULTS / "exp2_scaling.csv", index=False)
    print("\nTable 3 (CVI timing table):")
    print(df_scaling.to_string(index=False))

    if fig6_rows:
        df_fig6 = pd.DataFrame(fig6_rows)
        df_fig6.to_csv(RESULTS / "exp2_fig6.csv", index=False)
        plot_runtime(df_fig6, RESULTS / "exp2_runtime_vs_knots.png")
        plot_chi2(df_fig6, RESULTS / "exp2_chi2_vs_knots.png")
        plot_sparsity(df_fig6, RESULTS / "exp2_sparsity_vs_knots.png")
    else:
        # Still plot from the scaling sweep so the figures exist.
        plot_runtime(df_scaling, RESULTS / "exp2_runtime_vs_knots.png")
        plot_chi2(df_scaling, RESULTS / "exp2_chi2_vs_knots.png")
        plot_sparsity(df_scaling, RESULTS / "exp2_sparsity_vs_knots.png")


if __name__ == "__main__":
    main()
