# Repository guide for contributors

This repository replicates *Convex Volatility Interpolation* (Deschâtres, 2026).
It fits a listed-option implied-volatility surface as a convex quadratic program
in variance space with linear positivity / calendar / tail constraints and
iteratively linearized butterfly-arbitrage constraints.

## Layout

```
src/            CVI library modules (flat, no subdirectories)
  spline.py       cubic-spline basis and linear operators (B_v, B_s, B_c)
  preprocess.py   flat quote table -> per-expiry inputs (F, sigma*, z, v_*, vega)
  calibrate.py   QP assembly + Clarabel solve + iterative linearization
  config.py       all tunable parameters as UPPER_CASE module-level variables
exp/            experiment scripts (run from the repo root as `python -m exp.<name>`)
tests/          pytest suite (unit tests on src/ + a market-data fixture)
data/           data_loader.py: Massive REST options chain + NBBO quote fetcher
results/        aggregate CSV/JSON metrics and static .png figures
                plus RESULTS.md (paper comparison, deviations, not-reproduced)
reproduce.sh    runs the tests and both experiments with the README commands
```

`src/`, `exp/` and `data/` each contain an empty `__init__.py`. There are no
subdirectories under them.

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env            # fill MASSIVE_API_KEY if real data is used
```

## Commands

```bash
# Tests
python -m pytest -m "not market_data" -q   # offline (no API key needed)
python -m pytest -q                         # all tests (needs MASSIVE_API_KEY)

# Experiments
python -m exp.exp1_calibrate                # Experiment 1: calibrate surface
python -m exp.exp2_scaling                  # Experiment 2: knot/runtime scaling
python -m exp.exp2_scaling --full           # scaling with the paper knot grid
```

`reproduce.sh` chains the test + experiment commands above.

## Conventions

* **Parameters** live only in `src/config.py` as module-level UPPER_CASE
  variables; each variable has a comment with its meaning, unit and the paper
  section or equation that justifies it (or "not specified in the paper").
* **Data** is loaded only through `data/data_loader.py`. Downloaded data is
  never committed. The results directory contains only aggregate summary tables
  (CSV/JSON indexed by model, knot count or expiry — never by date or bar) and
  static `.png` / `.pdf` figures. No `.html` tearsheets, no time series.
* **Tests** marked `@pytest.mark.market_data` require `MASSIVE_API_KEY` and are
  skipped otherwise. They load a reduced slice through `data/data_loader.py`
  and check paper-stated properties (signs, directions, calendar ordering,
  butterfly inequality, calibration time trend), not reported numbers.
* **API keys** come from the environment; `.env` is git-ignored and
  `.env.example` holds only empty placeholders.
* After a rerun, the experiment commands overwrite the files in `results/`.
  The numbers and verdicts in `results/RESULTS.md` must be checked against the
  new files and updated by hand.

## Paper reference

Deschâtres, F. *Convex Volatility Interpolation*. The paper is the single source
of truth for every formula, parameter, constraint and benchmark table. See
`results/RESULTS.md` for the per-experiment comparison, deviations and the list
of items not reproduced.
