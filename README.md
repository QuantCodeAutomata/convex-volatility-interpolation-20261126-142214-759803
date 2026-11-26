# Convex Volatility Interpolation

Replication of the methodology of

> Deschâtres, F. *Convex Volatility Interpolation*.

The paper fits a listed-option implied-volatility surface, expiry by expiry and
across expiries jointly, by solving a convex quadratic program with linear
positivity, calendar-arbitrage and strike-arbitrage constraints, and iteratively
linearized butterfly-arbitrage constraints.

## Repository layout

```
src/            CVI library modules (flat): spline, preprocess, calibrate, config
exp/            Experiment scripts (run from the repository root with `python -m exp.<name>`)
tests/          pytest suite (targets src/ components)
data/           data_loader.py: pulls option quotes from Massive REST
results/        Aggregate metrics (CSV/JSON) and static .png figures; RESULTS.md
reproduce.sh    Runs the tests and both experiments with the commands below
```

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env         # fill MASSIVE_API_KEY (listed-option entitlement)
```

The replication fetches real listed-option quotes through Massive REST
(`https://api.massive.com`); the API key is read from the environment and never
appears in the repository.

## Reproduce

```bash
# Tests
python -m pytest -m "not market_data" -q           # offline tests (no API)
python -m pytest -q                                 # all tests (needs MASSIVE_API_KEY)

# Experiments
python -m exp.exp1_calibrate                        # Experiment 1
python -m exp.exp2_scaling                          # Experiment 2
python -m exp.exp2_scaling --full                   # paper-size knot grid
```

After a rerun, the experiment commands overwrite the files in `results/`. The
numbers and verdicts in `results/RESULTS.md` are checked against the new files
and updated by hand.

## Experiments

* **Experiment 1 – CVI surface calibration**: implements the quadratic
  program and iterative linearization of butterfly-arbitrage constraints
  described in "Analysis of the Problem" and "Putting It All Together"
  (pp. 1–6). Writes the fitted variance surface and a per-expiry fit summary
  to `results/exp1_*`.
* **Experiment 2 – Clarabel scaling**: reproduces the knot-count scaling and
  runtime study of "Numerical Results" (p. 6, Tables 2–3). Writes
  `results/exp2_scaling.csv` and `results/exp2_*.png`.

See `results/RESULTS.md` for the comparison with the paper's numbers,
deviations, and limitations.
