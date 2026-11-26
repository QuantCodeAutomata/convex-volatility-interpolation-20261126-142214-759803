#!/usr/bin/env bash
# Reproduce the replication from a clean checkout.
#
# Expects Python 3.11+ and MASSIVE_API_KEY in the environment (or in a local
# .env file, see .env.example).
set -euo pipefail

cd "$(dirname "$0")"

python -m pip install -r requirements.txt

# Offline tests always run. The market_data tests need MASSIVE_API_KEY.
python -m pytest -m "not market_data" -q
if [[ -n "${MASSIVE_API_KEY:-}" ]]; then
    python -m pytest -q
fi

python -m exp.exp1_calibrate
python -m exp.exp2_scaling
