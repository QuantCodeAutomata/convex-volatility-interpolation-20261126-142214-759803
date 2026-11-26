"""Unit tests for src.preprocess."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.preprocess import preprocess_expiry


def _synthetic_smile(F: float = 100.0, T: float = 0.25,
                     atm: float = 0.20, skew: float = -0.05,
                     curv: float = 0.4, spread: float = 0.01) -> pd.DataFrame:
    """A U-shaped smile with a constant bid/ask spread in volatility."""
    K = F * np.exp(np.linspace(-0.4, 0.4, 21))
    k = np.log(K / F)
    sigma = atm + skew * k + curv * k ** 2
    return pd.DataFrame({
        "strike": K,
        "sigma_bid": sigma - 0.5 * spread,
        "sigma_ask": sigma + 0.5 * spread,
        "contract_type": ["call"] * len(K),
    })


def test_preprocess_basic_shapes_and_z() -> None:
    quotes = _synthetic_smile()
    ex = preprocess_expiry(quotes, tenor_years=0.25, forward=100.0)
    assert ex.z.shape == ex.strikes.shape
    # z = k / (sigma_star sqrt(T)); check it is strictly increasing with K.
    assert np.all(np.diff(ex.z) > 0)
    assert ex.v_bid.shape == ex.z.shape
    # Variance equals the squared vol.
    assert np.allclose(ex.v_bid, ex.sigma_bid ** 2)
    assert np.allclose(ex.v_ask, ex.sigma_ask ** 2)
    # ATM anchor is near the input atm vol.
    assert abs(ex.sigma_star - 0.20) < 5e-3


def test_preprocess_drops_rows_with_both_sides_missing() -> None:
    df = _synthetic_smile()
    df.loc[0, ["sigma_bid", "sigma_ask"]] = np.nan
    ex = preprocess_expiry(df, tenor_years=0.25, forward=100.0)
    assert len(ex.strikes) == len(df) - 1


def test_preprocess_keeps_single_sided_quotes() -> None:
    """Rows with only bid or only ask must be kept for the one-sided penalty."""
    df = _synthetic_smile()
    df.loc[0, "sigma_ask"] = np.nan                      # bid-only row
    df.loc[len(df) - 1, "sigma_bid"] = np.nan            # ask-only row
    ex = preprocess_expiry(df, tenor_years=0.25, forward=100.0)
    assert len(ex.strikes) == len(df)
    # The one-sided rows must have has_mid = False but has_bid or has_ask True.
    assert ex.has_bid.sum() == len(df) - 1
    assert ex.has_ask.sum() == len(df) - 1
    assert ex.has_mid.sum() == len(df) - 2


def test_preprocess_tightest_quote_per_strike() -> None:
    """When call and put are both present, the tightest-spread row wins."""
    F, T = 100.0, 0.5
    df = pd.DataFrame({
        "strike": [100.0, 100.0],
        "sigma_bid": [0.18, 0.19],
        "sigma_ask": [0.22, 0.21],       # second row spread = 0.02; first = 0.04
        "contract_type": ["call", "put"],
    })
    ex = preprocess_expiry(df, tenor_years=T, forward=F)
    assert len(ex.strikes) == 1
    # Tight pair is the put.
    assert ex.sigma_bid[0] == pytest.approx(0.19)
    assert ex.sigma_ask[0] == pytest.approx(0.21)


def test_preprocess_vega_weights_are_bell_shaped() -> None:
    ex = preprocess_expiry(_synthetic_smile(), tenor_years=0.25, forward=100.0)
    vega = ex.vega_ask
    assert np.all(vega >= 0)
    # Peak is near z = 0.
    i_peak = int(np.argmax(vega))
    assert abs(ex.z[i_peak]) < 1.0
