"""Market preprocessing for the CVI calibration.

Converts a flat quote table (one row per contract) into per-expiry inputs:
forward ``F(T)``, anchor volatility ``sigma_star(T)``, log-moneyness ``k``,
normalized log-moneyness ``z``, and bid/mid/ask variances with vega weights.

The paper ("Analysis of the Problem", p. 1) uses variance
``v = sigma^2`` as the fitted quantity and the normalized log-moneyness
``z = k / (sigma_star sqrt(T))``. Puts and calls are not differentiated: the
tightest bid/ask pair in volatility terms is kept for each strike.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class ExpiryData:
    """Calibration inputs for one expiry."""

    tenor_years: float              # T in years
    forward: float                  # F(T)
    sigma_star: float               # sigma_*(T), anchor ATM volatility
    strikes: np.ndarray             # K_i
    k: np.ndarray                   # log(K/F)
    z: np.ndarray                   # k / (sigma_star sqrt(T))
    sigma_bid: np.ndarray           # bid implied vol (NaN when missing)
    sigma_ask: np.ndarray           # ask implied vol (NaN when missing)
    sigma_mid: np.ndarray           # mid implied vol (NaN when either side missing)
    v_bid: np.ndarray               # sigma_bid^2
    v_ask: np.ndarray               # sigma_ask^2
    v_mid: np.ndarray               # sigma_mid^2
    vega_ask: np.ndarray            # Black vega at ask IV
    has_bid: np.ndarray             # bool
    has_ask: np.ndarray             # bool
    has_mid: np.ndarray             # has_bid & has_ask
    meta: dict = field(default_factory=dict)

    @property
    def v_star(self) -> float:
        return self.sigma_star ** 2


def _black_vega(F: float, K: np.ndarray, T: float, sigma: np.ndarray, r: float = 0.0) -> np.ndarray:
    """Undiscounted Black-76 vega, per unit volatility (not per percent).

    Returned value is ``F * sqrt(T) * phi(d1)`` without a discount factor; the
    paper's weighting uses vega as a relative weight so the discount factor
    cancels across contracts of the same expiry.
    """
    K = np.asarray(K, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        sqrtT = np.sqrt(T)
        d1 = (np.log(F / K) + 0.5 * sigma ** 2 * T) / (sigma * sqrtT)
        phi = np.exp(-0.5 * d1 ** 2) / np.sqrt(2 * np.pi)
        vega = F * sqrtT * phi
    vega = np.where(np.isfinite(vega) & (vega > 0), vega, 0.0)
    return vega


def _tightest_per_strike(df: pd.DataFrame) -> pd.DataFrame:
    """For each strike keep the call/put row with the tightest IV spread.

    Expects columns ``strike, sigma_bid, sigma_ask, contract_type``. Rows where
    either side is missing are kept only when no complete pair exists for that
    strike (the paper applies the above-ask / below-bid penalty to strikes with
    a bid or an ask but not both, p. 3).
    """
    df = df.copy()
    df["spread"] = df["sigma_ask"] - df["sigma_bid"]
    df["spread_sort"] = df["spread"].where(df["spread"].notna() & (df["spread"] >= 0), np.inf)
    out = (
        df.sort_values(["strike", "spread_sort"])
          .drop_duplicates(subset="strike", keep="first")
          .drop(columns=["spread", "spread_sort"])
          .reset_index(drop=True)
    )
    return out


def preprocess_expiry(
    quotes: pd.DataFrame,
    *,
    tenor_years: float,
    forward: float,
    sigma_star: float | None = None,
) -> ExpiryData:
    """Build the calibration inputs for one expiry.

    Parameters
    ----------
    quotes : pd.DataFrame
        One row per listed contract with columns
        ``strike, sigma_bid, sigma_ask, contract_type``. Missing legs may be
        NaN; both missing drops the row.
    tenor_years : float
        Time to expiry in years.
    forward : float
        Forward price ``F(T)``.
    sigma_star : float, optional
        Anchor ATM volatility. If ``None`` it is estimated by interpolating
        the mid implied-volatility smile at ``k = 0`` (the ATM log-moneyness).
    """
    if tenor_years <= 0:
        raise ValueError("tenor_years must be positive")
    if forward <= 0:
        raise ValueError("forward must be positive")

    quotes = quotes.dropna(subset=["strike"]).copy()
    quotes = quotes[(quotes["sigma_bid"].notna()) | (quotes["sigma_ask"].notna())]
    if quotes.empty:
        raise ValueError("no usable quotes for this expiry")
    quotes = _tightest_per_strike(quotes)
    quotes = quotes.sort_values("strike").reset_index(drop=True)

    strikes = quotes["strike"].to_numpy(dtype=float)
    sigma_bid = quotes["sigma_bid"].to_numpy(dtype=float)
    sigma_ask = quotes["sigma_ask"].to_numpy(dtype=float)
    k = np.log(strikes / forward)

    has_bid = np.isfinite(sigma_bid) & (sigma_bid > 0)
    has_ask = np.isfinite(sigma_ask) & (sigma_ask > 0)
    has_mid = has_bid & has_ask
    sigma_mid = np.where(has_mid, 0.5 * (sigma_bid + sigma_ask), np.nan)

    if sigma_star is None:
        mid_mask = has_mid
        if not mid_mask.any():
            # Fallback: use whichever side exists at the strike closest to ATM.
            fallback = np.where(has_ask, sigma_ask, sigma_bid)
            i = int(np.argmin(np.abs(k)))
            sigma_star = float(fallback[i])
        else:
            k_mid = k[mid_mask]
            s_mid = sigma_mid[mid_mask]
            # Linear interpolation in k at k = 0; clamp to the convex hull.
            if k_mid.min() <= 0.0 <= k_mid.max():
                sigma_star = float(np.interp(0.0, k_mid, s_mid))
            else:
                i = int(np.argmin(np.abs(k_mid)))
                sigma_star = float(s_mid[i])

    if not np.isfinite(sigma_star) or sigma_star <= 0:
        raise ValueError("could not derive a positive anchor volatility")

    z = k / (sigma_star * np.sqrt(tenor_years))

    v_bid = np.where(has_bid, sigma_bid ** 2, np.nan)
    v_ask = np.where(has_ask, sigma_ask ** 2, np.nan)
    v_mid = np.where(has_mid, sigma_mid ** 2, np.nan)

    # Vega weights for the above-ask penalty use Black vega at the ask IV
    # (paper "Above Ask Penalty", p. 3).
    vega_sigma = np.where(has_ask, sigma_ask, sigma_star)
    vega_ask = _black_vega(forward, strikes, tenor_years, vega_sigma)
    vega_ask = np.where(has_ask, vega_ask, 0.0)

    return ExpiryData(
        tenor_years=float(tenor_years),
        forward=float(forward),
        sigma_star=float(sigma_star),
        strikes=strikes,
        k=k,
        z=z,
        sigma_bid=sigma_bid,
        sigma_ask=sigma_ask,
        sigma_mid=sigma_mid,
        v_bid=v_bid,
        v_ask=v_ask,
        v_mid=v_mid,
        vega_ask=vega_ask,
        has_bid=has_bid,
        has_ask=has_ask,
        has_mid=has_mid,
    )
