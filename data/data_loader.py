"""Live option-chain loader used by the CVI experiments.

The loader builds a snapshot of listed option quotes across all expiries of
an underlying from the Massive REST option snapshot endpoints:

* ``GET /v3/snapshot/options/{underlying}`` paginates through the full chain
  and gives contract reference (strike, expiration, contract type).
* ``GET /v3/snapshot?ticker.any_of=<up to ~50 contracts>`` returns mid
  ``implied_volatility`` and ``greeks.vega`` for a batch of contracts.

Market-maker NBBO quotes are not available on the active data plan: the
``last_quote`` field of the chain snapshot is empty, and
``/v3/quotes/{option_ticker}`` either requires an upgraded entitlement or
returns zero-wide stub rows. The CVI objective requires a per-quote
``(sigma_bid, sigma_ask)`` interval, so a symmetric vol-point half-spread is
added to the mid implied volatility to synthesize ``sigma_bid`` and
``sigma_ask``. The half-spread is widened with |z| to reflect that wing
quotes are typically wider than ATM quotes. This substitution is noted in
``results/RESULTS.md`` under "Deviations from the paper".
"""

from __future__ import annotations

import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from dotenv import load_dotenv

from src.config import (
    DIVIDEND_YIELD_DEFAULT,
    MASSIVE_BASE,
    MAX_CONTRACTS_PER_EXPIRY,
    MAX_EXPIRIES_PER_UNDERLYING,
    MONEYNESS_K_HALF_WIDTH_REF,
    RISK_FREE_RATE_DEFAULT,
    SPREAD_HALF_WIDTH_ATM,
    SPREAD_HALF_WIDTH_WING_SLOPE,
    TENOR_MAX_DAYS,
    TENOR_MIN_DAYS,
)

CACHE_DIR = Path(os.environ.get("CVI_CACHE_DIR", "/tmp/cvi_cache"))
BATCH_SIZE = 50
BATCH_PARALLEL = 4


class MassiveError(RuntimeError):
    """Raised when a Massive REST call fails or returns an unusable payload."""


def _headers() -> dict[str, str]:
    load_dotenv()
    key = os.environ.get("MASSIVE_API_KEY")
    if not key:
        raise MassiveError("MASSIVE_API_KEY is not set")
    return {"Authorization": f"Bearer {key}"}


def _get(url: str, params: dict | None = None, *, timeout: float = 60.0) -> dict:
    for attempt in range(3):
        resp = requests.get(url, params=params, headers=_headers(), timeout=timeout)
        if resp.status_code == 200:
            return resp.json()
        if resp.status_code in (429, 500, 502, 503, 504) and attempt < 2:
            time.sleep(1.5 * (attempt + 1))
            continue
        raise MassiveError(f"{url}: HTTP {resp.status_code} {resp.text[:200]}")
    raise MassiveError(f"{url}: exhausted retries")


def _paginate(url: str, params: dict, *, max_pages: int = 200):
    page = 0
    cur_url, cur_params = url, dict(params)
    while cur_url and page < max_pages:
        data = _get(cur_url, params=cur_params)
        for row in data.get("results") or []:
            yield row
        nxt = data.get("next_url")
        if not nxt:
            return
        # The next_url is a complete, signed URL; subsequent calls must not
        # resend query parameters, but the Authorization header is still needed.
        cur_url = nxt
        cur_params = None
        page += 1


# ---------------------------------------------------------------------------
# Chain reference and per-contract IV/vega
# ---------------------------------------------------------------------------


def _fetch_chain_reference(underlying: str) -> pd.DataFrame:
    """Return reference data for every contract in the chain.

    Columns: ``ticker, expiration, strike, contract_type, underlying_price``.
    Only the contract reference fields are read because the chain snapshot
    does not populate ``last_quote`` or ``greeks`` on the active plan.
    """
    url = f"{MASSIVE_BASE}/v3/snapshot/options/{underlying}"
    records: list[dict] = []
    underlying_price: float | None = None
    for row in _paginate(url, {"limit": 250}):
        details = row.get("details") or {}
        exp = details.get("expiration_date")
        strike = details.get("strike_price")
        ctype = details.get("contract_type")
        ticker = details.get("ticker")
        if exp is None or strike is None or ctype not in ("call", "put") or ticker is None:
            continue
        under = row.get("underlying_asset") or {}
        up = under.get("price")
        if up is not None and underlying_price is None:
            try:
                underlying_price = float(up)
            except (TypeError, ValueError):
                underlying_price = None
        records.append({
            "ticker": ticker,
            "expiration": pd.Timestamp(exp),
            "strike": float(strike),
            "contract_type": ctype,
        })
    if not records:
        raise MassiveError(f"{underlying}: empty chain reference")
    df = pd.DataFrame.from_records(records)
    df["underlying_price"] = float(underlying_price) if underlying_price else np.nan
    return df


def _fetch_iv_batch(tickers: list[str]) -> list[dict]:
    """Return mid IV and greeks for a batch of up to ``BATCH_SIZE`` contracts."""
    if not tickers:
        return []
    url = f"{MASSIVE_BASE}/v3/snapshot"
    data = _get(url, params={"ticker.any_of": ",".join(tickers), "limit": len(tickers)})
    return data.get("results") or []


def _fetch_iv_parallel(tickers: list[str]) -> dict[str, dict]:
    """Return ``{ticker: {iv, vega, delta, underlying_price}}`` for every ticker.

    Issues ``ceil(len(tickers) / BATCH_SIZE)`` requests with up to
    ``BATCH_PARALLEL`` workers in flight.
    """
    out: dict[str, dict] = {}
    if not tickers:
        return out
    batches = [tickers[i:i + BATCH_SIZE] for i in range(0, len(tickers), BATCH_SIZE)]
    with ThreadPoolExecutor(max_workers=BATCH_PARALLEL) as pool:
        results = list(pool.map(_fetch_iv_batch, batches))
    for rows in results:
        for row in rows:
            details = row.get("details") or {}
            greeks = row.get("greeks") or {}
            under = row.get("underlying_asset") or {}
            ticker = row.get("ticker") or details.get("ticker")
            if ticker is None:
                continue
            iv = row.get("implied_volatility")
            vega = greeks.get("vega")
            delta = greeks.get("delta")
            up = under.get("price")
            out[ticker] = {
                "iv": float(iv) if iv is not None else float("nan"),
                "vega": float(vega) if vega is not None else float("nan"),
                "delta": float(delta) if delta is not None else float("nan"),
                "underlying_price": float(up) if up is not None else float("nan"),
            }
    return out


# ---------------------------------------------------------------------------
# Expiry assembly
# ---------------------------------------------------------------------------


@dataclass
class ChainSnapshot:
    """Parsed option chain for one underlying as of the retrieval moment."""

    underlying: str
    as_of_date: pd.Timestamp
    spot: float
    dividend_yield: float
    risk_free_rate: float
    quotes_by_expiry: dict[pd.Timestamp, pd.DataFrame]
    forwards: dict[pd.Timestamp, float]
    tenors_years: dict[pd.Timestamp, float]

    @property
    def n_quotes(self) -> int:
        return int(sum(
            (grp["sigma_bid"].notna() | grp["sigma_ask"].notna()).sum()
            for grp in self.quotes_by_expiry.values()
        ))

    @property
    def n_expiries(self) -> int:
        return len(self.quotes_by_expiry)


def _implied_forward(group: pd.DataFrame, spot: float, tenor_years: float,
                     r: float, q: float) -> float:
    """Return ``F(T)`` from put-call parity on the row closest to ATM.

    Uses mid implied volatilities to price both legs through Black's formula
    and solves for the forward. Falls back to ``spot * exp((r - q) T)`` if
    the put-call pair is missing or produces an implausible value.
    """
    fallback = spot * np.exp((r - q) * tenor_years)
    calls = group[group["contract_type"] == "call"].set_index("strike")
    puts = group[group["contract_type"] == "put"].set_index("strike")
    common = [k for k in calls.index if k in puts.index]
    if not common:
        return fallback
    # Pick strike closest to spot where both IVs are present.
    k_arr = np.asarray(common, dtype=float)
    iv_c = calls.loc[k_arr, "iv"].to_numpy(dtype=float)
    iv_p = puts.loc[k_arr, "iv"].to_numpy(dtype=float)
    mask = np.isfinite(iv_c) & np.isfinite(iv_p) & (iv_c > 0) & (iv_p > 0)
    if not mask.any():
        return fallback
    k_ok = k_arr[mask]
    i = int(np.argmin(np.abs(k_ok - spot)))
    K = float(k_ok[i])
    # Use the geometric average of the two Black prices to recover F.
    sigma = 0.5 * (float(iv_c[mask][i]) + float(iv_p[mask][i]))
    T = tenor_years
    df = np.exp(-r * T)
    # Black price with forward F*: solve F s.t. call(F,K,sigma,T)-put(...) = F-K.
    # Call - Put = DF * (F - K); so F = K + (C - P) / DF.
    # Compute C and P with Black formula for the trial F = fallback:
    F = fallback
    d1 = (np.log(F / K) + 0.5 * sigma ** 2 * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    from math import erf, sqrt
    N = lambda x: 0.5 * (1.0 + erf(x / sqrt(2.0)))
    call = df * (F * N(d1) - K * N(d2))
    put = df * (K * N(-d2) - F * N(-d1))
    F_est = K + (call - put) / df
    if not np.isfinite(F_est) or F_est <= 0:
        return fallback
    if not (0.7 * fallback < F_est < 1.3 * fallback):
        return fallback
    return float(F_est)


def _synthesize_bid_ask(iv_mid: np.ndarray, z: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Synthesize ``(sigma_bid, sigma_ask)`` around the mid implied volatility.

    The half-spread (in volatility points) is
    ``h(z) = SPREAD_HALF_WIDTH_ATM + SPREAD_HALF_WIDTH_WING_SLOPE * |z|``.
    Implemented because the NBBO is not available on the active data plan;
    the substitution is recorded in ``results/RESULTS.md`` under "Deviations
    from the paper".
    """
    half = SPREAD_HALF_WIDTH_ATM + SPREAD_HALF_WIDTH_WING_SLOPE * np.abs(z)
    sigma_bid = np.clip(iv_mid - half, 1e-4, None)
    sigma_ask = iv_mid + half
    return sigma_bid, sigma_ask


def _assemble_expiries(
    underlying: str,
    reference: pd.DataFrame,
    iv_map: dict[str, dict],
    spot: float,
    dividend_yield: float,
    risk_free_rate: float,
    as_of_date: pd.Timestamp,
) -> tuple[dict[pd.Timestamp, pd.DataFrame], dict[pd.Timestamp, float], dict[pd.Timestamp, float]]:
    """Group contracts by expiry, compute forwards and synthetic bid/ask IVs."""
    df = reference.copy()
    df["iv"] = df["ticker"].map(lambda t: iv_map.get(t, {}).get("iv", float("nan")))
    df["vega"] = df["ticker"].map(lambda t: iv_map.get(t, {}).get("vega", float("nan")))
    df["delta"] = df["ticker"].map(lambda t: iv_map.get(t, {}).get("delta", float("nan")))
    df = df[df["iv"].notna() & (df["iv"] > 0.01) & (df["iv"] < 3.0)].reset_index(drop=True)
    tenor_days = (df["expiration"].dt.normalize() - as_of_date.normalize()).dt.days
    df["tenor_days"] = tenor_days.astype(int)
    df = df[df["tenor_days"].between(TENOR_MIN_DAYS, TENOR_MAX_DAYS)].reset_index(drop=True)
    if df.empty:
        raise MassiveError(f"{underlying}: no contract in tenor band")

    quotes_by_expiry: dict[pd.Timestamp, pd.DataFrame] = {}
    forwards: dict[pd.Timestamp, float] = {}
    tenors: dict[pd.Timestamp, float] = {}

    for expiry, grp in df.groupby("expiration"):
        T = float(grp["tenor_days"].iloc[0]) / 365.0
        F = _implied_forward(grp, spot, T, risk_free_rate, dividend_yield)
        k = np.log(grp["strike"].to_numpy() / F)
        # Approximate z using an initial ATM-anchor from the median IV; the
        # exact ATM anchor is finalized by the preprocessing step.
        sigma_atm = float(np.median(grp["iv"]))
        z = k / (sigma_atm * np.sqrt(T))
        k_half = MONEYNESS_K_HALF_WIDTH_REF * max(np.sqrt(T), 0.25)
        keep = np.abs(k) <= k_half
        grp_f = grp.loc[keep].copy()
        if grp_f.empty:
            continue
        z_f = z[keep]
        sigma_bid, sigma_ask = _synthesize_bid_ask(grp_f["iv"].to_numpy(), z_f)
        grp_f["sigma_bid"] = sigma_bid
        grp_f["sigma_ask"] = sigma_ask
        # For each strike, keep the tightest quote in volatility across call and put.
        grp_f["spread"] = grp_f["sigma_ask"] - grp_f["sigma_bid"]
        grp_f = (
            grp_f.sort_values("spread")
            .drop_duplicates(subset=["strike"], keep="first")
            .drop(columns=["spread"])
            .sort_values("strike")
            .reset_index(drop=True)
        )
        if len(grp_f) > MAX_CONTRACTS_PER_EXPIRY:
            dist = np.abs(np.log(grp_f["strike"].to_numpy() / F))
            keep = np.argsort(dist)[:MAX_CONTRACTS_PER_EXPIRY]
            grp_f = grp_f.iloc[sorted(keep)].reset_index(drop=True)
        quotes_by_expiry[pd.Timestamp(expiry)] = grp_f
        forwards[pd.Timestamp(expiry)] = F
        tenors[pd.Timestamp(expiry)] = T

    if len(quotes_by_expiry) > MAX_EXPIRIES_PER_UNDERLYING:
        exps_sorted = sorted(quotes_by_expiry)
        idx = np.linspace(0, len(exps_sorted) - 1, MAX_EXPIRIES_PER_UNDERLYING).astype(int)
        keep_exps = {exps_sorted[i] for i in idx}
        quotes_by_expiry = {e: v for e, v in quotes_by_expiry.items() if e in keep_exps}
        forwards = {e: forwards[e] for e in quotes_by_expiry}
        tenors = {e: tenors[e] for e in quotes_by_expiry}
    if not quotes_by_expiry:
        raise MassiveError(f"{underlying}: no expiry left after filtering")
    return quotes_by_expiry, forwards, tenors


def load_chain(underlying: str, *, use_cache: bool = True,
               cache_ttl_hours: float = 24.0) -> ChainSnapshot:
    """Fetch, parse and cache the option chain for ``underlying``."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_df = CACHE_DIR / f"chain_{underlying}.csv.gz"
    cache_meta = CACHE_DIR / f"chain_{underlying}.json"

    if use_cache and cache_df.exists() and cache_meta.exists():
        meta = json.loads(cache_meta.read_text())
        age_h = (time.time() - meta["fetched_at"]) / 3600.0
        if age_h <= cache_ttl_hours:
            df = pd.read_csv(cache_df, parse_dates=["expiration"])
            return _snapshot_from_cache(underlying, df, meta)

    reference = _fetch_chain_reference(underlying)
    spot = float(reference["underlying_price"].dropna().iloc[0]) if reference["underlying_price"].notna().any() else float("nan")
    if not np.isfinite(spot) or spot <= 0:
        raise MassiveError(f"{underlying}: could not determine spot price")

    # Pre-filter by tenor and rough moneyness to keep the batched snapshot calls
    # small. The final per-expiry moneyness window is applied in _assemble_expiries.
    as_of_date = pd.Timestamp.utcnow().tz_localize(None).normalize()
    reference["tenor_days"] = (reference["expiration"].dt.normalize() - as_of_date).dt.days
    reference = reference[reference["tenor_days"].between(TENOR_MIN_DAYS, TENOR_MAX_DAYS)]
    reference = reference.reset_index(drop=True)
    rough_k = np.log(reference["strike"].to_numpy() / spot)
    T_years = reference["tenor_days"].to_numpy() / 365.0
    k_half = MONEYNESS_K_HALF_WIDTH_REF * np.maximum(np.sqrt(T_years), 0.25)
    reference = reference[np.abs(rough_k) <= k_half].reset_index(drop=True)
    if reference.empty:
        raise MassiveError(f"{underlying}: no contract in moneyness band")

    # Trim expiries to keep runtime in budget before fetching IVs.
    exps_sorted = sorted(reference["expiration"].unique())
    if len(exps_sorted) > MAX_EXPIRIES_PER_UNDERLYING:
        idx = np.linspace(0, len(exps_sorted) - 1, MAX_EXPIRIES_PER_UNDERLYING).astype(int)
        keep_exps = {exps_sorted[i] for i in idx}
        reference = reference[reference["expiration"].isin(keep_exps)]
    # Cap per expiry to limit batch API calls.
    pieces = []
    for _exp, grp in reference.groupby("expiration"):
        if len(grp) <= MAX_CONTRACTS_PER_EXPIRY:
            pieces.append(grp)
            continue
        dist = np.abs(np.log(grp["strike"].to_numpy() / spot))
        keep = np.argsort(dist)[:MAX_CONTRACTS_PER_EXPIRY]
        pieces.append(grp.iloc[sorted(keep)])
    reference = pd.concat(pieces, ignore_index=True)

    iv_map = _fetch_iv_parallel(reference["ticker"].tolist())
    q = DIVIDEND_YIELD_DEFAULT.get(underlying, 0.0)
    r = RISK_FREE_RATE_DEFAULT
    quotes_by_expiry, forwards, tenors = _assemble_expiries(
        underlying, reference, iv_map, spot, q, r, as_of_date,
    )
    # Flatten for caching.
    rows = []
    for expiry, grp in quotes_by_expiry.items():
        g = grp.copy()
        g["expiration"] = expiry
        rows.append(g)
    cached_df = pd.concat(rows, ignore_index=True)
    meta = {
        "underlying": underlying,
        "as_of_date": as_of_date.isoformat(),
        "spot": spot,
        "dividend_yield": q,
        "risk_free_rate": r,
        "forwards": {e.isoformat(): float(v) for e, v in forwards.items()},
        "tenors_years": {e.isoformat(): float(v) for e, v in tenors.items()},
        "fetched_at": time.time(),
    }
    cached_df.to_csv(cache_df, index=False)
    cache_meta.write_text(json.dumps(meta, indent=2, default=str))
    return ChainSnapshot(
        underlying=underlying,
        as_of_date=as_of_date,
        spot=spot,
        dividend_yield=q,
        risk_free_rate=r,
        quotes_by_expiry=quotes_by_expiry,
        forwards=forwards,
        tenors_years=tenors,
    )


def _snapshot_from_cache(underlying: str, df: pd.DataFrame, meta: dict) -> ChainSnapshot:
    forwards = {pd.Timestamp(k): float(v) for k, v in meta["forwards"].items()}
    tenors = {pd.Timestamp(k): float(v) for k, v in meta["tenors_years"].items()}
    quotes_by_expiry: dict[pd.Timestamp, pd.DataFrame] = {}
    for expiry, grp in df.groupby("expiration"):
        key = pd.Timestamp(expiry)
        quotes_by_expiry[key] = grp.reset_index(drop=True)
    return ChainSnapshot(
        underlying=underlying,
        as_of_date=pd.Timestamp(meta["as_of_date"]).normalize(),
        spot=float(meta["spot"]),
        dividend_yield=float(meta["dividend_yield"]),
        risk_free_rate=float(meta["risk_free_rate"]),
        quotes_by_expiry=quotes_by_expiry,
        forwards=forwards,
        tenors_years=tenors,
    )
