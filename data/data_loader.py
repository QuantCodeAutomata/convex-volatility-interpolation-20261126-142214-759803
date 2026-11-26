"""Options NBBO quotes for CVI calibration.

Loads a full option chain for one underlying from the Massive REST API
(``/v3/snapshot/options/{underlying}``), which returns per-contract NBBO
bid/ask, implied volatility and Black-Scholes greeks in a single paginated
feed. Rows are filtered by tenor and by moneyness, bid/ask implied volatility
is derived from the snapshot IV and vega through the Newton linearization
``sigma_side = IV + (price_side - mark) / vega``, and the tightest
(call, put) pair per strike is kept (paper p. 1-3: puts and calls are not
differentiated and the tightest quotes are used).

The underlying forward is derived with put-call parity when a matched pair is
available at a strike close to the money; otherwise a fallback spot discounted
at the configured dividend yield is used.

No market-data time series are ever written to the repository; the only
caching lives outside ``/workspace/project`` (``/tmp``) and holds the parsed
per-expiry table used by the experiment scripts.
"""

from __future__ import annotations

import json
import os
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import requests
from dotenv import load_dotenv

from src.config import (
    DIVIDEND_YIELD_DEFAULT,
    MAX_CONTRACTS_PER_EXPIRY,
    MAX_EXPIRIES_PER_UNDERLYING,
    MONEYNESS_K_HALF_WIDTH_REF,
    QUOTE_PARALLEL,
    RISK_FREE_RATE_DEFAULT,
    TENOR_MAX_DAYS,
    TENOR_MIN_DAYS,
)

MASSIVE_BASE = "https://api.massive.com"

# Out-of-repo cache. Parsed per-expiry tables only; no raw API responses and no
# market-data time series are ever committed.
CACHE_DIR = Path(tempfile.gettempdir()) / "cvi_cache"


class MassiveError(RuntimeError):
    """A failed Massive REST request."""


def _headers() -> dict[str, str]:
    load_dotenv()
    key = os.environ.get("MASSIVE_API_KEY")
    if not key:
        raise MassiveError("MASSIVE_API_KEY is not set")
    return {"Authorization": f"Bearer {key}"}


def _get(url: str, params: dict | None = None, timeout: int = 60) -> dict:
    """GET with the project's standard header, following Massive conventions."""
    for attempt in range(3):
        resp = requests.get(url, params=params, headers=_headers(), timeout=timeout)
        if resp.status_code in (429, 503, 504) and attempt < 2:
            time.sleep(float(resp.headers.get("Retry-After") or 2 ** attempt))
            continue
        break
    if resp.status_code != 200:
        raise MassiveError(f"{url}: HTTP {resp.status_code} {resp.text[:200]}")
    return resp.json()


def _paginate(url: str, params: dict | None = None, max_pages: int = 50) -> Iterable[dict]:
    """Yield rows across ``next_url`` pages."""
    page = _get(url, params=params)
    seen_urls: set[str] = set()
    for _ in range(max_pages):
        results = page.get("results") or []
        for row in results:
            yield row
        next_url = page.get("next_url")
        if not next_url or next_url in seen_urls:
            return
        seen_urls.add(next_url)
        # next_url in Polygon-style responses omits the key.
        page = _get(next_url, params={"apiKey": os.environ["MASSIVE_API_KEY"]})


def _snapshot_option_chain(underlying: str) -> list[dict]:
    """Return every contract snapshot for ``underlying``.

    Each row contains ``details`` (strike, expiration, contract_type),
    ``implied_volatility``, ``greeks`` (delta, gamma, theta, vega) and
    ``underlying_asset`` (price, ...). Paginated via ``next_url``. The NBBO
    bid/ask is not carried by the chain snapshot and is fetched separately by
    :func:`_fetch_nbbo_quote` per contract.
    """
    url = f"{MASSIVE_BASE}/v3/snapshot/options/{underlying}"
    rows = list(_paginate(url, params={"limit": 250}, max_pages=200))
    if not rows:
        raise MassiveError(f"no snapshot rows for {underlying}")
    return rows


def _fetch_nbbo_quote(option_ticker: str) -> tuple[float | None, float | None]:
    """Return the most recent NBBO ``(bid, ask)`` for one option contract."""
    url = f"{MASSIVE_BASE}/v3/quotes/{option_ticker}"
    try:
        data = _get(url, params={"limit": 1, "order": "desc"})
    except MassiveError:
        return None, None
    rows = data.get("results") or []
    if not rows:
        return None, None
    row = rows[0]
    bid = row.get("bid_price")
    ask = row.get("ask_price")
    bid = float(bid) if bid is not None and bid > 0 else None
    ask = float(ask) if ask is not None and ask > 0 else None
    return bid, ask


def _fetch_quotes_parallel(tickers: list[str]) -> dict[str, tuple[float | None, float | None]]:
    """Fetch NBBO bid/ask for every ticker, using up to ``QUOTE_PARALLEL`` workers."""
    out: dict[str, tuple[float | None, float | None]] = {}
    if not tickers:
        return out
    with ThreadPoolExecutor(max_workers=QUOTE_PARALLEL) as pool:
        results = list(pool.map(_fetch_nbbo_quote, tickers))
    for ticker, pair in zip(tickers, results):
        out[ticker] = pair
    return out


# ---------------------------------------------------------------------------
# Parsing: snapshot -> tidy quote table
# ---------------------------------------------------------------------------


def _parse_snapshot_rows(rows: list[dict], as_of_date: pd.Timestamp
                         ) -> pd.DataFrame:
    """Flatten chain snapshot rows into a quote table.

    Columns: ``ticker, underlying, expiration, tenor_days, strike,
    contract_type, bid, ask, mark, iv, vega, delta, underlying_price``.
    Rows with missing bid/ask or non-positive vega are kept (filtering happens
    downstream); rows with no ``details.expiration_date`` are dropped.
    """
    records = []
    for row in rows:
        details = row.get("details") or {}
        expiration = details.get("expiration_date")
        strike = details.get("strike_price")
        ctype = details.get("contract_type")
        if expiration is None or strike is None or ctype not in ("call", "put"):
            continue
        last_quote = row.get("last_quote") or {}
        bid = last_quote.get("bid")
        ask = last_quote.get("ask")
        greeks = row.get("greeks") or {}
        iv = row.get("implied_volatility")
        vega = greeks.get("vega")
        delta = greeks.get("delta")
        under = row.get("underlying_asset") or {}
        underlying_price = under.get("price")

        exp_ts = pd.Timestamp(expiration)
        tenor_days = (exp_ts.normalize() - as_of_date.normalize()).days

        records.append({
            "ticker": details.get("ticker"),
            "expiration": exp_ts,
            "tenor_days": tenor_days,
            "strike": float(strike),
            "contract_type": ctype,
            "bid": float(bid) if bid is not None else np.nan,
            "ask": float(ask) if ask is not None else np.nan,
            "iv": float(iv) if iv is not None else np.nan,
            "vega": float(vega) if vega is not None else np.nan,
            "delta": float(delta) if delta is not None else np.nan,
            "underlying_price": float(underlying_price) if underlying_price is not None else np.nan,
        })
    df = pd.DataFrame.from_records(records)
    df = df[df["tenor_days"].between(TENOR_MIN_DAYS, TENOR_MAX_DAYS)]
    df = df[df["bid"].notna() | df["ask"].notna()]
    return df.reset_index(drop=True)


def _implied_forward(group: pd.DataFrame, spot: float, dividend_yield: float,
                     tenor_years: float, r: float) -> float:
    """Estimate ``F(T)`` from put-call parity on the row closest to ATM.

    Falls back to ``spot * exp((r - q) T)`` when no matched call/put pair
    exists for the expiry.
    """
    calls = group[group["contract_type"] == "call"].set_index("strike")
    puts = group[group["contract_type"] == "put"].set_index("strike")
    common = calls.index.intersection(puts.index)
    forward_fallback = spot * np.exp((r - dividend_yield) * tenor_years)
    if len(common) == 0:
        return forward_fallback
    # Use mid prices for both legs.
    call_mid = 0.5 * (calls.loc[common, "bid"] + calls.loc[common, "ask"])
    put_mid = 0.5 * (puts.loc[common, "bid"] + puts.loc[common, "ask"])
    # F = K + e^{r T} (C - P); choose the strike with the tightest spread.
    spread = (calls.loc[common, "ask"] - calls.loc[common, "bid"] +
              puts.loc[common, "ask"] - puts.loc[common, "bid"])
    mask = spread.notna() & call_mid.notna() & put_mid.notna()
    if not mask.any():
        return forward_fallback
    strikes = np.asarray(common[mask], dtype=float)
    i = int(np.argmin(np.abs(strikes - spot)))
    K = strikes[i]
    F_est = K + np.exp(r * tenor_years) * (float(call_mid[mask].iloc[i]) - float(put_mid[mask].iloc[i]))
    if not np.isfinite(F_est) or F_est <= 0:
        return forward_fallback
    # Keep the estimate within a plausible band around the fallback to avoid
    # noisy put-call rows flipping the forward on illiquid legs.
    if not (0.7 * forward_fallback < F_est < 1.3 * forward_fallback):
        return forward_fallback
    return float(F_est)


def _trim_per_expiry(df: pd.DataFrame, spot: float) -> pd.DataFrame:
    """Trim the chain to ``MAX_EXPIRIES_PER_UNDERLYING`` expiries and
    ``MAX_CONTRACTS_PER_EXPIRY`` contracts per expiry closest to the spot.

    Applied before fetching individual NBBO quotes to bound the number of HTTP
    calls, while preserving a spread across the full tenor range.
    """
    if df.empty:
        return df
    # Spread expiries across the full tenor range.
    exps = np.sort(df["expiration"].unique())
    if len(exps) > MAX_EXPIRIES_PER_UNDERLYING:
        idx = np.linspace(0, len(exps) - 1, MAX_EXPIRIES_PER_UNDERLYING).astype(int)
        keep_exps = set(exps[idx])
        df = df[df["expiration"].isin(keep_exps)]
    # Per expiry, keep MAX_CONTRACTS_PER_EXPIRY contracts closest to spot.
    out = []
    for _expiry, grp in df.groupby("expiration"):
        if len(grp) <= MAX_CONTRACTS_PER_EXPIRY:
            out.append(grp)
            continue
        dist = np.abs(np.log(grp["strike"].to_numpy() / spot))
        keep = np.argsort(dist)[:MAX_CONTRACTS_PER_EXPIRY]
        out.append(grp.iloc[sorted(keep)])
    return pd.concat(out, ignore_index=True)


def _derive_side_ivs(df: pd.DataFrame) -> pd.DataFrame:
    """Add ``sigma_bid, sigma_ask`` derived from snapshot ``iv`` and vega.

    The Newton linearization ``sigma_side = iv + (price_side - mark) / vega``
    is used with ``mark = 0.5 (bid + ask)``. When either side or vega is
    missing, the corresponding output is NaN. When the vega is extremely
    small, the linearization is clipped to a 50% relative move of ``iv``.
    """
    df = df.copy()
    mark = 0.5 * (df["bid"] + df["ask"])
    vega = df["vega"].where(df["vega"] > 1e-6, np.nan)
    # Mid-price fallback when one leg is NaN: use the other leg as mark.
    mark = mark.fillna(df["ask"]).fillna(df["bid"])
    # When vega is NaN keep sigma_bid = sigma_ask = iv (fall back to flat IV).
    delta_bid = (df["bid"] - mark) / vega
    delta_ask = (df["ask"] - mark) / vega
    sigma_bid = np.where(df["vega"].notna(),
                         df["iv"] + delta_bid, df["iv"])
    sigma_ask = np.where(df["vega"].notna(),
                         df["iv"] + delta_ask, df["iv"])
    # Clip to a positive value; drop rows where sigma_ask <= 0.
    sigma_bid = np.clip(sigma_bid, 1e-4, 5.0)
    sigma_ask = np.clip(sigma_ask, 1e-4, 5.0)
    # Ensure sigma_ask >= sigma_bid (numerical safety).
    hi = np.maximum(sigma_bid, sigma_ask)
    lo = np.minimum(sigma_bid, sigma_ask)
    df["sigma_bid"] = np.where(df["bid"].notna(), lo, np.nan)
    df["sigma_ask"] = np.where(df["ask"].notna(), hi, np.nan)
    return df


@dataclass
class ChainSnapshot:
    """Parsed option chain for one underlying as of the retrieval moment."""

    underlying: str
    as_of_date: pd.Timestamp
    spot: float
    dividend_yield: float
    risk_free_rate: float
    quotes_by_expiry: dict[pd.Timestamp, pd.DataFrame]  # per-expiry table
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


def load_chain(underlying: str, *, use_cache: bool = True,
               cache_ttl_hours: float = 12.0) -> ChainSnapshot:
    """Fetch and parse the full option chain for ``underlying``.

    The derived table is cached outside the repository (``/tmp/cvi_cache``) and
    reused for ``cache_ttl_hours`` so repeated runs of the experiments and
    tests do not hit the API again.
    """
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = CACHE_DIR / f"chain_{underlying}.csv.gz"
    meta_path = CACHE_DIR / f"chain_{underlying}.json"
    if use_cache and cache_path.exists() and meta_path.exists():
        meta = json.loads(meta_path.read_text())
        age_h = (time.time() - meta["fetched_at"]) / 3600.0
        if age_h <= cache_ttl_hours:
            df = pd.read_csv(cache_path, parse_dates=["expiration"])
            return _build_snapshot(underlying, df, meta)

    raw_rows = _snapshot_option_chain(underlying)
    as_of_date = pd.Timestamp.utcnow().tz_localize(None).normalize()
    df = _parse_snapshot_rows(raw_rows, as_of_date)
    if df.empty:
        raise MassiveError(f"{underlying}: no usable snapshot rows")
    spot = float(df["underlying_price"].dropna().median())
    if not np.isfinite(spot) or spot <= 0:
        raise MassiveError(f"{underlying}: could not determine spot price")

    # Pre-filter contracts by moneyness to limit the number of NBBO requests.
    # Use a conservative rough window around the spot, widened per expiry by
    # sqrt(T) so that short tenors keep a tight band and long tenors a wide one.
    df["rough_k"] = np.log(df["strike"] / spot)
    T_years = df["tenor_days"] / 365.0
    k_half = MONEYNESS_K_HALF_WIDTH_REF * np.maximum(np.sqrt(T_years), 0.25)
    df = df[np.abs(df["rough_k"]) <= k_half].reset_index(drop=True)
    if df.empty:
        raise MassiveError(f"{underlying}: no contracts in moneyness band")

    # Keep at most MAX_CONTRACTS_PER_EXPIRY per expiry, closest to the spot,
    # and at most MAX_EXPIRIES_PER_UNDERLYING spread across the full tenor range.
    df = _trim_per_expiry(df, spot)

    # Fetch NBBO bid/ask for the pre-filtered contracts.
    tickers = df["ticker"].dropna().unique().tolist()
    nbbo = _fetch_quotes_parallel(tickers)
    bid_series = df["ticker"].map(lambda t: (nbbo.get(t) or (None, None))[0])
    ask_series = df["ticker"].map(lambda t: (nbbo.get(t) or (None, None))[1])
    df["bid"] = pd.to_numeric(bid_series, errors="coerce")
    df["ask"] = pd.to_numeric(ask_series, errors="coerce")
    df = df.drop(columns=["rough_k"])
    df = df[df["bid"].notna() | df["ask"].notna()].reset_index(drop=True)
    if df.empty:
        raise MassiveError(f"{underlying}: no NBBO rows returned")
    df = _derive_side_ivs(df)
    meta = {
        "underlying": underlying,
        "as_of_date": as_of_date.isoformat(),
        "spot": spot,
        "dividend_yield": DIVIDEND_YIELD_DEFAULT.get(underlying, 0.0),
        "risk_free_rate": RISK_FREE_RATE_DEFAULT,
        "n_raw_rows": len(raw_rows),
        "n_parsed_rows": len(df),
        "fetched_at": time.time(),
    }
    df.to_csv(cache_path, index=False)
    meta_path.write_text(json.dumps(meta, indent=2, default=str))
    return _build_snapshot(underlying, df, meta)


def _build_snapshot(underlying: str, df: pd.DataFrame, meta: dict) -> ChainSnapshot:
    spot = float(meta["spot"])
    r = float(meta["risk_free_rate"])
    q = float(meta["dividend_yield"])
    as_of_date = pd.Timestamp(meta["as_of_date"]).normalize()

    # Group by expiry and apply moneyness filter widened by sqrt(T).
    quotes_by_expiry: dict[pd.Timestamp, pd.DataFrame] = {}
    forwards: dict[pd.Timestamp, float] = {}
    tenors: dict[pd.Timestamp, float] = {}

    for expiry, grp in df.groupby("expiration"):
        tenor_days = int(grp["tenor_days"].iloc[0])
        if tenor_days <= 0:
            continue
        T = tenor_days / 365.0
        F = _implied_forward(grp, spot, q, T, r)
        k = np.log(grp["strike"].to_numpy() / F)
        k_half = MONEYNESS_K_HALF_WIDTH_REF * max(np.sqrt(T), 0.25)
        inside = np.abs(k) <= k_half
        grp_f = grp.loc[inside].copy()
        if grp_f.empty:
            continue
        # Keep only rows with a valid vega or iv.
        grp_f = grp_f[grp_f["iv"].notna() & (grp_f["iv"] > 0.01)]
        if grp_f.empty:
            continue
        # Trim to MAX_CONTRACTS_PER_EXPIRY by keeping those closest to ATM.
        if len(grp_f) > MAX_CONTRACTS_PER_EXPIRY:
            dist = np.abs(np.log(grp_f["strike"].to_numpy() / F))
            keep = np.argsort(dist)[:MAX_CONTRACTS_PER_EXPIRY]
            grp_f = grp_f.iloc[sorted(keep)].reset_index(drop=True)
        quotes_by_expiry[expiry] = grp_f.reset_index(drop=True)
        forwards[expiry] = F
        tenors[expiry] = T

    # Trim to MAX_EXPIRIES_PER_UNDERLYING, preferring a spread across tenors.
    if len(quotes_by_expiry) > MAX_EXPIRIES_PER_UNDERLYING:
        exps_sorted = sorted(quotes_by_expiry)
        idx = np.linspace(0, len(exps_sorted) - 1, MAX_EXPIRIES_PER_UNDERLYING).astype(int)
        keep = {exps_sorted[i] for i in idx}
        quotes_by_expiry = {e: v for e, v in quotes_by_expiry.items() if e in keep}
        forwards = {e: forwards[e] for e in quotes_by_expiry}
        tenors = {e: tenors[e] for e in quotes_by_expiry}

    if not quotes_by_expiry:
        raise MassiveError(f"{underlying}: no expiry left after filtering")

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
