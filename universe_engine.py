#!/usr/bin/env python3
"""
Universe Engine v3 - Efficient Stock Universe Builder

Builds a filtered universe from S&P 500 + Nasdaq 100, enriched with
market cap, average volume, and momentum data via batch yfinance downloads.
"""

import os
import json
import logging
from datetime import datetime, timedelta
from typing import Optional

import pandas as pd
import yfinance as yf

# Configuration
MIN_MARKET_CAP = float(os.getenv("UNI_MIN_MARKET_CAP", "2000000000"))       # $2B
MIN_AVG_VOLUME = int(os.getenv("UNI_MIN_AVG_VOLUME", "500000"))             # 500K shares/day
CACHE_DAYS = int(os.getenv("UNI_CACHE_DAYS", "7"))
MOMENTUM_PERIOD = os.getenv("UNI_MOMENTUM_PERIOD", "1mo")

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger("universe_engine")

CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache")
os.makedirs(CACHE_DIR, exist_ok=True)

_STORAGE_OPTS = {"User-Agent": "Mozilla/5.0 (compatible; GrowthScanner/3.0)"}


class UniverseEngine:
    """Builds and filters a stock universe from major US indices."""

    def _cache_path(self, name: str) -> str:
        return os.path.join(CACHE_DIR, f"{name}.json")

    def _read_cache(self, name: str) -> Optional[pd.DataFrame]:
        path = self._cache_path(name)
        if not os.path.exists(path):
            return None
        age = datetime.now() - datetime.fromtimestamp(os.path.getmtime(path))
        if age > timedelta(days=CACHE_DAYS):
            return None
        try:
            with open(path, "r") as f:
                return pd.DataFrame(json.load(f))
        except Exception:
            return None

    def _write_cache(self, name: str, df: pd.DataFrame):
        try:
            with open(self._cache_path(name), "w") as f:
                json.dump(df.to_dict(orient="records"), f)
        except Exception:
            pass

    @staticmethod
    def _normalize_ticker(t: str) -> str:
        return str(t).strip().upper().replace(".", "-")

    # -- Index Fetchers --

    def sp500(self) -> pd.DataFrame:
        cached = self._read_cache("sp500")
        if cached is not None:
            return cached

        try:
            tables = pd.read_html(
                "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies",
                storage_options=_STORAGE_OPTS,
            )
            df = tables[0].rename(columns=lambda c: str(c).strip())
            df = df[["Symbol", "GICS Sector"]].copy()
            df["Symbol"] = df["Symbol"].map(self._normalize_ticker)
            df["Source"] = "sp500"
            df["GICS Sector"] = df["GICS Sector"].fillna("Unknown")
            self._write_cache("sp500", df)
            logger.info(f"S&P 500: {len(df)} tickers")
            return df
        except Exception as e:
            logger.error(f"S&P 500 fetch failed: {e}")
            return pd.DataFrame(columns=["Symbol", "GICS Sector", "Source"])

    def nasdaq100(self) -> pd.DataFrame:
        cached = self._read_cache("nasdaq100")
        if cached is not None:
            return cached

        fallback = [
            "NVDA", "AMD", "AVGO", "AAPL", "MSFT", "META", "TSLA", "GOOGL",
            "AMZN", "NFLX", "COST", "ADBE", "CRM", "CSCO", "QCOM", "INTC",
            "AMAT", "MU", "LRCX", "KLAC", "SNPS", "CDNS", "MRVL", "PANW",
            "CRWD", "FTNT", "DDOG", "ZS", "WDAY", "TEAM",
        ]
        try:
            tables = pd.read_html(
                "https://en.wikipedia.org/wiki/Nasdaq-100",
                storage_options=_STORAGE_OPTS,
            )
            for t in tables:
                if 80 <= len(t) <= 110:
                    col = next(
                        (c for c in t.columns
                         if "symbol" in str(c).lower() or "ticker" in str(c).lower()),
                        None,
                    )
                    if col:
                        tickers = t[col].dropna().astype(str).map(self._normalize_ticker).tolist()
                        df = pd.DataFrame({"Symbol": tickers})
                        df["GICS Sector"] = "Unknown"
                        df["Source"] = "nasdaq100"
                        self._write_cache("nasdaq100", df)
                        logger.info(f"Nasdaq 100: {len(df)} tickers")
                        return df
        except Exception:
            pass

        df = pd.DataFrame({"Symbol": fallback})
        df["GICS Sector"] = "Unknown"
        df["Source"] = "nasdaq100"
        self._write_cache("nasdaq100", df)
        logger.info(f"Nasdaq 100: using {len(df)} fallback tickers")
        return df

    # -- Batch Data Enrichment --

    def _enrich_market_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Batch-fetch market cap and average volume for all tickers."""
        tickers = df["Symbol"].tolist()
        logger.info(f"Fetching market data for {len(tickers)} tickers...")

        market_caps = {}
        avg_volumes = {}

        for sym in tickers:
            try:
                info = yf.Ticker(sym).fast_info
                market_caps[sym] = getattr(info, "market_cap", 0) or 0
                avg_volumes[sym] = getattr(info, "three_month_average_volume", 0) or 0
            except Exception:
                market_caps[sym] = 0
                avg_volumes[sym] = 0

        df["market_cap"] = df["Symbol"].map(market_caps)
        df["avg_volume"] = df["Symbol"].map(avg_volumes)
        return df

    def _add_momentum(self, df: pd.DataFrame) -> pd.DataFrame:
        """Batch-download price data and compute momentum for all tickers at once."""
        tickers = df["Symbol"].tolist()
        logger.info(f"Calculating momentum for {len(tickers)} tickers (period={MOMENTUM_PERIOD})...")

        try:
            data = yf.download(tickers, period=MOMENTUM_PERIOD, progress=False, group_by="ticker")
        except Exception as e:
            logger.error(f"Momentum download failed: {e}")
            df["momentum_pct"] = 0.0
            return df

        returns = {}
        multi = isinstance(data.columns, pd.MultiIndex)
        for t in tickers:
            try:
                closes = data[t]["Close"] if multi else data["Close"]
                closes = closes.dropna()
                if len(closes) >= 2:
                    returns[t] = (closes.iloc[-1] / closes.iloc[0] - 1) * 100
                else:
                    returns[t] = 0.0
            except Exception:
                returns[t] = 0.0

        df["momentum_pct"] = df["Symbol"].map(returns).round(2)
        return df

    # -- Main Build --

    def build(self, include_momentum: bool = True, apply_liquidity: bool = True) -> pd.DataFrame:
        """
        Build the stock universe.

        1. Fetch S&P 500 + Nasdaq 100 tickers
        2. Deduplicate
        3. Enrich with market cap & volume (batch)
        4. Filter by liquidity thresholds
        5. Add momentum scores (batch)
        6. Sort by momentum descending
        """
        sp = self.sp500()
        ndx = self.nasdaq100()
        df = pd.concat([sp, ndx], ignore_index=True)
        df = df.drop_duplicates("Symbol").reset_index(drop=True)
        logger.info(f"Combined universe: {len(df)} tickers (before filtering)")

        df = self._enrich_market_data(df)

        if apply_liquidity:
            before = len(df)
            df = df[(df["market_cap"] >= MIN_MARKET_CAP) & (df["avg_volume"] >= MIN_AVG_VOLUME)]
            df = df.reset_index(drop=True)
            logger.info(f"Liquidity filter: {before} -> {len(df)} tickers "
                        f"(min cap=${MIN_MARKET_CAP/1e9:.1f}B, min vol={MIN_AVG_VOLUME:,})")

        if include_momentum:
            df = self._add_momentum(df)
            df = df.sort_values("momentum_pct", ascending=False).reset_index(drop=True)

        logger.info(f"Final universe: {len(df)} tickers")
        return df


if __name__ == "__main__":
    engine = UniverseEngine()
    universe = engine.build()
    print(f"\nUniverse built with {len(universe)} stocks")
    cols = ["Symbol", "GICS Sector", "market_cap", "avg_volume", "momentum_pct"]
    available = [c for c in cols if c in universe.columns]
    print(universe[available].head(20).to_string(index=False))
