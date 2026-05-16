#!/usr/bin/env python3
"""
Elite Hybrid Scanner v3 - Dual-Mode Breakout + Growth Scanner

Two distinct scanning modes:
  BREAKOUT: Price breaking above resistance with volume confirmation
  GROWTH:   Strong revenue growth + earnings momentum + reasonable valuation

Each stock gets scored independently for both modes.
Telegram alerts show breakout picks and growth picks separately.
"""

import os
import logging
from datetime import datetime
from typing import Dict, List, Optional

import yfinance as yf
import pandas as pd
import requests

from universe_engine import UniverseEngine

# ========================= CONFIG =========================
# Growth filters
MIN_REVENUE_GROWTH = float(os.getenv("GF_MIN_REVENUE_GROWTH", "15"))
MAX_PE_RATIO = float(os.getenv("GF_MAX_PE_RATIO", "60"))
MIN_PROFIT_MARGIN = float(os.getenv("GF_MIN_PROFIT_MARGIN", "5"))

# Breakout filters
MIN_RS = float(os.getenv("GF_MIN_RS", "3"))
VOL_SPIKE = float(os.getenv("GF_VOL_SPIKE", "1.3"))

# Telegram
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN") or os.getenv("GF_TELEGRAM_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID") or os.getenv("GF_TELEGRAM_CHAT_ID", "")

TOP_N = 10
PRICE_PERIOD = "6mo"
# ==========================================================

LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
os.makedirs(LOG_DIR, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(LOG_DIR, f"scanner_{datetime.now().strftime('%Y-%m-%d')}.log")),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger("elite_scanner")


class EliteHybridScanner:
    def __init__(self):
        self.universe_engine = UniverseEngine()
        self._spy_cache: Optional[pd.DataFrame] = None

    # -- Market context (fetched once) --

    def _get_spy(self) -> pd.DataFrame:
        if self._spy_cache is None:
            try:
                self._spy_cache = yf.download("SPY", period="1y", progress=False)
            except Exception:
                self._spy_cache = pd.DataFrame()
        return self._spy_cache

    def market_regime(self) -> str:
        spy = self._get_spy()
        if spy.empty or len(spy) < 200:
            return "UNKNOWN"
        try:
            close = spy["Close"]
            ma50 = close.rolling(50).mean().iloc[-1]
            ma200 = close.rolling(200).mean().iloc[-1]
            price = close.iloc[-1]
            if price > ma50 and price > ma200:
                return "RISK_ON"
            elif price > ma200:
                return "TRANSITION"
            return "RISK_OFF"
        except Exception:
            return "UNKNOWN"

    # -- Batch data loading --

    def _load_price_data(self, tickers: List[str]) -> Dict[str, pd.DataFrame]:
        """Download price history for all tickers in a single batch call."""
        logger.info(f"Batch-downloading {PRICE_PERIOD} price data for {len(tickers)} tickers...")
        try:
            raw = yf.download(tickers, period=PRICE_PERIOD, progress=False, group_by="ticker")
        except Exception as e:
            logger.error(f"Batch price download failed: {e}")
            return {}

        result = {}
        multi = isinstance(raw.columns, pd.MultiIndex)
        for t in tickers:
            try:
                df = raw[t] if multi else raw
                if df is not None and len(df) >= 20:
                    result[t] = df
            except Exception:
                pass

        logger.info(f"Got price data for {len(result)}/{len(tickers)} tickers")
        return result

    def _load_fundamentals(self, tickers: List[str]) -> Dict[str, dict]:
        """Fetch fundamental data for all tickers."""
        logger.info(f"Fetching fundamentals for {len(tickers)} tickers...")
        result = {}
        for t in tickers:
            try:
                info = yf.Ticker(t).info
                result[t] = info
            except Exception:
                result[t] = {}
        return result

    # -- Technical indicators --

    def _relative_strength_vs_spy(self, ticker: str, data: pd.DataFrame) -> float:
        spy = self._get_spy()
        if spy.empty or data.empty:
            return 0.0
        try:
            period = min(len(data), len(spy), 21)
            stock_close = data["Close"].dropna()
            spy_close = spy["Close"].dropna()
            if len(stock_close) < period or len(spy_close) < period:
                return 0.0
            stock_ret = (stock_close.iloc[-1] / stock_close.iloc[-period] - 1) * 100
            spy_ret = (spy_close.iloc[-1] / spy_close.iloc[-period] - 1) * 100
            return stock_ret - spy_ret
        except Exception:
            return 0.0

    def _breakout_signal(self, data: pd.DataFrame) -> dict:
        """Detect price breakout above 20-day high with volume confirmation."""
        if len(data) < 20:
            return {"is_breakout": False}

        close = data["Close"]
        high = data["High"]
        volume = data["Volume"]

        high_20 = high.rolling(20).max()
        price = close.iloc[-1]
        prev_high = high_20.iloc[-3]

        is_breakout = price > prev_high

        vol_20_avg = volume.rolling(20).mean().iloc[-1]
        vol_today = volume.iloc[-1]
        vol_ratio = vol_today / vol_20_avg if vol_20_avg > 0 else 0

        ma20 = close.rolling(20).mean().iloc[-1]
        ma50 = close.rolling(50).mean().iloc[-1] if len(data) >= 50 else ma20
        stop_loss = round(float(ma20), 2)

        return {
            "is_breakout": bool(is_breakout),
            "vol_ratio": round(float(vol_ratio), 2),
            "above_20ma": bool(price > ma20),
            "above_50ma": bool(price > ma50) if len(data) >= 50 else None,
            "suggested_stop": stop_loss,
        }

    def _trend_strength(self, data: pd.DataFrame) -> dict:
        """Evaluate trend alignment and acceleration."""
        if len(data) < 50:
            return {"aligned": False, "acceleration": 0.0}

        close = data["Close"]
        ma10 = close.rolling(10).mean().iloc[-1]
        ma20 = close.rolling(20).mean().iloc[-1]
        ma50 = close.rolling(50).mean().iloc[-1]
        price = close.iloc[-1]

        aligned = price > ma10 > ma20 > ma50

        ret = close.pct_change()
        recent_avg = ret.iloc[-10:].mean()
        prior_avg = ret.iloc[-20:-10].mean()
        acceleration = float((recent_avg - prior_avg) * 1000)

        return {
            "aligned": bool(aligned),
            "acceleration": round(acceleration, 2),
        }

    def _volume_analysis(self, data: pd.DataFrame) -> dict:
        """Analyze volume patterns for accumulation signals."""
        if len(data) < 20:
            return {"vol_5_20_ratio": 0.0, "accumulating": False}

        volume = data["Volume"]
        close = data["Close"]

        vol_5 = volume.rolling(5).mean().iloc[-1]
        vol_20 = volume.rolling(20).mean().iloc[-1]
        ratio = vol_5 / vol_20 if vol_20 > 0 else 0

        up_days = (close.diff() > 0).iloc[-20:]
        up_vol = volume.iloc[-20:][up_days].mean()
        down_vol = volume.iloc[-20:][~up_days].mean()
        accumulating = up_vol > down_vol if pd.notna(up_vol) and pd.notna(down_vol) else False

        return {
            "vol_5_20_ratio": round(float(ratio), 2),
            "accumulating": bool(accumulating),
        }

    # -- Scoring --

    def _score_breakout(self, rs, breakout, trend, volume, regime):
        score = 0.0
        if regime == "RISK_ON":
            score += 1.5
        elif regime == "TRANSITION":
            score += 0.5
        if breakout["is_breakout"]:
            score += 2.5
        if breakout["vol_ratio"] > 2.0:
            score += 2.0
        elif breakout["vol_ratio"] > VOL_SPIKE:
            score += 1.0
        if rs > 15:
            score += 2.0
        elif rs > 8:
            score += 1.5
        elif rs > MIN_RS:
            score += 1.0
        if trend["aligned"]:
            score += 1.5
        if trend["acceleration"] > 0:
            score += 0.5
        if volume["accumulating"]:
            score += 0.5
        return round(min(score, 10.0), 1)

    def _score_growth(self, info, rs, trend, regime):
        score = 0.0
        rev_growth = (info.get("revenueGrowth") or 0) * 100
        earnings_growth = (info.get("earningsGrowth") or 0) * 100
        profit_margin = (info.get("profitMargins") or 0) * 100
        pe = info.get("forwardPE") or info.get("trailingPE") or 999

        if regime == "RISK_ON":
            score += 1.0
        if rev_growth > 40:
            score += 3.0
        elif rev_growth > 25:
            score += 2.5
        elif rev_growth > MIN_REVENUE_GROWTH:
            score += 1.5
        if earnings_growth > 30:
            score += 1.5
        elif earnings_growth > 15:
            score += 1.0
        if profit_margin > 20:
            score += 1.0
        elif profit_margin > MIN_PROFIT_MARGIN:
            score += 0.5
        if 0 < pe < 25:
            score += 1.5
        elif 0 < pe < MAX_PE_RATIO:
            score += 0.5
        if trend["aligned"]:
            score += 1.0
        if rs > 5:
            score += 0.5
        return round(min(score, 10.0), 1)

    # -- Main scan --

    def scan(self) -> dict:
        """Run dual-mode scan. Returns {"breakout": [...], "growth": [...], "regime": str}."""
        universe = self.universe_engine.build(include_momentum=True, apply_liquidity=True)
        logger.info(f"Universe loaded: {len(universe)} stocks")

        regime = self.market_regime()
        logger.info(f"Market Regime: {regime}")

        tickers = universe["Symbol"].tolist()
        price_data = self._load_price_data(tickers)
        fundamentals = self._load_fundamentals(list(price_data.keys()))

        breakout_candidates = []
        growth_candidates = []

        for _, row in universe.iterrows():
            ticker = row["Symbol"]
            data = price_data.get(ticker)
            if data is None or len(data) < 50:
                continue

            info = fundamentals.get(ticker, {})
            price = info.get("currentPrice") or info.get("regularMarketPrice") or 0

            rs = self._relative_strength_vs_spy(ticker, data)
            breakout = self._breakout_signal(data)
            trend = self._trend_strength(data)
            volume = self._volume_analysis(data)

            # -- Breakout mode --
            if breakout["is_breakout"] and breakout["vol_ratio"] >= VOL_SPIKE and rs >= MIN_RS:
                b_score = self._score_breakout(rs, breakout, trend, volume, regime)
                if b_score >= 4.0:
                    breakout_candidates.append({
                        "ticker": ticker,
                        "score": b_score,
                        "price": round(price, 2),
                        "rs": round(rs, 1),
                        "vol_ratio": breakout["vol_ratio"],
                        "trend_aligned": trend["aligned"],
                        "acceleration": trend["acceleration"],
                        "accumulating": volume["accumulating"],
                        "suggested_stop": breakout["suggested_stop"],
                        "sector": row.get("GICS Sector", "Unknown"),
                        "mode": "BREAKOUT",
                    })

            # -- Growth mode --
            rev_growth = (info.get("revenueGrowth") or 0) * 100
            profit_margin = (info.get("profitMargins") or 0) * 100
            pe = info.get("forwardPE") or info.get("trailingPE") or 999

            if rev_growth >= MIN_REVENUE_GROWTH and pe <= MAX_PE_RATIO and profit_margin >= MIN_PROFIT_MARGIN:
                g_score = self._score_growth(info, rs, trend, regime)
                if g_score >= 4.0:
                    growth_candidates.append({
                        "ticker": ticker,
                        "score": g_score,
                        "price": round(price, 2),
                        "rev_growth": round(rev_growth, 1),
                        "earnings_growth": round((info.get("earningsGrowth") or 0) * 100, 1),
                        "profit_margin": round(profit_margin, 1),
                        "pe": round(pe, 1) if pe < 999 else "N/A",
                        "rs": round(rs, 1),
                        "trend_aligned": trend["aligned"],
                        "sector": row.get("GICS Sector", "Unknown"),
                        "mode": "GROWTH",
                    })

        breakout_candidates.sort(key=lambda x: x["score"], reverse=True)
        growth_candidates.sort(key=lambda x: x["score"], reverse=True)

        return {
            "breakout": breakout_candidates[:TOP_N],
            "growth": growth_candidates[:TOP_N],
            "regime": regime,
        }

    # -- Telegram --

    def send_telegram(self, results: dict):
        if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
            logger.warning("Telegram credentials not set - skipping alert")
            return

        regime = results["regime"]
        breakouts = results["breakout"]
        growth = results["growth"]
        ts = datetime.now().strftime("%Y-%m-%d %H:%M")

        lines = ["<b>ELITE HYBRID SCANNER</b>", f"Time: {ts} | Regime: {regime}", ""]

        if breakouts:
            lines.append("<b>--- BREAKOUT PICKS ---</b>")
            for r in breakouts:
                lines.append(
                    f"<b>{r['ticker']}</b> {r['score']}/10 | ${r['price']} | "
                    f"RS:{r['rs']} | Vol:{r['vol_ratio']}x | Stop:${r['suggested_stop']}"
                )
            lines.append("")

        if growth:
            lines.append("<b>--- GROWTH PICKS ---</b>")
            for r in growth:
                lines.append(
                    f"<b>{r['ticker']}</b> {r['score']}/10 | ${r['price']} | "
                    f"Rev:{r['rev_growth']}% | Earn:{r['earnings_growth']}% | PE:{r['pe']}"
                )
            lines.append("")

        if not breakouts and not growth:
            lines.append("No signals today.")

        msg = "\n".join(lines)
        try:
            requests.post(
                f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage",
                json={"chat_id": TELEGRAM_CHAT_ID, "text": msg, "parse_mode": "HTML"},
                timeout=15,
            )
            logger.info("Telegram alert sent")
        except Exception as e:
            logger.error(f"Telegram failed: {e}")

    # -- Entry point --

    def run(self) -> dict:
        logger.info("=" * 70)
        logger.info("ELITE HYBRID SCANNER v3 STARTED")
        logger.info("=" * 70)

        results = self.scan()

        breakouts = results["breakout"]
        growth = results["growth"]

        if breakouts:
            logger.info("=== TOP BREAKOUT SIGNALS ===")
            for i, r in enumerate(breakouts, 1):
                logger.info(f"  {i:2d}. {r['ticker']:5s} | Score: {r['score']:.1f} | RS: {r['rs']} | Vol: {r['vol_ratio']}x | Stop: ${r['suggested_stop']}")

        if growth:
            logger.info("=== TOP GROWTH SIGNALS ===")
            for i, r in enumerate(growth, 1):
                logger.info(f"  {i:2d}. {r['ticker']:5s} | Score: {r['score']:.1f} | Rev: {r['rev_growth']}% | PE: {r['pe']}")

        self.send_telegram(results)

        total = len(breakouts) + len(growth)
        print(f"Scan completed | {len(breakouts)} breakout + {len(growth)} growth = {total} signals")
        return results


if __name__ == "__main__":
    scanner = EliteHybridScanner()
    scanner.run()
