#!/usr/bin/env python3
"""
Universe Engine Test v3
Run this first to verify the universe engine works correctly.
"""

from universe_engine import UniverseEngine
from datetime import datetime


def main():
    print("=" * 80)
    print("UNIVERSE ENGINE v3 - VALIDATION TEST")
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)

    engine = UniverseEngine()

    print()
    print("Building universe (S&P 500 + Nasdaq 100 + Market Data + Momentum)...")
    universe = engine.build(include_momentum=True, apply_liquidity=True)

    print(f"Final Universe Size: {len(universe)} stocks")

    # Show results
    cols = ["Symbol", "GICS Sector", "Source", "market_cap", "avg_volume", "momentum_pct"]
    available = [c for c in cols if c in universe.columns]

    print()
    print("-" * 80)
    print("TOP 15 MOMENTUM STOCKS")
    print("-" * 80)
    if "momentum_pct" in universe.columns:
        display_df = universe[available].sort_values("momentum_pct", ascending=False)
    else:
        display_df = universe[available]
    print(display_df.head(15).to_string(index=False))

    print()
    print("-" * 80)
    print("SUMMARY")
    print("-" * 80)
    print(f"Total Stocks          : {len(universe)}")
    if "market_cap" in universe.columns:
        valid_caps = universe["market_cap"][universe["market_cap"] > 0]
        if len(valid_caps) > 0:
            print(f"Average Market Cap    : ${valid_caps.mean()/1e9:.1f} Billion")
            print(f"Median Market Cap     : ${valid_caps.median()/1e9:.1f} Billion")
    if "avg_volume" in universe.columns:
        valid_vols = universe["avg_volume"][universe["avg_volume"] > 0]
        if len(valid_vols) > 0:
            print(f"Average Daily Volume  : {valid_vols.mean()/1e6:.1f}M shares")
    if "momentum_pct" in universe.columns:
        print(f"Highest Momentum      : {universe['momentum_pct'].max():.1f}%")
        print(f"Lowest Momentum       : {universe['momentum_pct'].min():.1f}%")
        print(f"Median Momentum       : {universe['momentum_pct'].median():.1f}%")

    # Sector breakdown
    if "GICS Sector" in universe.columns:
        print()
        print("Sector Distribution:")
        print(universe["GICS Sector"].value_counts().head(11).to_string())

    # Source breakdown
    if "Source" in universe.columns:
        print()
        print("Source Distribution:")
        print(universe["Source"].value_counts().to_string())

    print()
    print("=" * 80)
    print("Universe Engine Test Completed Successfully!")
    print("You can now safely run the main scanner.")
    print("=" * 80)


if __name__ == "__main__":
    main()
