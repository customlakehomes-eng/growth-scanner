#!/usr/bin/env python3
"""
Elite Hybrid Scanner Scheduler
Runs 3 times daily: Pre-market, Midday, Post-market
"""

from elite_hybrid_scanner import EliteHybridScanner
from apscheduler.schedulers.background import BackgroundScheduler
import time
from datetime import datetime


def main():
    scanner = EliteHybridScanner()

    scheduler = BackgroundScheduler(timezone="America/New_York")

    # 3x Daily Schedule
    scheduler.add_job(scanner.run, "cron", hour=6, minute=30, name="Pre-Market")
    scheduler.add_job(scanner.run, "cron", hour=11, minute=30, name="Midday")
    scheduler.add_job(scanner.run, "cron", hour=16, minute=15, name="Post-Market")

    scheduler.start()

    print(f"Elite Hybrid Scheduler STARTED at {datetime.now()}")
    print("   Schedule: 6:30 AM | 11:30 AM | 4:15 PM (New York Time)")
    print("   Dual-mode: Breakout + Growth scanning")
    print("   Using your existing Telegram credentials")
    print()

    try:
        while True:
            time.sleep(60)
    except (KeyboardInterrupt, SystemExit):
        scheduler.shutdown()
        print("Scheduler stopped gracefully.")


if __name__ == "__main__":
    main()
