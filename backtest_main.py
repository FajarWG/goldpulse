#!/usr/bin/env python3
"""Run or read the WG GoldPulse historical backtest."""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

from forex.backtest import (
    format_backtest,
    load_backtest_data,
    run_momentum_backtest,
    save_backtest,
)
from forex.config import Config
from forex.notify import send_telegram
from forex.product import DEFAULT_STRATEGY_VERSION, STRATEGY_VERSIONS


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser()
    parser.add_argument("--refresh", action="store_true", help="refresh cached candles from Twelve Data")
    parser.add_argument("--notify", action="store_true", help="send the result to Telegram")
    parser.add_argument(
        "--strategy",
        choices=("momentum_v1", "momentum_v3", "momentum_v3_improved", "all"),
        default="all",
    )
    parser.add_argument("--lookback-days", type=int, default=None)
    parser.add_argument("--output-label", default="")
    parser.add_argument("--momentum-rr", type=float, default=None, help="override momentum reward:risk")
    parser.add_argument(
        "--fetch-only",
        action="store_true",
        help="refresh/cache candles without replaying a strategy",
    )
    args = parser.parse_args()
    config = Config.from_env()
    logging.basicConfig(
        level=getattr(logging, config.log_level, logging.INFO),
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
    )
    state_dir = Path(os.getenv("FOREX_STATE_DIR", "/var/lib/xauusd-analysis"))
    backtest_dir = state_dir / "backtest"
    frames, fetched = load_backtest_data(
        backtest_dir / "cache",
        refresh=args.refresh,
        lookback_days=args.lookback_days or int(os.getenv("BACKTEST_LOOKBACK_DAYS", "90")),
    )
    if args.fetch_only:
        counts = ", ".join(f"{timeframe}={len(frame):,}" for timeframe, frame in frames.items())
        print(f"Cached candles: {counts}")
        print(f"Data fetched: {', '.join(fetched) if fetched else 'none (cache)'}")
        return 0

    target_versions = STRATEGY_VERSIONS if args.strategy == "all" else (args.strategy,)
    results = {}
    trades_map = {}

    timeout = int(os.getenv("SIGNAL_TIMEOUT_MINUTES", "240"))
    max_day = int(os.getenv("SIGNAL_MAX_MOMENTUM_PER_DAY", "5"))
    cooldown = int(os.getenv("SIGNAL_COOLDOWN_MINUTES", "30"))
    max_active = int(os.getenv("SIGNAL_MAX_ACTIVE_MOMENTUM", "3"))

    for version in target_versions:
        default_rr = 1.0 if version == "momentum_v1" else 2.0
        rr = args.momentum_rr if args.momentum_rr is not None else default_rr
        summary, trades = run_momentum_backtest(
            frames,
            strategy_version=version,
            timeout_minutes=timeout,
            max_per_day=max_day,
            cooldown_minutes=cooldown,
            max_active=max_active,
            reward_r=rr,
            session_filter=(version in ("momentum_v3", "momentum_v3_improved")),
        )
        label = args.output_label if len(target_versions) == 1 and args.output_label else version
        save_backtest(backtest_dir / label, summary, trades)
        if version == DEFAULT_STRATEGY_VERSION:
            save_backtest(backtest_dir, summary, trades)
        results[version] = summary
        trades_map[version] = trades

    sections = []
    for version in target_versions:
        if version == "momentum_v3_improved":
            label = "🔥 Momentum MTF Pro · momentum_v3_improved (Optimized)"
        elif version == "momentum_v3":
            label = "🎯 Momentum MTF · momentum_v3 (Two Candles)"
        else:
            label = "⚡ Momentum Candle · momentum_v1 (Standar)"
        sections.append(f"📋 {label}\n" + format_backtest(results[version], trades_map[version]))

    text = "\n\n".join(sections)
    print(text)
    print(f"Data fetched: {', '.join(fetched) if fetched else 'none (cache)'}")
    if args.notify and config.telegram.enabled:
        if not send_telegram(text, config.telegram):
            return 1
    return 0



if __name__ == "__main__":
    raise SystemExit(main())
