from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from forex.backtest import (
    BacktestSummary,
    BacktestTrade,
    _resolve_trade,
    format_backtest,
    load_backtest_data,
    load_latest_summary,
    monte_carlo_pvalue,
    save_backtest,
)


def _future(highs, lows):
    return pd.DataFrame(
        {"open": [100.0] * len(highs), "high": highs, "low": lows, "close": [100.0] * len(highs)},
        index=pd.date_range("2026-01-01", periods=len(highs), freq="5min", tz="UTC"),
    )


def test_resolve_trade_uses_first_subsequent_touch():
    frame = _future([101.0, 102.1], [99.5, 99.5])
    result, result_r, closed, ambiguous = _resolve_trade(
        frame, "LONG", 99.0, 102.0, frame.index[-1]
    )
    assert (result, result_r, ambiguous) == ("win", 2.0, False)
    assert closed == frame.index[-1]


def test_resolve_trade_counts_same_candle_as_loss():
    frame = _future([102.1], [98.9])
    result, result_r, _, ambiguous = _resolve_trade(
        frame, "LONG", 99.0, 102.0, frame.index[-1]
    )
    assert (result, result_r, ambiguous) == ("loss", -1.0, True)


def test_resolve_trade_uses_configured_reward_r():
    frame = _future([101.3], [99.5])
    result, result_r, _, _ = _resolve_trade(
        frame, "LONG", 99.0, 101.25, frame.index[-1], reward_r=1.25
    )
    assert (result, result_r) == ("win", 1.25)


def test_backtest_summary_round_trip_and_format(tmp_path):
    summary = BacktestSummary(
        generated_at=datetime.now(timezone.utc).isoformat(),
        strategy_version="v1",
        period_start="2026-01-01T00:00:00+00:00",
        period_end="2026-01-31T00:00:00+00:00",
        m5_candles=5000,
        evaluations=4500,
        signals=3,
        wins=2,
        losses=1,
        expired=0,
        win_rate=66.666,
        total_r=3.0,
        profit_factor=4.0,
        max_drawdown_r=1.0,
        max_candidate_score=80,
    )
    path, _ = save_backtest(Path(tmp_path), summary, [])
    loaded = load_latest_summary(path)
    assert loaded == summary
    text = format_backtest(loaded)
    assert "66.7%" in text
    assert "+3.0R" in text



def test_backtest_summary_format_shows_new_fields(tmp_path):
    summary = BacktestSummary(
        generated_at=datetime.now(timezone.utc).isoformat(),
        strategy_version="v1",
        period_start="2026-01-01T00:00:00+00:00",
        period_end="2026-01-31T00:00:00+00:00",
        m5_candles=5000,
        evaluations=4500,
        signals=10,
        wins=6,
        losses=4,
        expired=0,
        win_rate=60.0,
        total_r=8.0,
        profit_factor=2.0,
        max_drawdown_r=1.5,
        max_candidate_score=80,
        warmup_bars=500,
        monte_carlo_p_value=0.02,
        regime_breakdown={"normal": 8, "high_vol": 2},
    )
    text = format_backtest(summary)
    assert "P-value: 0.020" in text
    assert "Total: +8.0R" in text
    assert "DD 1.5R" in text


def test_monte_carlo_pvalue_deterministic_and_bounded():
    trades = [
        BacktestTrade(
            opened_at="2026-01-01T00:00:00+00:00",
            closed_at="2026-01-01T01:00:00+00:00",
            direction="LONG",
            entry=100.0,
            stop_loss=99.0,
            take_profit=102.0,
            score=70,
            macro_bias="up",
            m15_structure="bullish",
            m5_structure="bullish",
            rsi=50.0,
            liquidity_event=None,
            fvg=None,
            candle_pattern=None,
            result="win" if i % 2 == 0 else "loss",
            result_r=2.0 if i % 2 == 0 else -1.0,
            ambiguous=False,
        )
        for i in range(12)
    ]
    first = monte_carlo_pvalue(trades, iterations=100)
    second = monte_carlo_pvalue(trades, iterations=100)
    assert first == second  # seeded -> deterministic
    assert 0.0 <= first <= 1.0
    assert first < 1.0


def test_monte_carlo_pvalue_none_for_few_trades():
    assert monte_carlo_pvalue([]) is None


def test_cached_replay_respects_lookback_without_deleting_history(tmp_path):
    cache = tmp_path / "cache"
    cache.mkdir()
    now = pd.Timestamp.now(tz="UTC").floor("5min")
    index = pd.date_range(now - pd.Timedelta(days=120), now, periods=121)
    frame = pd.DataFrame(
        {
            "open": range(121),
            "high": range(1, 122),
            "low": range(121),
            "close": range(1, 122),
        },
        index=index,
    )
    for timeframe in ("M5", "H1", "H4", "D1"):
        frame.to_csv(cache / f"XAUUSD_{timeframe}.csv")

    frames, fetched = load_backtest_data(cache, lookback_days=30)

    assert fetched == []
    assert frames["M5"].index.min() >= now - pd.Timedelta(days=30)
    assert len(pd.read_csv(cache / "XAUUSD_M5.csv")) == 121


def test_monte_carlo_pvalue_detects_strong_positive_expectancy():
    trades = [
        BacktestTrade(
            opened_at="2026-01-01T00:00:00+00:00",
            closed_at="2026-01-01T01:00:00+00:00",
            direction="LONG",
            entry=100.0,
            stop_loss=99.0,
            take_profit=102.0,
            score=70,
            macro_bias="up",
            m15_structure="bullish",
            m5_structure="bullish",
            rsi=50.0,
            liquidity_event=None,
            fvg=None,
            candle_pattern=None,
            result="win" if i < 10 else "loss",
            result_r=2.0 if i < 10 else -1.0,
            ambiguous=False,
        )
        for i in range(12)
    ]
    assert monte_carlo_pvalue(trades, iterations=5000) < 0.05
