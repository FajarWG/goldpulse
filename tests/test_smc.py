import numpy as np
import pandas as pd

from signal_main import _momentum_text
from forex.providers import resample
from forex.smc import (
    candle_pattern,
    evaluate_momentum,
    market_structure,
    momentum_candle,
    momentum_candle_v2,
)


def _trend_frame(direction=1, rows=180, frequency="5min", start="2026-01-01 00:00:00"):
    index = pd.date_range(start, periods=rows, freq=frequency, tz="UTC")
    sequence = np.arange(rows, dtype=float)
    close = 2000 + direction * sequence * 0.25 + np.sin(sequence / 3) * 1.5
    open_ = close - direction * 0.15
    high = np.maximum(open_, close) + 0.25
    low = np.minimum(open_, close) - 0.25
    for i in range(rows - 5, rows):
        close[i] = close[i - 1] + direction * 1.0
        open_[i] = close[i] - direction * 0.9
        high[i] = max(open_[i], close[i]) + 0.05
        low[i] = min(open_[i], close[i]) - 0.05
    return pd.DataFrame(
        {
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
        },
        index=index,
    )


def test_market_structure_bullish_and_bearish():
    m5_bull = _trend_frame(direction=1)
    assert market_structure(m5_bull).direction == "bullish"
    m5_bear = _trend_frame(direction=-1)
    assert market_structure(m5_bear).direction == "bearish"


def test_momentum_v1_baseline_confirms_bullish():
    m5 = _trend_frame(direction=1)
    m15 = resample(m5, "15min")
    reading = momentum_candle(m5, m15, score_threshold=70)
    assert reading.action == "LONG"
    assert reading.strategy_version == "momentum_v1"
    assert reading.stop_loss < reading.entry < reading.take_profit
    assert reading.risk_reward == 1.0


def test_momentum_v2_session_filter():
    # Outside London/NY session (ends at 03:00 UTC)
    m5_asian = _trend_frame(direction=1, start="2026-01-01 12:00:00")
    m15_asian = resample(m5_asian, "15min")
    reading = momentum_candle_v2(m5_asian, m15_asian, session_filter=True)
    assert reading.action == "WAIT"
    assert any("Outside London/NY session" in c for c in reading.cautions)

    # In London/NY session (ends at 15:00 UTC)
    m5_london = _trend_frame(direction=1, start="2026-01-01 00:00:00")
    m15_london = resample(m5_london, "15min")
    reading_london = momentum_candle_v2(m5_london, m15_london, score_threshold=70, session_filter=True)
    assert reading_london.action == "LONG"
    assert reading_london.strategy_version == "momentum_v2"


def test_momentum_v2_exhaustion_cap():
    m5 = _trend_frame(direction=1, start="2026-01-01 00:00:00")
    # Spike the last candle to 5x normal size
    last_idx = m5.index[-1]
    m5.loc[last_idx, "open"] = 2000.0
    m5.loc[last_idx, "close"] = 2025.0
    m5.loc[last_idx, "high"] = 2026.0
    m5.loc[last_idx, "low"] = 1999.0
    m15 = resample(m5, "15min")
    reading = momentum_candle_v2(m5, m15, max_body_atr=2.2)
    assert reading.action == "WAIT"
    assert any("Exhaustion" in c for c in reading.cautions)


def test_momentum_v2_rejection_wick_filter():
    m5 = _trend_frame(direction=1, start="2026-01-01 00:00:00")
    # Add a massive upper rejection wick to a bullish candle
    last_idx = m5.index[-1]
    m5.loc[last_idx, "open"] = 2000.0
    m5.loc[last_idx, "close"] = 2001.0
    m5.loc[last_idx, "high"] = 2010.0  # huge upper wick
    m5.loc[last_idx, "low"] = 1999.8
    m15 = resample(m5, "15min")
    reading = momentum_candle_v2(m5, m15, max_wick_ratio=0.30)
    assert reading.action == "WAIT"
    assert any("Rejection wick" in c for c in reading.cautions)


def test_evaluate_momentum_dispatcher():
    m5 = _trend_frame(direction=1, start="2026-01-01 00:00:00")
    m15 = resample(m5, "15min")
    r1 = evaluate_momentum(m5, m15, strategy_version="momentum_v1")
    assert r1.strategy_version == "momentum_v1"
    assert r1.risk_reward == 1.0

    r2 = evaluate_momentum(m5, m15, strategy_version="momentum_v2")
    assert r2.strategy_version == "momentum_v2"
    assert r2.risk_reward == 1.25


def test_momentum_telegram_text_formatting():
    m5 = _trend_frame(direction=1, start="2026-01-01 00:00:00")
    m15 = resample(m5, "15min")
    reading = evaluate_momentum(m5, m15, strategy_version="momentum_v2", reward_r=1.25)
    text = _momentum_text(reading)
    assert "Momentum Improved (v2)" in text
    assert "BUY" in text
    assert "Entry:" in text
    assert "SL:" in text and "TP:" in text

