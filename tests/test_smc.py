import numpy as np
import pandas as pd

from signal_main import _momentum_text
from forex.providers import resample
from forex.smc import (
    candle_pattern,
    evaluate_momentum,
    market_structure,
    momentum_candle,
    momentum_candle_v3,
    momentum_candle_v3_improved,
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


def test_momentum_v3_improved_success():
    m30_idx = pd.date_range("2026-01-01 07:00:00", periods=10, freq="30min", tz="UTC")
    m30_df = pd.DataFrame(
        {
            "open": [2000.0] * 8 + [2000.0, 2010.0],
            "high": [2005.0] * 8 + [2010.0, 2020.0],
            "low": [1995.0] * 8 + [1999.0, 2008.0],
            "close": [2001.0] * 8 + [2010.0, 2019.0],
        },
        index=m30_idx,
    )

    m5_idx = pd.date_range("2026-01-01 08:30:00", periods=40, freq="5min", tz="UTC")
    m5_df = pd.DataFrame(
        {
            "open": [2000.0] * 37 + [2018.0, 2014.0, 2013.0],
            "high": [2002.0] * 37 + [2019.0, 2015.0, 2018.0],
            "low": [1998.0] * 37 + [2013.0, 2012.0, 2012.5],
            "close": [2001.0] * 37 + [2014.0, 2013.0, 2017.0],
        },
        index=m5_idx,
    )
    reading = momentum_candle_v3_improved(m5_df, m30=m30_df, session_filter=True)
    assert reading.action == "LONG"
    assert reading.strategy_version == "momentum_v3_improved"
    assert reading.risk_reward == 2.0
    assert reading.stop_loss < reading.entry < reading.take_profit


def test_momentum_v3_improved_ny_open_spike_filter():
    # 14:00 UTC spike filter
    m30_idx = pd.date_range("2026-01-01 11:00:00", periods=10, freq="30min", tz="UTC")
    m30_df = pd.DataFrame(
        {
            "open": [2000.0] * 8 + [2000.0, 2010.0],
            "high": [2005.0] * 8 + [2010.0, 2020.0],
            "low": [1995.0] * 8 + [1999.0, 2008.0],
            "close": [2001.0] * 8 + [2010.0, 2019.0],
        },
        index=m30_idx,
    )
    # 36 periods from 11:10 UTC ends at 14:05 UTC (hour == 14)
    m5_spike_idx = pd.date_range("2026-01-01 11:10:00", periods=36, freq="5min", tz="UTC")
    m5_spike_df = pd.DataFrame(
        {
            "open": [2000.0] * 33 + [2018.0, 2014.0, 2013.0],
            "high": [2002.0] * 33 + [2019.0, 2015.0, 2018.0],
            "low": [1998.0] * 33 + [2013.0, 2012.0, 2012.5],
            "close": [2001.0] * 33 + [2014.0, 2013.0, 2017.0],
        },
        index=m5_spike_idx,
    )
    reading = momentum_candle_v3_improved(m5_spike_df, m30=m30_df, avoid_open_hour=True)
    assert reading.action == "WAIT"
    assert any("NY Open 14:00 UTC spike" in c for c in reading.cautions)


def test_momentum_v3_improved_anti_deep_retrace():
    m30_idx = pd.date_range("2026-01-01 07:00:00", periods=10, freq="30min", tz="UTC")
    m30_df = pd.DataFrame(
        {
            "open": [2000.0] * 8 + [2000.0, 2010.0],
            "high": [2005.0] * 8 + [2010.0, 2020.0],
            "low": [1995.0] * 8 + [1999.0, 2008.0],
            "close": [2001.0] * 8 + [2010.0, 2019.0],
        },
        index=m30_idx,
    )
    # C1 low is 1999.0. If M5 retrace goes below 1999.0 (e.g. 1995.0), it should be blocked
    m5_idx = pd.date_range("2026-01-01 08:30:00", periods=40, freq="5min", tz="UTC")
    m5_df = pd.DataFrame(
        {
            "open": [2000.0] * 37 + [2018.0, 2014.0, 1996.0],
            "high": [2002.0] * 37 + [2019.0, 2015.0, 2018.0],
            "low": [1998.0] * 37 + [2013.0, 1995.0, 1996.0], # 1995.0 < 1999.0
            "close": [2001.0] * 37 + [2014.0, 1996.0, 2017.0],
        },
        index=m5_idx,
    )
    reading = momentum_candle_v3_improved(m5_df, m30=m30_df)
    assert reading.action == "WAIT"
    assert any("Deep retracement" in c for c in reading.cautions)


def test_evaluate_momentum_dispatcher():
    m5 = _trend_frame(direction=1, start="2026-01-01 00:00:00")
    m15 = resample(m5, "15min")
    r1 = evaluate_momentum(m5, m15, strategy_version="momentum_v1")
    assert r1.strategy_version == "momentum_v1"
    assert r1.risk_reward == 1.0

    r3 = evaluate_momentum(m5, m15, strategy_version="momentum_v3")
    assert r3.strategy_version == "momentum_v3"
    assert r3.risk_reward == 2.0

    r3_pro = evaluate_momentum(m5, m15, strategy_version="momentum_v3_improved")
    assert r3_pro.strategy_version == "momentum_v3_improved"
    assert r3_pro.risk_reward == 2.0

    # Legacy fallback: momentum_v2 routes to momentum_v3_improved
    r2 = evaluate_momentum(m5, m15, strategy_version="momentum_v2")
    assert r2.strategy_version == "momentum_v3_improved"
    assert r2.risk_reward == 2.0


def test_momentum_telegram_text_formatting():
    m5 = _trend_frame(direction=1, start="2026-01-01 00:00:00")
    m15 = resample(m5, "15min")
    reading = evaluate_momentum(m5, m15, strategy_version="momentum_v3_improved", reward_r=2.0)
    text = _momentum_text(reading)
    assert "Momentum MTF Pro (v3-Pro)" in text
    assert "Entry:" in text or "WAIT" in text


def test_momentum_v3_bullish_and_bearish():
    m30_idx = pd.date_range("2026-01-01 07:00:00", periods=10, freq="30min", tz="UTC")
    m30_df = pd.DataFrame(
        {
            "open": [2000.0] * 8 + [2000.0, 2010.0],
            "high": [2005.0] * 8 + [2010.0, 2020.0],
            "low": [1995.0] * 8 + [1999.0, 2008.0],
            "close": [2001.0] * 8 + [2010.0, 2019.0],
        },
        index=m30_idx,
    )

    m5_idx = pd.date_range("2026-01-01 08:30:00", periods=40, freq="5min", tz="UTC")
    m5_df = pd.DataFrame(
        {
            "open": [2000.0] * 37 + [2018.0, 2014.0, 2013.0],
            "high": [2002.0] * 37 + [2019.0, 2015.0, 2018.0],
            "low": [1998.0] * 37 + [2013.0, 2012.0, 2012.5],
            "close": [2001.0] * 37 + [2014.0, 2013.0, 2017.0],
        },
        index=m5_idx,
    )
    reading = momentum_candle_v3(m5_df, m30=m30_df, session_filter=True)
    assert reading.action == "LONG"
    assert reading.strategy_version == "momentum_v3"
    assert reading.risk_reward == 2.0
    assert reading.stop_loss < reading.entry < reading.take_profit
    assert reading.stop_loss < 2008.0


def test_momentum_v3_bearish():
    m30_idx = pd.date_range("2026-01-01 07:00:00", periods=10, freq="30min", tz="UTC")
    m30_df = pd.DataFrame(
        {
            "open": [2000.0] * 8 + [2020.0, 2010.0],
            "high": [2005.0] * 8 + [2021.0, 2012.0],
            "low": [1995.0] * 8 + [2010.0, 2000.0],
            "close": [2001.0] * 8 + [2010.0, 2001.0],
        },
        index=m30_idx,
    )

    m5_idx = pd.date_range("2026-01-01 08:30:00", periods=40, freq="5min", tz="UTC")
    m5_df = pd.DataFrame(
        {
            "open": [2000.0] * 37 + [2002.0, 2006.0, 2007.0],
            "high": [2002.0] * 37 + [2007.5, 2008.0, 2008.0],
            "low": [1998.0] * 37 + [2001.0, 2004.0, 2002.0],
            "close": [2001.0] * 37 + [2006.5, 2007.0, 2003.0],
        },
        index=m5_idx,
    )
    reading = momentum_candle_v3(m5_df, m30=m30_df, session_filter=True)
    assert reading.action == "SHORT"
    assert reading.strategy_version == "momentum_v3"
    assert reading.risk_reward == 2.0
    assert reading.take_profit < reading.entry < reading.stop_loss
    assert reading.stop_loss > 2012.0


def test_momentum_v3_waiting_retracement():
    m30_idx = pd.date_range("2026-01-01 07:00:00", periods=10, freq="30min", tz="UTC")
    m30_df = pd.DataFrame(
        {
            "open": [2000.0] * 8 + [2000.0, 2010.0],
            "high": [2005.0] * 8 + [2010.0, 2020.0],
            "low": [1995.0] * 8 + [1999.0, 2008.0],
            "close": [2001.0] * 8 + [2010.0, 2019.0],
        },
        index=m30_idx,
    )
    m5_idx = pd.date_range("2026-01-01 08:30:00", periods=40, freq="5min", tz="UTC")
    m5_df = pd.DataFrame(
        {
            "open": [2000.0] * 37 + [2018.0, 2018.5, 2019.0],
            "high": [2002.0] * 37 + [2021.0, 2022.0, 2023.0],
            "low": [1998.0] * 37 + [2018.0, 2018.2, 2018.5],
            "close": [2001.0] * 37 + [2019.0, 2020.0, 2021.0],
        },
        index=m5_idx,
    )
    reading = momentum_candle_v3(m5_df, m30=m30_df)
    assert reading.action == "WAIT"
    assert any("Waiting for M5 retracement" in c for c in reading.cautions)


def test_evaluate_momentum_v3_and_formatting():
    m5 = _trend_frame(direction=1, start="2026-01-01 00:00:00")
    m15 = resample(m5, "15min")
    r3 = evaluate_momentum(m5, m15, strategy_version="momentum_v3")
    assert r3.strategy_version == "momentum_v3"
    assert r3.risk_reward == 2.0

    text = _momentum_text(r3)
    assert "Momentum MTF 2-Candle (v3)" in text

