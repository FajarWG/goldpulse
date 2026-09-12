"""Momentum candle analysis engine for XAUUSD (M5 / M15 / M30).

Supports three versions:
  - momentum_v1: Baseline candle-action momentum strategy.
  - momentum_v3: Multi-timeframe 2-candle momentum strategy (M30 roadmap + M5
                 50% retracement entry) with 1:2 risk-to-reward ratio.
  - momentum_v3_improved: MTF 2-candle Pro with M30 macro trend filter (EMA 50),
                          volatility regime gating, NY open spike filter, and anti-deep retrace.

No broker, account, position sizing, or order execution code belongs here.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from .analysis import volatility_regime


@dataclass(frozen=True)
class Structure:
    direction: str
    break_type: Optional[str]
    last_swing_high: Optional[float]
    last_swing_low: Optional[float]


@dataclass(frozen=True)
class MomentumReading:
    """Lightweight signal from pure candle-action momentum scoring."""
    action: str
    score: int
    price: float
    entry: Optional[float]
    stop_loss: Optional[float]
    take_profit: Optional[float]
    risk_reward: float
    rsi: float
    atr: float
    candle_pattern: Optional[str]
    m5_body_ratio: float
    ema_aligned: bool
    m15_aligned: bool
    volume_aligned: bool
    regime: str
    reasons: List[str]
    cautions: List[str]
    strategy_version: str = "momentum_v1"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)



def _swings(frame: pd.DataFrame, lookback: int = 3) -> Tuple[List[Tuple[int, float]], List[Tuple[int, float]]]:
    window = lookback * 2 + 1
    high_values = frame["high"].astype(float).reset_index(drop=True)
    low_values = frame["low"].astype(float).reset_index(drop=True)
    high_mask = high_values.eq(high_values.rolling(window, center=True).max()).fillna(False)
    low_mask = low_values.eq(low_values.rolling(window, center=True).min()).fillna(False)
    highs = [(int(index), float(high_values.iloc[index])) for index in high_values.index[high_mask]]
    lows = [(int(index), float(low_values.iloc[index])) for index in low_values.index[low_mask]]
    return highs, lows


def market_structure(frame: pd.DataFrame) -> Structure:
    if len(frame) < 35:
        return Structure("neutral", None, None, None)
    recent = frame.tail(100).reset_index(drop=True)
    highs, lows = _swings(recent)
    if len(highs) < 2 or len(lows) < 2:
        return Structure("neutral", None, None, None)

    previous_high, last_high = highs[-2][1], highs[-1][1]
    previous_low, last_low = lows[-2][1], lows[-1][1]
    close = float(recent.iloc[-1]["close"])
    ema_fast = float(recent["close"].ewm(span=10, adjust=False).mean().iloc[-1])
    ema_slow = float(recent["close"].ewm(span=30, adjust=False).mean().iloc[-1])

    bullish_sequence = last_high > previous_high and last_low > previous_low
    bearish_sequence = last_high < previous_high and last_low < previous_low
    bullish_break = close > last_high
    bearish_break = close < last_low

    if bullish_break and ema_fast > ema_slow:
        return Structure("bullish", "BOS", last_high, last_low)
    if bearish_break and ema_fast < ema_slow:
        return Structure("bearish", "BOS", last_high, last_low)
    if bullish_sequence and ema_fast > ema_slow:
        return Structure("bullish", None, last_high, last_low)
    if bearish_sequence and ema_fast < ema_slow:
        return Structure("bearish", None, last_high, last_low)
    return Structure("neutral", None, last_high, last_low)


def _atr(frame: pd.DataFrame, period: int = 14) -> float:
    previous_close = frame["close"].shift(1)
    true_range = pd.concat(
        [
            frame["high"] - frame["low"],
            (frame["high"] - previous_close).abs(),
            (frame["low"] - previous_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    value = true_range.ewm(alpha=1 / period, adjust=False).mean().iloc[-1]
    return float(value)


def _rsi(frame: pd.DataFrame, period: int = 14) -> float:
    delta = frame["close"].diff()
    gains = delta.clip(lower=0).ewm(alpha=1 / period, adjust=False).mean()
    losses = (-delta.clip(upper=0)).ewm(alpha=1 / period, adjust=False).mean()
    loss = float(losses.iloc[-1])
    if loss == 0:
        return 100.0
    value = 100 - 100 / (1 + float(gains.iloc[-1]) / loss)
    return float(value)


def candle_pattern(frame: pd.DataFrame) -> Optional[str]:
    if len(frame) < 2:
        return None
    previous, current = frame.iloc[-2], frame.iloc[-1]
    previous_bullish = float(previous["close"]) > float(previous["open"])
    current_bullish = float(current["close"]) > float(current["open"])
    if not previous_bullish and current_bullish and float(current["open"]) <= float(previous["close"]) and float(current["close"]) >= float(previous["open"]):
        return "bullish_engulfing"
    if previous_bullish and not current_bullish and float(current["open"]) >= float(previous["close"]) and float(current["close"]) <= float(previous["open"]):
        return "bearish_engulfing"
    body = abs(float(current["close"]) - float(current["open"]))
    if body == 0:
        return None
    upper_wick = float(current["high"]) - max(float(current["open"]), float(current["close"]))
    lower_wick = min(float(current["open"]), float(current["close"])) - float(current["low"])
    if lower_wick >= body * 2 and upper_wick <= body:
        return "hammer"
    if upper_wick >= body * 2 and lower_wick <= body:
        return "shooting_star"
    return None




def _candle_body_ratio(candle: pd.Series) -> float:
    """Body / total-range ratio of a single candle. 1.0 = full-body marubozu."""
    total = float(candle["high"]) - float(candle["low"])
    if total <= 0:
        return 0.0
    body = abs(float(candle["close"]) - float(candle["open"]))
    return body / total


def momentum_candle(
    m5: pd.DataFrame,
    m15: pd.DataFrame,
    score_threshold: int = 80,
    reward_r: float = 1.0,
) -> MomentumReading:
    """Lightweight momentum signal based on candle action — no macro bias needed.

    The last closed M5 candle sets the direction; every other factor confirms
    that direction (not a separate long/short race). ``high_vol`` is a caution
    only, never a block: strong momentum often happens inside volatility spikes.

    Confirmation scoring (max 100):
      M5 candle body ratio ≥ 0.6 (≥ 0.4 partial)   +35 / +18
      M5 candle body vs ATR ≥ 0.3 (≥ 0.15 partial) +20 / +10
      RSI momentum with candle direction            +20 / +10
      EMA 12/26 aligned with candle direction       +20
      M15 structure aligned with candle direction   +15

    A meaningful candle body is required and EMA must agree with the candle
    direction; trend alone cannot trigger it.
    """
    if len(m5) < 35 or len(m15) < 20:
        raise ValueError("insufficient candles for momentum candle evaluation")

    m5 = m5.copy().dropna(subset=["open", "high", "low", "close"])
    m15 = m15.copy().dropna(subset=["open", "high", "low", "close"])

    price = float(m5.iloc[-1]["close"])
    atr_val = _atr(m5)
    rsi_val = _rsi(m5)
    regime = volatility_regime(m5["close"])

    # --- M5 candle analysis: the candle direction drives the signal ---
    last = m5.iloc[-1]
    body_ratio = _candle_body_ratio(last)
    body_size = abs(float(last["close"]) - float(last["open"]))
    is_bullish = float(last["close"]) > float(last["open"])
    is_bearish = not is_bullish
    direction = "long" if is_bullish else "short"

    # --- M15 structure ---
    m15_struct = market_structure(m15)
    m15_aligned = m15_struct.direction == ("bullish" if is_bullish else "bearish")

    # --- EMA alignment (fast 12 / slow 26) ---
    ema_fast = float(m5["close"].ewm(span=12, adjust=False).mean().iloc[-1])
    ema_slow = float(m5["close"].ewm(span=26, adjust=False).mean().iloc[-1])
    ema_aligned = (ema_fast > ema_slow) if is_bullish else (ema_fast < ema_slow)

    # --- Volume ---
    vol_df = m5 if "volume" in m5.columns else pd.DataFrame()
    volume_aligned = False
    if not vol_df.empty and vol_df["volume"].astype(float).nunique() > 1:
        from .analysis import obv as obv_fn
        obv_series = obv_fn(vol_df)
        if len(obv_series.dropna()) >= 25:
            slope = float(obv_series.iloc[-1]) - float(obv_series.iloc[-25])
            volume_aligned = (slope > 0) if is_bullish else (slope < 0)

    # --- Confirmation scoring: the candle sets direction, the rest agrees ---
    score = 0
    reasons: List[str] = []
    cautions: List[str] = []

    # 1. Candle body ratio
    if body_ratio >= 0.6:
        score += 35
        reasons.append(f"M5 strong {'bullish' if is_bullish else 'bearish'} candle (body {body_ratio:.0%})")
    elif body_ratio >= 0.4:
        score += 18
        reasons.append(f"M5 decent {'bullish' if is_bullish else 'bearish'} candle (body {body_ratio:.0%})")

    # 2. Candle directional body size (relative to ATR)
    body_atr = 0.0
    if atr_val > 0:
        body_atr = body_size / atr_val
        if body_atr >= 0.3:
            score += 20
            reasons.append(f"M5 {'bullish' if is_bullish else 'bearish'} body {body_atr:.1f}x ATR")
        elif body_atr >= 0.15:
            score += 10
            reasons.append(f"M5 {'bullish' if is_bullish else 'bearish'} body {body_atr:.1f}x ATR")

    # 3. EMA alignment
    if ema_aligned:
        score += 20
        reasons.append(f"M5 EMA 12/26 {'bullish' if is_bullish else 'bearish'} alignment")

    # 4. RSI momentum confirmation (trend-following, not contrarian)
    if is_bullish and rsi_val >= 55:
        score += 20 if rsi_val >= 65 else 10
        reasons.append(f"M5 RSI bullish momentum ({rsi_val:.1f})")
    elif is_bearish and rsi_val <= 45:
        score += 20 if rsi_val <= 35 else 10
        reasons.append(f"M5 RSI bearish momentum ({rsi_val:.1f})")

    # 5. M15 structure alignment
    if m15_aligned:
        score += 15
        reasons.append(f"M15 {'bullish' if is_bullish else 'bearish'} structure{f' {m15_struct.break_type}' if m15_struct.break_type else ''}")

    meaningful_body = body_ratio >= 0.4 or body_atr >= 0.15

    if regime == "high_vol":
        cautions.append("high-volatility regime; momentum relies on strong price action")

    action = "WAIT"
    if meaningful_body and score >= score_threshold and ema_aligned:
        action = direction.upper()
    elif not meaningful_body:
        cautions.append(f"last M5 candle body too small for a momentum signal (body {body_ratio:.0%})")
    elif not ema_aligned:
        cautions.append("EMA 12/26 does not confirm the candle direction")
    else:
        cautions.append(f"momentum score {score}/100 below {score_threshold} threshold")

    # Entry / SL / TP
    entry = stop = target = None
    risk_reward = reward_r
    if action == "LONG":
        entry = price
        # SL below the signal candle low minus ATR buffer
        stop = float(last["low"]) - atr_val * 0.3
        target = price + (price - stop) * risk_reward
    elif action == "SHORT":
        entry = price
        # SL above the signal candle high plus ATR buffer
        stop = float(last["high"]) + atr_val * 0.3
        target = price - (stop - price) * risk_reward

    return MomentumReading(
        action=action,
        score=min(score, 100),
        price=price,
        entry=entry,
        stop_loss=stop,
        take_profit=target,
        risk_reward=risk_reward,
        rsi=rsi_val,
        atr=atr_val,
        candle_pattern=candle_pattern(m5),
        m5_body_ratio=round(body_ratio, 3),
        ema_aligned=ema_aligned,
        m15_aligned=m15_aligned,
        volume_aligned=volume_aligned,
        regime=regime,
        reasons=reasons,
        cautions=cautions,
        strategy_version="momentum_v1",
    )


def momentum_candle_v3_improved(
    m5: pd.DataFrame,
    m30: Optional[pd.DataFrame] = None,
    score_threshold: int = 80,
    reward_r: float = 2.0,
    session_filter: bool = True,
    max_opposing_wick_ratio: float = 0.35,
    min_c1_body_ratio: float = 0.45,
    avoid_open_hour: bool = True,
    macro_ema_period: int = 50,
) -> MomentumReading:
    """Optimized multi-timeframe 2-candle momentum strategy (momentum_v3_improved).

    Built on empirical findings from 133 backtest trades:
      1. HTF 2-Candle Roadmap (M30):
         - C1 impulsive + C2 continuation with low opposing wick (<35%).
      2. M30 Macro Trend Confluence (EMA 50):
         - Long only when C2 close >= M30 EMA 50.
         - Short only when C2 close <= M30 EMA 50.
         - Solves directional asymmetry (historically Short 56.5% WR vs Long 41.1% WR).
      3. Anti-Deep Retracement:
         - Invalidates setup if M5 breaks past C1 invalidation boundary (C1 low for Long, C1 high for Short).
      4. Volatility Regime Gate:
         - Rejects setups during high-volatility regimes (backtest showed 31.2% WR, -1.0R in high_vol).
      5. NY Open Spike Filter:
         - Filters out hour 14:00 UTC (Wall Street open whipsaw, backtest showed 37.5% WR).
      6. 50% Equilibrium Retracement + M5 confirmation candle.
      7. Risk Management:
         - Stop Loss placed outside C2 range (+ 0.2 ATR buffer).
         - Target 2.0R default.
    """
    if len(m5) < 35:
        raise ValueError("insufficient candles for momentum candle evaluation")

    m5 = m5.copy().dropna(subset=["open", "high", "low", "close"])
    if m30 is None:
        from .providers import resample
        m30 = resample(m5, "30min")
    else:
        m30 = m30.copy().dropna(subset=["open", "high", "low", "close"])

    if len(m30) < 5:
        raise ValueError("insufficient candles for momentum candle evaluation")

    c1 = m30.iloc[-2]
    c2 = m30.iloc[-1]
    c2_timestamp = m30.index[-1]

    c1_open = float(c1["open"])
    c1_close = float(c1["close"])
    c1_high = float(c1["high"])
    c1_low = float(c1["low"])
    c1_range = c1_high - c1_low
    c1_body = abs(c1_close - c1_open)
    c1_body_ratio = (c1_body / c1_range) if c1_range > 0 else 0.0

    c2_open = float(c2["open"])
    c2_close = float(c2["close"])
    c2_high = float(c2["high"])
    c2_low = float(c2["low"])
    c2_range = c2_high - c2_low
    c2_body = abs(c2_close - c2_open)
    c2_body_ratio = (c2_body / c2_range) if c2_range > 0 else 0.0

    atr30 = _atr(m30)
    atr5 = _atr(m5)
    rsi5 = _rsi(m5)
    regime = volatility_regime(m5["close"])

    c1_is_bullish = c1_close > c1_open
    c1_is_bearish = c1_close < c1_open
    c1_impulsive = (c1_body_ratio >= min_c1_body_ratio) and (c1_body >= 0.35 * atr30 if atr30 > 0 else True)

    c2_is_bullish = c2_close > c2_open
    c2_is_bearish = c2_close < c2_open

    upper_wick_c2 = c2_high - max(c2_open, c2_close)
    lower_wick_c2 = min(c2_open, c2_close) - c2_low
    opposing_wick_c2 = upper_wick_c2 if c2_is_bullish else lower_wick_c2
    opposing_wick_ratio = (opposing_wick_c2 / c2_range) if c2_range > 0 else 0.0
    c2_no_rejection = opposing_wick_ratio <= max_opposing_wick_ratio

    bullish_setup = (
        c1_is_bullish
        and c1_impulsive
        and c2_is_bullish
        and c2_no_rejection
        and (c2_close >= c1_close or c2_high >= c1_high)
    )
    bearish_setup = (
        c1_is_bearish
        and c1_impulsive
        and c2_is_bearish
        and c2_no_rejection
        and (c2_close <= c1_close or c2_low <= c1_low)
    )

    htf_bias = "LONG" if bullish_setup else ("SHORT" if bearish_setup else None)

    # Session & Hour check
    last_m5 = m5.iloc[-1]
    last_timestamp = m5.index[-1]
    in_session = True
    is_ny_open_spike = False
    if isinstance(last_timestamp, pd.Timestamp):
        utc_time = last_timestamp.tz_convert("UTC") if last_timestamp.tzinfo else last_timestamp.tz_localize("UTC")
        in_session = (7 <= utc_time.hour <= 17) and (utc_time.dayofweek < 5)
        is_ny_open_spike = (utc_time.hour == 14) and avoid_open_hour

    # M30 Macro Trend Confluence (EMA 50 on M30)
    ema_span = min(macro_ema_period, len(m30))
    m30_ema = float(m30["close"].ewm(span=ema_span, adjust=False).mean().iloc[-1])
    macro_aligned = False
    if htf_bias == "LONG":
        macro_aligned = c2_close >= m30_ema
    elif htf_bias == "SHORT":
        macro_aligned = c2_close <= m30_ema

    # 50% Equilibrium level of C2
    c2_mid = (c2_high + c2_low) / 2.0

    # Determine M5 bars of the current developing M30 candle
    if isinstance(m5.index, pd.DatetimeIndex) and isinstance(c2_timestamp, pd.Timestamp):
        curr_m5_bars = m5.loc[m5.index > c2_timestamp]
    else:
        curr_m5_bars = m5.tail(6)

    if curr_m5_bars.empty:
        curr_m5_bars = m5.tail(1)

    m5_close = float(last_m5["close"])
    m5_open = float(last_m5["open"])
    m5_high = float(last_m5["high"])
    m5_low = float(last_m5["low"])
    m5_range = m5_high - m5_low
    m5_body = abs(m5_close - m5_open)
    m5_body_ratio = (m5_body / m5_range) if m5_range > 0 else 0.0

    # Retracement & Anti-deep retrace check
    retrace_happened = False
    deep_retrace = False
    m5_confirmed = False
    if htf_bias == "LONG":
        min_low = float(curr_m5_bars["low"].min())
        retrace_happened = min_low <= c2_mid
        deep_retrace = min_low < c1_low
        m5_confirmed = m5_close > m5_open
    elif htf_bias == "SHORT":
        max_high = float(curr_m5_bars["high"].max())
        retrace_happened = max_high >= c2_mid
        deep_retrace = max_high > c1_high
        m5_confirmed = m5_close < m5_open

    ema_fast = float(m5["close"].ewm(span=12, adjust=False).mean().iloc[-1])
    ema_slow = float(m5["close"].ewm(span=26, adjust=False).mean().iloc[-1])
    ema_aligned = (ema_fast > ema_slow) if (htf_bias == "LONG") else ((ema_fast < ema_slow) if (htf_bias == "SHORT") else False)

    vol_df = m5 if "volume" in m5.columns else pd.DataFrame()
    volume_aligned = False
    if not vol_df.empty and vol_df["volume"].astype(float).nunique() > 1:
        from .analysis import obv as obv_fn
        obv_series = obv_fn(vol_df)
        if len(obv_series.dropna()) >= 25:
            slope = float(obv_series.iloc[-1]) - float(obv_series.iloc[-25])
            volume_aligned = (slope > 0) if (htf_bias == "LONG") else (slope < 0)

    score = 0
    reasons: List[str] = []
    cautions: List[str] = []

    if in_session:
        reasons.append(f"High-liquidity session ({utc_time.hour:02d}:00 UTC)" if isinstance(last_timestamp, pd.Timestamp) else "In session")
        score += 10
    elif session_filter:
        cautions.append("Outside London/NY session (07:00-17:00 UTC)")

    if is_ny_open_spike:
        cautions.append("NY Open 14:00 UTC spike window filtered out")

    if regime == "high_vol":
        cautions.append("High-volatility regime filtered out (empirical loss rate >68%)")
    else:
        score += 10
        reasons.append(f"Favorable volatility regime ({regime})")

    if htf_bias is not None:
        score += 30
        reasons.append(f"M30 2-candle {htf_bias.lower()} roadmap confirmed (C1 body {c1_body_ratio:.0%})")
    else:
        cautions.append("M30 2-candle roadmap setup not met")

    if macro_aligned:
        score += 20
        reasons.append(f"M30 Macro Trend aligned (close vs EMA {ema_span})")
    elif htf_bias is not None:
        cautions.append(f"Opposes M30 Macro Trend (EMA {ema_span})")

    if retrace_happened:
        score += 15
        reasons.append(f"M5 retraced to 50% equilibrium ({c2_mid:.2f})")
    else:
        cautions.append("Waiting for M5 retracement to 50% equilibrium")

    if deep_retrace:
        cautions.append("Deep retracement breached C1 invalidation boundary")

    if m5_confirmed:
        score += 15
        reasons.append(f"M5 candle confirms {htf_bias.lower()} continuation")
    else:
        cautions.append("M5 candle does not confirm continuation")

    if ema_aligned:
        score += 10
        reasons.append("M5 EMA aligns with M30 roadmap")

    # Hard Gates for v3_improved
    gates_passed = (
        (htf_bias in ("LONG", "SHORT"))
        and retrace_happened
        and (not deep_retrace)
        and macro_aligned
        and m5_confirmed
        and (in_session or not session_filter)
        and (not is_ny_open_spike)
        and (regime != "high_vol")
        and score >= score_threshold
    )

    action = htf_bias if gates_passed else "WAIT"

    price = m5_close
    entry = stop = target = None
    risk_reward = reward_r

    if action == "LONG":
        entry = price
        stop = c2_low - atr5 * 0.2
        risk = entry - stop
        if risk > 0:
            target = entry + risk * reward_r
        else:
            action = "WAIT"
            cautions.append("Stop loss is above entry price")
    elif action == "SHORT":
        entry = price
        stop = c2_high + atr5 * 0.2
        risk = stop - entry
        if risk > 0:
            target = entry - risk * reward_r
        else:
            action = "WAIT"
            cautions.append("Stop loss is below entry price")

    return MomentumReading(
        action=action,
        score=min(score, 100),
        price=price,
        entry=entry,
        stop_loss=stop,
        take_profit=target,
        risk_reward=risk_reward,
        rsi=rsi5,
        atr=atr5,
        candle_pattern=candle_pattern(m5),
        m5_body_ratio=round(m5_body_ratio, 3),
        ema_aligned=ema_aligned,
        m15_aligned=(htf_bias is not None),
        volume_aligned=volume_aligned,
        regime=regime,
        reasons=reasons,
        cautions=cautions,
        strategy_version="momentum_v3_improved",
    )


def momentum_candle_v3(
    m5: pd.DataFrame,
    m30: Optional[pd.DataFrame] = None,
    score_threshold: int = 75,
    reward_r: float = 2.0,
    session_filter: bool = True,
    retrace_min_pct: float = 0.50,
    max_opposing_wick_ratio: float = 0.35,
    min_c1_body_ratio: float = 0.45,
) -> MomentumReading:
    """Multi-timeframe 2-candle momentum strategy (momentum_v3).

    Based on the MTF 2-Candle Price Action method:
      1. HTF Roadmap (M30):
         - Identifies the last 2 closed M30 candles.
         - Candle 1: Impulsive candle with strong directional body.
         - Candle 2: Continuation candle in the same direction, with low opposing wick (<35%).
         - Sets HTF bias (LONG or SHORT) and defines the equilibrium (50% pivot) of Candle 2.
      2. LTF Execution (M5):
         - Waits for M5 price to retrace into the discount (for LONG) or premium (for SHORT)
           area of the M30 setup candle (at or beyond 50% equilibrium).
         - Triggers entry on M5 reversal confirmation candle.
      3. Risk Management:
         - Stop Loss placed outside the M30 setup candle range (with ATR buffer).
         - Take Profit with 1:2 risk-to-reward ratio (default 2.0R).
      4. Session Filter:
         - Focus on London & New York high-liquidity sessions (07:00-17:00 UTC).
    """
    if len(m5) < 35:
        raise ValueError("insufficient candles for momentum candle evaluation")

    m5 = m5.copy().dropna(subset=["open", "high", "low", "close"])
    if m30 is None:
        from .providers import resample
        m30 = resample(m5, "30min")
    else:
        m30 = m30.copy().dropna(subset=["open", "high", "low", "close"])

    if len(m30) < 5:
        raise ValueError("insufficient candles for momentum candle evaluation")

    c1 = m30.iloc[-2]
    c2 = m30.iloc[-1]
    c2_timestamp = m30.index[-1]

    c1_open = float(c1["open"])
    c1_close = float(c1["close"])
    c1_high = float(c1["high"])
    c1_low = float(c1["low"])
    c1_range = c1_high - c1_low
    c1_body = abs(c1_close - c1_open)
    c1_body_ratio = (c1_body / c1_range) if c1_range > 0 else 0.0

    c2_open = float(c2["open"])
    c2_close = float(c2["close"])
    c2_high = float(c2["high"])
    c2_low = float(c2["low"])
    c2_range = c2_high - c2_low
    c2_body = abs(c2_close - c2_open)
    c2_body_ratio = (c2_body / c2_range) if c2_range > 0 else 0.0

    atr30 = _atr(m30)
    atr5 = _atr(m5)
    rsi5 = _rsi(m5)
    regime = volatility_regime(m5["close"])

    c1_is_bullish = c1_close > c1_open
    c1_is_bearish = c1_close < c1_open
    c1_impulsive = (c1_body_ratio >= min_c1_body_ratio) and (c1_body >= 0.35 * atr30 if atr30 > 0 else True)

    c2_is_bullish = c2_close > c2_open
    c2_is_bearish = c2_close < c2_open

    upper_wick_c2 = c2_high - max(c2_open, c2_close)
    lower_wick_c2 = min(c2_open, c2_close) - c2_low
    opposing_wick_c2 = upper_wick_c2 if c2_is_bullish else lower_wick_c2
    opposing_wick_ratio = (opposing_wick_c2 / c2_range) if c2_range > 0 else 0.0
    c2_no_rejection = opposing_wick_ratio <= max_opposing_wick_ratio

    bullish_setup = (
        c1_is_bullish
        and c1_impulsive
        and c2_is_bullish
        and c2_no_rejection
        and (c2_close >= c1_close or c2_high >= c1_high)
    )
    bearish_setup = (
        c1_is_bearish
        and c1_impulsive
        and c2_is_bearish
        and c2_no_rejection
        and (c2_close <= c1_close or c2_low <= c1_low)
    )

    htf_bias = "LONG" if bullish_setup else ("SHORT" if bearish_setup else None)

    # Session check
    last_m5 = m5.iloc[-1]
    last_timestamp = m5.index[-1]
    if isinstance(last_timestamp, pd.Timestamp):
        utc_time = last_timestamp.tz_convert("UTC") if last_timestamp.tzinfo else last_timestamp.tz_localize("UTC")
        in_session = (7 <= utc_time.hour <= 17) and (utc_time.dayofweek < 5)
    else:
        in_session = True

    # 50% Equilibrium level of C2
    c2_mid = (c2_high + c2_low) / 2.0

    # Determine M5 bars of the current developing M30 candle
    if isinstance(m5.index, pd.DatetimeIndex) and isinstance(c2_timestamp, pd.Timestamp):
        curr_m5_bars = m5.loc[m5.index > c2_timestamp]
    else:
        curr_m5_bars = m5.tail(6)

    if curr_m5_bars.empty:
        curr_m5_bars = m5.tail(1)

    m5_close = float(last_m5["close"])
    m5_open = float(last_m5["open"])
    m5_high = float(last_m5["high"])
    m5_low = float(last_m5["low"])
    m5_range = m5_high - m5_low
    m5_body = abs(m5_close - m5_open)
    m5_body_ratio = (m5_body / m5_range) if m5_range > 0 else 0.0

    retrace_happened = False
    m5_confirmed = False
    if htf_bias == "LONG":
        retrace_happened = float(curr_m5_bars["low"].min()) <= c2_mid
        m5_confirmed = m5_close > m5_open
    elif htf_bias == "SHORT":
        retrace_happened = float(curr_m5_bars["high"].max()) >= c2_mid
        m5_confirmed = m5_close < m5_open

    ema_fast = float(m5["close"].ewm(span=12, adjust=False).mean().iloc[-1])
    ema_slow = float(m5["close"].ewm(span=26, adjust=False).mean().iloc[-1])
    ema_aligned = (ema_fast > ema_slow) if (htf_bias == "LONG") else ((ema_fast < ema_slow) if (htf_bias == "SHORT") else False)

    vol_df = m5 if "volume" in m5.columns else pd.DataFrame()
    volume_aligned = False
    if not vol_df.empty and vol_df["volume"].astype(float).nunique() > 1:
        from .analysis import obv as obv_fn
        obv_series = obv_fn(vol_df)
        if len(obv_series.dropna()) >= 25:
            slope = float(obv_series.iloc[-1]) - float(obv_series.iloc[-25])
            volume_aligned = (slope > 0) if (htf_bias == "LONG") else (slope < 0)

    score = 0
    reasons: List[str] = []
    cautions: List[str] = []

    if in_session:
        reasons.append(f"High-liquidity session ({utc_time.hour:02d}:00 UTC)" if isinstance(last_timestamp, pd.Timestamp) else "In session")
        score += 15
    elif session_filter:
        cautions.append("Outside London/NY session (07:00-17:00 UTC)")

    if htf_bias is not None:
        score += 35
        reasons.append(f"M30 2-candle {htf_bias.lower()} roadmap confirmed (C1 body {c1_body_ratio:.0%})")
    else:
        cautions.append("M30 2-candle roadmap setup not met")

    if retrace_happened:
        score += 25
        reasons.append(f"M5 retraced to 50% equilibrium ({c2_mid:.2f})")
    else:
        cautions.append("Waiting for M5 retracement to 50% equilibrium")

    if m5_confirmed:
        score += 15
        reasons.append(f"M5 candle confirms {htf_bias.lower()} continuation")
    else:
        cautions.append("M5 candle does not confirm continuation")

    if ema_aligned:
        score += 10
        reasons.append("M5 EMA aligns with M30 roadmap")

    if regime == "high_vol":
        cautions.append("High-volatility regime")

    gates_passed = (
        (htf_bias in ("LONG", "SHORT"))
        and retrace_happened
        and m5_confirmed
        and (in_session or not session_filter)
        and score >= score_threshold
    )

    action = htf_bias if gates_passed else "WAIT"

    price = m5_close
    entry = stop = target = None
    risk_reward = reward_r

    if action == "LONG":
        entry = price
        stop = c2_low - atr5 * 0.2
        risk = entry - stop
        if risk > 0:
            target = entry + risk * reward_r
        else:
            action = "WAIT"
            cautions.append("Stop loss is above entry price")
    elif action == "SHORT":
        entry = price
        stop = c2_high + atr5 * 0.2
        risk = stop - entry
        if risk > 0:
            target = entry - risk * reward_r
        else:
            action = "WAIT"
            cautions.append("Stop loss is below entry price")

    return MomentumReading(
        action=action,
        score=min(score, 100),
        price=price,
        entry=entry,
        stop_loss=stop,
        take_profit=target,
        risk_reward=risk_reward,
        rsi=rsi5,
        atr=atr5,
        candle_pattern=candle_pattern(m5),
        m5_body_ratio=round(m5_body_ratio, 3),
        ema_aligned=ema_aligned,
        m15_aligned=(htf_bias is not None),
        volume_aligned=volume_aligned,
        regime=regime,
        reasons=reasons,
        cautions=cautions,
        strategy_version="momentum_v3",
    )


def evaluate_momentum(
    m5: pd.DataFrame,
    m15: pd.DataFrame,
    strategy_version: str = "momentum_v3_improved",
    reward_r: Optional[float] = None,
    session_filter: bool = True,
    m30: Optional[pd.DataFrame] = None,
) -> MomentumReading:
    """Evaluate momentum with the specified version."""
    if strategy_version == "momentum_v1":
        r = reward_r if reward_r is not None else 1.0
        return momentum_candle(m5, m15, reward_r=r)
    if strategy_version == "momentum_v3":
        r = reward_r if reward_r is not None else 2.0
        return momentum_candle_v3(
            m5,
            m30=m30,
            reward_r=r,
            session_filter=session_filter,
        )
    # momentum_v3_improved (and fallback for legacy momentum_v2)
    r = reward_r if reward_r is not None else 2.0
    return momentum_candle_v3_improved(
        m5,
        m30=m30,
        reward_r=r,
        session_filter=session_filter,
    )


