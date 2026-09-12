"""Momentum candle analysis engine for XAUUSD (M5 / M15).

Supports two versions:
  - momentum_v1: Baseline candle-action momentum strategy.
  - momentum_v2: Improved momentum with session filter (London/NY killzones),
                 exhaustion/climax cap, rejection wick filter, strict M15 trend
                 alignment, and adaptive risk-to-reward ratio.

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


def momentum_candle_v2(
    m5: pd.DataFrame,
    m15: pd.DataFrame,
    score_threshold: int = 80,
    reward_r: float = 1.25,
    session_filter: bool = True,
    max_body_atr: float = 2.2,
    max_wick_ratio: float = 0.30,
) -> MomentumReading:
    """Improved momentum signal (momentum_v2).

    Key enhancements:
      1. Session Filter: Focus on London and New York sessions (07:00-17:00 UTC).
         Filters out Asian and rollover low-liquidity fakeouts.
      2. Exhaustion/Climax Filter: Blocks candles with body > 2.2x ATR (news spikes/exhaustion).
      3. Rejection Wick Filter: Blocks candles with opposing wick > 30% of total range.
      4. Strict M15 Trend Confluence: M15 structure MUST align with candle direction.
      5. Enhanced Risk-to-Reward: Default 1.25R.
    """
    if len(m5) < 35 or len(m15) < 20:
        raise ValueError("insufficient candles for momentum candle evaluation")

    m5 = m5.copy().dropna(subset=["open", "high", "low", "close"])
    m15 = m15.copy().dropna(subset=["open", "high", "low", "close"])

    price = float(m5.iloc[-1]["close"])
    atr_val = _atr(m5)
    rsi_val = _rsi(m5)
    regime = volatility_regime(m5["close"])

    last = m5.iloc[-1]
    body_ratio = _candle_body_ratio(last)
    body_size = abs(float(last["close"]) - float(last["open"]))
    total_range = float(last["high"]) - float(last["low"])
    is_bullish = float(last["close"]) > float(last["open"])
    direction = "long" if is_bullish else "short"

    # --- 1. Session Filter ---
    last_timestamp = m5.index[-1]
    utc_time = last_timestamp.tz_convert("UTC") if last_timestamp.tzinfo else last_timestamp.tz_localize("UTC")
    in_session = (7 <= utc_time.hour <= 17) and (utc_time.dayofweek < 5)

    # --- 2. Rejection Wick Check ---
    rejection_wick = False
    upper_wick = float(last["high"]) - max(float(last["open"]), float(last["close"]))
    lower_wick = min(float(last["open"]), float(last["close"])) - float(last["low"])
    if total_range > 0:
        if is_bullish and (upper_wick / total_range) > max_wick_ratio:
            rejection_wick = True
        elif not is_bullish and (lower_wick / total_range) > max_wick_ratio:
            rejection_wick = True

    # --- 3. Body vs ATR (Exhaustion / Climax) ---
    body_atr = (body_size / atr_val) if atr_val > 0 else 0.0
    is_exhaustion = body_atr > max_body_atr

    # --- 4. M15 Structure ---
    m15_struct = market_structure(m15)
    m15_aligned = m15_struct.direction == ("bullish" if is_bullish else "bearish")

    # --- 5. EMA alignment (fast 12 / slow 26) ---
    ema_fast = float(m5["close"].ewm(span=12, adjust=False).mean().iloc[-1])
    ema_slow = float(m5["close"].ewm(span=26, adjust=False).mean().iloc[-1])
    ema_aligned = (ema_fast > ema_slow) if is_bullish else (ema_fast < ema_slow)

    # --- 6. Volume ---
    vol_df = m5 if "volume" in m5.columns else pd.DataFrame()
    volume_aligned = False
    if not vol_df.empty and vol_df["volume"].astype(float).nunique() > 1:
        from .analysis import obv as obv_fn

        obv_series = obv_fn(vol_df)
        if len(obv_series.dropna()) >= 25:
            slope = float(obv_series.iloc[-1]) - float(obv_series.iloc[-25])
            volume_aligned = (slope > 0) if is_bullish else (slope < 0)

    # --- Scoring ---
    score = 0
    reasons: List[str] = []
    cautions: List[str] = []

    if in_session:
        reasons.append(f"High-liquidity session ({utc_time.hour:02d}:00 UTC)")
    elif session_filter:
        cautions.append("Outside London/NY session (07:00-17:00 UTC)")

    if body_ratio >= 0.6:
        score += 35
        reasons.append(f"M5 strong {'bullish' if is_bullish else 'bearish'} candle (body {body_ratio:.0%})")
    elif body_ratio >= 0.45:
        score += 20
        reasons.append(f"M5 solid {'bullish' if is_bullish else 'bearish'} candle (body {body_ratio:.0%})")

    if 0.3 <= body_atr <= max_body_atr:
        score += 20
        reasons.append(f"M5 body {body_atr:.1f}x ATR (healthy momentum)")
    elif 0.15 <= body_atr < 0.3:
        score += 10
        reasons.append(f"M5 body {body_atr:.1f}x ATR")
    elif is_exhaustion:
        cautions.append(f"Exhaustion / climax candle ({body_atr:.1f}x ATR > {max_body_atr}x limit)")

    if rejection_wick:
        cautions.append(f"Rejection wick opposes direction (>{max_wick_ratio:.0%})")

    if ema_aligned:
        score += 20
        reasons.append(f"M5 EMA 12/26 {'bullish' if is_bullish else 'bearish'} alignment")
    else:
        cautions.append("EMA 12/26 does not confirm candle direction")

    if is_bullish and rsi_val >= 55:
        score += 20 if rsi_val >= 65 else 10
        reasons.append(f"M5 RSI bullish momentum ({rsi_val:.1f})")
    elif not is_bullish and rsi_val <= 45:
        score += 20 if rsi_val <= 35 else 10
        reasons.append(f"M5 RSI bearish momentum ({rsi_val:.1f})")

    if m15_aligned:
        score += 15
        reasons.append(f"M15 {'bullish' if is_bullish else 'bearish'} structure aligned")
    else:
        cautions.append(f"M15 structure is not {direction}")

    if volume_aligned:
        score += 10
        reasons.append("Volume confirms momentum")

    if regime == "high_vol":
        cautions.append("High-volatility regime")

    # Hard Gates for v2
    meaningful_body = body_ratio >= 0.45 and body_atr >= 0.20
    gates_passed = (
        meaningful_body
        and (not is_exhaustion)
        and (not rejection_wick)
        and (in_session or not session_filter)
        and m15_aligned
        and ema_aligned
        and score >= score_threshold
    )

    action = direction.upper() if gates_passed else "WAIT"

    # Entry / SL / TP
    entry = stop = target = None
    risk_reward = reward_r
    if action == "LONG":
        entry = price
        stop = float(last["low"]) - atr_val * 0.3
        target = price + (price - stop) * risk_reward
    elif action == "SHORT":
        entry = price
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
        strategy_version="momentum_v2",
    )


def evaluate_momentum(
    m5: pd.DataFrame,
    m15: pd.DataFrame,
    strategy_version: str = "momentum_v2",
    reward_r: Optional[float] = None,
    session_filter: bool = True,
) -> MomentumReading:
    """Evaluate momentum with the specified version."""
    if strategy_version == "momentum_v1":
        r = reward_r if reward_r is not None else 1.0
        return momentum_candle(m5, m15, reward_r=r)
    r = reward_r if reward_r is not None else 1.25
    return momentum_candle_v2(m5, m15, reward_r=r, session_filter=session_filter)

