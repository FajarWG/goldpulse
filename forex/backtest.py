"""Historical replay of the live XAUUSD strategy using cached market data."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .analysis import align_timeframes, analyse_timeframe
from .instruments import parse_symbol
from .product import CURRENT_STRATEGY_VERSION, DEFAULT_STRATEGY_VERSION
from .providers import TwelveDataProvider, resample
from .smc import (
    evaluate_momentum,
    momentum_candle,
    momentum_candle_v3,
)



@dataclass(frozen=True)
class BacktestTrade:
    opened_at: str
    closed_at: str
    direction: str
    entry: float
    stop_loss: float
    take_profit: float
    score: int
    macro_bias: str
    m15_structure: str
    m5_structure: str
    rsi: float
    liquidity_event: Optional[str]
    fvg: Optional[str]
    candle_pattern: Optional[str]
    result: str
    result_r: float
    ambiguous: bool
    regime: str = "normal"
    bias_source: str = "alignment"
    hysteresis: str = "off"
    volume_confirm: Optional[bool] = None
    signal_type: str = "full"
    ema_aligned: Optional[bool] = None
    m15_aligned: Optional[bool] = None
    volume_aligned: Optional[bool] = None


@dataclass(frozen=True)
class BacktestSummary:
    generated_at: str
    strategy_version: str
    period_start: str
    period_end: str
    m5_candles: int
    evaluations: int
    signals: int
    wins: int
    losses: int
    expired: int
    win_rate: Optional[float]
    total_r: float
    profit_factor: Optional[float]
    max_drawdown_r: float
    max_candidate_score: int
    warmup_bars: int = 0
    monte_carlo_p_value: Optional[float] = None
    regime_breakdown: Dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


_FETCH_BARS = {"H1": 3000, "H4": 1000, "D1": 400}


def _read_frame(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, index_col=0, parse_dates=True)
    index = pd.DatetimeIndex(frame.index)
    frame.index = index.tz_localize("UTC") if index.tz is None else index.tz_convert("UTC")
    return frame.sort_index()


def _write_frame(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.sort_index().to_csv(path)


def load_backtest_data(
    cache_dir: Path,
    refresh: bool = False,
    provider: Optional[TwelveDataProvider] = None,
    lookback_days: int = 90,
) -> Tuple[Dict[str, pd.DataFrame], List[str]]:
    """Load cached candles, fetching each timeframe only when requested/missing."""
    provider = provider or TwelveDataProvider(timeout=60)
    instrument = parse_symbol("XAUUSD")
    frames: Dict[str, pd.DataFrame] = {}
    fetched: List[str] = []
    m5_path = cache_dir / "XAUUSD_M5.csv"
    old_m5 = _read_frame(m5_path) if m5_path.exists() else pd.DataFrame()
    now = pd.Timestamp.now(tz="UTC").floor("5min")
    desired_start = now - timedelta(days=max(int(lookback_days), 14))
    if not refresh and not old_m5.empty:
        frames["M5"] = old_m5.loc[old_m5.index >= desired_start]
    else:
        pieces = [old_m5] if not old_m5.empty else []
        ranges = []
        if old_m5.empty or old_m5.index.min() > desired_start + timedelta(days=1):
            backfill_end = old_m5.index.min() if not old_m5.empty else now
            ranges.append((desired_start, backfill_end))
        if not old_m5.empty:
            ranges.append((old_m5.index.max() - timedelta(days=1), now))
        for range_start, range_end in ranges:
            cursor = range_start
            while cursor < range_end:
                chunk_end = min(cursor + timedelta(days=14), range_end)
                pieces.append(provider.fetch_range(instrument, "M5", cursor, chunk_end))
                cursor = chunk_end
        combined = pd.concat(pieces).sort_index()
        combined = combined[~combined.index.duplicated(keep="last")]
        # Keep older cached history for future research; lookback controls the
        # replay window, not whether historical data is destroyed on refresh.
        _write_frame(m5_path, combined)
        frames["M5"] = combined.loc[combined.index >= desired_start]
        fetched.append("M5")

    for timeframe, bars in _FETCH_BARS.items():
        path = cache_dir / f"XAUUSD_{timeframe}.csv"
        if path.exists() and not refresh:
            frames[timeframe] = _read_frame(path)
            continue
        frame = provider.fetch(instrument, timeframe, bars)
        if path.exists():
            old = _read_frame(path)
            frame = pd.concat([old, frame]).sort_index()
            frame = frame[~frame.index.duplicated(keep="last")].tail(bars)
        _write_frame(path, frame)
        frames[timeframe] = frame
        fetched.append(timeframe)
    return frames, fetched





def _resolve_trade(
    future: pd.DataFrame,
    direction: str,
    stop: float,
    target: float,
    deadline: pd.Timestamp,
    reward_r: float = 2.0,
) -> Tuple[str, float, pd.Timestamp, bool]:
    for timestamp, candle in future.loc[future.index <= deadline].iterrows():
        high, low = float(candle["high"]), float(candle["low"])
        if direction == "LONG":
            hit_target, hit_stop = high >= target, low <= stop
        else:
            hit_target, hit_stop = low <= target, high >= stop
        if hit_target and hit_stop:
            return "loss", -1.0, timestamp, True
        if hit_stop:
            return "loss", -1.0, timestamp, False
        if hit_target:
            return "win", reward_r, timestamp, False
    return "expired", 0.0, deadline, False





def monte_carlo_pvalue(
    trades: List[BacktestTrade],
    iterations: int = 500,
    seed: int = 42,
) -> Optional[float]:
    """One-sided sign-randomisation p-value for positive expectancy.

    Shuffling trade order cannot change total R, so the null distribution keeps
    each trade's absolute R and randomises only its win/loss sign. The returned
    value estimates how often a zero-edge process produces mean R at least as
    large as the observed sample. Returns ``None`` for fewer than eight trades.
    """
    if len(trades) < 8:
        return None
    if iterations < 1:
        raise ValueError("iterations must be at least 1")
    results = np.array([float(t.result_r) for t in trades], dtype=float)
    observed = float(results.mean())
    magnitudes = np.abs(results)
    rng = np.random.default_rng(seed)
    beats = 0
    for _ in range(iterations):
        signs = rng.choice((-1.0, 1.0), size=len(results))
        null_mean = float((magnitudes * signs).mean())
        if null_mean >= observed:
            beats += 1
    return round((beats + 1) / (iterations + 1), 3)


def run_momentum_backtest(
    frames: Dict[str, pd.DataFrame],
    strategy_version: str = "momentum_v1",
    timeout_minutes: int = 240,
    max_per_day: int = 5,
    cooldown_minutes: int = 30,
    max_active: int = 3,
    reward_r: Optional[float] = None,
    session_filter: bool = True,
) -> Tuple[BacktestSummary, List[BacktestTrade]]:
    """Backtest momentum strategies (momentum_v1 or momentum_v3)."""
    default_rr = 1.0 if strategy_version == "momentum_v1" else 2.0
    eff_rr = reward_r if reward_r is not None else default_rr
    m5 = frames["M5"].copy().sort_index()
    now = pd.Timestamp.now(tz="UTC")
    if not m5.empty and m5.index[-1] + timedelta(minutes=5) > now:
        m5 = m5.iloc[:-1]
    m15 = resample(m5, "15min")
    m30 = resample(m5, "30min")
    trades: List[BacktestTrade] = []
    daily_counts: Dict[str, int] = {}
    next_allowed: Optional[pd.Timestamp] = None
    open_end_times: List[pd.Timestamp] = []
    evaluations = 0
    max_score = 0
    warmup_bars = 500
    regime_counts: Dict[str, int] = {}

    for index in range(500, len(m5) - 1):
        timestamp = m5.index[index]
        open_end_times = [end for end in open_end_times if end > timestamp]
        if len(open_end_times) >= max_active:
            continue
        if next_allowed is not None and timestamp < next_allowed:
            continue
        day = timestamp.tz_convert("UTC").date().isoformat()
        if daily_counts.get(day, 0) >= max_per_day:
            continue
        m5_window = m5.iloc[: index + 1].tail(500)
        m15_end = int(m15.index.searchsorted(timestamp, side="right"))
        m15_window = m15.iloc[max(0, m15_end - 200) : m15_end]
        m30_end = int(m30.index.searchsorted(timestamp, side="right"))
        m30_window = m30.iloc[max(0, m30_end - 200) : m30_end]
        if len(m15_window) < 20 or len(m5_window) < 35 or (strategy_version == "momentum_v3" and len(m30_window) < 5):
            continue
        try:
            reading = evaluate_momentum(
                m5_window,
                m15_window,
                strategy_version=strategy_version,
                reward_r=eff_rr,
                session_filter=session_filter,
                m30=m30_window,
            )
        except ValueError:
            continue
        evaluations += 1
        max_score = max(max_score, reading.score)
        if reading.action not in ("LONG", "SHORT"):
            continue

        opened_at = timestamp
        deadline = opened_at + timedelta(minutes=timeout_minutes)
        future = m5.iloc[index + 1 :]
        target = float(reading.take_profit)
        result, result_r, closed_at, ambiguous = _resolve_trade(
            future,
            reading.action,
            float(reading.stop_loss),
            target,
            deadline,
            reward_r=eff_rr,
        )
        trades.append(
            BacktestTrade(
                opened_at=opened_at.isoformat(),
                closed_at=closed_at.isoformat(),
                direction=reading.action,
                entry=float(reading.entry),
                stop_loss=float(reading.stop_loss),
                take_profit=target,
                score=reading.score,
                macro_bias="N/A",
                m15_structure="N/A",
                m5_structure="N/A",
                rsi=round(reading.rsi, 2),
                liquidity_event=None,
                fvg=None,
                candle_pattern=reading.candle_pattern[0] if isinstance(reading.candle_pattern, tuple) else reading.candle_pattern,
                result=result,
                result_r=result_r,
                ambiguous=ambiguous,
                regime=reading.regime,
                signal_type="momentum",
                ema_aligned=reading.ema_aligned,
                m15_aligned=reading.m15_aligned,
                volume_aligned=reading.volume_aligned,
            )
        )
        regime_counts[reading.regime] = regime_counts.get(reading.regime, 0) + 1
        daily_counts[day] = daily_counts.get(day, 0) + 1
        open_end_times.append(closed_at)
        next_allowed = max(opened_at + timedelta(minutes=cooldown_minutes), closed_at)

    wins = sum(trade.result == "win" for trade in trades)
    losses = sum(trade.result == "loss" for trade in trades)
    expired = sum(trade.result == "expired" for trade in trades)
    completed = wins + losses
    total_r = sum(trade.result_r for trade in trades)
    cumulative = 0.0
    peak = 0.0
    max_drawdown = 0.0
    for trade in trades:
        cumulative += trade.result_r
        peak = max(peak, cumulative)
        max_drawdown = max(max_drawdown, peak - cumulative)
    summary = BacktestSummary(
        generated_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        strategy_version=strategy_version,
        period_start=m5.index[0].isoformat(),
        period_end=m5.index[-1].isoformat(),
        m5_candles=len(m5),
        evaluations=evaluations,
        signals=len(trades),
        wins=wins,
        losses=losses,
        expired=expired,
        win_rate=(wins / completed * 100) if completed else None,
        total_r=round(total_r, 2),
        profit_factor=(sum(max(trade.result_r, 0) for trade in trades) / losses) if losses else None,
        max_drawdown_r=round(max_drawdown, 2),
        max_candidate_score=max_score,
        warmup_bars=warmup_bars,
        monte_carlo_p_value=monte_carlo_pvalue(trades),
        regime_breakdown=regime_counts,
    )
    return summary, trades


run_backtest = run_momentum_backtest



def save_backtest(
    output_dir: Path,
    summary: BacktestSummary,
    trades: List[BacktestTrade],
) -> Tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "latest.json"
    temp = summary_path.with_suffix(".json.tmp")
    temp.write_text(json.dumps(summary.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
    temp.replace(summary_path)
    trades_path = output_dir / "latest_trades.csv"
    pd.DataFrame([asdict(trade) for trade in trades]).to_csv(trades_path, index=False)
    return summary_path, trades_path


def format_backtest(summary: BacktestSummary, trades: Optional[List[BacktestTrade]] = None) -> str:
    win_rate = "Belum tersedia" if summary.win_rate is None else f"{summary.win_rate:.1f}%"
    factor = "—" if summary.profit_factor is None else f"{summary.profit_factor:.2f}"
    p_value = "—" if summary.monte_carlo_p_value is None else f"{summary.monte_carlo_p_value:.3f}"
    lines = [
        "🧪 WG GOLDPULSE — BACKTEST",
        "━━━━━━━━━━━━━━━━",
        f"Periode: {summary.period_start[:10]} → {summary.period_end[:10]}",
        f"Evaluasi: {summary.evaluations:,} · M5: {summary.m5_candles:,} candle",
        "",
        f"Signal: {summary.signals} · Win rate: {win_rate}",
        f"✅ {summary.wins}  ❌ {summary.losses}  ⌛ {summary.expired}",
        f"Total: {summary.total_r:+.1f}R · PF {factor} · DD {summary.max_drawdown_r:.1f}R",
        f"P-value: {p_value}",
    ]
    lines.extend([
        "━━━━━━━━━━━━━━━━",
        "Hasil historis bukan jaminan performa berikutnya.",
    ])
    return "\n".join(lines)


def format_comparison(v1: BacktestSummary, v2: BacktestSummary) -> str:
    def rate(value: Optional[float]) -> str:
        return "—" if value is None else f"{value:.1f}%"

    verdict = "V2 LEBIH BAIK" if v2.total_r > v1.total_r else "V2 BELUM LEBIH BAIK"
    return "\n".join(
        [
            "🧪 WG GOLDPULSE — V1 VS V2",
            "━━━━━━━━━━━━━━━━",
            f"Periode: {v2.period_start[:10]} → {v2.period_end[:10]}",
            "",
            "V1",
            f"Signal {v1.signals} · WR {rate(v1.win_rate)} · {v1.total_r:+.1f}R · DD {v1.max_drawdown_r:.1f}R",
            "",
            "V2",
            f"Signal {v2.signals} · WR {rate(v2.win_rate)} · {v2.total_r:+.1f}R · DD {v2.max_drawdown_r:.1f}R",
            "",
            f"Kesimpulan: {verdict}",
            "━━━━━━━━━━━━━━━━",
            "V2 belum dipasang live sebelum lolos validasi out-of-sample.",
        ]
    )


def load_latest_summary(path: Path) -> Optional[BacktestSummary]:
    try:
        return BacktestSummary(**json.loads(path.read_text(encoding="utf-8")))
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        return None
