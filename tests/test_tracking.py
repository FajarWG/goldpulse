from datetime import datetime, timedelta, timezone

import pandas as pd

from forex.smc import MomentumReading
from forex.tracking import SignalTracker, append_stats_footer, format_stats_footer


def _reading(
    direction="LONG",
    entry=100.0,
    stop_loss=None,
    take_profit=None,
    score=85,
    strategy_version="momentum_v1",
):
    sl = stop_loss if stop_loss is not None else (99.0 if direction == "LONG" else 101.0)
    tp = take_profit if take_profit is not None else (102.0 if direction == "LONG" else 98.0)
    return MomentumReading(
        action=direction,
        score=score,
        price=entry,
        entry=entry,
        stop_loss=sl,
        take_profit=tp,
        risk_reward=2.0,
        rsi=50.0,
        atr=1.0,
        candle_pattern="bullish_momentum" if direction == "LONG" else "bearish_momentum",
        m5_body_ratio=0.7,
        ema_aligned=True,
        m15_aligned=True,
        volume_aligned=True,
        regime="normal",
        reasons=["test"],
        cautions=[],
        strategy_version=strategy_version,
    )


def _frame(timestamp, high, low, close=100.0):
    return pd.DataFrame(
        [{"open": 100.0, "high": high, "low": low, "close": close}],
        index=pd.DatetimeIndex([timestamp]),
    )


def test_long_target_is_a_win(tmp_path):
    tracker = SignalTracker(tmp_path / "signals.db")
    candle = datetime(2026, 1, 1, 10, 0, tzinfo=timezone.utc)
    assert tracker.create_signal(_reading(), candle, candle) is not None
    assert tracker.evaluate(_frame(candle + timedelta(minutes=5), 102.1, 99.5), now=candle + timedelta(minutes=5)) == 1
    stats = tracker.stats()
    assert (stats.completed, stats.wins, stats.losses, stats.win_rate, stats.total_r) == (1, 1, 0, 100.0, 2.0)


def test_short_stop_is_a_loss(tmp_path):
    tracker = SignalTracker(tmp_path / "signals.db")
    candle = datetime(2026, 1, 1, 10, 0, tzinfo=timezone.utc)
    tracker.create_signal(_reading("SHORT"), candle, candle)
    tracker.evaluate(_frame(candle + timedelta(minutes=5), 101.1, 99.0), now=candle + timedelta(minutes=5))
    stats = tracker.stats()
    assert (stats.completed, stats.wins, stats.losses, stats.total_r) == (1, 0, 1, -1.0)


def test_same_candle_target_and_stop_is_conservative_loss(tmp_path):
    tracker = SignalTracker(tmp_path / "signals.db")
    candle = datetime(2026, 1, 1, 10, 0, tzinfo=timezone.utc)
    tracker.create_signal(_reading(), candle, candle)
    tracker.evaluate(_frame(candle + timedelta(minutes=5), 102.1, 98.9), now=candle + timedelta(minutes=5))
    assert tracker.stats().losses == 1
    with tracker._connect() as connection:
        assert connection.execute("SELECT ambiguous FROM paper_signals").fetchone()[0] == 1


def test_expired_is_excluded_from_win_rate(tmp_path):
    tracker = SignalTracker(tmp_path / "signals.db")
    candle = datetime(2026, 1, 1, 10, 0, tzinfo=timezone.utc)
    tracker.create_signal(_reading(), candle, candle)
    tracker.evaluate(
        _frame(candle + timedelta(minutes=5), 100.5, 99.5),
        timeout_minutes=240,
        now=candle + timedelta(minutes=241),
    )
    stats = tracker.stats()
    assert stats.expired == 1
    assert stats.completed == 0
    assert stats.win_rate is None
    assert "Belum tersedia" in format_stats_footer(stats)


def test_target_after_timeout_does_not_turn_expired_signal_into_win(tmp_path):
    tracker = SignalTracker(tmp_path / "signals.db")
    candle = datetime(2026, 1, 1, 10, 0, tzinfo=timezone.utc)
    tracker.create_signal(_reading(), candle, candle)
    tracker.evaluate(
        _frame(candle + timedelta(minutes=245), 102.1, 99.5),
        timeout_minutes=240,
        now=candle + timedelta(minutes=245),
    )
    stats = tracker.stats()
    assert stats.expired == 1
    assert stats.wins == 0


def test_active_signal_blocks_another(tmp_path):
    tracker = SignalTracker(tmp_path / "signals.db")
    candle = datetime(2026, 1, 1, 10, 0, tzinfo=timezone.utc)
    tracker.create_signal(_reading(), candle, candle)
    assert tracker.can_create(now=candle + timedelta(hours=1)) is False


def test_strategy_versions_have_isolated_limits_and_statistics(tmp_path):
    path = tmp_path / "signals.db"
    v1_tracker = SignalTracker(path, strategy_version="momentum_v1")
    v2_tracker = SignalTracker(path, strategy_version="momentum_v2")
    v3_tracker = SignalTracker(path, strategy_version="momentum_v3")
    candle = datetime(2026, 1, 1, 10, 0, tzinfo=timezone.utc)
    v1_tracker.create_signal(_reading(strategy_version="momentum_v1"), candle, candle)

    assert v1_tracker.stats().total == 1
    assert v2_tracker.stats().total == 0
    assert v3_tracker.stats().total == 0
    assert v2_tracker.can_create(now=candle + timedelta(minutes=5)) is True
    assert v3_tracker.can_create(now=candle + timedelta(minutes=5)) is True


def test_footer_is_appended_when_database_is_configured(tmp_path, monkeypatch):
    path = tmp_path / "signals.db"
    SignalTracker(path)
    monkeypatch.setenv("SIGNAL_TRACKING_DB", str(path))
    result = append_stats_footer("Laporan")
    assert "Laporan" in result
    assert "✅ Benar: 0" in result
    assert "❌ Salah: 0" in result
    assert append_stats_footer(result) == result


def test_manual_decision_is_saved_once_for_active_signal(tmp_path):
    tracker = SignalTracker(tmp_path / "signals.db")
    candle = datetime(2026, 1, 1, 10, 0, tzinfo=timezone.utc)
    signal_id = tracker.create_signal(_reading(), candle, candle)
    assert tracker.record_decision(signal_id, "take", "123") == "saved"
    assert tracker.record_decision(signal_id, "skip", "123") == "already:take"


def test_state_round_trip(tmp_path):
    tracker = SignalTracker(tmp_path / "signals.db")
    assert tracker.get_state("offset") is None
    tracker.set_state("offset", "42")
    assert tracker.get_state("offset") == "42"


def test_momentum_signal_created_and_tracked(tmp_path):
    tracker = SignalTracker(tmp_path / "signals.db", strategy_version="momentum_v2")
    candle = datetime(2026, 1, 1, 10, 0, tzinfo=timezone.utc)
    reading = _reading(strategy_version="momentum_v2")
    assert tracker.can_create(now=candle, signal_type="momentum")
    assert tracker.create_signal(reading, candle, candle, signal_type="momentum") is not None
    assert tracker.stats_by_type("momentum").active == 1
    assert not tracker.can_create(now=candle, signal_type="momentum")


def test_all_mode_evaluates_and_resolves_all_versions(tmp_path):
    path = tmp_path / "signals.db"
    tracker = SignalTracker(path, strategy_version="all")
    candle = datetime(2026, 1, 1, 10, 0, tzinfo=timezone.utc)
    s1 = tracker.create_signal(_reading(strategy_version="momentum_v1"), candle, candle, strategy_version="momentum_v1")
    s2 = tracker.create_signal(_reading(strategy_version="momentum_v2"), candle, candle, strategy_version="momentum_v2")
    s3 = tracker.create_signal(_reading(strategy_version="momentum_v3"), candle, candle, strategy_version="momentum_v3")
    assert s1 and s2 and s3

    stats_all = tracker.stats()
    assert stats_all.active == 3
    assert stats_all.total == 3

    resolved = tracker.evaluate(_frame(candle + timedelta(minutes=5), 102.5, 99.5), now=candle + timedelta(minutes=5))
    assert resolved == 3
    stats_resolved = tracker.stats()
    assert stats_resolved.completed == 3
    assert stats_resolved.wins == 3
    assert stats_resolved.losses == 0
