#!/usr/bin/env python3
"""One-shot XAUUSD momentum signal confirmation run for a five-minute systemd timer.

Evaluates momentum strategies (momentum_v1 or momentum_v3) according to the
active Telegram toggle and environment settings.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from forex.config import Config
from forex.instruments import parse_symbol
from forex.notify import send_telegram, signal_keyboard
from forex.product import bot_name
from forex.providers import TwelveDataProvider, resample
from forex.smc import MomentumReading, evaluate_momentum
from forex.tracking import SignalTracker


logger = logging.getLogger("xauusd.signal")


def _atomic_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def _drop_open_candle(frame, minutes: int):
    if frame.empty:
        return frame
    now = datetime.now(timezone.utc)
    index = frame.index
    if index.tz is None:
        last_open = index[-1].to_pydatetime().replace(tzinfo=timezone.utc)
    else:
        last_open = index[-1].to_pydatetime().astimezone(timezone.utc)
    if last_open + timedelta(minutes=minutes) > now:
        return frame.iloc[:-1]
    return frame


def _append_event(path: Path, event: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(event, ensure_ascii=False) + "\n")



def _momentum_text(reading: MomentumReading, generated_at: Optional[datetime] = None) -> str:
    generated_at = generated_at or datetime.now(timezone.utc)
    if reading.action == "LONG":
        action_str = "🟢 BUY (LONG)"
    elif reading.action == "SHORT":
        action_str = "🔴 SELL (SHORT)"
    else:
        action_str = "⏳ WAIT"

    if reading.strategy_version == "momentum_v3":
        version_label = "🎯 Momentum MTF 2-Candle (v3)"
    else:
        version_label = "⚡ Momentum Standar (v1)"
    lines = [
        f"🥇 {bot_name().upper()}",
        "━━━━━━━━━━━━━━━━━━━━",
        f"🎯 {action_str}",
        f"📋 {version_label}",
        f"⏰ {generated_at:%Y-%m-%d %H:%M} UTC",
    ]
    if (
        reading.action in ("LONG", "SHORT")
        and reading.entry is not None
        and reading.stop_loss is not None
        and reading.take_profit is not None
    ):
        sl_dist = abs(reading.entry - reading.stop_loss)
        tp_dist = abs(reading.take_profit - reading.entry)
        sl_sign = "-" if reading.action == "LONG" else "+"
        tp_sign = "+" if reading.action == "LONG" else "-"
        lines.extend([
            "━━━━━━━━━━━━━━━━━━━━",
            f"📍 Entry: {reading.entry:.2f}",
            f"🛑 SL: {reading.stop_loss:.2f} ({sl_sign}{sl_dist:.2f})",
            f"🎯 TP: {reading.take_profit:.2f} ({tp_sign}{tp_dist:.2f})",
            f"⚖️ RR: 1:{reading.risk_reward:.2f}",
        ])
    lines.append("━━━━━━━━━━━━━━━━━━━━")
    return "\n".join(lines)


def main() -> int:
    config = Config.from_env()
    logging.basicConfig(
        level=getattr(logging, config.log_level, logging.INFO),
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
    )
    provider = TwelveDataProvider()
    if not provider.is_available():
        logger.error("TWELVEDATA_API_KEY is not configured")
        return 2

    state_dir = Path(os.getenv("FOREX_STATE_DIR", "/var/lib/xauusd-analysis"))
    tracker = SignalTracker.from_env() or SignalTracker(state_dir / "signals.sqlite3")
    instrument = parse_symbol("XAUUSD")
    m5 = _drop_open_candle(provider.fetch(instrument, "M5", 500), 5)
    m15 = resample(m5, "15min")

    # Resolve previous signals
    stats_before = tracker.stats()
    resolved = tracker.evaluate(
        m5,
        timeout_minutes=int(os.getenv("SIGNAL_TIMEOUT_MINUTES", "240")),
        strategy_version="all",
    )
    stats_after_evaluation = tracker.stats()

    now = datetime.now(timezone.utc)
    event_path = state_dir / "signals" / f"{now.date().isoformat()}.jsonl"
    max_momentum = int(os.getenv("SIGNAL_MAX_MOMENTUM_PER_DAY", "5"))
    max_active_momentum = int(os.getenv("SIGNAL_MAX_ACTIVE_MOMENTUM", "3"))
    cooldown = int(os.getenv("SIGNAL_COOLDOWN_MINUTES", "30"))

    # Mode & Strategy Settings (from Telegram state or env)
    db_enabled = tracker.get_state("signal_enabled")
    if db_enabled is not None:
        signal_enabled = db_enabled.strip().lower() == "on"
    else:
        signal_enabled = os.getenv("SIGNAL_ENABLED", "true").strip().lower() in ("1", "true", "yes", "on")

    db_strategy = tracker.get_state("signal_strategy")
    if db_strategy in ("momentum_v1", "momentum_v3", "all"):
        active_strategy = db_strategy
    elif db_strategy in ("momentum_v2", "momentum_v3_improved"):
        active_strategy = "momentum_v3"
    else:
        active_strategy = os.getenv("SIGNAL_STRATEGY", "momentum_v1").strip()

    # --- Momentum candle signals (single or all) ---
    target_strategies = (
        ("momentum_v1", "momentum_v3")
        if active_strategy == "all"
        else (active_strategy,)
    )
    m30 = resample(m5, "30min")
    strategy_events = []

    for strat in target_strategies:
        default_rr = 1.0 if strat == "momentum_v1" else 2.0
        momentum_reward_r = float(os.getenv("SIGNAL_MOMENTUM_REWARD_R", str(default_rr)))
        reading = evaluate_momentum(
            m5,
            m15,
            strategy_version=strat,
            reward_r=momentum_reward_r,
            m30=m30,
        )
        sig_created = False
        sig_id = None
        sig_notif = False

        if signal_enabled and reading.action in ("LONG", "SHORT") and tracker.can_create(
            now=now,
            max_per_day=max_momentum,
            cooldown_minutes=cooldown,
            signal_type="momentum",
            max_active=max_active_momentum,
            strategy_version=strat,
        ):
            sig_id = tracker.create_signal(
                reading,
                m5.index[-1],
                created_at=now,
                signal_type="momentum",
                strategy_version=strat,
            )
            sig_created = sig_id is not None

        if sig_created and config.telegram.enabled:
            sig_notif = send_telegram(
                _momentum_text(reading, now),
                config.telegram,
                reply_markup=signal_keyboard(),
            )

        strategy_events.append({
            "strategy_version": strat,
            "reading": reading.to_dict(),
            "signal_id": sig_id,
            "created": sig_created,
            "notification_sent": sig_notif,
        })

    any_signal_created = any(item["created"] for item in strategy_events)

    # --- Result notification ---
    result_notification_sent = False
    if resolved and not any_signal_created and config.telegram.enabled:
        changes = []
        won = stats_after_evaluation.wins - stats_before.wins
        lost = stats_after_evaluation.losses - stats_before.losses
        expired = stats_after_evaluation.expired - stats_before.expired
        if won:
            changes.append(f"✅ Benar: +{won}")
        if lost:
            changes.append(f"❌ Salah: +{lost}")
        if expired:
            changes.append(f"⌛ Kedaluwarsa: +{expired}")
        result_notification_sent = send_telegram(
            "\n".join([
                "🔄 HASIL SIGNAL SELESAI DINILAI",
                "━━━━━━━━━━━━━━━━━━━━",
                *changes,
                "",
                "Tekan Statistik untuk melihat rekap lengkap.",
            ]),
            config.telegram,
        )

    # --- Event log ---
    first_ev = strategy_events[0] if strategy_events else {}
    event = {
        "generated_at": now.isoformat(timespec="seconds"),
        "signal_enabled": signal_enabled,
        "active_strategy": active_strategy,
        "strategies": strategy_events,
        "momentum_candle": {
            **(first_ev.get("reading") or {}),
            "signal_id": first_ev.get("signal_id"),
            "created": first_ev.get("created", False),
            "notification_sent": first_ev.get("notification_sent", False),
        },
        "resolved_previous_signals": resolved,
        "tracking_stats": tracker.stats().__dict__,
        "result_notification_sent": result_notification_sent,
    }
    _append_event(event_path, event)
    _atomic_json(state_dir / "latest_signal.json", event)

    logger.info(
        "strategy=%s enabled=%s action=%s score=%s resolved=%d",
        active_strategy,
        signal_enabled,
        momentum_reading.action,
        momentum_reading.score,
        resolved,
    )
    return 0



if __name__ == "__main__":
    raise SystemExit(main())
