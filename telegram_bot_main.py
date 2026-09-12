#!/usr/bin/env python3
"""Long-poll Telegram callback buttons for the XAUUSD analysis bot."""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, Optional

import requests

from forex.config import Config, TelegramConfig
from forex.notify import mode_keyboard, send_telegram, stats_keyboard
from forex.backtest import format_backtest, load_latest_summary
from forex.llm import generate_commentary
from forex.product import (
    BOT_DESCRIPTION,
    BOT_SHORT_DESCRIPTION,
    DEFAULT_STRATEGY_VERSION,
    bot_name,
)
from forex.tracking import SignalTracker, format_stats_footer
from forex.usage import tracker_from_env


logger = logging.getLogger("xauusd.telegram")
_TELEGRAM_API = "https://api.telegram.org"


def _post(config: TelegramConfig, method: str, payload: Dict[str, Any], timeout: int = 20):
    response = requests.post(
        f"{_TELEGRAM_API}/bot{config.bot_token}/{method}",
        json=payload,
        timeout=timeout,
    )
    response.raise_for_status()
    return response.json()


def _answer(config: TelegramConfig, callback_id: str, text: str, alert: bool = False) -> None:
    _post(
        config,
        "answerCallbackQuery",
        {"callback_query_id": callback_id, "text": text, "show_alert": alert},
    )


def _help_text() -> str:
    return "\n".join(
        [
            f"🥇 {bot_name().upper()}",
            "Asisten analisis momentum XAUUSD—tanpa auto-trading.",
            "",
            "📋 DAFTAR PERINTAH",
            "• /mode — Pengaturan status sinyal (ON/OFF) & versi strategi",
            "• /stats — Rekap statistik forward test real",
            "• /backtest — Laporan historical backtest (v1 & v2)",
            "• /usage — Pemantauan kuota API Twelve Data",
            "• /ai — Analisis naratif kondisi market (on-demand)",
            "• /help — Panduan ini",
            "",
            "Semua sinyal dicatat dan dinilai otomatis (TP/SL).",
            "Bot tidak mengeksekusi transaksi.",
        ]
    )


def _mode_text(tracker: SignalTracker) -> str:
    db_enabled = tracker.get_state("signal_enabled")
    is_enabled = (db_enabled != "off") if db_enabled is not None else True
    db_strat = tracker.get_state("signal_strategy") or DEFAULT_STRATEGY_VERSION
    if db_strat == "all":
        strat_label = "🌟 SEMUA AKTIF (v1 + v3)"
    elif db_strat == "momentum_v3":
        strat_label = "🎯 Momentum MTF (v3)"
    else:
        strat_label = "⚡ Momentum Standar (v1)"
    status_label = "🟢 AKTIF (ON)" if is_enabled else "🔴 NONAKTIF (OFF)"
    return "\n".join([
        "⚙️ PENGATURAN STRATEGI",
        "━━━━━━━━━━━━━━━━━━━━",
        f"Status Sinyal : {status_label}",
        f"Strategi Aktif: {strat_label}",
        "",
        "Pilihan Versi:",
        "• all (Multi)     : Kedua strategi aktif bersamaan secara paralel",
        "• v1 (Standar)    : M5 candle action, EMA 12/26, 1.0R (Terbukti +28.0R)",
        "• v3 (Two Candles): M30 2-candle bias, M5 swing retracement entry, 1.5R",
        "━━━━━━━━━━━━━━━━━━━━",
        "Tekan tombol di bawah untuk mengubah pengaturan:",
    ])


def _send_mode(config: TelegramConfig, tracker: SignalTracker) -> None:
    db_enabled = tracker.get_state("signal_enabled")
    is_enabled = (db_enabled != "off") if db_enabled is not None else True
    db_strat = tracker.get_state("signal_strategy") or DEFAULT_STRATEGY_VERSION
    send_telegram(_mode_text(tracker), config, reply_markup=mode_keyboard(enabled=is_enabled, strategy=db_strat))


def _update_mode_message(
    config: TelegramConfig,
    tracker: SignalTracker,
    chat_id: str,
    message_id: int,
) -> None:
    db_enabled = tracker.get_state("signal_enabled")
    is_enabled = (db_enabled != "off") if db_enabled is not None else True
    db_strat = tracker.get_state("signal_strategy") or DEFAULT_STRATEGY_VERSION
    try:
        _post(
            config,
            "editMessageText",
            {
                "chat_id": chat_id,
                "message_id": message_id,
                "text": _mode_text(tracker),
                "reply_markup": mode_keyboard(enabled=is_enabled, strategy=db_strat),
            },
        )
    except Exception:
        _send_mode(config, tracker)


def _format_backtest_card(label: str, summary: Optional[Any]) -> str:
    if summary is None:
        return f"{label}\n• Status: Belum tersedia"
    win_rate = "Belum tersedia" if summary.win_rate is None else f"{summary.win_rate:.1f}%"
    factor = "—" if summary.profit_factor is None else f"{summary.profit_factor:.2f}"
    p_val = "—" if summary.monte_carlo_p_value is None else f"{summary.monte_carlo_p_value:.3f}"
    return "\n".join([
        label,
        f"• Sinyal   : {summary.signals} (✅ {summary.wins} · ❌ {summary.losses} · ⌛ {summary.expired})",
        f"• Win Rate : {win_rate}",
        f"• Total R  : {summary.total_r:+.1f}R",
        f"• PF / DD  : {factor} / {summary.max_drawdown_r:.1f}R",
        f"• P-value  : {p_val}",
    ])


def _backtest_text() -> str:
    state_dir = Path(os.getenv("FOREX_STATE_DIR", "/var/lib/xauusd-analysis"))
    root = state_dir / "backtest"
    v1_summary = load_latest_summary(root / "momentum_v1" / "latest.json")
    v3_summary = load_latest_summary(root / "momentum_v3" / "latest.json")

    period = "90 hari terakhir"
    if v1_summary:
        period = f"{v1_summary.period_start[:10]} → {v1_summary.period_end[:10]}"
    elif v3_summary:
        period = f"{v3_summary.period_start[:10]} → {v3_summary.period_end[:10]}"

    lines = [
        f"🧪 {bot_name().upper()} — BACKTEST",
        "━━━━━━━━━━━━━━━━━━━━",
        f"📅 Periode: {period}",
        "",
        _format_backtest_card("⚡ Momentum Standar (v1)", v1_summary),
        "",
        _format_backtest_card("🎯 Momentum MTF 2-Candle (v3)", v3_summary),
        "━━━━━━━━━━━━━━━━━━━━",
        "Hasil historis bukan jaminan performa berikutnya.",
    ]
    return "\n".join(lines)


def _usage_text() -> str:
    tracker = tracker_from_env()
    if tracker is None:
        return "Pemantauan Twelve Data belum dikonfigurasi."
    usage = tracker.summary()
    pct = (usage.estimated_credits / usage.daily_limit * 100) if usage.daily_limit else 0.0
    filled = min(10, max(0, int(pct / 10)))
    bar = "■" * filled + "□" * (10 - filled)
    return "\n".join(
        [
            "📡 PEMAKAIAN TWELVE DATA",
            "━━━━━━━━━━━━━━━━━━━━",
            f"📅 {usage.day_utc} (UTC)",
            f"💳 {usage.estimated_credits} / {usage.daily_limit} kredit ({pct:.1f}%)",
            f"📊 [{bar}]",
            f"🛡️ Sisa aman: {usage.remaining_estimate} kredit",
            f"🔄 Request: {usage.requests} (Gagal: {usage.failures})",
            "━━━━━━━━━━━━━━━━━━━━",
            "Dua API key otomatis rotasi & failover.",
        ]
    )


def _load_latest_payload() -> Optional[Dict[str, Any]]:
    """Read the latest computed analysis state written by the pipeline."""
    path = os.getenv("FOREX_STATE_PATH", "").strip()
    if not path:
        state_dir = Path(os.getenv("FOREX_STATE_DIR", "/var/lib/xauusd-analysis"))
        path = str(state_dir / "latest.json")
    try:
        with open(path, "r", encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, json.JSONDecodeError):
        return None


def _ai_text() -> str:
    """Generate AI commentary over the latest computed payload (on demand)."""
    config = Config.from_env()
    if not config.llm.enabled:
        return (
            "🤖 AI belum dikonfigurasi (tidak ada API key). "
            "Analisis deterministik tetap berjalan; AI hanya on-demand."
        )
    payload = _load_latest_payload()
    if not payload or not payload.get("pairs"):
        return (
            "Belum ada hasil analisis untuk dijelaskan. "
            "Jalankan `python main.py` dulu (analisis terjadwal otomatis), "
            "lalu minta AI lagi."
        )
    try:
        metadata: Dict[str, Any] = {}
        commentary = generate_commentary(payload, config.llm, metadata=metadata)
    except Exception as exc:
        logger.warning("AI commentary failed (%s)", type(exc).__name__)
        return "🤖 Gagal menghasilkan analisis AI. Coba lagi nanti."
    if not commentary:
        return "🤖 Penyedia AI tidak membalas. Coba lagi nanti."
    provider = ""
    if metadata.get("provider"):
        provider = f"\nSumber: {metadata['provider']} · {metadata.get('model', 'model default')}"
    return f"🤖 ANALISIS AI (XAUUSD & pasangan){provider}\n\n{commentary}"


def _send_command(parts: list[str], config: TelegramConfig, tracker: SignalTracker) -> None:
    command = parts[0].lower()
    if command in ("/start", "/help"):
        send_telegram(_help_text(), config)
    elif command in ("/mode", "/strategy", "/settings", "/setting", "/menu"):
        if len(parts) > 1:
            arg = parts[1].lower()
            if arg in ("on", "enable", "1"):
                tracker.set_state("signal_enabled", "on")
                send_telegram("🟢 Sinyal berhasil DIAKTIFKAN (ON).", config)
                _send_mode(config, tracker)
            elif arg in ("off", "disable", "0"):
                tracker.set_state("signal_enabled", "off")
                send_telegram("🔴 Sinyal berhasil DINONAKTIFKAN (OFF).", config)
                _send_mode(config, tracker)
            elif arg in ("v1", "momentum_v1"):
                tracker.set_state("signal_strategy", "momentum_v1")
                send_telegram("⚡ Strategi aktif diubah ke: Momentum Standar (v1).", config)
                _send_mode(config, tracker)
            elif arg in ("v3", "momentum_v3", "v3_improved", "pro"):
                tracker.set_state("signal_strategy", "momentum_v3")
                send_telegram("🎯 Strategi aktif diubah ke: Momentum MTF 2-Candle (v3).", config)
                _send_mode(config, tracker)
            elif arg in ("all", "multi", "semua"):
                tracker.set_state("signal_strategy", "all")
                send_telegram("🌟 Strategi aktif diubah ke: SEMUA AKTIF (v1 + v3).", config)
                _send_mode(config, tracker)
            else:
                _send_mode(config, tracker)
        else:
            _send_mode(config, tracker)
    elif command in ("/status", "/stats"):
        send_telegram(format_stats_footer(tracker.stats()), config)
    elif command == "/backtest":
        send_telegram(_backtest_text(), config)
    elif command == "/usage":
        send_telegram(_usage_text(), config)
    elif command == "/ai":
        send_telegram(_ai_text(), config)


def handle_message(message: Dict[str, Any], config: TelegramConfig, tracker: SignalTracker) -> None:
    chat_id = str((message.get("chat") or {}).get("id", ""))
    if chat_id != str(config.chat_id):
        return
    text = str(message.get("text", "")).strip()
    if not text:
        return
    parts = text.split()
    command = parts[0].split("@", 1)[0].lower()
    if command.startswith("/"):
        parts[0] = command
        _send_command(parts, config, tracker)


def handle_callback(
    query: Dict[str, Any],
    config: TelegramConfig,
    tracker: SignalTracker,
) -> None:
    callback_id = str(query.get("id", ""))
    message = query.get("message") or {}
    chat_id = str(message.get("chat", {}).get("id", ""))
    if chat_id != str(config.chat_id):
        _answer(config, callback_id, "Chat ini tidak diizinkan.", alert=True)
        return

    data = str(query.get("data", ""))
    if data == "mode":
        _answer(config, callback_id, "Membuka pengaturan mode")
        _send_mode(config, tracker)
        return
    if data == "mode:toggle":
        db_enabled = tracker.get_state("signal_enabled")
        new_state = "off" if (db_enabled != "off") else "on"
        tracker.set_state("signal_enabled", new_state)
        label = "🔴 NONAKTIF (OFF)" if new_state == "off" else "🟢 AKTIF (ON)"
        _answer(config, callback_id, f"Status: {label}")
        msg_id = message.get("message_id")
        if msg_id:
            _update_mode_message(config, tracker, chat_id, msg_id)
        else:
            _send_mode(config, tracker)
        return
    if data.startswith("strat:"):
        strat = data.split(":", 1)[1]
        if strat in ("momentum_v1", "momentum_v3", "all", "momentum_v3_improved", "momentum_v2"):
            if strat in ("momentum_v3_improved", "momentum_v2"):
                strat = "momentum_v3"
            tracker.set_state("signal_strategy", strat)
            if strat == "all":
                label = "🌟 Semua Aktif (v1 + v3)"
            elif strat == "momentum_v3":
                label = "🎯 Momentum v3 (Two Candles)"
            else:
                label = "⚡ Momentum v1 (Standar)"
            _answer(config, callback_id, f"Strategi: {label}")
            msg_id = message.get("message_id")
            if msg_id:
                _update_mode_message(config, tracker, chat_id, msg_id)
            else:
                _send_mode(config, tracker)
            return
    if data == "ai":
        _answer(config, callback_id, "Menghasilkan analisis AI…")
        send_telegram(_ai_text(), config)
        return
    if data == "stats":
        _answer(config, callback_id, "Statistik diperbarui")
        send_telegram(format_stats_footer(tracker.stats()), config)
        return
    if data == "backtest":
        _answer(config, callback_id, "Membuka backtest terakhir")
        send_telegram(_backtest_text(), config)
        return
    if data == "help":
        _answer(config, callback_id, "Membuka bantuan")
        send_telegram(_help_text(), config)
        return
    _answer(config, callback_id, "Tombol tidak dikenali.", alert=True)


def main() -> int:
    config = Config.from_env()
    logging.basicConfig(
        level=getattr(logging, config.log_level, logging.INFO),
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
    )
    if not config.telegram.enabled:
        logger.error("Telegram bot token or chat id is not configured")
        return 2
    tracker = SignalTracker.from_env()
    if tracker is None:
        logger.error("SIGNAL_TRACKING_DB is not configured")
        return 2

    offset_value = tracker.get_state("telegram_update_offset")
    offset = int(offset_value) if offset_value else None
    try:
        _post(config.telegram, "setMyName", {"name": bot_name()})
        _post(config.telegram, "setMyDescription", {"description": BOT_DESCRIPTION})
        _post(config.telegram, "setMyShortDescription", {"short_description": BOT_SHORT_DESCRIPTION})
        _post(
            config.telegram,
            "setMyCommands",
            {
                "commands": [
                    {"command": "mode", "description": "Atur status ON/OFF & versi strategi"},
                    {"command": "stats", "description": "Rekap statistik forward test real"},
                    {"command": "backtest", "description": "Hasil backtest komparasi v1 & v2"},
                    {"command": "usage", "description": "Pemantauan kuota Twelve Data"},
                    {"command": "ai", "description": "Analisis AI pasar terkini (on-demand)"},
                    {"command": "help", "description": "Panduan & cara memakai bot"},
                ]
            },
        )
    except Exception as exc:
        logger.warning("Unable to update Telegram profile (%s)", type(exc).__name__)

    logger.info("Telegram command and button listener started")
    while True:
        payload: Dict[str, Any] = {
            "timeout": 45,
            "allowed_updates": ["callback_query", "message"],
        }
        if offset is not None:
            payload["offset"] = offset
        try:
            response = _post(config.telegram, "getUpdates", payload, timeout=55)
            for update in response.get("result", []):
                callback = update.get("callback_query")
                if callback:
                    try:
                        handle_callback(callback, config.telegram, tracker)
                    except Exception as exc:
                        logger.warning("Callback handling failed (%s)", type(exc).__name__)
                message = update.get("message")
                if message:
                    try:
                        handle_message(message, config.telegram, tracker)
                    except Exception as exc:
                        logger.warning("Message handling failed (%s)", type(exc).__name__)
                offset = int(update["update_id"]) + 1
                tracker.set_state("telegram_update_offset", str(offset))
        except Exception as exc:
            logger.warning("Telegram polling failed (%s)", type(exc).__name__)
            time.sleep(5)


if __name__ == "__main__":
    raise SystemExit(main())
