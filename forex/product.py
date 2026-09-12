"""Product identity shared by reports, Telegram, and documentation."""

from __future__ import annotations

import os


def bot_name() -> str:
    return os.getenv("BOT_DISPLAY_NAME", "WG GoldPulse").strip() or "WG GoldPulse"


DEFAULT_STRATEGY_VERSION = "momentum_v2"
CURRENT_STRATEGY_VERSION = DEFAULT_STRATEGY_VERSION
STRATEGY_VERSIONS = ("momentum_v1", "momentum_v2")


BOT_DESCRIPTION = (
    "Asisten analisis momentum XAUUSD (M5/M15), validasi hasil signal, "
    "historical backtest, AI commentary, dan kontrol strategi lewat Telegram. "
    "Tidak melakukan transaksi otomatis."
)

BOT_SHORT_DESCRIPTION = "Analisis dan validasi signal momentum XAUUSD—tanpa auto-trading."

