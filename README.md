# GoldPulse

GoldPulse adalah asisten analisis pasar dan validasi sinyal momentum XAU/USD (Gold) multi-timeframe untuk Linux/VPS. Sistem mengambil data pasar secara berkala, mengevaluasi momentum candle M5 & konfirmasi tren M15 secara deterministik, memvalidasi hasil sinyal secara otomatis (forward test), menjalankan historical backtest komparatif, menyediakan penjelasan AI opsional, serta memberikan kontrol penuh melalui Telegram bot.

> **⚠️ Catatan Penting:** Proyek ini adalah alat bantu analisis teknikal independen dan **bukan robot trading otomatis**. Sistem tidak terhubung ke akun broker dan tidak melakukan eksekusi order. Semua referensi entry, stop loss, dan take profit harus selalu dicocokkan dengan manajemen risiko masing-masing.

---

## ⚡ Strategi Momentum

Sistem berfokus pada strategi momentum candle murni tanpa ketergantungan indikator lag berlebih:

| Versi | Tipe | Deskripsi & Filter Kunci | Reward:Risk |
| :--- | :--- | :--- | :--- |
| **`momentum_v1`** | Standar | Aksi harga candle M5, body-to-range ratio, alignment EMA 12/26, skor momentum $\ge 80$. | **1.0R** |
| **`momentum_v2`** | Improved | Peningkatan dari v1 dengan **Session Filter** (London & NY killzones 07:00–17:00 UTC), **Exhaustion Cap** (`body_atr <= 2.2`), **Rejection Wick Filter** (`opposing_wick / range <= 0.30`), dan **Strict M15 Trend Confluence** (`m15_aligned`). | **1.25R** |

Keduanya dapat dipilih dan diganti secara instan kapan saja langsung melalui Telegram tanpa perlu me-restart daemon service.

---

## 🚀 Fitur Utama

1. **Sinyal Momentum Candle (M5 / M15):**
   - Timing eksekusi candle M5 dengan konfirmasi struktur pasar M15.
   - Deteksi pola candle, rasio body vs range, filter ekor perlawanan, dan capping volatilitas ekstrem.
   - Sinyal terformat rapi: Aksi (`BUY` / `SELL`), Skor, Entry, Batas Salah (SL), Target (TP), serta alasan teknikal.

2. **Kontrol Interaktif Telegram (`/mode`):**
   - **Toggle On/Off:** Menyalakan atau mematikan pengiriman notifikasi sinyal secara live (`/mode on` atau `/mode off`).
   - **Ganti Strategi:** Berpindah antara `momentum_v1` dan `momentum_v2` secara instan (`/mode v1` atau `/mode v2`).
   - **Inline Keyboard:** Tombol interaktif langsung di Telegram untuk kemudahan kontrol pengguna.
   - Perubahan disimpan di SQLite (`telegram_state`) dan langsung diterapkan pada evaluasi 5-menitan berikutnya.

3. **Forward Test Otomatis (Dynamic R):**
   - Setiap sinyal dicatat dan dievaluasi candle-by-candle secara real-time.
   - Target TP tersentuh lebih dulu: Menang sesuai reward rasio sebenarnya (`+1.0R` atau `+1.25R`).
   - Stop Loss tersentuh lebih dulu: Kalah (`-1.0R`).
   - TP & SL tersentuh pada candle M5 yang sama: Dihitung kalah secara konservatif.
   - Statistik mencakup Win Rate, Total R, status aktif, dan rincian per versi (`/stats`).

4. **Historical Backtest Engine Komparatif:**
   - Replay data historis 90 hari dengan incremental cache Twelve Data.
   - Perhitungan Monte Carlo randomisation p-value, profit factor, win rate, dan maximum drawdown.
   - Perbandingan performa `momentum_v1` vs `momentum_v2` via CLI atau bot Telegram (`/backtest`).

5. **Analisis Pasar Multi-Timeframe (H1, H4, D1):**
   - Pemetaan tren besar berkala setiap jam.
   - Indikator teknikal: EMA 20/50/200, RSI, MACD, ATR, Bollinger Bands, Donchian Channels, Support/Resistance, dan Volatility Regime.

6. **AI Market Commentary On-Demand:**
   - Penjelasan kondisi teknikal pasar menggunakan LLM (urutan fallback: Groq → Gemini → DeepSeek).
   - Berjalan murni **on-demand** (via perintah `/ai` atau tombol 🤖), tidak memakan kuota token saat evaluasi rutin.

7. **Resilient Data Architecture:**
   - Dual Twelve Data API Key dengan auto-failover, pembatasan kuota otomatis, retry mechanism, dan pencatatan pemakaian harian di database SQLite.

---

## 🛠️ Quick Start Lokal

### Prasyarat
- Python 3.10+ (disarankan Python 3.11 atau 3.13)
- Akun Twelve Data API Key (gratis)
- Bot Telegram (via @BotFather)

### Instalasi

```bash
git clone https://github.com/FajarWG/goldpulse.git
cd goldpulse

python3 -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'

cp deploy/xauusd-analysis.env.example .env
```

Sesuaikan konfigurasi di file `.env`:

```dotenv
TWELVEDATA_API_KEY=your_key_1
TWELVEDATA_API_KEY2=your_key_2
TELEGRAM_BOT_TOKEN=your_bot_token
TELEGRAM_CHAT_ID=your_chat_id

# Mode & Strategi Awal
SIGNAL_ENABLED=on
SIGNAL_STRATEGY=momentum_v2
SIGNAL_MOMENTUM_REWARD_R=1.25

# AI Opsional
GROQ_API_KEY=
GEMINI_API_KEY=
DEEPSEEK_API_KEY=
```

### Menjalankan Modul

```bash
# 1. Analisis tren besar H1/H4/D1 ke terminal
python main.py --symbols XAUUSD --timeframes H1,H4,D1 --stdout

# 2. Evaluasi sinyal momentum M5/M15 satu kali (seperti systemd timer)
python signal_main.py

# 3. Jalankan bot Telegram interaktif (listener)
python telegram_bot_main.py

# 4. Replay backtest data historis (komparasi v1 & v2)
python backtest_main.py --strategy all
```

---

## 🖥️ Panduan Perintah Bot Telegram

| Perintah | Deskripsi |
| :--- | :--- |
| `/mode` | Menampilkan panel kontrol mode & status strategi aktif dengan tombol inline. |
| `/mode on` / `/mode off` | Menyalakan atau menonaktifkan pengiriman sinyal momentum. |
| `/mode v1` / `/mode v2` | Mengganti strategi aktif ke `momentum_v1` (Standar) atau `momentum_v2` (Improved). |
| `/stats` | Melihat statistik performa forward test real beserta breakdown per versi. |
| `/backtest` | Melihat laporan ringkasan komparasi historical backtest terbaru. |
| `/ai` | Meminta analisis kondisi teknikal pasar saat ini menggunakan AI. |
| `/help` | Menampilkan panduan penggunaan dan daftar perintah. |

---

## 🌐 Instalasi di Ubuntu/Debian VPS

```bash
sudo mkdir -p /opt/xauusd-analysis /var/lib/xauusd-analysis
sudo chown ubuntu:ubuntu /opt/xauusd-analysis /var/lib/xauusd-analysis

git clone https://github.com/FajarWG/goldpulse.git /opt/xauusd-analysis
cd /opt/xauusd-analysis
python3 -m venv .venv
.venv/bin/pip install -e .

# Konfigurasi Environment VPS
sudo install -o root -g root -m 0600 \
  deploy/xauusd-analysis.env.example /etc/xauusd-analysis.env
sudoedit /etc/xauusd-analysis.env

# Pasang Systemd Services & Timers
sudo install -m 0644 deploy/*.service deploy/*.timer /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now \
  xauusd-analysis.timer \
  xauusd-signal.timer \
  xauusd-backtest.timer \
  xauusd-telegram.service
```

### Memeriksa Status Layanan

```bash
systemctl status xauusd-analysis.timer xauusd-signal.timer xauusd-backtest.timer
systemctl status xauusd-telegram.service
journalctl -u xauusd-signal.service -n 50 --no-pager
```

---

## 🧪 Testing

Project dilengkapi dengan automated test suite offline deterministik tanpa ketergantungan jaringan eksternal:

```bash
.venv/bin/pytest -v
```

Semua pengujian mencakup verifikasi matematis indikator teknikal, deteksi momentum candle v1 & v2, dynamic R tracking, format notifikasi Telegram, dan kalkulasi Monte Carlo p-value.

---

## 📁 Struktur Repository

```text
├── forex/
│   ├── analysis.py          # Indikator teknikal multi-timeframe & kalkulasi ATR/RSI/EMA
│   ├── backtest.py          # Engine replay historis komparatif v1 vs v2 & simulasi trade
│   ├── config.py            # Parser konfigurasi lingkungan (.env)
│   ├── instruments.py       # Parser simbol instrumen
│   ├── llm.py               # Integrasi AI multi-provider (Groq, Gemini, DeepSeek)
│   ├── notify.py            # Formatter notifikasi Telegram & inline keyboards
│   ├── pipeline.py          # Orkestrasi analisis pasar berkala
│   ├── product.py           # Konstanta strategi, versi, dan display name bot
│   ├── providers.py         # Client Twelve Data, caching, resample & quota limiter
│   ├── report.py            # Formatter laporan teks & markdown
│   ├── smc.py               # Engine sinyal momentum candle (v1 & v2)
│   └── tracking.py          # SQLite forward test tracker & dynamic R calculator
├── deploy/                  # Template systemd timer/service dan contoh file konfigurasi
├── docs/                    # Dokumentasi panduan operasional (GUIDE.md)
├── tests/                   # Test suite unit & integrasi offline
├── main.py                  # Runner analisis tren H1/H4/D1
├── signal_main.py           # Runner konfirmasi sinyal momentum M5/M15 (5-menitan)
├── telegram_bot_main.py     # Bot Telegram listener & handler interaktif
├── backtest_main.py         # CLI runner historical replay
├── research_main.py         # Analisis atribut trade historis
├── research_rr.py           # Sweep reward:risk statistik
└── research_walkforward.py  # Walk-forward out-of-sample validation
```

---

## 🔒 Lisensi & Keamanan

- **Keamanan Kredensial:** Jangan pernah melakukan commit untuk file `.env`, file database SQLite (`*.sqlite3`), cache lokal, atau private keys.
- **Sanitasi Log:** Error Twelve Data disanitasi otomatis agar API key tidak terekspos di log aplikasi.
- **Lisensi:** MIT License.

