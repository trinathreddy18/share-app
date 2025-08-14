# app.py
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime
from ta.trend import SMAIndicator, EMAIndicator, MACD, ADXIndicator
from ta.momentum import RSIIndicator, StochRSIIndicator
from ta.volatility import AverageTrueRange

# --------------------------- CONFIG ---------------------------
FIB_TOLERANCE = 0.015   # ±1.5% tolerance around Fib 61.8
VOL_SPIKE_MULT = 1.5    # Volume spike threshold vs 20-day avg
CPR_LOOKBACK = 1        # Use previous bar to compute CPR
FIB_LOOKBACK_DAYS = 120 # Lookback for Fib range
RSI_PERIOD = 14
ADX_PERIOD = 14

# --------------------------- UI ---------------------------
st.set_page_config(page_title="📊 Market Scanner – Technical + Fundamental", layout="wide")
st.title("📊 Market Scanner – Technical + Fundamental")

with st.sidebar:
    st.header("⚙️ Scanner Settings")
    default_tickers = "TCS.NS, INFY.NS, RELIANCE.NS, HDFCBANK.NS, ICICIBANK.NS, SBIN.NS, LT.NS, HINDUNILVR.NS, ASIANPAINT.NS"
    tickers_text = st.text_area("Tickers (comma-separated)", value=default_tickers, height=120)
    period = st.selectbox("History period", ["6mo","1y","2y"], index=1)
    interval = st.selectbox("Interval", ["1d","1h"], index=0)

    # Technical thresholds
    st.subheader("Technical Thresholds")
    adx_min = st.slider("ADX min", 10, 40, 25)
    rsi_floor = st.slider("RSI floor (bullish)", 40, 60, 50)
    rising_days = st.slider("Rising window (days)", 15, 90, 30)
    rising_tol = st.slider("Rising tolerance (down steps allowed)", 0, 15, 6)

    # Fundamental thresholds
    st.subheader("Fundamental Thresholds")
    mcap_min_cr = st.number_input("Min Market Cap (₹ Cr)", value=1000, step=100)
    pe_max = st.number_input("Max P/E (0=ignore)", value=0, step=1)
    roe_min_pct = st.number_input("Min ROE % (0=ignore)", value=0, step=1)
    de_max = st.number_input("Max D/E (0=ignore)", value=0.0, step=0.1, format="%.1f")

    # Scoring weights
    st.subheader("Scoring Weights")
    w_trend = st.slider("Trend components (SMA/EMA/MACD/RSI)", 0, 5, 3)
    w_adx = st.slider("ADX / DI+", 0, 5, 2)
    w_macd_cross = st.slider("MACD Bull Cross", 0, 5, 2)
    w_rsi_bull = st.slider("RSI > floor", 0, 5, 1)
    w_vol_spike = st.slider("Volume Spike", 0, 5, 1)
    w_stoch = st.slider("StochRSI Buy", 0, 5, 1)
    w_cpr = st.slider("Price > CPR Top", 0, 5, 1)
    w_fib = st.slider("Fib 61.8 Bounce", 0, 5, 1)
    w_ao = st.slider("AO Bullish", 0, 5, 1)
    w_fund = st.slider("Fundamental pass", 0, 5, 2)

    run_btn = st.button("🚀 Run Scan")

# --------------------------- Helpers ---------------------------
@st.cache_data(show_spinner=False)
def download_history_multi(tickers, period, interval):
    """
    Use yfinance to download multiple tickers in one call.
    Returns the raw df (single or multi-index columns depending on number of tickers).
    """
    if not tickers:
        return pd.DataFrame()
    return yf.download(tickers, period=period, interval=interval, auto_adjust=True, progress=False, group_by='ticker')

def compute_indicators(df):
    """Compute indicators for a single-ticker DataFrame. Returns df with new columns."""
    if df is None or df.empty:
        return pd.DataFrame()

    df = df.copy()

    # Safeguard - ensure numeric types
    for c in ["Open", "High", "Low", "Close", "Volume"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # SMA / EMA
    if "Close" in df.columns:
        df["SMA20"] = SMAIndicator(df["Close"], 20).sma_indicator()
        df["EMA13"] = EMAIndicator(df["Close"], 13).ema_indicator()

    # MACD
    macd_obj = MACD(df["Close"])
    df["MACD"] = macd_obj.macd()
    df["Signal"] = macd_obj.macd_signal()

    # RSI
    rsi_obj = RSIIndicator(df["Close"], RSI_PERIOD)
    df["RSI"] = rsi_obj.rsi()

    # StochRSI (k, d)
    # StochRSI
    stoch_obj = StochRSIIndicator(df["Close"], window=14, smooth1=3, smooth2=3)
    df["StochRSI"] = stoch_obj.stochrsi()
    # build K/D (simple smoothing of the raw stochrsi)
    df["StochRSI_%K"] = df["StochRSI"].rolling(3).mean()
    df["StochRSI_%D"] = df["StochRSI_%K"].rolling(3).mean()
    df["StochRSI_Buy_Signal"] = (
        (df["StochRSI_%K"].shift(1) < df["StochRSI_%D"].shift(1))
        & (df["StochRSI_%K"] > df["StochRSI_%D"])
        & (df["StochRSI_%K"] < 0.8)
    )

    
    # ta's stochrsi returns raw value; we create K/D equivalents using rolling means
    df["StochRSI"] = stoch_obj.stochrsi()
    df["StochRSI_%K"] = df["StochRSI"].rolling(3).mean()
    df["StochRSI_%D"] = df["StochRSI_%K"].rolling(3).mean()
    df["StochRSI_Buy_Signal"] = (df["StochRSI_%K"].shift(1) < df["StochRSI_%D"].shift(1)) & \
                                 (df["StochRSI_%K"] > df["StochRSI_%D"]) & \
                                 (df["StochRSI_%K"] < 0.8)

    # Volume spike vs 20-day avg
    df["Volume Spike"] = df["Volume"] > (VOL_SPIKE_MULT * df["Volume"].rolling(20).mean())

    # ATR
    atr_obj = AverageTrueRange(df["High"], df["Low"], df["Close"], window=14)
    df["ATR"] = atr_obj.average_true_range()

    # CPR (use previous row)
    df["Pivot"] = (df["High"].shift(CPR_LOOKBACK) + df["Low"].shift(CPR_LOOKBACK) + df["Close"].shift(CPR_LOOKBACK)) / 3
    df["CPR_Top"] = (df["High"].shift(CPR_LOOKBACK) + df["Low"].shift(CPR_LOOKBACK)) / 2

    # Fib 61.8 across lookback window
    high_price = df["High"].rolling(FIB_LOOKBACK_DAYS).max()
    low_price = df["Low"].rolling(FIB_LOOKBACK_DAYS).min()
    df["Fib_61.8"] = high_price - (high_price - low_price) * 0.618

    # ADX / DI
    adx_obj = ADXIndicator(df["High"], df["Low"], df["Close"], window=ADX_PERIOD)
    df["ADX"] = adx_obj.adx()
    df["+DI"] = adx_obj.adx_pos()
    df["-DI"] = adx_obj.adx_neg()

    # Awesome Oscillator
    df["MedianPrice"] = (df["High"] + df["Low"]) / 2
    df["AO"] = df["MedianPrice"].rolling(5).mean() - df["MedianPrice"].rolling(34).mean()
    df["AO_Signal"] = np.where(
        (df["AO"] > 0) & (df["AO"].diff() > 0), "🟢 Bullish",
        np.where((df["AO"] < 0) & (df["AO"].diff() < 0), "🔴 Bearish", "⚪ Neutral")
    )

    return df

def is_rising(series: pd.Series, days: int, tolerance: int) -> bool:
    """Return True if the series is mostly rising over 'days' allowing 'tolerance' down steps."""
    if series is None or len(series.dropna()) < days:
        return False
    recent = series.dropna().iloc[-days:]
    dips = sum(1 for a, b in zip(recent, recent[1:]) if b < a)
    return dips <= tolerance

@st.cache_data
def fundamental_snapshot(ticker: str):
    """Get quick fundamentals from yfinance with fallbacks. Cached to reduce rate usage."""
    t = yf.Ticker(ticker)
    info = {}
    try:
        fi = getattr(t, "fast_info", {}) or {}
        info["marketCap"] = fi.get("market_cap") or None
    except Exception:
        info["marketCap"] = None

    try:
        i = t.info or {}
    except Exception:
        i = {}

    for k in ["marketCap", "returnOnEquity", "trailingPE", "debtToEquity", "totalCash"]:
        info[k] = i.get(k, info.get(k))

    mc = info.get("marketCap")
    cash = info.get("totalCash")
    info["cash_gt_mcap"] = bool(cash and mc and cash > mc)
    return info

def passes_fundamentals(info: dict, mcap_min_cr: float, pe_max: float, roe_min_pct: float, de_max: float) -> bool:
    try:
        mc_ok = (info.get("marketCap") or 0) >= (mcap_min_cr * 1e7)  # ₹Cr -> ₹
        pe_ok = (pe_max == 0) or ((info.get("trailingPE") is not None) and info.get("trailingPE") <= pe_max)
        roe_ok = (roe_min_pct == 0) or ((info.get("returnOnEquity") is not None) and (info.get("returnOnEquity") * 100 >= roe_min_pct))
        de_ok = (de_max == 0) or ((info.get("debtToEquity") is not None) and (info.get("debtToEquity") <= de_max))
        return all([mc_ok, pe_ok, roe_ok, de_ok])
    except Exception:
        return False

def decision_from_score(score: float) -> str:
    if score >= 9:
        return "🟢 Strong Buy"
    if score >= 6:
        return "🟢 Buy"
    if score <= -3:
        return "🔴 Sell"
    return "⚪ Hold"

import textwrap

def build_breakdown(components: dict) -> (str, dict):
    lines = []
    out = {}
    for label, (weight, awarded, points) in components.items():
        awarded_mark = "✅" if awarded else "❌"
        lines.append(f"{label}: {awarded_mark} ({points:.2f})")
        out[label] = {
            "weight": weight,
            "awarded": bool(awarded),
            "points": round(points, 2)
        }
    return "\n".join(lines), out



# --------------------------- ANALYSIS / SCORING ---------------------------
def analyze_ticker(ticker: str, df_ticker: pd.DataFrame):
    if df_ticker is None or df_ticker.empty:
        return None

    df = compute_indicators(df_ticker).dropna(subset=["Close"])
    if df is None or df.empty or len(df) < 40:
        return None

    latest = df.iloc[-1]
    prev = df.iloc[-2] if len(df) >= 2 else latest

    # Fundamentals
    finfo = fundamental_snapshot(ticker)
    fund_ok = passes_fundamentals(finfo, mcap_min_cr, pe_max, roe_min_pct, de_max)

    # Signals (boolean checks)
    trend_components_bools = [
        ("SMA20 Rising", is_rising(df.get('SMA20'), rising_days, rising_tol)),
        ("EMA13 Rising", is_rising(df.get('EMA13'), rising_days, rising_tol)),
        ("MACD Rising", is_rising(df.get('MACD'), rising_days, max(6, rising_tol))),
        ("RSI Rising", is_rising(df.get('RSI'), rising_days, max(10, rising_tol))),
    ]
    trend_count = sum(1 for _, v in trend_components_bools if v)
    trend_score = w_trend * (trend_count / len(trend_components_bools))  # normalized to [0, w_trend]
    
    # Convert boolean signals into ✅ or ❌
    trend_signals_dict = {name: "✅" if ok else "❌" for name, ok in trend_components_bools}

    # Merge into one comma-separated string
    trend_signals_str = ", ".join(f"{name}: {signal}" for name, signal in trend_signals_dict.items())

    adx_ok = (latest.get("ADX", 0) > adx_min) and (latest.get("+DI", 0) > latest.get("-DI", 0))
    macd_cross = (df["MACD"].iloc[-2] < df["Signal"].iloc[-2]) and (latest["MACD"] > latest["Signal"])
    rsi_bull = latest.get("RSI", 0) > rsi_floor
    vol_spike = bool(latest.get("Volume Spike", False))
    stoch_buy = bool(latest.get("StochRSI_Buy_Signal", False))

    price_above_cpr = False
    try:
        price_above_cpr = latest["Close"] > latest.get("CPR_Top", np.nan)
    except Exception:
        price_above_cpr = False

    fib_bounce = False
    try:
        fib_val = latest.get("Fib_61.8", np.nan)
        if not np.isnan(fib_val):
            fib_bounce = abs(latest["Close"] - fib_val) <= latest["Close"] * FIB_TOLERANCE
    except Exception:
        fib_bounce = False

    ao_bull = (latest.get("AO_Signal") == "🟢 Bullish")

    # Compose components and weights
    components = {
        "Trend (normalized)": (w_trend, trend_count >= 2, trend_score),
        "ADX & DI+": (w_adx, adx_ok, w_adx if adx_ok else 0),
        "MACD Bull Cross": (w_macd_cross, macd_cross, w_macd_cross if macd_cross else 0),
        "RSI > floor": (w_rsi_bull, rsi_bull, w_rsi_bull if rsi_bull else 0),
        "Volume Spike": (w_vol_spike, vol_spike, w_vol_spike if vol_spike else 0),
        "StochRSI Buy": (w_stoch, stoch_buy, w_stoch if stoch_buy else 0),
        "Above CPR Top": (w_cpr, price_above_cpr, w_cpr if price_above_cpr else 0),
        "Fib 61.8 Bounce": (w_fib, fib_bounce, w_fib if fib_bounce else 0),
        "AO Bullish": (w_ao, ao_bull, w_ao if ao_bull else 0),
        "Fundamentals": (w_fund, fund_ok, w_fund if fund_ok else 0),
    }

    # Sum points (note: trend_score already scaled to [0,w_trend])
    total_score = 0.0
    # Start by using the trend_score numeric (not count)
    total_score += trend_score
    # Add rest numeric awards
    for key, (w, awarded, pts) in components.items():
        if key == "Trend (normalized)":
            continue
        total_score += pts

    # Build breakdown: replace trend component with numeric points
    components_for_breakdown = components.copy()
    components_for_breakdown["Trend (normalized)"] = (w_trend, trend_count >= 2, trend_score)

    breakdown_str, breakdown_obj = build_breakdown(components_for_breakdown)
    
    breakdown_str = "\n".join(
    f"{label}: {'✅' if awarded else '❌'} ({points:.2f})"
    for label, (weight, awarded, points) in components.items()
    )

    # Basic formatting values for output
    def fmt(x, nd=2):
        try:
            return round(float(x), nd)
        except Exception:
            return "N/A"

    mcap_cr = (finfo.get("marketCap") / 1e7) if finfo.get("marketCap") else None
    # Build fundamentals string
    fundamentals_str = (
        f"MktCap(₹ Cr): {mcap_cr:,.0f}" if mcap_cr else "MktCap(₹ Cr): N/A"
    ) + ", " + \
        f"P/E: {fmt(finfo.get('trailingPE') or 'N/A')}, " + \
        f"ROE %: {fmt((finfo.get('returnOnEquity') or 0) * 100)}, " + \
        f"D/E: {fmt(finfo.get('debtToEquity') if finfo.get('debtToEquity') is not None else 'N/A')}"

    row = {
        "Symbol": ticker,
        "Last": fmt(latest.get("Close", np.nan)),
        "RSI": fmt(latest.get("RSI", np.nan)),
        "ADX": fmt(latest.get("ADX", np.nan)),
        "+DI": fmt(latest.get("+DI", np.nan)),
        "-DI": fmt(latest.get("-DI", np.nan)),
        "AO": fmt(latest.get("AO", np.nan)),
        #"AO Signal": latest.get("AO_Signal", "⚪ Neutral"),
        #"Vol Spike": "✅" if vol_spike else "❌",
        #"StochRSI Buy": "✅" if stoch_buy else "❌",
        #"Above CPR Top": "✅" if price_above_cpr else "❌",
        #"Fib 61.8 Bounce": "✅" if fib_bounce else "❌",
        "Tech OK": "✅" if trend_count >= 3 else "❌",
        "Fund OK": "✅" if fund_ok else "❌",
        #"Cash>MCAP": "⚠️" if finfo.get("cash_gt_mcap") else "—",
        "MktCap (₹ Cr)": f"{mcap_cr:,.0f}" if mcap_cr else "N/A",
        #"P/E": fmt(finfo.get("trailingPE") or "N/A"),
        #"ROE %": fmt((finfo.get("returnOnEquity") or 0) * 100),
        #"D/E": fmt(finfo.get("debtToEquity") if finfo.get("debtToEquity") is not None else "N/A"),
        "Score": round(total_score, 2),
        "Decision": decision_from_score(total_score),
        "Score Breakdown": breakdown_str,
        "Technical": trend_signals_str,
        "Fundamentals": fundamentals_str,
        "Score Components": breakdown_obj,
    }
    return row


# --------------------------- EXECUTION ---------------------------
if run_btn:
    tickers = [t.strip().upper() for t in tickers_text.split(",") if t.strip()]
    if not tickers:
        st.warning("Enter at least one ticker.")
    else:
        df_all = download_history_multi(tickers, period, interval)
        results = []
        with st.spinner("Scanning..."):
            for tk in tickers:
                try:
                    # Handle multi-index group_by result vs single-ticker
                    if isinstance(df_all.columns, pd.MultiIndex):
                        # Some tickers may be missing; guard with try
                        try:
                            df_tk = df_all[tk].dropna(how="all")
                        except Exception:
                            df_tk = pd.DataFrame()
                    else:
                        # Single ticker download returns a single DataFrame
                        df_tk = df_all.copy()

                    row = analyze_ticker(tk, df_tk)

                    if row:
                        results.append(row)
                    else:
                        results.append({"Symbol": tk, "Error": "Not enough data or indicators could not be computed."})
                except Exception as e:
                    results.append({"Symbol": tk, "Error": str(e)})

        if not results:
            st.warning("No results.")
        else:
            df_out = pd.DataFrame(results)

            # Optional: filter by Decision
            decisions_to_keep = st.multiselect(
                "Filter by Decision", ["🟢 Strong Buy", "🟢 Buy", "⚪ Hold", "🔴 Sell"],
                default=["🟢 Strong Buy", "🟢 Buy"]
            )
            if "Decision" in df_out.columns:
                df_out = df_out[df_out["Decision"].isin(decisions_to_keep)]

            # Sort and display
            if "Score" in df_out.columns:
                df_out = df_out.sort_values(by=["Score"], ascending=False)

            st.caption(f"Scanned: {len(tickers)} • Shown: {len(df_out)} • Time: {datetime.now().strftime('%H:%M:%S')}")
            # Show main table but hide huge Score Components column by default
            short_display = df_out.copy()
            if "Score Components" in short_display.columns:
                short_display = short_display.drop(columns=["Score Components"])
            st.dataframe(short_display, use_container_width=True)
            
            # Expandable detail for breakdowns
            with st.expander("Show raw Score Components (JSON per row)"):
                st.write(df_out[["Symbol", "Score", "Decision", "Score Components"]])

            # CSV download
            csv = df_out.to_csv(index=False).encode("utf-8")
            st.download_button("⬇️ Download CSV", data=csv, file_name="market_scan_with_breakdown.csv", mime="text/csv")
else:
    st.info("Set parameters in the sidebar and click **Run Scan**.")
