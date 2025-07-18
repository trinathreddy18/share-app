import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go

st.set_page_config(layout="wide")
st.title("📊 Indian Stock Technical Indicator Dashboard")

# --- Time Selection ---
st.markdown("### ⏱️ Select Time Range")

time_options = {
    "1 Day": ("1d", "5m"),
    "Weeks (Weekly)": ("1y", "5d"),
    "1 Month": ("2mo", "1d"),
    "3 Months": ("4mo", "1d"),
    "6 Months": ("7mo", "1d"),
    "1 Year": ("13mo", "1d"),
    "2 Years": ("25mo", "1d")
}

time_labels = list(time_options.keys())
cols = st.columns(len(time_labels))

if "selected_label" not in st.session_state:
    st.session_state.selected_label = "1 Month"

for i, label in enumerate(time_labels):
    if cols[i].button(label, use_container_width=True):
        st.session_state.selected_label = label

selected_label = st.session_state.selected_label
period, interval = time_options[selected_label]

st.markdown(f"**Selected Time Range:** `{selected_label}`  |  Period: `{period}`, Interval: `{interval}`")

# --- Load Data ---
@st.cache_data(ttl=3600)
def load_data(ticker, period, interval):
    df = yf.download(ticker, period=period, interval=interval, progress=False)
    if df.empty:
        return df

    # Timezone
    try:
        if df.index.tz is None:
            df.index = df.index.tz_localize("UTC").tz_convert("Asia/Kolkata")
        else:
            df.index = df.index.tz_convert("Asia/Kolkata")
    except TypeError:
        df.index = df.index.tz_convert("Asia/Kolkata")

    df = df[df.index.dayofweek < 5]  # Remove weekends

    # SMA
    df['SMA'] = df['Close'].rolling(window=20).mean()

    # EMA
    df['EMA20'] = df['Close'].ewm(span=20, adjust=False).mean()

    # MACD
    df['EMA12'] = df['Close'].ewm(span=12, adjust=False).mean()
    df['EMA26'] = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = df['EMA12'] - df['EMA26']
    df['Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['Histogram'] = df['MACD'] - df['Signal']

    # RSI
    delta = df['Close'].diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    rs = avg_gain / avg_loss
    df['RSI'] = 100 - (100 / (1 + rs))

    # Bollinger Bands
    df['BB_Mid'] = df['Close'].rolling(window=20).mean()
    df['BB_Std'] = df['Close'].rolling(window=20).std()
    df['BB_Upper'] = df['BB_Mid'] + 2 * df['BB_Std']
    df['BB_Lower'] = df['BB_Mid'] - 2 * df['BB_Std']

    df.dropna(inplace=True)
    return df

# --- Stock List ---
# --- Dynamic Stock Input ---
default_stocks = "TCS.NS, RELIANCE.NS, INFY.NS"
stock_input = st.text_input("🧾 Enter stock tickers (comma-separated)", default_stocks)

# Process input
stock_list = [ticker.strip().upper() for ticker in stock_input.split(",") if ticker.strip()]


# --- Main Loop ---
for ticker in stock_list:
    df = load_data(ticker, period, interval)

    if df.empty:
        st.warning(f"No data for {ticker}.")
        continue

    col1, col_price, col_macd, col_rsi, col_data = st.columns([1, 2.2, 2.2, 2.2, 1])

    with col1:
        st.subheader(ticker)

# SMA Line Chart ---
    with col_price:
        fig_sma = go.Figure()
        fig_sma.add_trace(go.Scatter(
            x=df.index,
            y=df['SMA'],  # Make sure 'SMA' column exists in df
            mode='lines',
            name='SMA',
            line=dict(color='blue', width=2)
        ))
        fig_sma.update_layout(
            title="Simple Moving Average (SMA)",
            xaxis_title="Date",
            yaxis_title="SMA (₹)",
            height=200,
            width=400,
            margin=dict(l=10, r=10, t=30, b=10),
            showlegend=False,
        )
        st.plotly_chart(fig_sma, use_container_width=True, key=f"sma_chart_{ticker}")
        
# EMA20 Line Chart ---
    with col_price:
        fig_sma = go.Figure()
        fig_sma.add_trace(go.Scatter(
            x=df.index,
            y=df['EMA20'],  # Make sure 'EMA' column exists in df
            mode='lines',
            name='EMA20',
            line=dict(color='blue', width=2)
        ))
        fig_sma.update_layout(
            title="Exponential Moving Average (EMA)",
            xaxis_title="Date",
            yaxis_title="EMA20 (₹)",
            height=200,
            width=400,
            margin=dict(l=10, r=10, t=30, b=10),
            showlegend=False,
        )
        st.plotly_chart(fig_sma, use_container_width=True, key=f"ema_chart_{ticker}")
                
    # --- MACD ---
    with col_macd:
        macd_hist_colors = ['green' if val >= 0 else 'red' for val in df['Histogram']]
        fig_macd = go.Figure()

        # MACD line
        fig_macd.add_trace(go.Scatter(
            x=df.index, y=df['MACD'],
            mode='lines', name='MACD',
            line=dict(color='blue', width=1)
        ))

        # Signal line
        fig_macd.add_trace(go.Scatter(
            x=df.index, y=df['Signal'],
            mode='lines', name='Signal',
            line=dict(color='orange', width=1)
        ))

        # Histogram bars with dynamic colors
        fig_macd.add_trace(go.Bar(
            x=df.index,
            y=df['Histogram'],
            name='Histogram',
            marker_color=macd_hist_colors,
            opacity=0.5
        ))

        fig_macd.update_layout(
            title="MACD Indicator",
            height=250,
            margin=dict(l=10, r=10, t=30, b=10),
            showlegend=True,
            xaxis_title="Date",
            yaxis_title="MACD",
        )

        st.plotly_chart(fig_macd, use_container_width=True, key=f"macd_chart_{ticker}")

    # --- RSI ---
    with col_rsi:
        fig_rsi = go.Figure()
        fig_rsi.add_trace(go.Scatter(x=df.index, y=df['RSI'], name='RSI', line=dict(color='purple')))
        fig_rsi.add_hline(y=70, line=dict(color='red', dash='dot'))
        fig_rsi.add_hline(y=30, line=dict(color='green', dash='dot'))
        fig_rsi.update_layout(title="RSI", height=250, margin=dict(t=30, b=20), showlegend=False)
        st.plotly_chart(fig_rsi, use_container_width=True, key=f"rsi_chart_{ticker}")

    with col_data:
        with st.expander("🔍 Data"):
            st.dataframe(df.tail(5), use_container_width=True)
