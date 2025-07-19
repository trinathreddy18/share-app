import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import feedparser

# NEW IMPORTS
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

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

    try:
        if df.index.tz is None:
            df.index = df.index.tz_localize("UTC").tz_convert("Asia/Kolkata")
        else:
            df.index = df.index.tz_convert("Asia/Kolkata")
    except TypeError:
        df.index = df.index.tz_convert("Asia/Kolkata")

    df = df[df.index.dayofweek < 5]

    df['SMA'] = df['Close'].rolling(window=20).mean()
    df['EMA20'] = df['Close'].ewm(span=20, adjust=False).mean()
    df['EMA12'] = df['Close'].ewm(span=12, adjust=False).mean()
    df['EMA26'] = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = df['EMA12'] - df['EMA26']
    df['Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['Histogram'] = df['MACD'] - df['Signal']

    delta = df['Close'].diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    rs = avg_gain / avg_loss
    df['RSI'] = 100 - (100 / (1 + rs))

    df['BB_Mid'] = df['Close'].rolling(window=20).mean()
    df['BB_Std'] = df['Close'].rolling(window=20).std()
    df['BB_Upper'] = df['BB_Mid'] + 2 * df['BB_Std']
    df['BB_Lower'] = df['BB_Mid'] - 2 * df['BB_Std']

    df.dropna(inplace=True)
    return df

# --- LSTM Forecast Function ---
def lstm_forecast(df, days=1):
    data = df[['Close']].copy()
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data)

    window_size = 60
    if len(scaled_data) < window_size + 1:
        return None

    X = []
    for i in range(window_size, len(scaled_data)):
        X.append(scaled_data[i-window_size:i, 0])
    X = np.array(X)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))

    y = scaled_data[window_size:]

    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(X.shape[1], 1)))
    model.add(LSTM(50))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X, y, epochs=5, batch_size=32, verbose=0)

    input_data = scaled_data[-window_size:]
    input_data = input_data.reshape(1, window_size, 1)

    prediction = model.predict(input_data)
    forecast_price = scaler.inverse_transform(prediction)[0][0]
    return round(forecast_price, 2)


# --- Sentiment Analysis Function ---
# --- Sentiment Analysis Function ---
def get_sentiment_score_from_google_news(stock_name):
    # Google News RSS feed URL for the stock
    rss_url = f"https://news.google.com/rss/search?q={stock_name}+stock&hl=en-IN&gl=IN&ceid=IN:en"
    feed = feedparser.parse(rss_url)
    
    analyzer = SentimentIntensityAnalyzer()
    
    # Extract top 5 headlines
    headlines = [entry.title for entry in feed.entries[:5]]
    
    # Calculate sentiment scores
    scores = [analyzer.polarity_scores(headline)['compound'] for headline in headlines]
    
    return round(np.mean(scores), 2) if scores else 0

# --- Stock Input ---
default_stocks = "TCS.NS, RELIANCE.NS, INFY.NS"
stock_input = st.text_input("🧾 Enter stock tickers (comma-separated)", default_stocks)
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

    with col_price:
        fig_sma = go.Figure()
        fig_sma.add_trace(go.Scatter(
            x=df.index, y=df['SMA'],
            mode='lines', name='SMA', line=dict(color='blue', width=2)
        ))
        fig_sma.update_layout(
            title="Simple Moving Average (SMA)",
            xaxis_title="Date", yaxis_title="SMA (₹)",
            height=200, width=400,
            margin=dict(l=10, r=10, t=30, b=10), showlegend=False,
        )
        st.plotly_chart(fig_sma, use_container_width=True, key=f"sma_chart_{ticker}")

        fig_ema = go.Figure()
        fig_ema.add_trace(go.Scatter(
            x=df.index, y=df['EMA20'],
            mode='lines', name='EMA20', line=dict(color='blue', width=2)
        ))
        fig_ema.update_layout(
            title="Exponential Moving Average (EMA)",
            xaxis_title="Date", yaxis_title="EMA20 (₹)",
            height=200, width=400,
            margin=dict(l=10, r=10, t=30, b=10), showlegend=False,
        )
        st.plotly_chart(fig_ema, use_container_width=True, key=f"ema_chart_{ticker}")

    with col_macd:
        macd_hist_colors = ['green' if val >= 0 else 'red' for val in df['Histogram']]
        fig_macd = go.Figure()
        fig_macd.add_trace(go.Scatter(x=df.index, y=df['MACD'], mode='lines', name='MACD', line=dict(color='blue')))
        fig_macd.add_trace(go.Scatter(x=df.index, y=df['Signal'], mode='lines', name='Signal', line=dict(color='orange')))
        fig_macd.add_trace(go.Bar(x=df.index, y=df['Histogram'], name='Histogram', marker_color=macd_hist_colors, opacity=0.5))
        fig_macd.update_layout(
            title="MACD Indicator", height=250,
            margin=dict(l=10, r=10, t=30, b=10),
            showlegend=True, xaxis_title="Date", yaxis_title="MACD",
        )
        st.plotly_chart(fig_macd, use_container_width=True, key=f"macd_chart_{ticker}")

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
            
        # --- Last Close Price ---
        last_close = df['Close'].iloc[-1]
        last_close_rounded = round(last_close)  # or use int() if preferred
        
         # --   st.markdown(f"💰 **Last Close Price**: ₹ `{float(last_close)}`")
        st.markdown(
            f"💰 **Last Close Price**: ₹ <span style='color:orange; font-weight:bold;'>{float(last_close):.2f}</span>",
            unsafe_allow_html=True
        )

        # --- LSTM Forecast ---
        forecast_price = lstm_forecast(df)
        if forecast_price:
            # Determine color and emoji
            if forecast_price > float(last_close):
                st.markdown(
                    f"📈 **Forecasted (Next Day)**: ₹ <span style='color:green; font-weight:bold;'>{forecast_price:.2f}</span>",
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    f"📉 **Forecasted (Next Day)**: ₹ <span style='color:red; font-weight:bold;'>{forecast_price:.2f}</span>",
                    unsafe_allow_html=True
                )        
        else:
            st.warning("📉 Not enough data for prediction (at least 3 months of data is required)")

        # --- Sentiment Score (Mock Data) ---
        sentiment = get_sentiment_score_from_google_news(ticker)
        sentiment_label = "Positive" if sentiment > 0.2 else "Negative" if sentiment < -0.2 else "Neutral"
        st.markdown(f"📰 **Sentiment Score**: <span style='font-weight:bold;'>{sentiment} ({sentiment_label})</span>",
                    unsafe_allow_html=True)
