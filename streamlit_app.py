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
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Display top headlines at the top of your dashboard
def fetch_headlines(rss_url, limit=3):
    feed = feedparser.parse(rss_url)
    return [entry.title for entry in feed.entries[:limit]]

# Add this near the top of your Streamlit app (before st.title)
st.markdown("## 📢 Latest Market News & Headlines")

# Define Indian-focused RSS URLs
rss_sources = {
    "Moneycontrol Markets": "https://www.moneycontrol.com/rss/MCtopnews.xml",
    "Zee Business": "https://zeenews.india.com/rss/business.xml",
    "Economic Times Markets": "https://b2b.economictimes.indiatimes.com/rss/markets",  # published by ET :contentReference[oaicite:1]{index=1}
}

# Show headlines inline
for label, url in rss_sources.items():
    headlines = fetch_headlines(url, limit=2)
    if headlines:
        st.markdown(f"**{label}:**")
        for hl in headlines:
            st.markdown(f"- {hl}")
    else:
        st.markdown(f"**{label}:** No headlines found.")

st.markdown("---")


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

st.markdown(f"**Selected Time Range:** {selected_label}  |  Period: {period}, Interval: {interval}")

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

    X, y = [], []
    for i in range(window_size, len(scaled_data)):
        X.append(scaled_data[i-window_size:i, 0])
        y.append(scaled_data[i, 0])

    X, y = np.array(X), np.array(y)
    X = X.reshape((X.shape[0], X.shape[1], 1))

    # Split
    split = int(len(X) * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    # Model
    model = Sequential([
        LSTM(100, return_sequences=True, input_shape=(X.shape[1], 1)),
        LSTM(50),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X_train, y_train, epochs=15, batch_size=32, verbose=0)

    # Evaluate
    y_pred_scaled = model.predict(X_test)
    y_pred = scaler.inverse_transform(y_pred_scaled)
    y_actual = scaler.inverse_transform(y_test.reshape(-1, 1))

    rmse = np.sqrt(mean_squared_error(y_actual, y_pred))
    mae = mean_absolute_error(y_actual, y_pred)
    r2 = r2_score(y_actual, y_pred)

    # Forecast next `days` prices
    last_60 = scaled_data[-window_size:].reshape(1, window_size, 1)
    predictions = []

    for _ in range(days):
        pred_scaled = model.predict(last_60)
        predictions.append(pred_scaled[0][0])
        last_60 = np.append(last_60[:, 1:, :], [[[pred_scaled[0][0]]]], axis=1)

    forecasted_prices = scaler.inverse_transform(np.array(predictions).reshape(-1, 1)).flatten()
    return forecasted_prices, round(rmse, 2), round(mae, 2), round(r2, 2)


# --- Sentiment Analysis Functions ---
# --- Sentiment Analysis Functions ---

def get_sentiment_score_from_google_news(stock_name, alt_name=None):
    """
    Fetch sentiment score based on news headlines for the stock using Google News RSS.
    Accepts both stock ticker (e.g., INFY) and alternative company name (e.g., Infosys).
    """
    rss_url = f"https://news.google.com/rss/search?q={stock_name}+stock&hl=en-IN&gl=IN&ceid=IN:en"
    feed = feedparser.parse(rss_url)
    analyzer = SentimentIntensityAnalyzer()

    # Consider alternative company name as well if provided
    all_headlines = feed.entries[:10]  # Fetch more in case filtering removes some
    headlines = []
    for entry in all_headlines:
        title = entry.title.lower()
        if stock_name.lower() in title or (alt_name and alt_name.lower() in title):
            headlines.append(entry.title)
        if len(headlines) >= 5:
            break

    if not headlines:
        return 0, []

    scores = [analyzer.polarity_scores(title)['compound'] for title in headlines]
    return round(np.mean(scores), 2), headlines

def get_sentiment_from_rss(rss_url, stock_name, alt_name=None):
    """
    Fetch sentiment from any RSS source using both ticker and company name match.
    """
    feed = feedparser.parse(rss_url)
    analyzer = SentimentIntensityAnalyzer()

    # Possible variations of the company name to search for
    possible_names = [stock_name.lower()]
    if ".NS" in stock_name:
        short = stock_name.split(".")[0]
        possible_names.append(short.lower())
    if alt_name:
        possible_names.append(alt_name.lower())

    headlines = [
        entry.title for entry in feed.entries
        if any(name in entry.title.lower() for name in possible_names)
    ][:5]

    if not headlines:
        return 0, []

    scores = [analyzer.polarity_scores(title)['compound'] for title in headlines]
    return round(np.mean(scores), 2), headlines


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
        fig_sma.add_trace(go.Scatter(x=df.index, y=df['SMA'], mode='lines', name='SMA', line=dict(color='blue')))
        fig_sma.update_layout(title="Simple Moving Average (SMA)", height=200, width=400, margin=dict(l=10, r=10, t=30, b=10), showlegend=False)
        st.plotly_chart(fig_sma, use_container_width=True, key=f"sma_chart_{ticker}")

        fig_ema = go.Figure()
        fig_ema.add_trace(go.Scatter(x=df.index, y=df['EMA20'], mode='lines', name='EMA20', line=dict(color='blue')))
        fig_ema.update_layout(title="Exponential Moving Average (EMA)", height=200, width=400, margin=dict(l=10, r=10, t=30, b=10), showlegend=False)
        st.plotly_chart(fig_ema, use_container_width=True, key=f"ema_chart_{ticker}")

    with col_macd:
        macd_hist_colors = ['green' if val >= 0 else 'red' for val in df['Histogram']]
        fig_macd = go.Figure()
        fig_macd.add_trace(go.Scatter(x=df.index, y=df['MACD'], name='MACD', line=dict(color='blue')))
        fig_macd.add_trace(go.Scatter(x=df.index, y=df['Signal'], name='Signal', line=dict(color='orange')))
        fig_macd.add_trace(go.Bar(x=df.index, y=df['Histogram'], marker_color=macd_hist_colors, opacity=0.5))
        fig_macd.update_layout(title="MACD Indicator", height=250, margin=dict(l=10, r=10, t=30, b=10), showlegend=True)
        st.plotly_chart(fig_macd, use_container_width=True, key=f"macd_chart_{ticker}")

        last_close = df['Close'].iloc[-1]
        last_close_rounded = round(last_close)  # or use int() if preferred
        
         # --   st.markdown(f"💰 **Last Close Price**: ₹ {float(last_close)}")
        st.markdown(
            f"💰 **Last Close Price**: ₹ <span style='color:orange; font-weight:bold;'>{float(last_close):.2f}</span>",
            unsafe_allow_html=True
        )
        
        # Existing stock selection
        ticker_data = yf.Ticker(ticker)

        # Always fetch 1-minute interval for today, to get the latest price
        df_intraday = ticker_data.history(period="1d", interval="1m")

        # Ensure it's not empty
        if not df_intraday.empty:
            current_price = df_intraday['Close'].iloc[-1]
        else:
            current_price = df['Close'].iloc[-1]  # fallback

        # Display dynamically colored current traded price
        st.markdown(
            f"💰 **Current Traded Price**: ₹ <span style='color:green; font-weight:bold;'>{float(current_price):.2f}</span>",
            unsafe_allow_html=True
        )
        
        forecast_result = lstm_forecast(df)
        if forecast_result:
            forecast_1d = lstm_forecast(df, days=1)
            forecast_1w = lstm_forecast(df, days=5)
            forecast_1m = lstm_forecast(df, days=21)

            if forecast_1d and forecast_1w and forecast_1m:
                price_1d, rmse1, mae1, r2_1 = forecast_1d[0][-1], forecast_1d[1], forecast_1d[2], forecast_1d[3]
                price_1w = forecast_1w[0][-1]
                price_1m = forecast_1m[0][-1]

                # Display Forecasts
                st.markdown("### 📈 **LSTM Price Forecasts**")

                col_a, col_b, col_c = st.columns(3)
                col_a.markdown(f"**Next Day (1D):** ₹ <span style='color:green; font-weight:bold;'>{price_1d:.2f}</span>", unsafe_allow_html=True)
                col_b.markdown(f"**Next Week (1W):** ₹ <span style='color:orange; font-weight:bold;'>{price_1w:.2f}</span>", unsafe_allow_html=True)
                col_c.markdown(f"**Next Month (1M):** ₹ <span style='color:red; font-weight:bold;'>{price_1m:.2f}</span>", unsafe_allow_html=True)

                st.markdown("### 📊 **LSTM Model Accuracy**")
                c1, c2, c3 = st.columns(3)
                c1.metric("RMSE", f"{rmse1}")
                c2.metric("MAE", f"{mae1}")
                c3.metric("R² Score", f"{r2_1:.4f}")
            else:
                st.warning("Not enough data to train LSTM model.")



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
            
        try:
        # Get company name from yfinance
            ticker_obj = yf.Ticker(ticker)
            company_name = ticker_obj.info.get("longName", ticker)
        except Exception:
            company_name = ticker

        sentiment_score, sentiment_headlines = get_sentiment_score_from_google_news(ticker, company_name)
        sentiment_label = "Positive" if sentiment_score > 0.2 else "Negative" if sentiment_score < -0.2 else "Neutral"
        st.markdown(f"📰 **Google News** Sentiment: <span style='font-weight:bold;'>{sentiment_score} ({sentiment_label})</span>", unsafe_allow_html=True)


        mc_sentiment, mc_headlines = get_sentiment_from_rss("https://www.moneycontrol.com/rss/MCtopnews.xml", ticker, company_name)
        mc_label = "Positive" if mc_sentiment > 0.2 else "Negative" if mc_sentiment < -0.2 else "Neutral"
        st.markdown(f"🗞️ **Moneycontrol** Sentiment: <span style='font-weight:bold;'>{mc_sentiment} ({mc_label})</span>", unsafe_allow_html=True)

        zee_sentiment, zee_headlines = get_sentiment_from_rss("https://zeenews.india.com/rss/business.xml", ticker, company_name)
        zee_label = "Positive" if zee_sentiment > 0.2 else "Negative" if zee_sentiment < -0.2 else "Neutral"
        st.markdown(f"📺 **Zee Business** Sentiment: <span style='font-weight:bold;'>{zee_sentiment} ({zee_label})</span>", unsafe_allow_html=True)

        with st.expander("📰 Related Headlines (Zee & Moneycontrol)"):
            if mc_headlines:
                st.markdown("**Moneycontrol:**")
                for headline in mc_headlines:
                    st.markdown(f"- {headline}")
            if zee_headlines:
                st.markdown("**Zee Business:**")
                for headline in zee_headlines:
                    st.markdown(f"- {headline}")
