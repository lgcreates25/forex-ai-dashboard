import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

st.set_page_config(page_title="Forex AI Dashboard", layout="wide")

st.title("🤖 Forex AI Analysis Dashboard")
st.write("Educational EUR/USD market analysis")

# Download data
@st.cache_data
def get_data():
    data = yf.download(
        "EURUSD=X",
        period="5y",
        interval="1d",
        auto_adjust=True,
        progress=False
    )

    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    return data

data = get_data()

if data.empty:
    st.error("Unable to download market data.")
    st.stop()

# Prepare indicators
df = data.copy()
close = df["Close"].astype(float)

df["Returns"] = close.pct_change()
df["MA_5"] = close.rolling(5).mean()
df["MA_20"] = close.rolling(20).mean()

delta = close.diff()
gain = delta.clip(lower=0)
loss = -delta.clip(upper=0)

avg_gain = gain.ewm(
    alpha=1/14,
    min_periods=14,
    adjust=False
).mean()

avg_loss = loss.ewm(
    alpha=1/14,
    min_periods=14,
    adjust=False
).mean()

rs = avg_gain / avg_loss
df["RSI"] = 100 - (100 / (1 + rs))

ema12 = close.ewm(span=12, adjust=False).mean()
ema26 = close.ewm(span=26, adjust=False).mean()

df["MACD"] = ema12 - ema26
df["Volatility"] = close.pct_change().rolling(14).std()

df["Target"] = (close.shift(-1) > close).astype(int)
df = df.dropna().iloc[:-1].copy()

# Features
features = [
    "Returns", "MA_5", "MA_20",
    "RSI", "MACD", "Volatility"
]

X = df[features]
y = df["Target"]

split = int(len(df) * 0.8)

X_train = X.iloc[:split]
X_test = X.iloc[split:]

y_train = y.iloc[:split]
y_test = y.iloc[split:]

# Train models
model = make_pipeline(
    StandardScaler(),
    LogisticRegression(max_iter=1000)
)

model.fit(X_train, y_train)

rf_model = RandomForestClassifier(
    n_estimators=200,
    max_depth=5,
    min_samples_leaf=10,
    random_state=42,
    class_weight="balanced"
)

rf_model.fit(X_train, y_train)

# Results
log_accuracy = accuracy_score(
    y_test, model.predict(X_test)
)

rf_accuracy = accuracy_score(
    y_test, rf_model.predict(X_test)
)

baseline = max(y_test.mean(), 1 - y_test.mean())

# Dashboard
st.subheader("EUR/USD Market Overview")

col1, col2, col3 = st.columns(3)

col1.metric("Latest Price", round(float(close.iloc[-1]), 5))
col2.metric("RSI", round(float(df["RSI"].iloc[-1]), 2))
col3.metric("Historical Records", len(df))

st.subheader("EUR/USD Price Chart")

st.line_chart(df["Close"])

st.subheader("Technical Indicators")

st.line_chart(df[["MA_5", "MA_20"]])
st.line_chart(df[["RSI", "MACD"]])

st.subheader("AI Model Performance")

st.write("Logistic Regression:", round(log_accuracy * 100, 2), "%")
st.write("Random Forest:", round(rf_accuracy * 100, 2), "%")
st.write("Baseline:", round(baseline * 100, 2), "%")

st.caption(
    "Educational research only. Historical model performance "
    "does not guarantee future results. Not a trading signal."
)

# AI FOREX SCREENSHOT ANALYZER

st.divider()

st.title("📸 AI Forex Screenshot Analyzer")

st.write(
    "Upload a Forex chart screenshot for educational analysis."
)

uploaded_image = st.file_uploader(
    "Upload your Forex chart",
    type=["png", "jpg", "jpeg"]
)

if uploaded_image is not None:

    from PIL import Image

    chart_image = Image.open(uploaded_image)

    st.image(
        chart_image,
        caption="Uploaded Forex Chart",
        use_container_width=True
    )

    st.success("Screenshot uploaded successfully!")

    st.info(
        "Image analysis will be connected in the next step."
)

# SIMULATED CHART ANALYSIS

st.subheader("📊 Educational Chart Observations")

trend = st.selectbox(
    "What trend do you observe?",
    ["Uptrend", "Downtrend", "Sideways"]
)

rsi = st.slider(
    "RSI value (from your chart)",
    0, 100, 50
)

macd = st.selectbox(
    "What does the MACD show?",
    ["Bullish", "Bearish", "Unclear"]
)

if st.button("Analyse Chart"):

    score = 0

    if trend == "Uptrend":
        score += 1
    elif trend == "Downtrend":
        score -= 1

    if rsi > 50:
        score += 1
    elif rsi < 50:
        score -= 1

    if macd == "Bullish":
        score += 1
    elif macd == "Bearish":
        score -= 1

    st.subheader("Analysis Result")

    if score >= 2:
        st.success("BUY - Simulated bullish conditions")
    elif score <= -2:
        st.error("SELL - Simulated bearish conditions")
    else:
        st.warning("WAIT - Conditions are mixed or unclear")

    st.write("Trend:", trend)
    st.write("RSI:", rsi)
    st.write("MACD:", macd)

    st.info(
        "This is a basic educational simulation using "
        "manually entered observations. It does not analyse "
        "the screenshot or predict actual market movements."
)
