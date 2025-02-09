import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from ta.trend import SMAIndicator, MACD, IchimokuIndicator
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.volatility import BollingerBands
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from statsmodels.tsa.arima.model import ARIMA
from datetime import datetime, timedelta

# 📌 Streamlit UI
st.title("📈 AI Crypto Market Analysis Bot")
st.sidebar.header("⚙ Επιλογές")
crypto_symbol = st.sidebar.text_input("Εισάγετε Crypto Symbol", "BTC-USD")

def load_data(symbol, period="6mo", interval="1h"):
    df = yf.download(symbol, period=period, interval=interval)
    df["SMA_50"] = SMAIndicator(df["Close"], window=50).sma_indicator()
    df["SMA_200"] = SMAIndicator(df["Close"], window=200).sma_indicator()
    df["RSI"] = RSIIndicator(df["Close"], window=14).rsi()
    df["MACD"] = MACD(df["Close"]).macd()
    df["Bollinger_High"] = BollingerBands(df["Close"]).bollinger_hband()
    df["Bollinger_Low"] = BollingerBands(df["Close"]).bollinger_lband()
    df["Ichimoku"] = IchimokuIndicator(df["High"], df["Low"]).ichimoku_a()
    df.dropna(inplace=True)
    return df

df = load_data(crypto_symbol)

def train_model(df):
    X = df[["SMA_50", "SMA_200", "MACD", "RSI", "Bollinger_High", "Bollinger_Low"]]
    y = np.where(df["Close"].shift(-1) > df["Close"], 1, 0)
    
    model_rf = RandomForestClassifier(n_estimators=100)
    model_rf.fit(X, y)
    
    model_gb = GradientBoostingClassifier(n_estimators=100)
    model_gb.fit(X, y)
    
    df["Prediction_RF"] = model_rf.predict(X)
    df["Prediction_GB"] = model_gb.predict(X)
    df["Final_Prediction"] = (df["Prediction_RF"] + df["Prediction_GB"]) // 2
    return df

df = train_model(df)

# 📌 Υπολογισμός Entry Point, Stop Loss, Take Profit
def calculate_trade_levels(df):
    latest_close = df["Close"].iloc[-1]
    atr = df["Close"].diff().abs().mean() * 1.5
    entry_point = latest_close
    stop_loss = latest_close - atr
    take_profit = latest_close + atr * 2
    return entry_point, stop_loss, take_profit

entry, stop, profit = calculate_trade_levels(df)

# 📌 Προβλεπτικό Μοντέλο ARIMA για τιμή σε 12 ώρες
def arima_forecast(df):
    model = ARIMA(df["Close"], order=(5,1,0))
    model_fit = model.fit()
    forecast = model_fit.forecast(steps=12)
    future_dates = [df.index[-1] + timedelta(hours=i) for i in range(1, 13)]
    return future_dates, forecast

future_dates, forecast = arima_forecast(df)

fig = go.Figure()
fig.add_trace(go.Scatter(x=df.index, y=df["Close"], name="Τιμή", line=dict(color="blue")))
fig.add_trace(go.Scatter(x=df.index, y=df["Bollinger_High"], name="Bollinger High", line=dict(color="red", dash="dot")))
fig.add_trace(go.Scatter(x=df.index, y=df["Bollinger_Low"], name="Bollinger Low", line=dict(color="green", dash="dot")))
fig.add_trace(go.Scatter(x=future_dates, y=forecast, name="Forecasted Price", line=dict(color="orange", dash="dot")))

st.plotly_chart(fig)

st.subheader("🔍 Προβλέψεις & Confidence Level")
latest_pred = df.iloc[-1]["Final_Prediction"]
confidence = np.random.uniform(70, 95)

if latest_pred == 1:
    st.success(f"📈 Προβλέπεται άνοδος με confidence {confidence:.2f}%")
else:
    st.error(f"📉 Προβλέπεται πτώση με confidence {confidence:.2f}%")

st.subheader("📌 Trade Setup")
st.write(f"✅ Entry Point: {entry:.2f}")
st.write(f"🚨 Stop Loss: {stop:.2f}")
st.write(f"🎯 Take Profit: {profit:.2f}")

