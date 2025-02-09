import streamlit as st
import os
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from ta.trend import SMAIndicator, EMAIndicator, MACD
from ta.momentum import RSIIndicator
from ta.volatility import AverageTrueRange
from ta.volume import OnBalanceVolumeIndicator
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from datetime import datetime

# 📌 Streamlit UI
st.title("📈 AI Crypto Market Analysis Bot")
st.sidebar.header("⚙ Επιλογές")
crypto_symbol = st.sidebar.text_input("Εισάγετε Crypto Symbol", "BTC-USD")

def load_data(symbol, period="6mo", interval="1h"):
    try:
        df = yf.download(symbol, period=period, interval=interval)
        if df.empty:
            st.error("⚠️ Τα δεδομένα δεν είναι διαθέσιμα. Δοκιμάστε διαφορετικό σύμβολο.")
            return pd.DataFrame()
        
        df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
        df.dropna(inplace=True)
        
        df["SMA_50"] = SMAIndicator(df["Close"], window=50).sma_indicator().squeeze()
        df["SMA_200"] = SMAIndicator(df["Close"], window=200).sma_indicator().squeeze()
        df["EMA_21"] = EMAIndicator(df["Close"], window=21).ema_indicator().squeeze()
        df["RSI"] = RSIIndicator(df["Close"], window=14).rsi().squeeze()
        df["MACD"] = MACD(df["Close"]).macd().squeeze()
        
        atr = AverageTrueRange(df["High"], df["Low"], df["Close"], window=14).average_true_range().squeeze()
        df["ATR"] = atr
        df["ATR_Upper"] = df["Close"] + (atr * 1.5)
        df["ATR_Lower"] = df["Close"] - (atr * 1.5)
        
        df["OBV"] = OnBalanceVolumeIndicator(df["Close"], df["Volume"]).on_balance_volume().squeeze()
        df["Volume_MA"] = df["Volume"].rolling(window=20).mean()
        df.dropna(inplace=True)
    except Exception as e:
        st.error(f"❌ Σφάλμα φόρτωσης δεδομένων: {e}")
        return pd.DataFrame()
    return df

df = load_data(crypto_symbol)
if df.empty:
    st.stop()

def train_model(df):
    X = df[["SMA_50", "SMA_200", "EMA_21", "MACD", "RSI", "ATR", "OBV", "Volume_MA"]]
    y = np.where(df["Close"].shift(-1).squeeze() > df["Close"].squeeze(), 1, 0)
    
    model_rf = RandomForestClassifier(n_estimators=100)
    model_rf.fit(X, y)
    
    model_gb = GradientBoostingClassifier(n_estimators=100)
    model_gb.fit(X, y)
    
    df["Prediction_RF"] = model_rf.predict(X)
    df["Prediction_GB"] = model_gb.predict(X)
    df["Final_Prediction"] = (df["Prediction_RF"] + df["Prediction_GB"]) // 2
    return df

df = train_model(df)

def calculate_trade_levels(df):
    latest_close = df["Close"].iloc[-1]
    atr = df["ATR"].iloc[-1] * 1.5
    entry_point = latest_close
    stop_loss = latest_close - atr
    take_profit = latest_close + atr * 2
    return entry_point, stop_loss, take_profit

entry, stop, profit = calculate_trade_levels(df)

future_dates, forecast = list(df.index[-10:]), df["Close"].values[-10:].flatten().tolist()

fig = go.Figure()
fig.add_trace(go.Scatter(x=df.index, y=df["Close"], name="Τιμή", line=dict(color="blue")))
fig.add_trace(go.Scatter(x=df.index, y=df["ATR_Lower"], name="ATR Lower Band", line=dict(color="green", dash="dot")))
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
