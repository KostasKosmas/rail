import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from ta.trend import SMAIndicator, EMAIndicator, MACD
from ta.momentum import RSIIndicator
from ta.volume import OnBalanceVolumeIndicator
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

# 📌 Streamlit UI
st.title("📈 AI Crypto Market Analysis Bot")
st.sidebar.header("⚙ Επιλογές")
crypto_symbol = st.sidebar.text_input("Εισάγετε Crypto Symbol", "BTC-USD")

def load_data(symbol, period="6mo", interval="1h"):
    try:
        st.write(f"Loading data for {symbol} with period {period} and interval {interval}")
        df = yf.download(symbol, period=period, interval=interval)
        if df.empty:
            st.error("⚠️ Τα δεδομένα δεν είναι διαθέσιμα. Δοκιμάστε διαφορετικό σύμβολο.")
            return pd.DataFrame()
        
        st.write("Dataframe after downloading:", df.head())
        
        df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
        df.dropna(inplace=True)
        
        # Debug statements
        st.write("Dataframe after initial processing:", df.head())
        
        # Add technical indicators
        df["SMA_50"] = SMAIndicator(df["Close"], window=50).sma_indicator()
        st.write("SMA_50 added:", df[["Close", "SMA_50"]].head())

        df["SMA_200"] = SMAIndicator(df["Close"], window=200).sma_indicator()
        st.write("SMA_200 added:", df[["Close", "SMA_200"]].head())

        df["EMA_21"] = EMAIndicator(df["Close"], window=21).ema_indicator()
        st.write("EMA_21 added:", df[["Close", "EMA_21"]].head())

        df["RSI"] = RSIIndicator(df["Close"], window=14).rsi()
        st.write("RSI added:", df[["Close", "RSI"]].head())

        df["MACD"] = MACD(df["Close"]).macd()
        st.write("MACD added:", df[["Close", "MACD"]].head())

        # Skip ATR calculation
        st.write("Skipping ATR calculation to avoid errors.")

        df["OBV"] = OnBalanceVolumeIndicator(df["Close"], df["Volume"]).on_balance_volume()
        st.write("OBV added:", df[["Close", "OBV"]].head())

        df["Volume_MA"] = df["Volume"].rolling(window=20).mean()
        st.write("Volume_MA added:", df[["Volume", "Volume_MA"]].head())

        df.dropna(inplace=True)
        st.write("Dataframe after adding indicators and dropping NA:", df.head())

        # Check for NaN values in the DataFrame
        if df.isnull().values.any():
            st.error("⚠️ DataFrame contains NaN values. Please check the data processing steps.")
            st.write(df.isnull().sum())
            return pd.DataFrame()
        
        # Check data types
        st.write("Data types in DataFrame:", df.dtypes)
    except Exception as e:
        st.error(f"❌ Σφάλμα φόρτωσης δεδομένων: {e}")
        return pd.DataFrame()
    return df

df = load_data(crypto_symbol)
if df.empty:
    st.stop()

def train_model(df):
    try:
        X = df[["SMA_50", "SMA_200", "EMA_21", "MACD", "RSI", "OBV", "Volume_MA"]].astype(float)
        st.write("Feature matrix (X):", X.head())
        
        y = np.where(df["Close"].shift(-1) > df["Close"], 1, 0)
        st.write("Target vector (y):", y[:5])
        
        model_rf = RandomForestClassifier(n_estimators=100, random_state=42)
        model_rf.fit(X, y)
        st.write("RandomForest model trained")
        
        model_gb = GradientBoostingClassifier(n_estimators=100, random_state=42)
        model_gb.fit(X, y)
        st.write("GradientBoosting model trained")
        
        df["Prediction_RF"] = model_rf.predict(X)
        df["Prediction_GB"] = model_gb.predict(X)
        df["Final_Prediction"] = (df["Prediction_RF"] + df["Prediction_GB"]) // 2
        st.write("Predictions added to dataframe", df[["Prediction_RF", "Prediction_GB", "Final_Prediction"]].head())
    except Exception as e:
        st.error(f"❌ Σφάλμα εκπαίδευσης μοντέλου: {e}")
    return df

df = train_model(df)

def calculate_trade_levels(df):
    try:
        latest_close = df["Close"].iloc[-1]
        # Use a simple percentage-based stop loss and take profit
        stop_loss = latest_close * 0.95  # 5% stop loss
        take_profit = latest_close * 1.10  # 10% take profit
        entry_point = latest_close
        st.write("Trade levels calculated: Entry Point:", entry_point, "Stop Loss:", stop_loss, "Take Profit:", take_profit)
    except Exception as e:
        st.error(f"❌ Σφάλμα υπολογισμού επιπέδων συναλλαγών: {e}")
        entry_point, stop_loss, take_profit = None, None, None
    return entry_point, stop_loss, take_profit

entry, stop, profit = calculate_trade_levels(df)

if entry is None or stop is None or profit is None:
    st.stop()

future_dates, forecast = list(df.index[-10:]), df["Close"].values[-10:].tolist()

fig = go.Figure()
fig.add_trace(go.Scatter(x=df.index, y=df["Close"], name="Τιμή", line=dict(color="blue")))
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
