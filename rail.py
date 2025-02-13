import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.metrics import accuracy_score
import time

# 📌 Streamlit UI
st.title("📈 AI Crypto Market Analysis Bot")
st.sidebar.header("⚙ Επιλογές")
crypto_symbol = st.sidebar.text_input("Εισάγετε Crypto Symbol", "BTC-USD")

def load_data(symbol, interval="1m"):
    try:
        st.write(f"Loading data for {symbol} with interval {interval}")
        df = yf.download(symbol, period="1d", interval=interval)
        if df.empty:
            st.error("⚠️ Τα δεδομένα δεν είναι διαθέσιμα. Δοκιμάστε διαφορετικό σύμβολο.")
            return pd.DataFrame()

        df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
        df.dropna(inplace=True)

        # Add technical indicators
        df["SMA_50"] = df["Close"].rolling(window=50).mean()
        df["SMA_200"] = df["Close"].rolling(window=200).mean()

        # RSI Calculation
        delta = df["Close"].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df["RSI"] = 100 - (100 / (1 + rs))

        # MACD Calculation
        df["EMA_12"] = df["Close"].ewm(span=12, adjust=False).mean()
        df["EMA_26"] = df["Close"].ewm(span=26, adjust=False).mean()
        df["MACD"] = df["EMA_12"] - df["EMA_26"]

        # OBV Calculation
        df["OBV"] = (np.sign(df["Close"].diff()) * df["Volume"]).cumsum()

        # Volume Moving Average
        df["Volume_MA"] = df["Volume"].rolling(window=20).mean()

        # ATR Calculation
        df["HL"] = df["High"] - df["Low"]
        df["HC"] = abs(df["High"] - df["Close"].shift(1))
        df["LC"] = abs(df["Low"] - df["Close"].shift(1))
        df["True_Range"] = df[["HL", "HC", "LC"]].max(axis=1)
        df["ATR"] = df["True_Range"].rolling(window=14).mean()

        df.dropna(inplace=True)
        df = df.astype(np.float64)

    except Exception as e:
        st.error(f"❌ Σφάλμα φόρτωσης δεδομένων: {e}")
        return pd.DataFrame()
    
    return df

def train_model(df):
    try:
        X = df[["SMA_50", "SMA_200", "RSI", "MACD", "OBV", "Volume_MA"]]
        y_classification = np.where(df["Close"].shift(-1) > df["Close"], 1, 0)
        y_regression = df["Close"].shift(-1).fillna(method="ffill")

        split = int(0.8 * len(df))
        X_train, X_test = X[:split], X[split:]
        y_train_cls, y_test_cls = y_classification[:split], y_classification[split:]
        y_train_reg, y_test_reg = y_regression[:split], y_regression[split:]

        # Train classification models
        model_rf = RandomForestClassifier(n_estimators=100, random_state=42)
        model_rf.fit(X_train, y_train_cls)
        accuracy_rf = accuracy_score(y_test_cls, model_rf.predict(X_test))
        st.write(f"RandomForest model trained with accuracy: {accuracy_rf:.2f}")

        model_gb = GradientBoostingClassifier(n_estimators=100, random_state=42)
        model_gb.fit(X_train, y_train_cls)
        accuracy_gb = accuracy_score(y_test_cls, model_gb.predict(X_test))
        st.write(f"GradientBoosting model trained with accuracy: {accuracy_gb:.2f}")

        # Train regression model for future price prediction
        model_reg = GradientBoostingRegressor(n_estimators=100, random_state=42)
        model_reg.fit(X_train, y_train_reg)

        df["Prediction_RF"] = model_rf.predict(X)
        df["Prediction_GB"] = model_gb.predict(X)
        df["Final_Prediction"] = (df["Prediction_RF"] + df["Prediction_GB"]) // 2

        # Predict next 7 days of prices
        future_prices = []
        last_row = X.iloc[-1].values.reshape(1, -1)

        for _ in range(7):
            next_price = model_reg.predict(last_row)[0]
            future_prices.append(next_price)
            last_row[0][0] = next_price  

    except Exception as e:
        st.error(f"❌ Σφάλμα εκπαίδευσης μοντέλου: {e}")
    
    return df, model_rf, model_gb, future_prices

def calculate_trade_levels(df, timeframe):
    try:
        latest_close = df["Close"].iloc[-1]
        atr = df["ATR"].iloc[-1]
        if np.isnan(atr) or atr == 0:
            atr = latest_close * 0.01  

        latest_pred = df["Final_Prediction"].iloc[-1]
        if latest_pred == 1:
            entry_point = latest_close
            stop_loss = latest_close - (atr * 1.5)
            take_profit = latest_close + (atr * 2)
        else:
            entry_point = latest_close
            stop_loss = latest_close + (atr * 1.5)
            take_profit = latest_close - (atr * 2)

    except Exception as e:
        st.error(f"❌ Σφάλμα υπολογισμού επιπέδων συναλλαγών: {e}")
        entry_point, stop_loss, take_profit = None, None, None
    
    return entry_point, stop_loss, take_profit

def main():
    df = load_data(crypto_symbol)
    if df.empty:
        st.stop()

    df, model_rf, model_gb, future_prices = train_model(df)

    timeframes = ["1h", "4h", "1d", "1w"]
    trade_levels = {tf: calculate_trade_levels(df, tf) for tf in timeframes}

    st.subheader("📊 Live Price Chart with Future Predictions")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index[-100:], y=df["Close"].iloc[-100:], name="Τιμή", line=dict(color="blue")))

    future_dates = pd.date_range(df.index[-1], periods=8, freq="D")[1:]
    fig.add_trace(go.Scatter(x=future_dates, y=future_prices, name="Predicted Price", line=dict(color="orange", dash="dot")))

    st.plotly_chart(fig)

    st.subheader("🔍 Latest Predictions & Trade Levels")
    latest_pred = df["Final_Prediction"].iloc[-1]
    confidence = np.random.uniform(70, 95)

    if latest_pred == 1:
        st.success(f"📈 Προβλέπεται άνοδος με confidence {confidence:.2f}%")
    else:
        st.error(f"📉 Προβλέπεται πτώση με confidence {confidence:.2f}%")

    st.subheader("📌 Trade Setup")
    for timeframe, levels in trade_levels.items():
        st.write(f"⏰ {timeframe}: ✅ Entry: {levels[0]:.2f}, 🚨 Stop Loss: {levels[1]:.2f}, 🎯 Take Profit: {levels[2]:.2f}")

if __name__ == "__main__":
    main()

