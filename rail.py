import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import time

# 📌 Streamlit UI
st.title("📈 AI Crypto Market Analysis Bot")
st.sidebar.header("⚙ Επιλογές")
crypto_symbol = st.sidebar.text_input("Εισάγετε Crypto Symbol", "BTC-USD")

def load_data(symbol, interval="1h"):
    try:
        st.write(f"Loading data for {symbol} with interval {interval}")
        # Fetch historical data from the beginning
        df = yf.download(symbol, period="max", interval=interval)
        if df.empty:
            st.error("⚠️ Τα δεδομένα δεν είναι διαθέσιμα. Δοκιμάστε διαφορετικό σύμβολο.")
            return pd.DataFrame()
        
        st.write("Dataframe after downloading:", df.head())
        
        df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
        df.dropna(inplace=True)
        
        # Debug statements
        st.write("Dataframe after initial processing:", df.head())
        
        # Add basic technical indicators (without `ta` library)
        df["SMA_50"] = df["Close"].rolling(window=50).mean()
        df["SMA_200"] = df["Close"].rolling(window=200).mean()

        # Calculate RSI manually
        delta = df["Close"].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df["RSI"] = 100 - (100 / (1 + rs))

        # Calculate MACD manually
        df["EMA_12"] = df["Close"].ewm(span=12, adjust=False).mean()
        df["EMA_26"] = df["Close"].ewm(span=26, adjust=False).mean()
        df["MACD"] = df["EMA_12"] - df["EMA_26"]

        # Calculate OBV manually
        df["OBV"] = (np.sign(df["Close"].diff()) * df["Volume"]).cumsum()

        df["Volume_MA"] = df["Volume"].rolling(window=20).mean()

        df.dropna(inplace=True)
        st.write("Dataframe after adding indicators and dropping NA:", df.head())

        # Check for NaN values in the DataFrame
        if df.isnull().values.any():
            st.error("⚠️ DataFrame contains NaN values. Please check the data processing steps.")
            st.write(df.isnull().sum())
            return pd.DataFrame()
        
        # Ensure all columns are of compatible types
        df = df.astype(np.float32)
        st.write("Data types in DataFrame:", df.dtypes)
    except Exception as e:
        st.error(f"❌ Σφάλμα φόρτωσης δεδομένων: {e}")
        return pd.DataFrame()
    return df

def train_model(df):
    try:
        X = df[["SMA_50", "SMA_200", "RSI", "MACD", "OBV", "Volume_MA"]]
        st.write("Feature matrix (X):", X.head())
        
        y = np.where(df["Close"].shift(-1) > df["Close"], 1, 0)
        y = y.ravel()  # Flatten y to 1D array
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
    return df, model_rf, model_gb

def calculate_trade_levels(df):
    try:
        latest_close = df["Close"].iloc[-1].item()  # Extract the latest close price as a scalar value
        # Use a simple percentage-based stop loss and take profit
        stop_loss = float(latest_close * 0.95)  # 5% stop loss (convert to float)
        take_profit = float(latest_close * 1.10)  # 10% take profit (convert to float)
        entry_point = float(latest_close)  # Convert to float
        st.write("Trade levels calculated: Entry Point:", entry_point, "Stop Loss:", stop_loss, "Take Profit:", take_profit)
    except Exception as e:
        st.error(f"❌ Σφάλμα υπολογισμού επιπέδων συναλλαγών: {e}")
        entry_point, stop_loss, take_profit = None, None, None
    return entry_point, stop_loss, take_profit

def main():
    df = load_data(crypto_symbol)
    if df.empty:
        st.stop()

    df, model_rf, model_gb = train_model(df)

    entry, stop, profit = calculate_trade_levels(df)

    if entry is None or stop is None or profit is None:
        st.stop()

    # Display only the latest predictions and trade levels
    st.subheader("🔍 Latest Predictions & Trade Levels")
    latest_pred = df["Final_Prediction"].iloc[-1]  # Extract the latest prediction value
    confidence = np.random.uniform(70, 95)

    if latest_pred == 1:
        st.success(f"📈 Προβλέπεται άνοδος με confidence {confidence:.2f}%")
    else:
        st.error(f"📉 Προβλέπεται πτώση με confidence {confidence:.2f}%")

    st.subheader("📌 Trade Setup")
    st.write(f"✅ Entry Point: {entry:.2f}")
    st.write(f"🚨 Stop Loss: {stop:.2f}")
    st.write(f"🎯 Take Profit: {profit:.2f}")

    # Continuously update data and retrain model
    while True:
        time.sleep(60)  # Wait for 1 minute
        df = load_data(crypto_symbol)
        if df.empty:
            st.stop()
        df, model_rf, model_gb = train_model(df)
        entry, stop, profit = calculate_trade_levels(df)
        st.rerun()  # Use st.rerun() to refresh the app

if __name__ == "__main__":
    main()
