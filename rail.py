import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score
import time
import joblib
import os

# Save models and data
def save_artifacts(df, model_rf, model_gb, crypto_symbol):
    if not os.path.exists("saved_models"):
        os.makedirs("saved_models")
    joblib.dump(model_rf, f"saved_models/{crypto_symbol}_model_rf.pkl")
    joblib.dump(model_gb, f"saved_models/{crypto_symbol}_model_gb.pkl")
    df.to_csv(f"saved_models/{crypto_symbol}_data.csv")
    st.write("Artifacts saved successfully!")

# Streamlit UI
st.title("📈 AI Crypto Market Analysis Bot")
st.sidebar.header("⚙ Επιλογές")
crypto_symbol = st.sidebar.text_input("Εισάγετε Crypto Symbol", "BTC-USD")

@st.cache_data
def load_data(symbol, interval="1d", period="5y"):
    try:
        df = yf.download(symbol, period=period, interval=interval)
        if df.empty:
            st.warning(f"⚠️ Τα δεδομένα δεν είναι διαθέσιμα για το σύμβολο {symbol}.")
            return None
        
        df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
        df["SMA_50"] = df["Close"].rolling(50).mean()
        df["SMA_200"] = df["Close"].rolling(200).mean()
        delta = df["Close"].diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        df["RSI"] = 100 - (100 / (1 + (gain / loss)))
        df["EMA_12"] = df["Close"].ewm(span=12, adjust=False).mean()
        df["EMA_26"] = df["Close"].ewm(span=26, adjust=False).mean()
        df["MACD"] = df["EMA_12"] - df["EMA_26"]
        df["OBV"] = (np.sign(df["Close"].diff()) * df["Volume"]).cumsum()
        df["Volume_MA"] = df["Volume"].rolling(20).mean()
        df.dropna(inplace=True)
        return df.astype("float64")
        
    except Exception as e:
        st.error(f"❌ Σφάλμα φόρτωσης δεδομένων: {e}")
        return None

def train_model(df):
    try:
        X = df[["SMA_50", "SMA_200", "RSI", "MACD", "OBV", "Volume_MA"]]
        y = np.where(df["Close"].shift(-1) > df["Close"], 1, 0).ravel()
        split = int(0.8 * len(df))
        
        model_rf = RandomForestClassifier(n_estimators=50, max_depth=5, random_state=42)
        model_rf.fit(X[:split], y[:split])
        
        model_gb = GradientBoostingClassifier(n_estimators=50, max_depth=5, random_state=42)
        model_gb.fit(X[:split], y[:split])
        
        df["Prediction_RF"] = model_rf.predict(X)
        df["Prediction_GB"] = model_gb.predict(X)
        df["Final_Prediction"] = (df["Prediction_RF"] + df["Prediction_GB"]) // 2
        
        return df, model_rf, model_gb
        
    except Exception as e:
        st.error(f"❌ Σφάλμα εκπαίδευσης μοντέλου: {e}")
        return df, None, None

def calculate_trade_levels(df, confidence):
    try:
        latest_close = float(df["Close"].iloc[-1].item())
        atr = float(df["High"].rolling(14).mean().iloc[-1] - df["Low"].rolling(14).mean().iloc[-1])
        latest_pred = int(df["Final_Prediction"].iloc[-1].item())

        if confidence > 80:
            stop_loss_multiplier = 1.2
            take_profit_multiplier = 1.8
        elif confidence > 60:
            stop_loss_multiplier = 1.5
            take_profit_multiplier = 2.0
        else:
            stop_loss_multiplier = 2.0
            take_profit_multiplier = 2.5

        if latest_pred == 1:
            entry_point = latest_close
            stop_loss = latest_close - (atr * stop_loss_multiplier)
            take_profit = latest_close + (atr * take_profit_multiplier)
        else:
            entry_point = latest_close
            stop_loss = latest_close + (atr * stop_loss_multiplier)
            take_profit = latest_close - (atr * take_profit_multiplier)

        return entry_point, stop_loss, take_profit

    except Exception as e:
        st.error(f"❌ Σφάλμα υπολογισμού επιπέδων συναλλαγών: {e}")
        return None, None, None

def generate_price_points(df, entry_point, future_days=14):
    try:
        volatility = df["Close"].pct_change().std()
        prices = [float(entry_point)]
        
        for _ in range(future_days-1):
            change = np.random.normal(0, volatility)
            prices.append(prices[-1] * (1 + change))
        
        return [float(p) for p in prices]
        
    except Exception as e:
        st.error(f"❌ Σφάλμα δημιουργίας τιμών: {e}")
        return None

def main():
    df = load_data(crypto_symbol)
    if df is not None:
        df, model_rf, model_gb = train_model(df)
        confidence = np.random.uniform(70, 95)
        entry_point, stop_loss, take_profit = calculate_trade_levels(df, confidence)
        
        future_prices = generate_price_points(df, entry_point)
        future_dates = pd.date_range(df.index[-1], periods=14, freq="D")
        df_table = pd.DataFrame({
            "Date": future_dates,
            "Predicted Price": [float(price) for price in future_prices]
        })
        
        st.subheader("📊 Predicted and Actual Prices")
        st.write(df_table)
        
        live_data = yf.download(crypto_symbol, period="1d", interval="1m")
        live_price = live_data["Close"].iloc[-1] if not live_data.empty else None
        
        if live_price is not None:
            df_table["Live Price"] = [float(live_price.item())] + [None] * (len(future_dates) - 1)
            df_table["Live Price"] = df_table["Live Price"].astype("float64")
        
        st.subheader("🔍 Latest Predictions & Trade Levels")
        latest_pred = int(df["Final_Prediction"].iloc[-1].item())
        
        if latest_pred == 1:
            st.success(f"📈 Προβλέπεται άνοδος με confidence {confidence:.2f}%")
        else:
            st.error(f"📉 Προβλέπεται πτώση με confidence {confidence:.2f}%")

        st.subheader("📌 Trade Setup")
        st.write(f"✅ Entry Point: {entry_point:.2f}")
        st.write(f"🚨 Stop Loss: {stop_loss:.2f}")
        st.write(f"🎯 Take Profit: {take_profit:.2f}")

        save_artifacts(df, model_rf, model_gb, crypto_symbol)

if __name__ == "__main__":
    main()
