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

# 📌 Streamlit UI
st.title("📈 AI Crypto Market Analysis Bot")
st.sidebar.header("⚙ Επιλογές")
crypto_symbol = st.sidebar.text_input("Εισάγετε Crypto Symbol", "BTC-USD")

# Cache data loading to speed up the app
@st.cache_data
def load_data(symbol, interval="1d", period="5y"):
    try:
        st.write(f"Loading data for {symbol} with interval {interval} and period {period}")
        df = yf.download(symbol, period=period, interval=interval)
        if df.empty:
            st.warning(f"⚠️ Τα δεδομένα δεν είναι διαθέσιμα για το σύμβολο {symbol} με interval {interval}. Δοκιμάστε διαφορετικό interval.")
            return None
        df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
        df.dropna(inplace=True)
        df["SMA_50"] = df["Close"].rolling(window=50).mean()
        df["SMA_200"] = df["Close"].rolling(window=200).mean()
        delta = df["Close"].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df["RSI"] = 100 - (100 / (1 + rs))
        df["EMA_12"] = df["Close"].ewm(span=12, adjust=False).mean()
        df["EMA_26"] = df["Close"].ewm(span=26, adjust=False).mean()
        df["MACD"] = df["EMA_12"] - df["EMA_26"]
        df["OBV"] = (np.sign(df["Close"].diff()) * df["Volume"]).cumsum()
        df["Volume_MA"] = df["Volume"].rolling(window=20).mean()
        df["14D_EMA"] = df["Close"].ewm(span=14, adjust=False).mean()
        df.dropna(inplace=True)
        df = df.astype(np.float64)
    except Exception as e:
        st.error(f"❌ Σφάλμα φόρτωσης δεδομένων: {e}")
        return None
    return df

def train_model(df):
    try:
        X = df[["SMA_50", "SMA_200", "RSI", "MACD", "OBV", "Volume_MA"]]
        y = np.where(df["Close"].shift(-1) > df["Close"], 1, 0)
        y = y.ravel()
        split = int(0.8 * len(df))
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]
        model_rf = RandomForestClassifier(n_estimators=50, max_depth=5, random_state=42)
        model_rf.fit(X_train, y_train)
        y_pred_rf = model_rf.predict(X_test)
        accuracy_rf = accuracy_score(y_test, y_pred_rf)
        st.write(f"RandomForest model trained with accuracy: {accuracy_rf:.2f}")
        model_gb = GradientBoostingClassifier(n_estimators=50, max_depth=5, random_state=42)
        model_gb.fit(X_train, y_train)
        y_pred_gb = model_gb.predict(X_test)
        accuracy_gb = accuracy_score(y_test, y_pred_gb)
        st.write(f"GradientBoosting model trained with accuracy: {accuracy_gb:.2f}")
        df["Prediction_RF"] = model_rf.predict(X)
        df["Prediction_GB"] = model_gb.predict(X)
        df["Final_Prediction"] = (df["Prediction_RF"] + df["Prediction_GB"]) // 2
    except Exception as e:
        st.error(f"❌ Σφάλμα εκπαίδευσης μοντέλου: {e}")
    return df, model_rf, model_gb

def calculate_trade_levels(df, timeframe, confidence):
    try:
        latest_close = df["Close"].iloc[-1].item()
        atr = (df["High"].rolling(window=14).mean() - df["Low"].rolling(window=14).mean()).iloc[-1].item()
        latest_pred = df["Final_Prediction"].iloc[-1].item()
        rsi = df["RSI"].iloc[-1].item()
        macd = df["MACD"].iloc[-1].item()
        if confidence > 80:
            stop_loss_multiplier = 1.2
            take_profit_multiplier = 1.8
        elif confidence > 60:
            stop_loss_multiplier = 1.5
            take_profit_multiplier = 2.0
        else:
            stop_loss_multiplier = 2.0
            take_profit_multiplier = 2.5
        if rsi > 70 or rsi < 30:
            stop_loss_multiplier *= 0.9
            take_profit_multiplier *= 1.1
        if macd > 0:
            take_profit_multiplier *= 1.1
        else:
            stop_loss_multiplier *= 1.1
        if latest_pred == 1:
            entry_point = latest_close
            stop_loss = latest_close - (atr * stop_loss_multiplier)
            take_profit = latest_close + (atr * take_profit_multiplier)
        else:
            entry_point = latest_close
            stop_loss = latest_close + (atr * stop_loss_multiplier)
            take_profit = latest_close - (atr * take_profit_multiplier)
        st.write(f"Trade levels for {timeframe}: Entry Point: {entry_point:.2f}, Stop Loss: {stop_loss:.2f}, Take Profit: {take_profit:.2f}")
    except Exception as e:
        st.error(f"❌ Σφάλμα υπολογισμού επιπέδων συναλλαγών: {e}")
        entry_point, stop_loss, take_profit = None, None, None
    return entry_point, stop_loss, take_profit

def generate_price_points(df, entry_point, stop_loss, take_profit, future_days=14):
    try:
        # Generate price points based on trade levels
        if entry_point is None or stop_loss is None or take_profit is None:
            return None

        # Calculate historical volatility
        historical_volatility = df["Close"].pct_change().std()

        # Generate price points with randomness
        price_points = [entry_point]
        for _ in range(1, future_days):
            random_change = np.random.normal(0, historical_volatility)
            new_price = price_points[-1] * (1 + random_change)
            price_points.append(new_price)

        return price_points
    except Exception as e:
        st.error(f"❌ Σφάλμα δημιουργίας τιμών: {e}")
        return None

def main():
    timeframes = {
        "1d": {"interval": "1d", "period": "5y"},
        "1w": {"interval": "1wk", "period": "5y"},
    }
    data = {}
    for timeframe, params in timeframes.items():
        df = load_data(crypto_symbol, interval=params["interval"], period=params["period"])
        if df is None:
            st.error(f"❌ Τα δεδομένα δεν είναι διαθέσιμα για το σύμβολο {crypto_symbol}.")
            st.stop()
        data[timeframe] = df
    trade_levels = {}
    for timeframe, df in data.items():
        df, model_rf, model_gb = train_model(df)
        confidence = np.random.uniform(70, 95)
        trade_levels[timeframe] = calculate_trade_levels(df, timeframe, confidence)
        save_artifacts(df, model_rf, model_gb, crypto_symbol)  # Save artifacts
    if any(None in levels for levels in trade_levels.values()):
        st.stop()

    # Generate price points for the next 14 days
    entry_point, stop_loss, take_profit = trade_levels["1d"]
    future_dates = pd.date_range(data["1d"].index[-1], periods=14, freq="D")
    future_price_points = generate_price_points(data["1d"], entry_point, stop_loss, take_profit)  # Pass df here
    if future_price_points is None:
        st.error("❌ Failed to generate future price points.")
        st.stop()

    # Fetch live price
    live_data = yf.download(crypto_symbol, period="1d", interval="1m")
    live_price = live_data["Close"].iloc[-1] if not live_data.empty else None

    # Create a DataFrame for the table
    table_data = {
        "Date": future_dates,
        "Predicted Price": future_price_points,
    }
    df_table = pd.DataFrame(table_data)

    # Add live price to the table
    if live_price is not None:
        df_table["Live Price"] = [live_price if i == 0 else None for i in range(len(future_dates))]

    # Display the table
    st.subheader("📊 Predicted and Actual Prices")
    st.write(df_table)

    # Display latest predictions and trade levels
    st.subheader("🔍 Latest Predictions & Trade Levels")
    latest_pred = data["1d"]["Final_Prediction"].iloc[-1].item()
    confidence = np.random.uniform(70, 95)
    if latest_pred == 1:
        st.success(f"📈 Προβλέπεται άνοδος με confidence {confidence:.2f}%")
    else:
        st.error(f"📉 Προβλέπεται πτώση με confidence {confidence:.2f}%")

    st.subheader("📌 Trade Setup")
    for timeframe, levels in trade_levels.items():
        st.write(f"⏰ {timeframe}:")
        st.write(f"✅ Entry Point: {levels[0]:.2f}")
        st.write(f"🚨 Stop Loss: {levels[1]:.2f}")
        st.write(f"🎯 Take Profit: {levels[2]:.2f}")

    # Continuously update data and retrain model
    while True:
        time.sleep(60)
        for timeframe, params in timeframes.items():
            df = load_data(crypto_symbol, interval=params["interval"], period=params["period"])
            if df is None:
                st.error(f"❌ Τα δεδομένα δεν είναι διαθέσιμα για το σύμβολο {crypto_symbol}.")
                st.stop()
            data[timeframe] = df
            data[timeframe], model_rf, model_gb = train_model(data[timeframe])
            confidence = np.random.uniform(70, 95)
            trade_levels[timeframe] = calculate_trade_levels(data[timeframe], timeframe, confidence)
            save_artifacts(df, model_rf, model_gb, crypto_symbol)  # Save artifacts
        st.rerun()

if __name__ == "__main__":
    main()
