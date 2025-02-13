import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
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

# Calculate Bollinger Bands
def calculate_bollinger_bands(df, window=20, num_std=2):
    df["SMA"] = df["Close"].rolling(window=window).mean()
    df["STD"] = df["Close"].rolling(window=window).std()
    df["Upper Band"] = df["SMA"] + (df["STD"] * num_std)
    df["Lower Band"] = df["SMA"] - (df["STD"] * num_std)
    return df

# Calculate MACD
def calculate_macd(df, short_window=12, long_window=26, signal_window=9):
    df["EMA_12"] = df["Close"].ewm(span=short_window, adjust=False).mean()
    df["EMA_26"] = df["Close"].ewm(span=long_window, adjust=False).mean()
    df["MACD"] = df["EMA_12"] - df["EMA_26"]
    df["Signal Line"] = df["MACD"].ewm(span=signal_window, adjust=False).mean()
    return df

# Calculate RSI
def calculate_rsi(df, window=14):
    delta = df["Close"].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    df["RSI"] = 100 - (100 / (1 + rs))
    return df

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
        df = calculate_bollinger_bands(df)
        df = calculate_macd(df)
        df = calculate_rsi(df)
        df.dropna(inplace=True)
        df = df.astype(np.float64)
    except Exception as e:
        st.error(f"❌ Σφάλμα φόρτωσης δεδομένων: {e}")
        return None
    return df

def train_model(df):
    try:
        # Use technical indicators as features
        X = df[["SMA", "Upper Band", "Lower Band", "MACD", "Signal Line", "RSI"]]
        y = df["Close"].shift(-1)  # Predict the next day's closing price
        y.dropna(inplace=True)
        X = X.iloc[:-1]  # Align X and y

        # Split data into training and testing sets
        split = int(0.8 * len(X))
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]

        # Train RandomForestRegressor
        model_rf = RandomForestRegressor(n_estimators=50, max_depth=5, random_state=42)
        model_rf.fit(X_train, y_train)
        y_pred_rf = model_rf.predict(X_test)
        mae_rf = mean_absolute_error(y_test, y_pred_rf)
        st.write(f"RandomForest model trained with MAE: {mae_rf:.2f}")

        # Train GradientBoostingRegressor
        model_gb = GradientBoostingRegressor(n_estimators=50, max_depth=5, random_state=42)
        model_gb.fit(X_train, y_train)
        y_pred_gb = model_gb.predict(X_test)
        mae_gb = mean_absolute_error(y_test, y_pred_gb)
        st.write(f"GradientBoosting model trained with MAE: {mae_gb:.2f}")

        # Add predictions to the DataFrame
        df["Prediction_RF"] = np.nan
        df["Prediction_RF"].iloc[split:] = model_rf.predict(X[split:])
        df["Prediction_GB"] = np.nan
        df["Prediction_GB"].iloc[split:] = model_gb.predict(X[split:])
    except Exception as e:
        st.error(f"❌ Σφάλμα εκπαίδευσης μοντέλου: {e}")
    return df, model_rf, model_gb

def generate_price_points(df, future_days=14):
    try:
        # Use the last row's predictions to generate future price points
        last_row = df.iloc[-1]
        if last_row["Prediction_RF"] > last_row["Prediction_GB"]:
            # Use RandomForest prediction as the base
            base_price = last_row["Prediction_RF"]
        else:
            # Use GradientBoosting prediction as the base
            base_price = last_row["Prediction_GB"]

        # Generate price points based on the base price
        price_points = np.linspace(base_price, base_price * 1.05, future_days)  # 5% increase over 14 days
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
        save_artifacts(df, model_rf, model_gb, crypto_symbol)  # Save artifacts

    # Generate price points for the next 14 days
    future_dates = pd.date_range(data["1d"].index[-1], periods=14, freq="D")
    future_price_points = generate_price_points(data["1d"])
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
            save_artifacts(df, model_rf, model_gb, crypto_symbol)  # Save artifacts
        st.rerun()

if __name__ == "__main__":
    main()
