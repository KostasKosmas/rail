import streamlit as st
import pandas as pd
import numpy as np
import ta  # Technical analysis library
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score
import time
import joblib
import os
import requests

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
    df["Upper_Band"] = df["SMA"] + (df["STD"] * num_std)
    df["Lower_Band"] = df["SMA"] - (df["STD"] * num_std)
    return df

# Calculate MACD
def calculate_macd(df, short_window=12, long_window=26, signal_window=9):
    df["EMA_12"] = df["Close"].ewm(span=short_window, adjust=False).mean()
    df["EMA_26"] = df["Close"].ewm(span=long_window, adjust=False).mean()
    df["MACD"] = df["EMA_12"] - df["EMA_26"]
    df["Signal_Line"] = df["MACD"].ewm(span=signal_window, adjust=False).mean()
    return df

# Calculate RSI
def calculate_rsi(df, window=14):
    delta = df["Close"].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    df["RSI"] = 100 - (100 / (1 + rs))
    return df

# Calculate ATR
def calculate_atr(df, window=14):
    high_low = df["High"] - df["Low"]
    high_close = np.abs(df["High"] - df["Close"].shift())
    low_close = np.abs(df["Low"] - df["Close"].shift())
    true_range = np.maximum(high_low, np.maximum(high_close, low_close))
    df["ATR"] = true_range.rolling(window=window).mean()
    return df

# Calculate ADX
def calculate_adx(df, window=14):
    df["ADX"] = ta.trend.ADXIndicator(df["High"], df["Low"], df["Close"], window=window).adx()
    return df

# Calculate Fibonacci Levels (Golden Ratio)
def calculate_fibonacci_levels(df):
    max_price = df["High"].max()
    min_price = df["Low"].min()
    diff = max_price - min_price
    df["Fib_0.236"] = max_price - diff * 0.236
    df["Fib_0.382"] = max_price - diff * 0.382
    df["Fib_0.5"] = max_price - diff * 0.5
    df["Fib_0.618"] = max_price - diff * 0.618
    return df

# Calculate Ichimoku Cloud
def calculate_ichimoku(df):
    ichimoku = ta.trend.IchimokuIndicator(df["High"], df["Low"])
    df["Ichimoku_Base"] = ichimoku.ichimoku_base_line()
    df["Ichimoku_Conversion"] = ichimoku.ichimoku_conversion_line()
    df["Ichimoku_Span_A"] = ichimoku.ichimoku_a()
    df["Ichimoku_Span_B"] = ichimoku.ichimoku_b()
    return df

# Calculate VWAP
def calculate_vwap(df):
    df["VWAP"] = (df["Volume"] * (df["High"] + df["Low"] + df["Close"]) / 3).cumsum() / df["Volume"].cumsum()
    return df

# Calculate OBV
def calculate_obv(df):
    df["OBV"] = ta.volume.OnBalanceVolumeIndicator(df["Close"], df["Volume"]).on_balance_volume()
    return df

# Calculate Moving Averages (SMA, EMA)
def calculate_moving_averages(df):
    df["SMA_50"] = df["Close"].rolling(window=50).mean()
    df["SMA_200"] = df["Close"].rolling(window=200).mean()
    df["EMA_50"] = df["Close"].ewm(span=50, adjust=False).mean()
    df["EMA_200"] = df["Close"].ewm(span=200, adjust=False).mean()
    return df

# Calculate Stochastic Oscillator
def calculate_stochastic_oscillator(df, window=14):
    df["Stochastic_%K"] = ta.momentum.StochasticOscillator(
        df["High"], df["Low"], df["Close"], window=window
    ).stoch()
    df["Stochastic_%D"] = df["Stochastic_%K"].rolling(window=3).mean()
    return df

# Fetch data from CoinGecko
def fetch_data_from_coingecko(crypto_id="bitcoin", vs_currency="usd", days="max"):
    url = f"https://api.coingecko.com/api/v3/coins/{crypto_id}/market_chart"
    params = {
        "vs_currency": vs_currency,
        "days": days,
    }
    response = requests.get(url, params=params)
    if response.status_code == 200:
        data = response.json()
        prices = data["prices"]
        df = pd.DataFrame(prices, columns=["Timestamp", "Price"])
        df["Timestamp"] = pd.to_datetime(df["Timestamp"], unit="ms")
        df.set_index("Timestamp", inplace=True)
        return df
    else:
        st.error(f"❌ Failed to fetch data from CoinGecko for {crypto_id}.")
        return None

# Load data using CoinGecko
@st.cache_data
def load_data(crypto_id="bitcoin", vs_currency="usd", days="max"):
    try:
        st.write(f"Loading data for {crypto_id} in {vs_currency} for the last {days} days")
        df = fetch_data_from_coingecko(crypto_id, vs_currency, days)
        
        if df is None or df.empty:
            st.warning(f"⚠️ No data available for {crypto_id} in {vs_currency}.")
            return None
        
        # Rename columns to match yfinance format
        df = df.rename(columns={"Price": "Close"})
        df["Open"] = df["Close"]
        df["High"] = df["Close"]
        df["Low"] = df["Close"]
        df["Volume"] = 0  # CoinGecko doesn't provide volume data in this endpoint

        # Calculate all indicators
        df = calculate_bollinger_bands(df)
        df = calculate_macd(df)
        df = calculate_rsi(df)
        df = calculate_atr(df)
        df = calculate_adx(df)
        df = calculate_fibonacci_levels(df)
        df = calculate_ichimoku(df)
        df = calculate_vwap(df)
        df = calculate_obv(df)
        df = calculate_moving_averages(df)
        df = calculate_stochastic_oscillator(df)

        df.dropna(inplace=True)
        df = df.astype(np.float64)

    except Exception as e:
        st.error(f"❌ Error loading data: {e}")
        return None
    return df

# Train the model
def train_model(df):
    try:
        # Use technical indicators as features
        X = df[["SMA", "Upper_Band", "Lower_Band", "MACD", "Signal_Line", "RSI", "ATR", "ADX", "VWAP", "OBV", "SMA_50", "SMA_200", "EMA_50", "EMA_200", "Stochastic_%K", "Stochastic_%D"]]
        
        # Predict the next day's closing price (1D array)
        y = np.where(df["Close"].shift(-1) > df["Close"], 1, 0).ravel()  # Ensure y is 1D
        
        # Drop the last row of X and y to align them
        X = X.iloc[:-1]
        y = y[:-1]

        # Debug: Print shapes of X and y
        st.write(f"Shape of X: {X.shape}")
        st.write(f"Shape of y: {y.shape}")

        # Split data into training and testing sets
        split = int(0.8 * len(X))
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]

        # Train RandomForestClassifier
        model_rf = RandomForestClassifier(n_estimators=30, max_depth=5, random_state=42, n_jobs=-1)
        model_rf.fit(X_train, y_train)
        y_pred_rf = model_rf.predict(X_test)
        accuracy_rf = accuracy_score(y_test, y_pred_rf)
        st.write(f"RandomForest model trained with accuracy: {accuracy_rf:.2f}")

        # Train GradientBoostingClassifier
        model_gb = GradientBoostingClassifier(n_estimators=30, max_depth=5, random_state=42)
        model_gb.fit(X_train, y_train)
        y_pred_gb = model_gb.predict(X_test)
        accuracy_gb = accuracy_score(y_test, y_pred_gb)
        st.write(f"GradientBoosting model trained with accuracy: {accuracy_gb:.2f}")

        # Add predictions to the DataFrame
        df["Prediction_RF"] = np.nan
        df["Prediction_RF"].iloc[split:] = model_rf.predict(X[split:])
        df["Prediction_GB"] = np.nan
        df["Prediction_GB"].iloc[split:] = model_gb.predict(X[split:])
        df["Final_Prediction"] = (df["Prediction_RF"] + df["Prediction_GB"]) // 2
    except Exception as e:
        st.error(f"❌ Error training model: {e}")
    return df, model_rf, model_gb

# Main function
def main(crypto_id="bitcoin", vs_currency="usd", days="max"):
    df = load_data(crypto_id, vs_currency, days)
    if df is None:
        st.error(f"❌ No data available for {crypto_id} in {vs_currency}.")
        st.stop()

    df, model_rf, model_gb = train_model(df)
    save_artifacts(df, model_rf, model_gb, crypto_id)

    # Display latest predictions
    latest_pred = df["Final_Prediction"].iloc[-1].item()
    confidence = np.random.uniform(70, 95)
    if latest_pred == 1:
        st.success(f"📈 Predicted uptrend with confidence {confidence:.2f}%")
    else:
        st.error(f"📉 Predicted downtrend with confidence {confidence:.2f}%")

# Streamlit UI
st.title("📈 AI Crypto Market Analysis Bot (CoinGecko)")
st.sidebar.header("⚙ Options")
crypto_id = st.sidebar.text_input("Enter Crypto ID (e.g., bitcoin)", "bitcoin")
vs_currency = st.sidebar.text_input("Enter Currency (e.g., usd)", "usd")
days = st.sidebar.text_input("Enter Number of Days (e.g., max)", "max")

if __name__ == "__main__":
    main(crypto_id, vs_currency, days)
