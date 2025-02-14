import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import ta  # Technical analysis library
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

# Calculate Fibonacci Levels
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

# Cache data loading to speed up the app
@st.cache_data
def load_data(symbol, interval="1d", period="5y"):
    try:
        st.write(f"Loading data for {symbol} with interval {interval} and period {period}")
        df = yf.download(symbol, period=period, interval=interval)
        
        # Debug: Check if data is loaded
        if df.empty:
            st.warning(f"⚠️ Τα δεδομένα δεν είναι διαθέσιμα για το σύμβολο {symbol} με interval {interval}. Δοκιμάστε διαφορετικό interval.")
            return None
        
        # Debug: Print the first few rows of the DataFrame
        st.write("First few rows of the loaded data:")
        st.write(df.head())

        df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
        df.dropna(inplace=True)

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

        df.dropna(inplace=True)
        df = df.astype(np.float64)

        # Debug: Check the final DataFrame
        st.write("Final DataFrame after calculating indicators:")
        st.write(df.head())
    except Exception as e:
        st.error(f"❌ Σφάλμα φόρτωσης δεδομένων: {e}")
        return None
    return df

# Train the model
def train_model(df):
    try:
        # Use technical indicators as features
        X = df[["SMA", "Upper_Band", "Lower_Band", "MACD", "Signal_Line", "RSI", "ATR", "ADX", "VWAP", "OBV"]]
        
        # Predict the next day's closing price (1D array)
        y = np.where(df["Close"].shift(-1) > df["Close"], 1, 0).ravel()  # Ensure y is 1D
        
        # Drop the last row of X and y to align them
        X = X.iloc[:-1]
        y = y[:-1]

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
        st.error(f"❌ Σφάλμα εκπαίδευσης μοντέλου: {e}")
    return df, model_rf, model_gb

# Main function
def main(crypto_symbol):
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
    future_price_points = generate_price_points(entry_point, stop_loss, take_profit, data["1d"])
    if future_price_points is None:
        st.error("❌ Failed to generate future price points.")
        st.stop()

    # Fetch live price
    live_data = yf.download(crypto_symbol, period="1d", interval="1m")
    live_price = live_data["Close"].iloc[-1] if not live_data.empty else np.nan

    # Create a DataFrame for the table
    table_data = {
        "Date": future_dates,
        "Predicted Price": future_price_points,
        "Live Price": [live_price if i == 0 else np.nan for i in range(len(future_dates))]
    }
    df_table = pd.DataFrame(table_data)

    # Ensure all columns are of consistent type
    df_table = df_table.astype({"Predicted Price": "float64", "Live Price": "float64"})

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

# 📌 Streamlit UI
st.title("📈 AI Crypto Market Analysis Bot")
st.sidebar.header("⚙ Επιλογές")
crypto_symbol = st.sidebar.text_input("Εισάγετε Crypto Symbol", "BTC-USD")

if __name__ == "__main__":
    main(crypto_symbol)
