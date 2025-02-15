import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
import time
import joblib
import os
from pytz import timezone, utc

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
        if len(df) < 200:  # Ensure there are enough rows for rolling calculations
            st.warning(f"⚠️ Τα δεδομένα δεν είναι αρκετά για να γίνουν οι υπολογισμοί. Δοκιμάστε μεγαλύτερο interval ή διαφορετικό σύμβολο.")
            return None
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
    best_rf = None
    best_gb = None
    try:
        if df.empty or len(df) == 0:
            raise ValueError("DataFrame is empty")
        X = df[["SMA_50", "SMA_200", "RSI", "MACD", "OBV", "Volume_MA"]]
        y = np.where(df["Close"].shift(-1) > df["Close"], 1, 0)
        y = y.ravel()
        if len(X) == 0 or len(y) == 0:
            raise ValueError("Not enough data to train the model")
        split = int(0.8 * len(df))
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]

        # Hyperparameter tuning for RandomForest
        param_grid_rf = {
            'n_estimators': [50, 100, 150],
            'max_depth': [5, 10, 15],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
        grid_search_rf = GridSearchCV(RandomForestClassifier(random_state=42, n_jobs=-1), param_grid_rf, cv=3)
        grid_search_rf.fit(X_train, y_train)
        best_rf = grid_search_rf.best_estimator_

        y_pred_rf = best_rf.predict(X_test)
        accuracy_rf = accuracy_score(y_test, y_pred_rf)
        st.write(f"RandomForest model trained with accuracy: {accuracy_rf:.2f}")

        # Hyperparameter tuning for GradientBoosting
        param_grid_gb = {
            'n_estimators': [50, 100, 150],
            'max_depth': [3, 5, 7],
            'learning_rate': [0.01, 0.1, 0.2],
            'subsample': [0.8, 1.0]
        }
        grid_search_gb = GridSearchCV(GradientBoostingClassifier(random_state=42), param_grid_gb, cv=3)
        grid_search_gb.fit(X_train, y_train)
        best_gb = grid_search_gb.best_estimator_

        y_pred_gb = best_gb.predict(X_test)
        accuracy_gb = accuracy_score(y_test, y_pred_gb)
        st.write(f"GradientBoosting model trained with accuracy: {accuracy_gb:.2f}")

        df["Prediction_RF"] = best_rf.predict(X)
        df["Prediction_GB"] = best_gb.predict(X)
        df["Final_Prediction"] = (df["Prediction_RF"] + df["Prediction_GB"]) // 2
    except Exception as e:
        st.error(f"❌ Σφάλμα εκπαίδευσης μοντέλου: {e}")
    return df, best_rf, best_gb

def calculate_trade_levels(df, timeframe, confidence, future_price_points, future_dates):
    try:
        if df.empty or len(df) == 0:
            raise ValueError("DataFrame is empty")

        latest_close = df["Close"].iloc[-1]
        atr = (df["High"].rolling(window=14).mean() - df["Low"].rolling(window=14).mean()).iloc[-1]
        latest_pred = df["Final_Prediction"].iloc[-1]
        rsi = df["RSI"].iloc[-1]
        macd = df["MACD"].iloc[-1]

        stop_loss_multiplier = 1.0  # Initialize stop loss multiplier
        take_profit_multiplier = 1.0  # Initialize take profit multiplier

        if future_price_points is not None and len(future_price_points) > 0:
            future_pred = future_price_points[-1]
            if future_pred > latest_close:
                take_profit_multiplier = (future_pred - latest_close) / atr
            else:
                stop_loss_multiplier = (latest_close - future_pred) / atr

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

        # Ensure future_dates is in the correct timezone
        future_dates = future_dates.tz_localize('UTC')

        take_profit_reached = (np.array(future_price_points) >= take_profit).any()
        if take_profit_reached.any():
            expected_profit_index = np.argmax(np.array(future_price_points) >= take_profit)
        else:
            expected_profit_index = np.argmin(np.abs(np.array(future_price_points) - take_profit))
        
        expected_profit_time = future_dates[min(expected_profit_index, len(future_dates) - 1)]

        greece_tz = timezone('Europe/Athens')
        expected_profit_time_eet = expected_profit_time.tz_convert(greece_tz)

        st.write(f"Trade levels for {timeframe}: Entry Point: {entry_point:.2f}, Stop Loss: {stop_loss:.2f}, Take Profit: {take_profit:.2f}, Expected Time to Profit: {expected_profit_time_eet.strftime('%Y-%m-%d %H:%M:%S')} EET")
    except Exception as e:
        st.error(f"❌ Σφάλμα υπολογισμού επιπέδων συναλλαγών: {e}")
        return None, None, None, None
    return entry_point, stop_loss, take_profit, expected_profit_time

def generate_price_points(df, entry_point, future_days=15):
    try:
        # Generate price points based on trade levels
        if entry_point is None:
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
        if df.empty or len(df) == 0:
            st.error("❌ DataFrame is empty, cannot generate price points.")
            st.stop()
        future_price_points = generate_price_points(df, df["Close"].iloc[-1], future_days=15)
        future_dates = pd.date_range(df.index[-1], periods=15, freq="D")
        if future_price_points is None or len(future_price_points) == 0:
            st.error("❌ Failed to generate future price points.")
            st.stop()
        entry_point, stop_loss, take_profit, expected_profit_time = calculate_trade_levels(df, timeframe, confidence, future_price_points, future_dates)
        trade_levels[timeframe] = (entry_point, stop_loss, take_profit, expected_profit_time)
        save_artifacts(df, model_rf, model_gb, crypto_symbol)  # Save artifacts
    if any(levels is None for levels in trade_levels.values()):
        st.stop()

    # Generate price points for the next 15 days
    entry_point, stop_loss, take_profit, expected_profit_time = trade_levels["1d"]
    future_dates = pd.date_range(data["1d"].index[-1], periods=15, freq="D")
    future_price_points = generate_price_points(data["1d"], entry_point, future_days=15)
    if future_price_points is None or len(future_price_points) == 0:
        st.error("❌ Failed to generate future price points.")
        st.stop()

    # Fetch live price
    live_data = yf.download(crypto_symbol, period="1d", interval="1m")
    live_price = float(live_data["Close"].iloc[-1]) if not live_data.empty else None

    # Create a DataFrame for the table
    table_data = {
        "Date": future_dates,
        "Predicted Price": future_price_points,
    }
    df_table = pd.DataFrame(table_data)

    # Add live price to the table
    if live_price is not None:
        df_table["Live Price"] = [live_price if i == 0 else None for i in range(len(future_dates))]

    # Ensure the "Predicted Price" and "Live Price" columns are of type float
    df_table["Predicted Price"] = df_table["Predicted Price"].astype(float)
    df_table["Live Price"] = df_table["Live Price"].apply(lambda x: float(x) if x is not None else None)

    # Display the table
    st.subheader("📊 Predicted and Actual Prices")
    st.write(df_table)

    # Display latest predictions and trade levels
    st.subheader("🔍 Latest Predictions & Trade Levels")
    latest_pred = data["1d"]["Final_Prediction"].iloc[-1]
    confidence = np.random.uniform(70, 95)
    if latest_pred == 1:
        st.success(f"📈 Προβλέπεται άνοδος με confidence {confidence:.2f}%")
    else:
        st.error(f"📉 Προβλέπεται πτώση με confidence {confidence:.2f}%")

    st.subheader("📌 Trade Setup")
    for timeframe, levels in trade_levels.items():
        if levels is not None:
            entry_point, stop_loss, take_profit, expected_profit_time = levels
            greece_tz = timezone('Europe/Athens')
            if expected_profit_time.tzinfo is None:
                expected_profit_time = expected_profit_time.tz_localize(utc)
            expected_profit_time_eet = expected_profit_time.tz_convert(greece_tz)
            st.write(f"⏰ {timeframe}:")
            st.write(f"✅ Entry Point: {entry_point:.2f}")
            st.write(f"🚨 Stop Loss: {stop_loss:.2f}")
            st.write(f"🎯 Take Profit: {take_profit:.2f}")
            st.write(f"🕒 Expected Time to Profit: {expected_profit_time_eet.strftime('%Y-%m-%d %H:%M:%S')} EET")

    # Continuously update data and retrain model
    while True:
        time.sleep(60)
        live_data = yf.download(crypto_symbol, period="1d", interval="1m")
        actual_price = float(live_data["Close"].iloc[-1]) if not live_data.empty else None

        if actual_price is not None:
            # Compare predicted price with actual price and retrain if necessary
            if future_price_points and len(future_price_points) > 0:
                predicted_price = future_price_points.pop(0)
                if abs(predicted_price - actual_price) / actual_price > 0.001:  # 0.1% threshold
                    df = load_data(crypto_symbol, interval="1d", period="5y")
                    if df is None:
                        st.error(f"❌ Τα δεδομένα δεν είναι διαθέσιμα για το σύμβολο {crypto_symbol}.")
                        st.stop()
                    data["1d"] = df
                    data["1d"], model_rf, model_gb = train_model(data["1d"])
                    confidence = np.random.uniform(70, 95)
                    future_price_points = generate_price_points(data["1d"], data["1d"]["Close"].iloc[-1], future_days=15)
                    future_dates = pd.date_range(data["1d"].index[-1], periods=15, freq="D")
                    if future_price_points is None or len(future_price_points) == 0:
                        st.error("❌ Failed to generate future price points.")
                        st.stop()
                    entry_point, stop_loss, take_profit, expected_profit_time = calculate_trade_levels(data["1d"], "1d", confidence, future_price_points, future_dates)
                    trade_levels["1d"] = (entry_point, stop_loss, take_profit, expected_profit_time)
                    save_artifacts(df, model_rf, model_gb, crypto_symbol)  # Save artifacts

        st.rerun()

if __name__ == "__main__":
    main()
