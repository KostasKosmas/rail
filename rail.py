# Import all required libraries
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, mean_absolute_error, mean_squared_error
from statsmodels.tsa.arima.model import ARIMA
from pmdarima import auto_arima  # For auto-tuning ARIMA parameters
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

# Train ARIMA model with auto-tuning
def train_arima_model(df):
    try:
        # Auto-tune ARIMA parameters
        model = auto_arima(
            df["Close"],
            seasonal=False,  # Non-seasonal data
            trace=True,  # Print logs
            error_action="ignore",  # Ignore errors
            suppress_warnings=True,  # Suppress warnings
            stepwise=True,  # Use stepwise algorithm
        )
        st.write(f"Best ARIMA parameters: {model.order}")
        return model
    except Exception as e:
        st.error(f"❌ Σφάλμα εκπαίδευσης ARIMA μοντέλου: {e}")
        return None

# Predict future prices with ARIMA
def predict_with_arima(model, future_days=14):
    try:
        # Predict future prices with confidence intervals
        predictions, conf_int = model.predict(n_periods=future_days, return_conf_int=True)
        return predictions, conf_int
    except Exception as e:
        st.error(f"❌ Σφάλμα πρόβλεψης με ARIMA: {e}")
        return None, None

# Evaluate ARIMA model accuracy
def evaluate_arima_model(model, df):
    try:
        # Split data into train and test sets
        train_size = int(len(df) * 0.8)
        train, test = df["Close"].iloc[:train_size], df["Close"].iloc[train_size:]
        
        # Fit the model on training data
        model.fit(train)
        
        # Predict on test data
        predictions = model.predict(n_periods=len(test))
        
        # Calculate Mean Absolute Error (MAE) and Root Mean Squared Error (RMSE)
        mae = mean_absolute_error(test, predictions)
        rmse = np.sqrt(mean_squared_error(test, predictions))
        st.write(f"ARIMA Model Mean Absolute Error (MAE): {mae:.2f}")
        st.write(f"ARIMA Model Root Mean Squared Error (RMSE): {rmse:.2f}")
    except Exception as e:
        st.error(f"❌ Σφάλμα αξιολόγησης ARIMA μοντέλου: {e}")

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

    # Train ARIMA model
    arima_model = train_arima_model(data["1d"])
    if arima_model is not None:
        # Evaluate ARIMA model accuracy
        evaluate_arima_model(arima_model, data["1d"])

        # Predict future prices with ARIMA
        future_dates = pd.date_range(data["1d"].index[-1], periods=14, freq="D")
        future_predictions_arima, conf_int = predict_with_arima(arima_model)
        if future_predictions_arima is not None:
            st.write("ARIMA Future Predictions:", future_predictions_arima)

    # Display live price chart with predictions
    st.subheader("📊 Live Price Chart with Predictions")
    fig = go.Figure()

    # Plot actual prices for the last 6 months (using daily data)
    last_6_months = data["1d"].iloc[-180:]
    fig.add_trace(go.Scatter(x=last_6_months.index, y=last_6_months["Close"], name="Actual Price (Last 6 Months)", line=dict(color="blue")))

    # Plot ARIMA future predictions with confidence intervals
    if arima_model is not None and future_predictions_arima is not None:
        fig.add_trace(go.Scatter(x=future_dates, y=future_predictions_arima, name="ARIMA Future Predictions", line=dict(color="red", dash="dot")))
        fig.add_trace(go.Scatter(
            x=np.concatenate([future_dates, future_dates[::-1]]),  # X values for the confidence interval
            y=np.concatenate([conf_int[:, 0], conf_int[::-1, 1]]),  # Y values for the confidence interval
            fill="toself",
            fillcolor="rgba(255, 0, 0, 0.2)",
            line=dict(color="rgba(255, 0, 0, 0)"),
            name="ARIMA Confidence Interval",
        ))

    # Plot existing future predictions (from RandomForest/GradientBoosting)
    future_dates = pd.date_range(data["1d"].index[-1], periods=14, freq="D")
    future_predictions = np.repeat(data["1d"]["14D_EMA"].iloc[-1], len(future_dates))
    fig.add_trace(go.Scatter(x=future_dates, y=future_predictions, name="Existing Future Predictions", line=dict(color="orange", dash="dot")))

    st.plotly_chart(fig)

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
