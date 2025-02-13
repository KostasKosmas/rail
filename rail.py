
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import time
import pytz

# 📌 Streamlit UI
st.title("📈 AI Crypto Market Analysis Bot")
st.sidebar.header("⚙ Επιλογές")
crypto_symbol = st.sidebar.text_input("Εισάγετε Crypto Symbol", "BTC-USD")

# Cache data loading to speed up the app
@st.cache_data(ttl=3600)  # Cache data for 1 hour
def load_data(symbol, interval="1d", period="5y"):
    try:
        df = yf.download(symbol, period=period, interval=interval)
        if df.empty:
            st.warning(f"⚠️ Τα δεδομένα δεν είναι διαθέσιμα για το σύμβολο {symbol} με interval {interval}. Δοκιμάστε διαφορετικό interval.")
            return None
        df.index = df.index.tz_localize("UTC").tz_convert("Europe/Athens")
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
        df.dropna(inplace=True)
        df = df.astype(np.float64)
    except Exception as e:
        st.error(f"❌ Σφάλμα φόρτωσης δεδομένων: {e}")
        return None
    return df

def train_model(df):
    try:
        # Use historical data to predict future prices
        X = df[["SMA_50", "SMA_200", "RSI", "MACD", "OBV", "Volume_MA"]]
        y = df["Close"]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        st.write(f"Model trained with Mean Squared Error: {mse:.2f}")
    except Exception as e:
        st.error(f"❌ Σφάλμα εκπαίδευσης μοντέλου: {e}")
        return None, None
    return model, X.columns

def predict_future_prices(model, last_row, feature_columns, future_days=14):
    try:
        future_predictions = []
        current_features = last_row[feature_columns].values.reshape(1, -1)
        for _ in range(future_days):
            # Predict the next day's price
            next_price = model.predict(current_features)[0]
            future_predictions.append(next_price)
            # Update features for the next prediction
            current_features[0][0] = next_price  # Update SMA_50 (simplified)
            current_features[0][1] = next_price  # Update SMA_200 (simplified)
            current_features[0][4] += np.random.normal(0, 100)  # Update OBV (simplified)
        return future_predictions
    except Exception as e:
        st.error(f"❌ Σφάλμα πρόβλεψης μελλοντικών τιμών: {e}")
        return None

def main():
    # Load historical data
    df = load_data(crypto_symbol)
    if df is None:
        st.error(f"❌ Τα δεδομένα δεν είναι διαθέσιμα για το σύμβολο {crypto_symbol}.")
        st.stop()

    # Train the model
    model, feature_columns = train_model(df)
    if model is None:
        st.stop()

    # Predict future prices
    last_row = df.iloc[-1]
    future_dates = pd.date_range(df.index[-1], periods=14, freq="D")
    future_predictions = predict_future_prices(model, last_row, feature_columns)
    if future_predictions is None:
        st.stop()

    # Display the chart
    st.subheader("📊 Live Price Chart with Predictions")
    fig = go.Figure()

    # Plot historical data (last 6 months)
    last_6_months = df.iloc[-180:]
    fig.add_trace(go.Scatter(x=last_6_months.index, y=last_6_months["Close"], name="Historical Prices (Last 6 Months)", line=dict(color="blue")))

    # Plot future predictions
    fig.add_trace(go.Scatter(x=future_dates, y=future_predictions, name="Future Predictions (Next 14 Days)", line=dict(color="orange", dash="dot")))

    # Continuously update with actual prices
    while True:
        # Fetch the latest data
        latest_data = yf.download(crypto_symbol, period="1d", interval="1m")
        if not latest_data.empty:
            latest_price = latest_data["Close"].iloc[-1]
            latest_time = latest_data.index[-1].tz_localize("UTC").tz_convert("Europe/Athens")
            # Update the chart with the latest price
            fig.add_trace(go.Scatter(x=[latest_time], y=[latest_price], name="Actual Price", mode="markers", marker=dict(color="green", size=10)))
            st.plotly_chart(fig)
        time.sleep(60)  # Wait for 1 minute

if __name__ == "__main__":
    main()
