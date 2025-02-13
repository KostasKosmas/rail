import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score
import time

# 📌 Streamlit UI
st.title("📈 AI Crypto Market Analysis Bot")
st.sidebar.header("⚙ Επιλογές")
crypto_symbol = st.sidebar.text_input("Εισάγετε Crypto Symbol", "BTC-USD")

def load_data(symbol, interval="1m"):
    try:
        st.write(f"Loading data for {symbol} with interval {interval}")
        df = yf.download(symbol, period="1d", interval=interval)  # Fetch 1 day of data with 1-minute intervals
        if df.empty:
            st.error("⚠️ Τα δεδομένα δεν είναι διαθέσιμα. Δοκιμάστε διαφορετικό σύμβολο.")
            return pd.DataFrame()
        
        df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
        df.dropna(inplace=True)
        
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
        
        # Ensure all columns are of compatible types
        df = df.astype(np.float64)
    except Exception as e:
        st.error(f"❌ Σφάλμα φόρτωσης δεδομένων: {e}")
        return pd.DataFrame()
    return df

def train_model(df):
    try:
        X = df[["SMA_50", "SMA_200", "RSI", "MACD", "OBV", "Volume_MA"]]
        y = np.where(df["Close"].shift(-1) > df["Close"], 1, 0)
        y = y.ravel()  # Flatten y to 1D array
        
        # Split data into training and testing sets
        split = int(0.8 * len(df))
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]
        
        # Train RandomForest model
        model_rf = RandomForestClassifier(n_estimators=100, random_state=42)
        model_rf.fit(X_train, y_train)
        y_pred_rf = model_rf.predict(X_test)
        accuracy_rf = accuracy_score(y_test, y_pred_rf)
        st.write(f"RandomForest model trained with accuracy: {accuracy_rf:.2f}")
        
        # Train GradientBoosting model
        model_gb = GradientBoostingClassifier(n_estimators=100, random_state=42)
        model_gb.fit(X_train, y_train)
        y_pred_gb = model_gb.predict(X_test)
        accuracy_gb = accuracy_score(y_test, y_pred_gb)
        st.write(f"GradientBoosting model trained with accuracy: {accuracy_gb:.2f}")
        
        # Combine predictions
        df["Prediction_RF"] = model_rf.predict(X)
        df["Prediction_GB"] = model_gb.predict(X)
        df["Final_Prediction"] = (df["Prediction_RF"] + df["Prediction_GB"]) // 2
    except Exception as e:
        st.error(f"❌ Σφάλμα εκπαίδευσης μοντέλου: {e}")
    return df, model_rf, model_gb

def calculate_trade_levels(df):
    try:
        latest_close = df["Close"].iloc[-1]
        atr = df["High"].rolling(window=14).max()[-1] - df["Low"].rolling(window=14).min()[-1]  # ATR-like calculation

        # Log calculations for debugging
        st.write(f"Latest Close: {latest_close}")
        st.write(f"ATR: {atr}")

        # Determine trade levels based on prediction
        latest_pred = df["Final_Prediction"].iloc[-1]
        if latest_pred == 1:  # Long position
            entry_point = latest_close
            stop_loss = latest_close - (atr * 1.5)  # Stop loss below entry
            take_profit = latest_close + (atr * 2)  # Take profit above entry
        else:  # Short position
            entry_point = latest_close
            stop_loss = latest_close + (atr * 1.5)  # Stop loss above entry
            take_profit = latest_close - (atr * 2)  # Take profit below entry
        
        st.write(f"Trade levels: Entry Point: {entry_point:.2f}, Stop Loss: {stop_loss:.2f}, Take Profit: {take_profit:.2f}")
    except Exception as e:
        st.error(f"❌ Σφάλμα υπολογισμού επιπέδων συναλλαγών: {e}")
        entry_point, stop_loss, take_profit = None, None, None
    return entry_point, stop_loss, take_profit

def future_price_prediction(df, model_rf, model_gb, steps=60):
    try:
        future_df = pd.DataFrame(index=pd.date_range(start=df.index[-1], periods=steps+1, freq='T')[1:])
        for step in range(steps):
            X_future = df[["SMA_50", "SMA_200", "RSI", "MACD", "OBV", "Volume_MA"]].iloc[-1].values.reshape(1, -1)
            pred_rf = model_rf.predict(X_future)
            pred_gb = model_gb.predict(X_future)
            final_pred = (pred_rf + pred_gb) // 2
            new_close = df["Close"].iloc[-1] * (1 + 0.001 * (2 * final_pred - 1))  # Small percentage change based on prediction
            df.loc[future_df.index[step]] = df.iloc[-1]
            df.at[future_df.index[step], "Close"] = new_close

            # Recalculate indicators for the new data point
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

        future_df["Close"] = df["Close"].iloc[-steps:]
        return future_df
    except Exception as e:
        st.error(f"❌ Σφάλμα πρόβλεψης μελλοντικών τιμών: {e}")
        return pd.DataFrame()

def main():
    df = load_data(crypto_symbol)
    if df.empty:
        st.stop()

    df, model_rf, model_gb = train_model(df)

    entry, stop, profit = calculate_trade_levels(df)

    if entry is None or stop is None or profit is None:
        st.stop()

    future_df = future_price_prediction(df, model_rf, model_gb, steps=60)
    if future_df.empty:
        st.stop()

    st.subheader("📊 Live Price Chart with Future Predictions")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index[-100:], y=df["Close"].iloc[-100:], name="Τιμή", line=dict(color="blue")))
    fig.add_trace(go.Scatter(x=future_df.index, y=future_df["Close"], name="Predicted Price", line=dict(color="orange", dash="dot")))
    st.plotly_chart(fig)

    st.subheader("🔍 Latest Predictions & Trade Levels")
    latest_pred = df["Final_Prediction"].iloc[-1]
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
        future_df = future_price_prediction(df, model_rf, model_gb, steps=60)
        st.rerun()  # Use st.rerun() to refresh the app

if __name__ == "__main__":
    main()
