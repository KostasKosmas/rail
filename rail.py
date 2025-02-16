# Install dependencies: pip install -r requirements.txt

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from sklearn.metrics import accuracy_score, f1_score
from sklearn.feature_selection import SelectFromModel
import joblib
import os
import time
from arch import arch_model
import logging

# Configuration
MAX_DATA_POINTS = 3000
INCREMENTAL_ESTIMATORS = 50
SAVE_PATH = "saved_models"
FORECAST_DAYS = 15
SIMULATIONS = 500

# Auto-create model directory
os.makedirs(SAVE_PATH, exist_ok=True)

# Logging configuration
logging.basicConfig(filename='app.log', level=logging.ERROR)

def safe_execute(func):
    try:
        return func()
    except Exception as e:
        logging.error(f"Error in {func.__name__}: {e}")
        return None

@st.cache_data
def load_data(symbol, interval="1d", period="5y"):
    try:
        df = yf.download(symbol, period=period, interval=interval)
        if df.empty or len(df) < 100:
            st.error(f"⚠️ Insufficient data for {symbol}")
            return None

        df = df[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
        
        # Technical indicators
        df["SMA_50"] = df["Close"].rolling(50).mean()
        df["SMA_200"] = df["Close"].rolling(200).mean()
        
        # RSI calculation
        delta = df['Close'].diff().dropna()
        gain = delta.clip(lower=0)
        loss = (-delta).clip(lower=0)
        avg_gain = gain.rolling(14).mean()
        avg_loss = loss.rolling(14).mean()
        rs = avg_gain / (avg_loss + 1e-10)
        df['RSI'] = 100 - (100 / (1 + rs))
        
        df['MACD'] = df['Close'].ewm(span=12).mean() - df['Close'].ewm(span=26).mean()
        df['OBV'] = (np.sign(df['Close'].diff()) * df['Volume']).fillna(0).cumsum()
        df['Volume_MA'] = df['Volume'].rolling(20).mean()
        
        df = df.dropna().iloc[-MAX_DATA_POINTS:].astype(np.float64)
        return df
    
    except Exception as e:
        st.error(f"Data error: {str(e)}")
        return None

def select_features(X, y):
    selector = SelectFromModel(RandomForestClassifier(n_estimators=100), threshold="median")
    selector.fit(X, y)
    return selector

def train_model(df, crypto_symbol):
    try:
        if df is None or len(df) < 200:
            raise ValueError("Insufficient training data")
            
        X = df[["SMA_50", "SMA_200", "RSI", "MACD", "OBV", "Volume_MA"]]
        y = np.where(df["Close"].shift(-1) > df["Close"], 1, 0).ravel()
        
        tscv = TimeSeriesSplit(n_splits=5)
        for train_idx, test_idx in tscv.split(X):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

        selector = select_features(X_train, y_train)
        X_train_sel = selector.transform(X_train)
        X_test_sel = selector.transform(X_test)

        rf_path = f"{SAVE_PATH}/{crypto_symbol}_model_rf.pkl"
        gb_path = f"{SAVE_PATH}/{crypto_symbol}_model_gb.pkl"
        
        # Random Forest
        model_rf = joblib.load(rf_path) if os.path.exists(rf_path) else None
        if model_rf:
            model_rf.n_estimators += INCREMENTAL_ESTIMATORS
            model_rf.fit(X_train_sel, y_train)
        else:
            model_rf = RandomizedSearchCV(
                RandomForestClassifier(n_jobs=-1, class_weight='balanced', warm_start=True),
                {'n_estimators': [200, 300], 'max_depth': [15, 20, None], 'min_samples_split': [2, 5]},
                n_iter=3, cv=3, scoring='f1'
            ).fit(X_train_sel, y_train).best_estimator_

        # Gradient Boosting
        model_gb = joblib.load(gb_path) if os.path.exists(gb_path) else None
        if model_gb:
            model_gb.n_estimators += INCREMENTAL_ESTIMATORS
            model_gb.fit(X_train_sel, y_train)
        else:
            model_gb = RandomizedSearchCV(
                GradientBoostingClassifier(warm_start=True),
                {'n_estimators': [200, 300], 'learning_rate': [0.05, 0.1], 'max_depth': [3, 5], 'subsample': [0.8, 1.0]},
                n_iter=3, cv=3, scoring='f1'
            ).fit(X_train_sel, y_train).best_estimator_

        y_pred = (model_rf.predict(X_test_sel) + model_gb.predict(X_test_sel)) // 2
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        
        joblib.dump(model_rf, rf_path)
        joblib.dump(model_gb, gb_path)
        
        return model_rf, model_gb, selector, df, accuracy, f1
        
    except Exception as e:
        st.error(f"Training error: {str(e)}")
        return None, None, None, None, None, None

def generate_price_points(df, days=FORECAST_DAYS, simulations=SIMULATIONS):
    try:
        if df.empty or len(df) < 100:
            raise ValueError("Insufficient historical data")
            
        returns = np.log(df['Close']).diff().dropna()
        garch_model = arch_model(returns, vol='Garch', p=1, q=1)
        garch_fitted = garch_model.fit(disp="off")
        forecast_volatility = garch_fitted.forecast(start=0).variance

        mu = returns.mean()
        sigma = np.sqrt(forecast_volatility.iloc[-1, 0])
        last_price = df['Close'].iloc[-1].item()
        
        forecast = np.zeros((days, simulations))
        for i in range(simulations):
            daily_returns = np.random.normal(mu, sigma, days)
            price_path = last_price * np.exp(np.cumsum(daily_returns))
            forecast[:, i] = price_path
        
        forecast_df = pd.DataFrame({
            'Day': range(1, days+1),
            'Median': np.median(forecast, axis=1),
            'Upper': np.percentile(forecast, 95, axis=1),
            'Lower': np.percentile(forecast, 5, axis=1)
        }).astype({'Day': 'int32', 'Median': 'float64', 'Upper': 'float64', 'Lower': 'float64'})
        
        return forecast_df.reset_index(drop=True)
        
    except Exception as e:
        st.error(f"Forecast error: {str(e)}")
        return pd.DataFrame()

def calculate_trade_levels(df, selector, model_rf, model_gb):
    try:
        features = ["SMA_50", "SMA_200", "RSI", "MACD", "OBV", "Volume_MA"]
        X = selector.transform(df[features][-30:])
        
        rf_proba = model_rf.predict_proba(X)[:, 1]
        gb_proba = model_gb.predict_proba(X)[:, 1]
        confidence = np.mean((rf_proba + gb_proba) / 2)
        
        high = df['High'].iloc[-14:].values
        low = df['Low'].iloc[-14:].values
        atr = np.mean(high - low).item()
        
        current_price = df['Close'].iloc[-1].item()
        sma_trend = df['SMA_50'].iloc[-1].item() > df['SMA_200'].iloc[-1].item()
        
        forecast = generate_price_points(df)
        if forecast.empty:
            raise ValueError("Price forecast failed")
        
        risk_multiplier = 1.5 if confidence > 0.7 else 1.0
        stop_loss = current_price - (atr * 1.5 * risk_multiplier)
        take_profit = current_price + (atr * 2.0 * risk_multiplier)
        
        return {
            'entry': current_price,
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'forecast': forecast,
            'confidence': confidence,
            'trend': 'Bullish' if sma_trend else 'Bearish'
        }
    except Exception as e:
        st.error(f"Trade error: {str(e)}")
        return None

def main():
    st.title("🚀 AI Crypto Trading System")
    
    crypto_symbol = st.sidebar.text_input("Crypto Pair", "BTC-USD")
    auto_refresh = st.sidebar.checkbox("Live Updates", True)
    
    df = load_data(crypto_symbol)
    if df is None:
        st.stop()
    
    if st.button("🔄 Refresh") or auto_refresh:
        with st.spinner("Analyzing market..."):
            model_rf, model_gb, selector, df, acc, f1 = train_model(df, crypto_symbol)
        
        if model_rf and model_gb:
            st.subheader("📊 Performance Metrics")
            cols = st.columns(2)
            cols[0].metric("Accuracy", f"{acc:.2%}")
            cols[1].metric("F1 Score", f"{f1:.2%}")
            
            levels = calculate_trade_levels(df, selector, model_rf, model_gb)
            if levels:
                st.subheader("📈 Trading Signals")
                
                cols = st.columns(3)
                cols[0].metric("Current Price", f"${levels['entry']:.2f}")
                cols[1].metric("Stop Loss", f"${levels['stop_loss']:.2f}", delta_color="inverse")
                cols[2].metric("Take Profit", f"${levels['take_profit']:.2f}")
                
                st.subheader("🔮 Price Forecast")
                forecast_df = levels['forecast'].set_index('Day').astype('float64')
                st.line_chart(forecast_df)
                
                st.progress(float(levels['confidence']))
                st.caption(f"Model Confidence: {levels['confidence']:.2%}")
                
                if levels['trend'] == 'Bullish':
                    st.success("📈 Bullish Trend Detected")
                else:
                    st.warning("📉 Bearish Market Conditions")
    
    try:
        live_data = yf.download(crypto_symbol, period='1d', interval='1m')
        if not live_data.empty and len(live_data) > 1:
            st.subheader("🔴 Live Market Feed")
            current = live_data.iloc[-1]
            prev = live_data.iloc[-2]
            
            price_now = current['Close'].item()
            price_change = price_now - prev['Close'].item()
            
            vol_now = current['Volume'].item()
            vol_prev = prev['Volume'].item()
            vol_change = vol_now - vol_prev
            
            cols = st.columns(4)
            cols[0].metric("Price", f"${price_now:.2f}", f"{price_change:.2f}")
            cols[1].metric("Volume", f"{vol_now:,.0f}", 
                          f"{vol_change:+,.0f}" if vol_change >=0 else f"({-vol_change:,.0f})")
            cols[2].metric("24H High", f"${live_data['High'].max().item():.2f}")
            cols[3].metric("24H Low", f"${live_data['Low'].min().item():.2f}")
    except Exception as e:
        st.error(f"Live feed error: {str(e)}")

    if auto_refresh:
        time.sleep(300)
        st.rerun()

if __name__ == "__main__":
    main()
