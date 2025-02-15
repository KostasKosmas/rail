import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from sklearn.metrics import accuracy_score, f1_score
from sklearn.feature_selection import SelectKBest, f_classif
import joblib
import os
import time
import ta
from pytz import timezone
from datetime import datetime, timedelta

# Configuration
MAX_DATA_POINTS = 5000
INCREMENTAL_ESTIMATORS = 100
SAVE_PATH = "saved_models"
FORECAST_DAYS = 15
SIMULATIONS = 1000
REFRESH_INTERVAL = 180  # 3 minutes
TARGET_TIMEZONE = timezone('Europe/Athens')

# Create save directory
os.makedirs(SAVE_PATH, exist_ok=True)

@st.cache_data
def load_data(symbol):
    try:
        # First try with 1h interval
        df = yf.download(
            symbol,
            period="max",
            interval="1h",
            progress=False,
            timeout=10
        )
        
        # Fallback to daily data if insufficient
        if len(df) < 200:
            df = yf.download(
                symbol,
                period="max",
                interval="1d",
                progress=False,
                timeout=10
            )
        
        # Validate data requirements
        if len(df) < 200:
            st.error(f"⚠️ Insufficient historical data for {symbol}")
            st.info("Try popular pairs: BTC-USD, ETH-USD, BNB-USD, XRP-USD")
            return None
            
        # Ensure required columns exist
        required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        if not all(col in df.columns for col in required_cols):
            st.error("Missing required price data columns")
            return None
            
        # Feature Engineering
        df = df[required_cols].copy()
        
        # Technical Indicators
        df['SMA_20'] = df['Close'].rolling(20, min_periods=5).mean()
        df['SMA_50'] = df['Close'].rolling(50, min_periods=10).mean()
        df['SMA_200'] = df['Close'].rolling(200, min_periods=50).mean()
        df['BB_upper'] = df['SMA_20'] + 2*df['Close'].rolling(20).std()
        df['BB_lower'] = df['SMA_20'] - 2*df['Close'].rolling(20).std()
        
        # Handle NaN values
        df = df.dropna()
        
        # Trim to max data points
        return df.iloc[-MAX_DATA_POINTS:].astype(np.float32)
    
    except Exception as e:
        st.error(f"Data error: {str(e)}")
        st.write("Troubleshooting tips:")
        st.write("- Check internet connection")
        st.write("- Verify symbol format (e.g., BTC-USD)")
        st.write("- Try again in a few minutes")
        return None

def train_model(df, crypto_symbol):
    try:
        if df is None or len(df) < 500:
            raise ValueError("Minimum 500 data points required")
            
        # Create target: 3-day price movement
        y = np.where(df["Close"].shift(-3) > df["Close"], 1, 0).ravel()
        X = df.drop(columns=['Close'], errors='ignore')
        
        # Time-series validation
        tscv = TimeSeriesSplit(n_splits=3)
        
        # Random Forest with enhanced params
        rf = RandomizedSearchCV(
            RandomForestClassifier(n_jobs=-1, class_weight='balanced'),
            {
                'n_estimators': [300, 500],
                'max_depth': [10, 20, None],
                'min_samples_split': [2, 5]
            },
            n_iter=15, cv=tscv, scoring='f1', error_score='raise'
        ).fit(X, y)
        
        # Gradient Boosting with validation
        gb = RandomizedSearchCV(
            GradientBoostingClassifier(
                n_iter_no_change=15,
                validation_fraction=0.2
            ),
            {
                'n_estimators': [300, 500],
                'learning_rate': [0.01, 0.05],
                'max_depth': [3, 5]
            },
            n_iter=15, cv=tscv, scoring='f1', error_score='raise'
        ).fit(X, y)
        
        # Save models
        joblib.dump(rf.best_estimator_, f"{SAVE_PATH}/{crypto_symbol}_rf.pkl")
        joblib.dump(gb.best_estimator_, f"{SAVE_PATH}/{crypto_symbol}_gb.pkl")
        
        return rf.best_estimator_, gb.best_estimator_, X.columns.tolist()
    
    except Exception as e:
        st.error(f"Training failed: {str(e)}")
        return None, None, None

def generate_price_forecast(df):
    try:
        # Handle timezone-aware timestamps
        last_date = pd.to_datetime(df.index[-1]).tz_convert(TARGET_TIMEZONE)
        future_dates = pd.date_range(
            start=last_date + pd.DateOffset(days=1),
            periods=FORECAST_DAYS,
            freq='D',
            tz=TARGET_TIMEZONE
        )
        
        # Volatility-adjusted simulation
        returns = np.log(df['Close']).diff().dropna()
        mu = returns.mean()
        sigma = returns.std() + 1e-8  # Prevent division by zero
        
        # Vectorized Monte Carlo
        daily_returns = np.random.normal(mu, sigma, (SIMULATIONS, FORECAST_DAYS))
        price_paths = df['Close'].iloc[-1] * np.exp(np.cumsum(daily_returns, axis=1))
        
        return pd.DataFrame({
            'Date': future_dates,
            'Low': np.percentile(price_paths, 5, axis=0),
            'Median': np.median(price_paths, axis=0),
            'High': np.percentile(price_paths, 95, axis=0)
        })
        
    except Exception as e:
        st.error(f"Forecast failed: {str(e)}")
        return pd.DataFrame()

def main():
    st.title("📈 AI Crypto Trading Assistant")
    
    # Input validation
    crypto_symbol = st.sidebar.text_input("Crypto Pair", "BTC-USD").upper()
    if '-' not in crypto_symbol:
        st.error("Invalid format - use pair like BTC-USD")
        st.stop()
        
    auto_refresh = st.sidebar.checkbox("Auto Refresh (3min)", True)
    
    # Data loading with retry
    df = load_data(crypto_symbol)
    if df is None:
        st.stop()
    
    # Analysis pipeline
    if st.button("🔄 Analyze") or auto_refresh:
        with st.spinner("Running analysis..."):
            model_rf, model_gb, features = train_model(df, crypto_symbol)
            forecast = generate_price_forecast(df)
            
        if model_rf and model_gb and not forecast.empty:
            # Calculate trading signals
            current_price = df['Close'].iloc[-1]
            atr = df['ATR'].iloc[-1] if 'ATR' in df.columns else df['High'].iloc[-1] - df['Low'].iloc[-1]
            
            take_profit = current_price + 1.5 * atr
            stop_loss = current_price - 0.8 * atr
            
            # Display results
            st.subheader("🚦 Trading Signals")
            cols = st.columns(3)
            cols[0].metric("Current Price", f"${current_price:.2f}")
            cols[1].metric("Take Profit", f"${take_profit:.2f}")
            cols[2].metric("Stop Loss", f"${stop_loss:.2f}", delta_color="inverse")
            
            # Show forecast
            st.subheader("🔮 Price Forecast")
            st.line_chart(forecast.set_index('Date'))
    
    # Live market data with fallback
    try:
        live_data = yf.download(crypto_symbol, period='1d', interval='15m')
        if not live_data.empty:
            st.subheader("🔴 Live Market")
            current = live_data.iloc[-1]
            cols = st.columns(3)
            cols[0].metric("Price", f"${current['Close']:.2f}")
            cols[1].metric("24H High", f"${live_data['High'].max():.2f}")
            cols[2].metric("24H Low", f"${live_data['Low'].min():.2f}")
    except Exception as e:
        st.warning("Live data temporarily unavailable")

    if auto_refresh:
        time.sleep(REFRESH_INTERVAL)
        st.rerun()

if __name__ == "__main__":
    main()
