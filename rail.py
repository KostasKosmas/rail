import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import TimeSeriesSplit
import joblib
import os
import time
from pytz import timezone

# Configuration
MAX_DATA_POINTS = 3000
SAVE_PATH = "saved_models"
FORECAST_DAYS = 14
REFRESH_INTERVAL = 180  # 3 minutes
TARGET_TZ = 'Europe/Athens'

# Setup directory
os.makedirs(SAVE_PATH, exist_ok=True)

@st.cache_data
def load_data(symbol):
    """Load data with fallback strategy and validation"""
    try:
        # Try hourly then daily data
        df = yf.download(symbol, period="60d", interval="1h", progress=False)
        if len(df) < 100:
            df = yf.download(symbol, period="730d", interval="1d", progress=False)
        
        # Validate data
        if len(df) < 50:
            st.error(f"⚠️ Insufficient data for {symbol}")
            st.info("Try popular pairs: BTC-USD, ETH-USD, BNB-USD")
            return None
            
        # Basic features
        df = df[['Close', 'Volume']].copy()
        df['SMA_20'] = df['Close'].rolling(20).mean()
        df['SMA_50'] = df['Close'].rolling(50).mean()
        df['Returns'] = np.log(df['Close']).diff()
        
        return df.dropna().iloc[-MAX_DATA_POINTS:]
        
    except Exception as e:
        st.error(f"Data error: {str(e)}")
        return None

def train_model(df, symbol):
    """Simplified model training with validation"""
    try:
        if df is None or len(df) < 100:
            return None, None
            
        # Create target: next day's return
        y = (df['Returns'].shift(-1) > 0
        y = y.dropna().astype(int)
        X = df[['SMA_20', 'SMA_50', 'Returns']].iloc[:-1]
        
        # Align X and y
        X = X.iloc[:len(y)]
        y = y.iloc[:len(X)]
        
        # Simple classifier
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=5,
            random_state=42,
            n_jobs=-1
        )
        model.fit(X, y)
        
        # Save model
        joblib.dump(model, os.path.join(SAVE_PATH, f"{symbol}_model.pkl"))
        return model, X.columns.tolist()
        
    except Exception as e:
        st.error(f"Training failed: {str(e)}")
        return None, None

def generate_forecast(df):
    """Simplified price forecasting"""
    try:
        last_close = df['Close'].iloc[-1]
        volatility = df['Returns'].std()
        
        # Generate possible paths
        days = np.arange(1, FORECAST_DAYS+1)
        simulations = last_close * np.exp(np.cumsum(
            np.random.normal(0, volatility, (1000, FORECAST_DAYS)),
            axis=1
        )
        
        # Calculate percentiles
        return pd.DataFrame({
            'Day': days,
            'Low': np.percentile(simulations, 5, axis=0),
            'Median': np.median(simulations, axis=0),
            'High': np.percentile(simulations, 95, axis=0)
        })
        
    except Exception as e:
        st.error(f"Forecast error: {str(e)}")
        return None

def main():
    st.title("Crypto Trading Assistant")
    
    # Input validation
    symbol = st.text_input("Crypto Pair (e.g., BTC-USD)", "BTC-USD").strip().upper()
    if '-' not in symbol:
        st.error("Invalid format. Use: XXX-XXX")
        st.stop()
    
    # Load data
    df = load_data(symbol)
    if df is None:
        st.stop()
    
    # Training and prediction
    model, features = train_model(df, symbol)
    
    if model:
        # Generate forecast
        forecast = generate_forecast(df)
        current_price = df['Close'].iloc[-1]
        last_change = df['Close'].iloc[-1] - df['Close'].iloc[-2]
        
        # Display results
        st.subheader(f"{symbol} Analysis")
        cols = st.columns(3)
        cols[0].metric("Current Price", f"${current_price:.2f}")
        cols[1].metric("24h Change", f"${last_change:.2f}")
        cols[2].metric("Volatility", f"{df['Returns'].std()*100:.2f}%")
        
        if forecast is not None:
            st.subheader("14-Day Forecast")
            st.line_chart(forecast.set_index('Day'))
            
            # Trading signals
            take_profit = forecast['High'].iloc[-1]
            stop_loss = forecast['Low'].min()
            
            cols = st.columns(2)
            cols[0].metric("Suggested Take Profit", f"${take_profit:.2f}")
            cols[1].metric("Recommended Stop Loss", f"${stop_loss:.2f}", 
                          delta_color="inverse")
    
    # Auto-refresh
    if st.button("Refresh") or st._is_running_with_streamlit:
        time.sleep(REFRESH_INTERVAL)
        st.rerun()

if __name__ == "__main__":
    main()
