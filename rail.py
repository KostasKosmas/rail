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

# Configuration
MAX_DATA_POINTS = 5000
INCREMENTAL_ESTIMATORS = 100
SAVE_PATH = "saved_models"
FORECAST_DAYS = 15
SIMULATIONS = 1000
REFRESH_INTERVAL = 180  # 3 minutes
TARGET_TIMEZONE = timezone('Europe/Athens')  # Change to your timezone

# Create save directory
os.makedirs(SAVE_PATH, exist_ok=True)

@st.cache_data
def load_data(symbol, interval="1h", period="10y"):
    try:
        df = yf.download(symbol, period=period, interval=interval)
        if df.empty or len(df) < 200:
            st.error(f"⚠️ Insufficient data for {symbol}")
            return None

        # Feature Engineering
        df = df[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
        
        # Technical Indicators
        df['SMA_20'] = df['Close'].rolling(20).mean()
        df['SMA_50'] = df['Close'].rolling(50).mean()
        df['SMA_200'] = df['Close'].rolling(200).mean()
        df['BB_upper'] = df['SMA_20'] + 2*df['Close'].rolling(20).std()
        df['BB_lower'] = df['SMA_20'] - 2*df['Close'].rolling(20).std()
        df['ATR'] = ta.volatility.average_true_range(df['High'], df['Low'], df['Close'], 14)
        df['RSI'] = ta.momentum.rsi(df['Close'], 14)
        df['MACD'] = ta.trend.macd_diff(df['Close'])
        df['OBV'] = ta.volume.on_balance_volume(df['Close'], df['Volume'])
        df['VWAP'] = ta.volume.volume_weighted_average_price(df['High'], df['Low'], df['Close'], df['Volume'])
        
        return df.dropna().iloc[-MAX_DATA_POINTS:].astype(np.float32)
    
    except Exception as e:
        st.error(f"Data error: {str(e)}")
        return None

def train_model(df, crypto_symbol):
    try:
        if df is None or len(df) < 500:
            raise ValueError("Insufficient training data")
            
        y = np.where(df["Close"].shift(-3) > df["Close"], 1, 0).ravel()
        X = df.drop(columns=['Close'], errors='ignore')
        
        tscv = TimeSeriesSplit(n_splits=5)
        
        # Random Forest
        rf = RandomizedSearchCV(
            RandomForestClassifier(n_jobs=-1, class_weight='balanced'),
            {
                'n_estimators': [500, 1000],
                'max_depth': [None, 20, 30],
                'min_samples_split': [2, 5]
            },
            n_iter=20, cv=tscv, scoring='f1'
        ).fit(X, y)
        
        # Gradient Boosting
        gb = RandomizedSearchCV(
            GradientBoostingClassifier(n_iter_no_change=10),
            {
                'n_estimators': [500, 1000],
                'learning_rate': [0.01, 0.05],
                'max_depth': [3, 5]
            },
            n_iter=20, cv=tscv, scoring='f1'
        ).fit(X, y)
        
        # Save models
        joblib.dump(rf.best_estimator_, f"{SAVE_PATH}/{crypto_symbol}_rf.pkl")
        joblib.dump(gb.best_estimator_, f"{SAVE_PATH}/{crypto_symbol}_gb.pkl")
        
        return rf.best_estimator_, gb.best_estimator_, X.columns.tolist()
    
    except Exception as e:
        st.error(f"Training error: {str(e)}")
        return None, None, None

def generate_price_forecast(df):
    try:
        # Generate future dates
        last_date = df.index[-1].tz_convert(TARGET_TIMEZONE)
        future_dates = pd.date_range(
            start=last_date + pd.DateOffset(days=1),
            periods=FORECAST_DAYS,
            freq='D',
            tz=TARGET_TIMEZONE
        )
        
        # Monte Carlo simulation
        returns = np.log(df['Close']).diff().dropna()
        mu = returns.mean()
        sigma = returns.std()
        last_price = df['Close'].iloc[-1]
        
        forecast = np.zeros((FORECAST_DAYS, SIMULATIONS))
        for i in range(SIMULATIONS):
            daily_returns = np.random.normal(mu, sigma, FORECAST_DAYS)
            forecast[:, i] = last_price * np.exp(np.cumsum(daily_returns))
        
        return pd.DataFrame({
            'Date': future_dates,
            'Low': np.percentile(forecast, 5, axis=1),
            'Median': np.median(forecast, axis=1),
            'High': np.percentile(forecast, 95, axis=1)
        })
        
    except Exception as e:
        st.error(f"Forecast error: {str(e)}")
        return pd.DataFrame()

def calculate_trade_signals(df, forecast):
    try:
        current_price = df['Close'].iloc[-1]
        atr = df['ATR'].iloc[-1]
        
        # Calculate targets
        take_profit = current_price + 1.5 * atr
        stop_loss = current_price - 0.8 * atr
        
        # Find target date
        target_info = forecast[forecast['High'] >= take_profit]
        if not target_info.empty:
            target_date = target_info['Date'].iloc[0].strftime('%Y-%m-%d')
            days_to_target = (target_info['Date'].iloc[0] - forecast['Date'].iloc[0]).days
            confidence = max(10, 100 - (days_to_target * 5))
        else:
            target_date = "Beyond 15-day forecast"
            days_to_target = FORECAST_DAYS
            confidence = 10
        
        return {
            'current_price': current_price,
            'take_profit': take_profit,
            'stop_loss': stop_loss,
            'target_date': target_date,
            'days_to_target': days_to_target,
            'confidence': confidence,
            'trend': 'Bullish' if current_price > df['SMA_200'].iloc[-1] else 'Bearish'
        }
    except Exception as e:
        st.error(f"Signal error: {str(e)}")
        return None

def main():
    st.title("📈 AI Crypto Trading Assistant")
    
    # User inputs
    crypto_symbol = st.sidebar.text_input("Crypto Pair", "BTC-USD")
    auto_refresh = st.sidebar.checkbox("Auto Refresh (3min)", True)
    
    # Data pipeline
    df = load_data(crypto_symbol)
    if df is None:
        st.stop()
    
    if st.button("🔄 Analyze") or auto_refresh:
        with st.spinner("Generating insights..."):
            model_rf, model_gb, features = train_model(df, crypto_symbol)
            forecast = generate_price_forecast(df)
            
        if model_rf and model_gb and not forecast.empty:
            signals = calculate_trade_signals(df, forecast)
            
            # Display trade signals
            st.subheader("🚦 Trading Signals")
            cols = st.columns(3)
            cols[0].metric("Current Price", f"${signals['current_price']:.2f}")
            cols[1].metric("Take Profit", f"${signals['take_profit']:.2f}",
                          f"Target: {signals['target_date']}")
            cols[2].metric("Stop Loss", f"${signals['stop_loss']:.2f}",
                          delta_color="inverse")
            
            # Confidence and timeline
            st.progress(signals['confidence']/100)
            st.write(f"""
                **Expected Timeline:**  
                - Days to Target: {signals['days_to_target']}  
                - Confidence Score: {signals['confidence']}/100  
                - Market Trend: {signals['trend']}
            """)
            
            # Forecast visualization
            st.subheader("🔮 Price Forecast")
            forecast_display = forecast.set_index('Date')[['Low', 'Median', 'High']]
            st.area_chart(forecast_display)
    
    # Live market monitor
    try:
        live_data = yf.download(crypto_symbol, period='1d', interval='5m')
        if not live_data.empty:
            st.subheader("🔴 Live Market Monitor")
            current = live_data.iloc[-1]
            prev = live_data.iloc[-2]
            
            cols = st.columns(4)
            cols[0].metric("Price", f"${current['Close']:.2f}", 
                          f"{current['Close'] - prev['Close']:.2f}")
            cols[1].metric("Volume", f"{current['Volume']:,.0f}", 
                          f"{(current['Volume']/1e6 - prev['Volume']/1e6):.1f}M")
            cols[2].metric("24H High", f"${live_data['High'].max():.2f}")
            cols[3].metric("24H Low", f"${live_data['Low'].min():.2f}")
    except Exception as e:
        st.error(f"Live feed error: {str(e)}")

    if auto_refresh:
        time.sleep(REFRESH_INTERVAL)
        st.rerun()

if __name__ == "__main__":
    main()
