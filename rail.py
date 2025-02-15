import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import accuracy_score, f1_score
from sklearn.feature_selection import SelectKBest, f_classif
import joblib
import os
import time
from pytz import timezone, utc

# Configuration
MAX_DATA_POINTS = 3000
INCREMENTAL_ESTIMATORS = 50
SAVE_PATH = "saved_models"
FORECAST_DAYS = 15

# Create save directory if not exists
if not os.path.exists(SAVE_PATH):
    os.makedirs(SAVE_PATH)

# Improved data loading with caching
@st.cache_data
def load_data(symbol, interval="1d", period="5y"):
    try:
        df = yf.download(symbol, period=period, interval=interval)
        if df.empty:
            st.error(f"⚠️ Data unavailable for {symbol} ({interval})")
            return None

        # Feature engineering
        df = df[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
        
        # Technical indicators
        df["SMA_50"] = df["Close"].rolling(50).mean()
        df["SMA_200"] = df["Close"].rolling(200).mean()
        
        # RSI calculation
        delta = df['Close'].diff()
        gain = delta.clip(lower=0)
        loss = (-delta).clip(lower=0)
        avg_gain = gain.rolling(14).mean()
        avg_loss = loss.rolling(14).mean()
        rs = avg_gain / avg_loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        df['MACD'] = df['Close'].ewm(span=12).mean() - df['Close'].ewm(span=26).mean()
        df['OBV'] = (np.sign(df['Close'].diff()) * df['Volume']).cumsum()
        df['Volume_MA'] = df['Volume'].rolling(20).mean()
        
        # Clean and format data
        df = df.dropna().iloc[-MAX_DATA_POINTS:]
        return df.astype(np.float32).reset_index(drop=True)
    
    except Exception as e:
        st.error(f"❌ Data loading error: {e}")
        return None

# Feature selection
def select_features(X, y):
    selector = SelectKBest(score_func=f_classif, k='all')
    selector.fit(X, y)
    return selector

# Advanced model training with incremental learning
def train_model(df, crypto_symbol):
    try:
        X = df[["SMA_50", "SMA_200", "RSI", "MACD", "OBV", "Volume_MA"]]
        y = np.where(df["Close"].shift(-1) > df["Close"], 1, 0).ravel()
        split = int(0.8 * len(df))
        X_train, X_test = X.iloc[:split], X.iloc[split:]
        y_train, y_test = y[:split], y[split:]

        # Feature selection
        selector = select_features(X_train, y_train)
        X_train_sel = selector.transform(X_train)
        X_test_sel = selector.transform(X_test)

        # Model paths
        rf_path = f"{SAVE_PATH}/{crypto_symbol}_model_rf.pkl"
        gb_path = f"{SAVE_PATH}/{crypto_symbol}_model_gb.pkl"

        # Load or initialize models
        model_rf = joblib.load(rf_path) if os.path.exists(rf_path) else None
        model_gb = joblib.load(gb_path) if os.path.exists(gb_path) else None

        # Random Forest training
        if model_rf:
            model_rf.n_estimators += INCREMENTAL_ESTIMATORS
            model_rf.fit(X_train_sel, y_train)
        else:
            param_dist_rf = {
                'n_estimators': [200, 300],
                'max_depth': [15, 20, None],
                'min_samples_split': [2, 5],
                'class_weight': ['balanced']
            }
            model_rf = RandomizedSearchCV(
                RandomForestClassifier(n_jobs=-1),
                param_dist_rf,
                n_iter=3,
                cv=3,
                scoring='f1'
            ).fit(X_train_sel, y_train).best_estimator_

        # Gradient Boosting training
        if model_gb:
            model_gb.n_estimators += INCREMENTAL_ESTIMATORS
            model_gb.fit(X_train_sel, y_train)
        else:
            param_dist_gb = {
                'n_estimators': [200, 300],
                'learning_rate': [0.05, 0.1],
                'max_depth': [3, 5],
                'subsample': [0.8, 1.0]
            }
            model_gb = RandomizedSearchCV(
                GradientBoostingClassifier(n_iter_no_change=5),
                param_dist_gb,
                n_iter=3,
                cv=3,
                scoring='f1'
            ).fit(X_train_sel, y_train).best_estimator_

        # Evaluate performance
        y_pred = (model_rf.predict(X_test_sel) + model_gb.predict(X_test_sel)) // 2
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        
        # Save updated models
        joblib.dump(model_rf, rf_path)
        joblib.dump(model_gb, gb_path)
        
        return model_rf, model_gb, selector, df, accuracy, f1
        
    except Exception as e:
        st.error(f"Training error: {e}")
        return None, None, None, None, None, None

# Enhanced price prediction with Monte Carlo simulation
def generate_price_points(df, days=FORECAST_DAYS, simulations=100):
    try:
        if df.empty or len(df) < 100:
            raise ValueError("Insufficient historical data")
            
        # Calculate daily returns
        returns = df['Close'].pct_change().dropna()
        
        # Simulation parameters
        mu = returns.mean()
        sigma = returns.std()
        last_price = df['Close'].iloc[-1]
        
        # Generate simulations
        price_paths = []
        for _ in range(simulations):
            daily_returns = np.random.normal(mu, sigma, days)
            price_path = last_price * (1 + daily_returns).cumprod()
            price_paths.append(price_path)
        
        # Calculate percentiles
        forecast = np.percentile(price_paths, [10, 50, 90], axis=0)
        return forecast[-1]  # Return median forecast
        
    except Exception as e:
        st.error(f"Price prediction error: {e}")
        return None

# Enhanced trading logic
def calculate_trade_levels(df, selector, model_rf, model_gb):
    try:
        # Prepare features
        features = ["SMA_50", "SMA_200", "RSI", "MACD", "OBV", "Volume_MA"]
        X = selector.transform(df[features][-30:])
        
        # Generate predictions
        rf_pred = model_rf.predict_proba(X)[:, 1]
        gb_pred = model_gb.predict_proba(X)[:, 1]
        combined_confidence = (rf_pred + gb_pred) / 2
        
        # Calculate dynamic ATR
        high_low = df['High'].iloc[-14:].values - df['Low'].iloc[-14:].values
        atr = float(np.mean(high_low))
        
        # Current market state
        current_price = float(df['Close'].iloc[-1].item())
        sma_trend = bool(df['SMA_50'].iloc[-1].item() > df['SMA_200'].iloc[-1].item())
        
        # Generate price forecast
        price_forecast = generate_price_points(df)
        if price_forecast is None:
            raise ValueError("Failed to generate price forecast")
        
        # Risk management parameters
        confidence = float(np.mean(combined_confidence))
        risk_multiplier = 1.5 if confidence > 0.7 else 1.0
        
        if sma_trend:
            stop_loss = current_price - (atr * 1.2 * risk_multiplier)
            take_profit = current_price + (atr * 2.5 * risk_multiplier)
        else:
            stop_loss = current_price + (atr * 1.2 * risk_multiplier)
            take_profit = current_price - (atr * 2.5 * risk_multiplier)
            
        return {
            'entry': current_price,
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'forecast': price_forecast,
            'confidence': confidence,
            'trend': 'Bullish' if sma_trend else 'Bearish'
        }
    except Exception as e:
        st.error(f"Trade calculation error: {e}")
        return None

# Streamlit UI
def main():
    st.title("🚀 AI Crypto Trading System")
    
    # User inputs
    crypto_symbol = st.sidebar.text_input("Crypto Pair", "BTC-USD")
    auto_refresh = st.sidebar.checkbox("Enable Live Updates", True)
    
    # Load data
    df = load_data(crypto_symbol)
    if df is None:
        st.stop()
    
    # Model training section
    if st.button("🔄 Refresh Analysis") or auto_refresh:
        with st.spinner("Optimizing trading models..."):
            model_rf, model_gb, selector, df, accuracy, f1 = train_model(df, crypto_symbol)
            
        if model_rf and model_gb:
            # Display performance metrics
            col1, col2 = st.columns(2)
            col1.metric("Model Accuracy", f"{accuracy:.2%}")
            col2.metric("F1 Score", f"{f1:.2%}")
            
            # Show trading signals
            levels = calculate_trade_levels(df, selector, model_rf, model_gb)
            if levels:
                st.subheader("📈 Trading Signals")
                
                cols = st.columns(3)
                cols[0].metric("Current Price", f"${levels['entry']:.2f}")
                cols[1].metric("Stop Loss", f"${levels['stop_loss']:.2f}", delta_color="inverse")
                cols[2].metric("Take Profit", f"${levels['take_profit']:.2f}")
                
                # Display price forecast
                st.subheader("🔮 Price Forecast")
                forecast_df = pd.DataFrame({
                    'Day': range(1, FORECAST_DAYS+1),
                    'Predicted Price': levels['forecast']
                })
                st.line_chart(forecast_df.set_index('Day'))
                
                st.progress(levels['confidence'])
                st.caption(f"Model Confidence: {levels['confidence']:.2%}")
                
                if levels['trend'] == 'Bullish':
                    st.success("📈 Bullish Trend Detected - Long Position Recommended")
                else:
                    st.warning("📉 Bearish Trend Detected - Short Position Recommended")
    
    # Live market data
    try:
        live_data = yf.download(crypto_symbol, period='1d', interval='1m')
        if not live_data.empty:
            st.subheader("🔴 Live Market Feed")
            current = live_data.iloc[-1]
            prev = live_data.iloc[-2]
            
            # Convert to native Python types
            current_close = current['Close'].item()
            prev_close = prev['Close'].item()
            current_vol = int(current['Volume'].item())
            prev_vol = int(prev['Volume'].item())
            high_24h = live_data['High'].max().item()
            low_24h = live_data['Low'].min().item()
            
            cols = st.columns(4)
            cols[0].metric("Price", f"${current_close:.2f}", f"{current_close - prev_close:.2f}")
            cols[1].metric("Volume", f"{current_vol:,}", f"{current_vol - prev_vol:,}")
            cols[2].metric("24H High", f"${high_24h:.2f}")
            cols[3].metric("24H Low", f"${low_24h:.2f}")
    except Exception as e:
        st.error(f"Live data error: {str(e)}")

    # Auto-refresh logic
    if auto_refresh:
        time.sleep(300)
        st.rerun()

if __name__ == "__main__":
    main()
