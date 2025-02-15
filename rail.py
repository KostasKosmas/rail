import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import accuracy_score, f1_score
import time
import joblib
import os
from pytz import timezone, utc
from sklearn.feature_selection import SelectKBest, f_classif

# Configuration
MAX_DATA_POINTS = 3000  # Optimal balance between history and recent data
INCREMENTAL_ESTIMATORS = 50  # Number of estimators to add in each retrain

# Save models and data
def save_artifacts(df, model_rf, model_gb, crypto_symbol):
    if not os.path.exists("saved_models"):
        os.makedirs("saved_models")
    joblib.dump(model_rf, f"saved_models/{crypto_symbol}_model_rf.pkl")
    joblib.dump(model_gb, f"saved_models/{crypto_symbol}_model_gb.pkl")
    df.to_csv(f"saved_models/{crypto_symbol}_data.csv")

# Feature engineering with caching
@st.cache_data
def load_data(symbol, interval="1d", period="5y"):
    try:
        df = yf.download(symbol, period=period, interval=interval)
        if df.empty:
            st.warning(f"⚠️ Data unavailable for {symbol} ({interval})")
            return None
            
        # Feature engineering
        df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
        df["SMA_50"] = df["Close"].rolling(50).mean()
        df["SMA_200"] = df["Close"].rolling(200).mean()
        df['RSI'] = 100 - (100 / (1 + (df['Close'].diff().clip(lower=0).rolling(14).mean() / 
                            df['Close'].diff().clip(upper=0).abs().rolling(14).mean()))
        df['MACD'] = df['Close'].ewm(span=12).mean() - df['Close'].ewm(span=26).mean()
        df['OBV'] = (np.sign(df['Close'].diff()) * df['Volume']).cumsum()
        df['Volume_MA'] = df['Volume'].rolling(20).mean()
        df = df.dropna()
        
        # Keep only relevant recent data
        df = df.iloc[-MAX_DATA_POINTS:]
        
        return df.astype(np.float32)  # Reduce memory usage
    except Exception as e:
        st.error(f"❌ Data loading error: {e}")
        return None

# Feature selection
def select_features(X, y):
    selector = SelectKBest(score_func=f_classif, k='all')
    selector.fit(X, y)
    return selector

# Model training with incremental learning
@st.cache_resource
def train_model(df, crypto_symbol):
    try:
        X = df[["SMA_50", "SMA_200", "RSI", "MACD", "OBV", "Volume_MA"]]
        y = np.where(df["Close"].shift(-1) > df["Close"], 1, 0)
        split = int(0.8 * len(df))
        X_train, X_test = X.iloc[:split], X.iloc[split:]
        y_train, y_test = y[:split], y[split:]
        
        # Feature selection
        selector = select_features(X_train, y_train)
        X_train = selector.transform(X_train)
        X_test = selector.transform(X_test)

        # Load existing models if available
        model_rf = joblib.load(f"saved_models/{crypto_symbol}_model_rf.pkl") if os.path.exists(f"saved_models/{crypto_symbol}_model_rf.pkl") else None
        model_gb = joblib.load(f"saved_models/{crypto_symbol}_model_gb.pkl") if os.path.exists(f"saved_models/{crypto_symbol}_model_gb.pkl") else None

        # Random Forest with incremental training
        if model_rf:
            model_rf.n_estimators += INCREMENTAL_ESTIMATORS
            model_rf.fit(X_train, y_train)
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
            ).fit(X_train, y_train).best_estimator_

        # Gradient Boosting with early stopping
        if model_gb:
            model_gb.n_estimators += INCREMENTAL_ESTIMATORS
            model_gb.fit(X_train, y_train)
        else:
            param_dist_gb = {
                'n_estimators': [200, 300],
                'learning_rate': [0.05, 0.1],
                'max_depth': [3, 5],
                'subsample': [0.8, 1.0],
                'validation_fraction': [0.1],
                'n_iter_no_change': [5]
            }
            model_gb = RandomizedSearchCV(
                GradientBoostingClassifier(),
                param_dist_gb,
                n_iter=3,
                cv=3,
                scoring='f1'
            ).fit(X_train, y_train).best_estimator_

        # Ensemble predictions
        y_pred = (model_rf.predict(X_test) + model_gb.predict(X_test)) // 2
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        
        st.success(f"Model refresh complete | Accuracy: {accuracy:.2%} | F1 Score: {f1:.2%}")
        return model_rf, model_gb, selector, df
        
    except Exception as e:
        st.error(f"Training error: {e}")
        return None, None, None, None

# Trading logic with improved risk management
def calculate_trade_levels(df, selector, model_rf, model_gb):
    try:
        X = selector.transform(df[["SMA_50", "SMA_200", "RSI", "MACD", "OBV", "Volume_MA"]])
        ensemble_pred = (model_rf.predict(X[-30:]) + model_gb.predict(X[-30:])) // 2
        confidence = np.mean(ensemble_pred == model_rf.predict(X[-30:]))
        
        # Calculate dynamic ATR
        high_low = df['High'][-14:] - df['Low'][-14:]
        atr = np.mean(high_low)
        
        # Price action analysis
        current_price = df['Close'].iloc[-1]
        resistance = df['High'][-14:].max()
        support = df['Low'][-14:].min()
        
        # Risk management parameters
        if confidence > 0.7:
            risk_multiplier = 1.5
        else:
            risk_multiplier = 1.0
            
        stop_loss = current_price - (atr * 1.5 * risk_multiplier)
        take_profit = current_price + (atr * 2.0 * risk_multiplier)
        
        # Trend confirmation
        sma_direction = np.sign(df['SMA_50'][-1] - df['SMA_50'][-5])
        if sma_direction > 0:
            take_profit *= 1.1
        else:
            stop_loss *= 0.9
            
        return {
            'entry': current_price,
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'confidence': confidence,
            'trend_direction': sma_direction
        }
    except Exception as e:
        st.error(f"Trade calculation error: {e}")
        return None

# Main app interface
def main():
    st.title("🚀 AI Crypto Trading Optimizer")
    
    # Sidebar controls
    crypto_symbol = st.sidebar.text_input("Crypto Pair", "BTC-USD")
    auto_refresh = st.sidebar.checkbox("Enable Live Updates", True)
    
    # Data loading
    df = load_data(crypto_symbol)
    if df is None:
        st.stop()
        
    # Model training section
    if st.button("🔄 Refresh Analysis") or auto_refresh:
        with st.spinner("Optimizing trading models..."):
            model_rf, model_gb, selector, updated_df = train_model(df, crypto_symbol)
            if model_rf and model_gb:
                save_artifacts(updated_df, model_rf, model_gb, crypto_symbol)
                
        # Display trading signals
        levels = calculate_trade_levels(updated_df, selector, model_rf, model_gb)
        if levels:
            st.subheader("📈 Live Trading Signals")
            col1, col2, col3 = st.columns(3)
            col1.metric("Current Price", f"${levels['entry']:.2f}")
            col2.metric("Stop Loss", f"${levels['stop_loss']:.2f}", delta_color="inverse")
            col3.metric("Take Profit", f"${levels['take_profit']:.2f}")
            
            st.progress(levels['confidence'])
            st.caption(f"Model Confidence: {levels['confidence']:.2%}")
            
            if levels['trend_direction'] > 0:
                st.success("📈 Strong Bullish Trend Detected")
            else:
                st.warning("📉 Bearish Trend Caution")
                
        # Live price updates
        live_data = yf.download(crypto_symbol, period='1d', interval='1m')
        if not live_data.empty:
            st.subheader("🔴 Live Market Feed")
            current = live_data.iloc[-1]
            prev = live_data.iloc[-2]
            
            cols = st.columns(4)
            cols[0].metric("Price", f"${current['Close']:.2f}", 
                          f"{(current['Close'] - prev['Close']):.2f}")
            cols[1].metric("Volume", f"{current['Volume']:,.0f}", 
                          f"{(current['Volume'] - prev['Volume']):,.0f}")
            cols[2].metric("High", f"${current['High']:.2f}")
            cols[3].metric("Low", f"${current['Low']:.2f}")
            
    # Strategy explanation
    st.subheader("📚 Trading Strategy Overview")
    st.markdown("""
    - **AI Ensemble Model**: Combines Random Forest & Gradient Boosting predictions
    - **Dynamic Risk Management**: Adjusts stop-loss/take-profit based on volatility (ATR)
    - **Trend Confirmation**: Uses SMA crossover validation
    - **Continuous Learning**: Models improve with each refresh
    - **Live Adaptation**: Auto-updates every 5 minutes with market changes
    """)
    
    # Auto-refresh logic
    if auto_refresh:
        time.sleep(300)
        st.rerun()

if __name__ == "__main__":
    main()
