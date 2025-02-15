# crypto_trading_advanced.py
# Install: pip install streamlit yfinance pandas numpy scikit-learn joblib ta

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, StackingClassifier
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.preprocessing import RobustScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
import joblib
import os
import time
from ta import add_all_ta_features
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.volatility import BollingerBands
from ta.trend import MACD, ADXIndicator
from ta.volume import MFIIndicator

# Configuration
MAX_DATA_POINTS = 5000
INCREMENTAL_ESTIMATORS = 75
SAVE_PATH = "saved_models"
FORECAST_DAYS = 21
SIMULATIONS = 1000
THRESHOLD_DAYS = 3

# Auto-create model directory
os.makedirs(SAVE_PATH, exist_ok=True)

@st.cache_data
def load_enhanced_data(symbol, interval="1d", period="5y"):
    try:
        df = yf.download(symbol, period=period, interval=interval, auto_adjust=True)
        if df.empty or len(df) < 100:
            st.error(f"⚠️ Insufficient data for {symbol}")
            return None

        df = add_all_ta_features(df, open="Open", high="High", low="Low", close="Close", volume="Volume")
        
        # Custom indicators
        bb = BollingerBands(df["Close"])
        df["BB_upper"] = bb.bollinger_hband()
        df["BB_lower"] = bb.bollinger_lband()
        df["BB_width"] = bb.bollinger_wband()
        
        stoch = StochasticOscillator(high=df["High"], low=df["Low"], close=df["Close"])
        df["Stoch_%K"] = stoch.stoch()
        df["Stoch_%D"] = stoch.stoch_signal()
        
        mfi = MFIIndicator(high=df["High"], low=df["Low"], close=df["Close"], volume=df["Volume"])
        df["MFI"] = mfi.money_flow_index()
        
        adx = ADXIndicator(high=df["High"], low=df["Low"], close=df["Close"])
        df["ADX"] = adx.adx()
        
        # Lagged features
        for lag in [1, 3, 5]:
            df[f"Return_{lag}d"] = df["Close"].pct_change(lag)
            df[f"Volatility_{lag}d"] = df["Close"].pct_change().rolling(lag).std()
        
        # Target variable with explicit 1D conversion
        future_returns = df["Close"].pct_change(THRESHOLD_DAYS).shift(-THRESHOLD_DAYS)
        df["Target"] = np.where(future_returns > 0.015, 1, 0).astype(np.int32).flatten()
        
        # Feature engineering
        df["RSI_Volume"] = df["rsi"] * df["volume_adi"]
        df["MACD_Signal_Ratio"] = df["macd"] / (df["macd_signal"] + 1e-10)
        
        # Clean data and ensure 1D target
        df = df.dropna().iloc[-MAX_DATA_POINTS:]
        df = df.astype(np.float32)
        
        if df["Target"].ndim != 1:
            df["Target"] = df["Target"].squeeze()
            
        return df
    
    except Exception as e:
        st.error(f"Data error: {str(e)}")
        return None

def create_feature_pipeline():
    return Pipeline([
        ('scaler', RobustScaler()),
        ('selector', SelectKBest(score_func=mutual_info_classif, k=20))
    ])

def train_enhanced_model(df, crypto_symbol):
    try:
        if df is None or len(df) < 300:
            raise ValueError("Insufficient training data")
            
        # Explicit 1D conversion for target
        y = np.ravel(df["Target"].values)
        X = df.drop(columns=["Target"])
        
        # Validation check
        if y.ndim != 1:
            st.error(f"Invalid target shape: {y.shape}. Must be 1D.")
            return None, None, None, None
        
        tscv = TimeSeriesSplit(n_splits=3)
        feature_pipeline = create_feature_pipeline()
        
        split = int(0.8 * len(df))
        X_train, X_test = X.iloc[:split], X.iloc[split:]
        y_train, y_test = y[:split], y[split:]
        
        X_train_trans = feature_pipeline.fit_transform(X_train, y_train)
        X_test_trans = feature_pipeline.transform(X_test)

        rf = RandomForestClassifier(n_jobs=-1, class_weight='balanced')
        gb = GradientBoostingClassifier(n_iter_no_change=10, validation_fraction=0.1)
        meta = LogisticRegression()
        
        model_path = f"{SAVE_PATH}/{crypto_symbol}_stacked_model.pkl"
        
        if os.path.exists(model_path):
            model = joblib.load(model_path)
            if hasattr(model, 'n_estimators'):
                model.n_estimators += INCREMENTAL_ESTIMATORS
            model.fit(X_train_trans, y_train)
        else:
            model = StackingClassifier(
                estimators=[
                    ('rf', RandomizedSearchCV(
                        rf,
                        {
                            'n_estimators': [300, 500],
                            'max_depth': [None, 20],
                            'min_samples_split': [2, 5]
                        },
                        n_iter=3, cv=tscv, scoring='f1'
                    )),
                    ('gb', RandomizedSearchCV(
                        gb,
                        {
                            'learning_rate': [0.01, 0.1],
                            'subsample': [0.8, 1.0],
                            'max_depth': [3, 5]
                        },
                        n_iter=3, cv=tscv, scoring='f1'
                    ))
                ],
                final_estimator=meta,
                stack_method='predict_proba'
            ).fit(X_train_trans, y_train)

        y_pred = model.predict(X_test_trans)
        
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'f1': f1_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred)
        }
        
        joblib.dump((model, feature_pipeline), model_path)
        return model, feature_pipeline, df, metrics
        
    except Exception as e:
        st.error(f"Training error: {str(e)}")
        return None, None, None, None

def generate_enhanced_forecast(df):
    try:
        returns = np.log(df['Close']).diff().dropna()
        volatility = returns.rolling(21).std().iloc[-1]
        last_price = df['Close'].iloc[-1].item()
        
        simulations = np.exp(
            np.random.normal(
                loc=0, 
                scale=volatility, 
                size=(SIMULATIONS, FORECAST_DAYS)
            ).cumsum(axis=1)
        
        price_paths = last_price * simulations
        
        return pd.DataFrame({
            'Day': range(1, FORECAST_DAYS+1),
            'Median': np.median(price_paths, axis=0),
            'Upper_95': np.percentile(price_paths, 95, axis=0),
            'Lower_95': np.percentile(price_paths, 5, axis=0),
            'Volatility': volatility
        })
    except Exception as e:
        st.error(f"Forecast error: {str(e)}")
        return pd.DataFrame()

def calculate_risk_levels(df, model, pipeline):
    try:
        current_features = pipeline.transform(df.iloc[-1:].drop(columns=["Target"]))
        proba = model.predict_proba(current_features)[0][1]
        
        volatility = df["Volatility_5d"].iloc[-1]
        adx_value = df["ADX"].iloc[-1]
        mfi_value = df["MFI"].iloc[-1]
        
        risk_params = {
            'base_atr': df['average_true_range'].iloc[-14:].mean(),
            'trend_strength': adx_value / 100,
            'volume_confirmation': mfi_value / 100
        }
        
        confidence = 0.4*proba + 0.3*risk_params['trend_strength'] + 0.3*risk_params['volume_confirmation']
        
        return {
            'entry': df['Close'].iloc[-1].item(),
            'stop_loss': df['Close'].iloc[-1] * (1 - (risk_params['base_atr'] * (2 - confidence))),
            'take_profit': df['Close'].iloc[-1] * (1 + (risk_params['base_atr'] * (1 + confidence))),
            'confidence': confidence,
            'volatility': volatility,
            'trend_strength': adx_value,
            'mfi_status': "Overbought" if mfi_value > 80 else "Oversold" if mfi_value < 20 else "Neutral"
        }
    except Exception as e:
        st.error(f"Risk calculation error: {str(e)}")
        return None

def main():
    st.title("🚀 Advanced AI Crypto Trading System")
    
    crypto_symbol = st.sidebar.text_input("Crypto Pair", "BTC-USD")
    auto_refresh = st.sidebar.checkbox("Live Updates", True)
    
    df = load_enhanced_data(crypto_symbol)
    if df is None:
        st.stop()
    
    if st.button("🔄 Refresh") or auto_refresh:
        with st.spinner("Training advanced model..."):
            model, pipeline, df, metrics = train_enhanced_model(df, crypto_symbol)
        
        if model and metrics:
            st.subheader("📊 Advanced Performance Metrics")
            cols = st.columns(4)
            cols[0].metric("Accuracy", f"{metrics['accuracy']:.2%}")
            cols[1].metric("F1 Score", f"{metrics['f1']:.2%}")
            cols[2].metric("Precision", f"{metrics['precision']:.2%}")
            cols[3].metric("Recall", f"{metrics['recall']:.2%}")
            
            levels = calculate_risk_levels(df, model, pipeline)
            forecast = generate_enhanced_forecast(df)
            
            if levels and not forecast.empty:
                st.subheader("📈 Smart Trading Signals")
                
                cols = st.columns(4)
                cols[0].metric("Current Price", f"${levels['entry']:.2f}")
                cols[1].metric("Stop Loss", f"${levels['stop_loss']:.2f}", delta_color="inverse")
                cols[2].metric("Take Profit", f"${levels['take_profit']:.2f}")
                cols[3].metric("Risk Score", f"{levels['confidence']:.2%}")
                
                st.subheader("📊 Market Conditions")
                cond_cols = st.columns(3)
                cond_cols[0].metric("Volatility", f"{levels['volatility']:.2%}")
                cond_cols[1].metric("Trend Strength", f"{levels['trend_strength']:.1f}/100")
                cond_cols[2].metric("MFI Status", levels['mfi_status'])
                
                st.subheader("🔮 Price Forecast")
                st.line_chart(forecast.set_index('Day')[['Median', 'Upper_95', 'Lower_95']])
                
                st.subheader("📉 Technical Overview")
                tab1, tab2, tab3 = st.tabs(["Bollinger Bands", "MACD", "Stochastic"])
                
                with tab1:
                    st.line_chart(df[['Close', 'BB_upper', 'BB_lower']].iloc[-100:])
                with tab2:
                    st.line_chart(df[['macd', 'macd_signal']].iloc[-100:])
                with tab3:
                    st.line_chart(df[['Stoch_%K', 'Stoch_%D']].iloc[-100:])

    try:
        live_data = yf.download(crypto_symbol, period='1d', interval='1m', auto_adjust=True)
        if not live_data.empty:
            st.subheader("🔴 Live Market Dashboard")
            current = live_data.iloc[-1]
            changes = live_data.pct_change().iloc[-1] * 100
            
            cols = st.columns(4)
            cols[0].metric("Price", f"${current['Close']:.2f}", f"{changes['Close']:.2f}%")
            cols[1].metric("Volume", f"{current['Volume']:,.0f}", f"{changes['Volume']:.2f}%")
            cols[2].metric("Spread", f"{(current['High'] - current['Low']):.2f}")
            cols[3].metric("VWAP", f"{np.sum(live_data['Close'] * live_data['Volume']) / np.sum(live_data['Volume']):.2f}")
            
    except Exception as e:
        st.error(f"Live feed error: {str(e)}")

    if auto_refresh:
        time.sleep(300)
        st.rerun()

if __name__ == "__main__":
    main()
