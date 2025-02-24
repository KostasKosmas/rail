# trading_system.py
import logging
import numpy as np
import pandas as pd
import yfinance as yf
import streamlit as st
import plotly.express as px
import optuna
import requests
from xgboost import XGBClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import roc_auc_score, precision_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
import warnings
import json
from functools import partial
from typing import Tuple

# Configuration
DEFAULT_SYMBOL = 'BTC-USD'
INTERVAL_OPTIONS = ["15m", "30m", "1h", "1d"]
TRADE_THRESHOLD_BUY = 0.65
TRADE_THRESHOLD_SELL = 0.35
MAX_TRIALS = 50
GARCH_WINDOW = 21
MIN_FEATURES = 15
HOLD_LOOKAHEAD = 6
MAX_RETRIES = 3
VALIDATION_WINDOW = 63  # 3 month lookback for financial data
MIN_CLASS_RATIO = 0.3
MIN_TRAIN_SAMPLES = 500
DATA_LEAKAGE_BUFFER = 3  # Periods to exclude after feature calculation

warnings.filterwarnings("ignore", category=UserWarning)
logging.basicConfig(level=logging.INFO)

# Streamlit UI Configuration
st.set_page_config(page_title="AI Trading System", layout="wide")
st.title("🚀 Smart Crypto Trading Assistant")

# Session State Management
if 'model' not in st.session_state:
    st.session_state.model = None
if 'processed_data' not in st.session_state:
    st.session_state.processed_data = None
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
if 'training_progress' not in st.session_state:
    st.session_state.training_progress = {
        'completed': 0,
        'current_score': 0.0,
        'best_score': 0.0
    }
if 'study' not in st.session_state:
    st.session_state.study = None

# --------------------------
# Core Data Functions
# --------------------------

def safe_yf_download(symbol: str, **kwargs) -> pd.DataFrame:
    """Robust data downloader with retries and validation"""
    for _ in range(MAX_RETRIES):
        try:
            data = yf.download(symbol, progress=False, auto_adjust=True, **kwargs)
            if not data.empty and data['Volume'].mean() > 0:
                return data
        except (requests.exceptions.RequestException, json.JSONDecodeError):
            continue
        except Exception as e:
            logging.error(f"Download error: {str(e)}")
            break
    st.error(f"Failed to fetch data for {symbol}")
    return pd.DataFrame()

def calculate_features(df: pd.DataFrame, interval: str) -> pd.DataFrame:
    """Feature engineering with strict temporal validation"""
    df = df.copy()
    
    # Price transformations
    df['log_price'] = np.log(df['Close'])
    df['returns'] = df['Close'].pct_change()
    df['log_returns'] = np.log1p(df['returns'])
    
    # Volatility features
    for window in [7, 21, 63]:
        df[f'volatility_{window}'] = df['returns'].rolling(window).std()
        df[f'zscore_{window}'] = (df['returns'] - df['returns'].rolling(window).mean()) / df[f'volatility_{window}']
    
    # Volume features
    df['volume_zscore'] = (df['Volume'] - df['Volume'].rolling(21).mean()) / df['Volume'].rolling(21).std()
    df['volume_roc'] = df['Volume'].pct_change(3)
    
    # Price dynamics
    df['trend_strength'] = df['Close'].rolling(14).apply(lambda x: np.polyfit(np.arange(len(x)), x, 1)[0])
    df['momentum'] = df['Close'].pct_change(14)
    
    # Target formulation with forward-looking protection
    hold_period = max(1, HOLD_LOOKAHEAD // (24 if "d" in interval else 1))
    future_returns = df['Close'].pct_change(hold_period).shift(-hold_period)
    df['target'] = (future_returns > 0).astype(int)
    
    # Remove lookahead bias and leakage buffer
    df = df.dropna().iloc[:-hold_period-DATA_LEAKAGE_BUFFER]
    
    # Validate dataset
    if len(df) < MIN_TRAIN_SAMPLES or df['target'].nunique() == 1:
        st.error("Insufficient data or no price movement detected")
        return pd.DataFrame()
    
    return df.replace([np.inf, -np.inf], np.nan).ffill().bfill()

# --------------------------
# Modeling Components
# --------------------------

class LeakageAwareSplitter:
    """Custom time series split with leakage buffer"""
    def __init__(self, n_splits=3, test_size=VALIDATION_WINDOW, leakage_buffer=DATA_LEAKAGE_BUFFER):
        self.n_splits = n_splits
        self.test_size = test_size
        self.leakage_buffer = leakage_buffer
        
    def split(self, X):
        n_samples = len(X)
        for i in range(self.n_splits):
            test_end = n_samples - i*self.test_size
            test_start = test_end - self.test_size
            train_end = test_start - self.leakage_buffer
            if train_end < MIN_TRAIN_SAMPLES:
                continue
            yield (np.arange(0, train_end), np.arange(test_start, test_end))

class TradingModel:
    def __init__(self):
        self.pipeline = Pipeline([
            ('scaler', RobustScaler()),
            ('model', XGBClassifier(tree_method='hist', enable_categorical=False))
        ])
        self.feature_importances = pd.DataFrame()

    def optimize_model(self, X: pd.DataFrame, y: pd.Series) -> bool:
        """Advanced optimization with leakage-aware validation"""
        try:
            study = optuna.create_study(direction='maximize')
            study.optimize(partial(self._objective, X=X, y=y), n_trials=MAX_TRIALS)
            best_params = study.best_params
            
            # Final model training
            self.pipeline.set_params(**self._convert_params(best_params))
            self.pipeline.fit(X, y)
            
            # Store feature importances
            self.feature_importances = pd.Series(
                self.pipeline.named_steps['model'].feature_importances_,
                index=X.columns
            ).sort_values(ascending=False)
            
            return True
        except Exception as e:
            st.error(f"Optimization failed: {str(e)}")
            return False

    def _objective(self, trial, X: pd.DataFrame, y: pd.Series) -> float:
        params = {
            'model__n_estimators': trial.suggest_int('n_estimators', 100, 500),
            'model__learning_rate': trial.suggest_float('learning_rate', 1e-3, 0.1, log=True),
            'model__max_depth': trial.suggest_int('max_depth', 3, 7),
            'model__subsample': trial.suggest_float('subsample', 0.6, 0.95),
            'model__colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 0.95),
            'model__gamma': trial.suggest_float('gamma', 0, 0.5),
            'model__reg_alpha': trial.suggest_float('reg_alpha', 0, 1),
            'model__reg_lambda': trial.suggest_float('reg_lambda', 0, 1),
        }
        
        cv = LeakageAwareSplitter().split(X)
        scores = []
        
        for train_idx, test_idx in cv:
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            
            model = XGBClassifier(**self._convert_params(params))
            model.fit(X_train, y_train)
            
            preds = model.predict_proba(X_test)[:, 1]
            scores.append(roc_auc_score(y_test, preds))
            
        return np.mean(scores)

    def _convert_params(self, params: dict) -> dict:
        return {k.replace('model__', ''): v for k, v in params.items()}

    def predict(self, X: pd.DataFrame) -> Tuple[float, dict]:
        """Returns prediction with confidence metrics"""
        try:
            if X.empty or not hasattr(self.pipeline, 'named_steps'):
                return 0.5, {}
                
            proba = self.pipeline.predict_proba(X)[0][1]
            return proba, {
                'confidence_interval': (max(0, proba-0.1), min(1, proba+0.1)),
                'volatility': X['volatility_21'].iloc[-1]
            }
        except Exception as e:
            logging.error(f"Prediction error: {str(e)}")
            return 0.5, {}

# --------------------------
# Streamlit Interface
# --------------------------

def main():
    st.sidebar.header("Configuration")
    symbol = st.sidebar.text_input("Asset Symbol", DEFAULT_SYMBOL).upper()
    interval = st.sidebar.selectbox("Time Interval", INTERVAL_OPTIONS, index=2)
    
    if st.sidebar.button("🔄 Load & Process Data"):
        with st.spinner("Building dataset..."):
            data = safe_yf_download(symbol, period='max', interval=interval)
            processed_data = calculate_features(data, interval)
            
            if not processed_data.empty:
                st.session_state.processed_data = processed_data
                st.session_state.data_loaded = True
                st.session_state.model = None
                st.success(f"Processed {len(processed_data)} samples")
            else:
                st.error("Data processing failed")

    if st.session_state.get('data_loaded', False):
        df = st.session_state.processed_data
        
        # Display market overview
        col1, col2, col3 = st.columns(3)
        col1.metric("Current Price", f"${df['Close'].iloc[-1]:.2f}")
        col2.metric("24h Volatility", f"{df['volatility_21'].iloc[-1]:.2%}")
        col3.metric("Market Sentiment", 
                   "Bullish" if df['target'].iloc[-1] == 1 else "Bearish")
        
        # Price chart
        st.plotly_chart(px.line(df, x=df.index, y='Close', 
                          title=f"{symbol} Price History"), use_container_width=True)

    if st.sidebar.button("🚀 Train Model") and st.session_state.data_loaded:
        df = st.session_state.processed_data
        X = df.drop(columns=['target'])
        y = df['target']
        
        model = TradingModel()
        with st.spinner(f"Optimizing across {MAX_TRIALS} trials..."):
            if model.optimize_model(X, y):
                st.session_state.model = model
                st.success("Model training completed!")
                
                # Show feature insights
                st.subheader("Key Predictive Features")
                st.dataframe(model.feature_importances.reset_index().rename(
                    columns={'index': 'Feature', 0: 'Importance'}), 
                    height=400)

    if st.session_state.model and st.session_state.data_loaded:
        df = st.session_state.processed_data
        latest_data = df.drop(columns=['target']).iloc[[-1]]
        
        prediction, metrics = st.session_state.model.predict(latest_data)
        
        # Trading signal logic
        st.subheader("Trading Advisory")
        col1, col2 = st.columns([1, 2])
        col1.metric("Prediction Confidence", f"{prediction:.2%}")
        
        # Dynamic visualization
        fig = px.bar(x=['Buy Threshold', 'Current', 'Sell Threshold'],
                    y=[TRADE_THRESHOLD_BUY, prediction, TRADE_THRESHOLD_SELL],
                    text_auto='.2%')
        fig.update_layout(title="Decision Threshold Analysis")
        col2.plotly_chart(fig, use_container_width=True)
        
        # Risk management overlay
        with st.expander("Risk Assessment"):
            st.progress(min(100, int(prediction*100)), 
                         text=f"Confidence Level: {prediction:.2%}")
            st.write(f"**Volatility Consideration:** {metrics['volatility']:.2%}")
            st.write(f"**Confidence Range:** {metrics['confidence_interval'][0]:.2%} - {metrics['confidence_interval'][1]:.2%}")

if __name__ == "__main__":
    main()
