# rail.py
import logging
import numpy as np
import pandas as pd
import yfinance as yf
import streamlit as st
import plotly.express as px
import optuna
import sqlite3
import os
from xgboost import XGBClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import roc_auc_score, f1_score
from sklearn.preprocessing import RobustScaler
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
import warnings
import json

# Configuration
DEFAULT_SYMBOL = 'BTC-USD'
INTERVAL_OPTIONS = ["15m", "30m", "1h", "1d"]
MAX_TRIALS = 200
GARCH_WINDOW = 21
MIN_FEATURES = 25
HOLD_LOOKAHEAD = 3
MAX_RETRIES = 3
VALIDATION_WINDOW = 63
MIN_CLASS_RATIO = 0.25
MIN_TRAIN_SAMPLES = 1000

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

class FeatureEngineer(BaseEstimator, TransformerMixin):
    """Manual feature engineering without external libraries"""
    def fit(self, X, y=None):
        return self
        
    def transform(self, X):
        try:
            df = X.copy()
            
            # Price-based features
            df['returns'] = df['close'].pct_change()
            df['log_returns'] = np.log1p(df['returns'])
            
            # Moving Averages
            for window in [10, 20, 50, 200]:
                df[f'ma_{window}'] = df['close'].rolling(window).mean()
            
            # RSI Calculation
            delta = df['close'].diff()
            gain = delta.where(delta > 0, 0)
            loss = -delta.where(delta < 0, 0)
            avg_gain = gain.rolling(14).mean()
            avg_loss = loss.rolling(14).mean()
            rs = avg_gain / avg_loss
            df['rsi'] = 100 - (100 / (1 + rs))
            
            # MACD Calculation
            exp12 = df['close'].ewm(span=12, adjust=False).mean()
            exp26 = df['close'].ewm(span=26, adjust=False).mean()
            df['macd'] = exp12 - exp26
            df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
            
            # Volatility Features
            for window in [7, 14, 21]:
                df[f'volatility_{window}'] = df['returns'].rolling(window).std()
                df[f'zscore_{window}'] = (df['close'] - df[f'ma_{window}']) / df[f'volatility_{window}']
            
            # Volume Features
            df['volume_ma'] = df['volume'].rolling(14).mean()
            df['volume_oscillator'] = (df['volume'] - df['volume_ma']) / df['volume_ma']
            
            # Lagged Features
            for lag in [1, 2, 3, 5, 8]:
                df[f'return_lag_{lag}'] = df['returns'].shift(lag)
            
            # Cleanup
            df = df.replace([np.inf, -np.inf], np.nan)
            return df.ffill().dropna()
            
        except Exception as e:
            st.error(f"Feature engineering failed: {str(e)}")
            return pd.DataFrame()

def safe_yf_download(symbol: str, **kwargs) -> pd.DataFrame:
    """Robust data downloader with error handling"""
    for _ in range(MAX_RETRIES):
        try:
            data = yf.download(
                symbol,
                group_by='ticker',
                progress=False,
                auto_adjust=True,
                **kwargs
            )
            return data if not data.empty else pd.DataFrame()
        except Exception as e:
            logging.error(f"Download error: {str(e)}")
    return pd.DataFrame()

@st.cache_data(ttl=300)
def fetch_data(symbol: str, interval: str) -> pd.DataFrame:
    """Data acquisition pipeline"""
    period_map = {'15m': '60d', '30m': '60d', '1h': '730d', '1d': 'max'}
    df = safe_yf_download(symbol, period=period_map.get(interval, '60d'), interval=interval)
    return df[['Open', 'High', 'Low', 'Close', 'Volume']].ffill().dropna().rename(
        columns=lambda x: x.lower()
    )

def calculate_target(df: pd.DataFrame) -> pd.Series:
    """Create target variable with volatility adjustment"""
    future_returns = df['close'].pct_change(HOLD_LOOKAHEAD).shift(-HOLD_LOOKAHEAD)
    volatility = df['volatility_21'].rolling(10).mean()
    threshold = volatility * 1.5
    target = (future_returns > threshold).astype(int)
    return target.dropna()

class TradingModel:
    """Trading model with manual feature engineering"""
    def __init__(self):
        self.pipeline = Pipeline([
            ('scaler', RobustScaler()),
            ('model', XGBClassifier(
                n_estimators=1000,
                early_stopping_rounds=50,
                eval_metric='auc',
                random_state=42
            ))
        ])
        self.best_threshold = 0.5
        self.feature_importances = pd.DataFrame()

    def optimize(self, X: pd.DataFrame, y: pd.Series, symbol: str, interval: str):
        study = optuna.create_study(direction='maximize')
        study.optimize(
            lambda trial: self.objective(trial, X, y),
            n_trials=MAX_TRIALS,
            callbacks=[self._update_progress]
        )
        self._train_final_model(X, y, study.best_params)
        self._optimize_threshold(X, y)

    def objective(self, trial, X: pd.DataFrame, y: pd.Series) -> float:
        params = {
            'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.3, log=True),
            'max_depth': trial.suggest_int('max_depth', 3, 9),
            'subsample': trial.suggest_float('subsample', 0.6, 0.95),
            'gamma': trial.suggest_float('gamma', 0, 1.0)
        }
        scores = []
        for train_idx, val_idx in TimeSeriesSplit(3).split(X):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            self.pipeline.set_params(**params)
            self.pipeline.fit(X_train, y_train)
            scores.append(roc_auc_score(y_val, self.pipeline.predict_proba(X_val)[:, 1]))
        return np.mean(scores)

    def _train_final_model(self, X: pd.DataFrame, y: pd.Series, params: dict):
        self.pipeline.set_params(**params)
        self.pipeline.fit(X, y)
        self.feature_importances = pd.Series(
            self.pipeline.named_steps['model'].feature_importances_,
            index=X.columns
        ).sort_values(ascending=False).head(MIN_FEATURES)

    def _optimize_threshold(self, X: pd.DataFrame, y: pd.Series):
        thresholds = np.linspace(0.3, 0.7, 50)
        best_score = 0
        for train_idx, val_idx in TimeSeriesSplit(3).split(X):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            self.pipeline.fit(X_train, y_train)
            y_proba = self.pipeline.predict_proba(X_val)[:, 1]
            for thresh in thresholds:
                score = f1_score(y_val, y_proba >= thresh)
                if score > best_score:
                    best_score = score
                    self.best_threshold = thresh

    def _update_progress(self, study, trial):
        st.session_state.training_progress = {
            'completed': len(study.trials),
            'current_score': trial.value,
            'best_score': study.best_value
        }

def main():
    st.sidebar.header("Configuration")
    symbol = st.sidebar.text_input("Asset Symbol", DEFAULT_SYMBOL).upper()
    interval = st.sidebar.selectbox("Interval", INTERVAL_OPTIONS, index=2)
    
    if st.sidebar.button("🔄 Load Data"):
        with st.spinner("Processing data..."):
            raw_data = fetch_data(symbol, interval)
            fe = FeatureEngineer()
            processed_data = fe.fit_transform(raw_data)
            processed_data['target'] = calculate_target(processed_data)
            st.session_state.processed_data = processed_data.dropna()
            st.session_state.data_loaded = True

    if st.session_state.get('data_loaded'):
        st.subheader(f"{symbol} Price Chart")
        fig = px.line(st.session_state.processed_data, y='close')
        st.plotly_chart(fig, use_container_width=True)

    if st.sidebar.button("🚀 Train Model") and st.session_state.data_loaded:
        X = st.session_state.processed_data.drop(columns=['target'])
        y = st.session_state.processed_data['target']
        model = TradingModel()
        model.optimize(X, y, symbol, interval)
        st.session_state.model = model
        st.success("Training completed!")
        
        st.subheader("Feature Importances")
        st.dataframe(model.feature_importances.reset_index().rename(
            columns={'index': 'Feature', 0: 'Importance'}
        ))

    if st.session_state.get('model'):
        latest_data = st.session_state.processed_data.drop(columns=['target']).iloc[[-1]]
        confidence = st.session_state.model.pipeline.predict_proba(latest_data)[0][1]
        st.metric("Prediction Confidence", f"{confidence:.2%}")
        st.caption(f"Optimal Threshold: {st.session_state.model.best_threshold:.2%}")

if __name__ == "__main__":
    main()
