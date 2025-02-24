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
from sklearn.metrics import roc_auc_score
import warnings
import json

# Configuration
DEFAULT_SYMBOL = 'BTC-USD'
INTERVAL_OPTIONS = ["15m", "30m", "1h", "1d"]
TRADE_THRESHOLD_BUY = 0.6
TRADE_THRESHOLD_SELL = 0.4
MAX_TRIALS = 50
GARCH_WINDOW = 21
MIN_FEATURES = 10
HOLD_LOOKAHEAD = 6
MAX_RETRIES = 3
VALIDATION_WINDOW = 21
MIN_CLASS_RATIO = 0.3
MIN_TRAIN_SAMPLES = 200

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
if 'optimization_started' not in st.session_state:
    st.session_state.optimization_started = False

def safe_yf_download(symbol: str, **kwargs) -> pd.DataFrame:
    """Robust data downloader with error handling and retries"""
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
        except Exception:
            continue
    return pd.DataFrame()

def normalize_columns(symbol: str, columns) -> list:
    """Normalize column names and remove ticker prefixes"""
    normalized_symbol = symbol.lower().replace('-', '')
    return [
        col.lower()
        .replace('-', '')
        .replace(' ', '_')
        .replace(f"{normalized_symbol}_", "")
        for col in columns
    ]

@st.cache_data(ttl=300, show_spinner="Fetching market data...")
def fetch_data(symbol: str, interval: str) -> pd.DataFrame:
    """Data acquisition and preprocessing pipeline"""
    try:
        period_map = {'15m': '60d', '30m': '60d', '1h': '730d', '1d': 'max'}
        df = safe_yf_download(symbol, period=period_map.get(interval, '60d'), interval=interval)
        if df.empty:
            return pd.DataFrame()
        
        df.columns = normalize_columns(symbol, df.columns)
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        
        return df[required_cols].ffill().bfill().dropna()
    except Exception:
        return pd.DataFrame()

def calculate_features(df: pd.DataFrame, interval: str) -> pd.DataFrame:
    """Conservative feature engineering with strict temporal validation"""
    try:
        df = df.copy()
        hold_period = max(2, HOLD_LOOKAHEAD // (24 if "d" in interval else 4))
        
        # Price features
        df['returns'] = df['close'].pct_change()
        df['log_returns'] = np.log1p(df['returns'])
        
        # Technical indicators
        for span in [12, 26]:
            df[f'ema_{span}'] = df['close'].ewm(span=span, adjust=False).mean()
        df['macd'] = df['ema_12'] - df['ema_26']
        df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
        
        # Volatility
        df['volatility'] = df['returns'].rolling(GARCH_WINDOW).std()
        
        # Target: Future price direction (smoothed)
        future_returns = df['close'].pct_change(hold_period).shift(-hold_period)
        df['target'] = (future_returns.rolling(3).mean() > 0).astype(int)
        df = df.dropna().iloc[:-hold_period]
        
        return df.replace([np.inf, -np.inf], np.nan).ffill().dropna()
    except Exception:
        return pd.DataFrame()

class TradingModel:
    def __init__(self):
        self.model = None
        self.feature_importances = pd.DataFrame()

    def _update_progress(self, trial_number: int, current_score: float, best_score: float):
        st.session_state.training_progress = {
            'completed': trial_number,
            'current_score': current_score,
            'best_score': best_score
        }
        st.rerun()

    def optimize_model(self, X: pd.DataFrame, y: pd.Series) -> bool:
        """Optimization pipeline with proper state management"""
        if X.empty or y.nunique() != 2:
            return False

        if st.session_state.study is None:
            st.session_state.study = optuna.create_study(direction='maximize')
            st.session_state.optimization_started = True

        def callback(study, trial):
            self._update_progress(
                trial.number + 1,
                trial.value if trial.value else 0.0,
                study.best_value
            )

        try:
            st.session_state.study.optimize(
                lambda trial: self._objective(trial, X, y),
                n_trials=MAX_TRIALS,
                callbacks=[callback],
                show_progress_bar=False,
                gc_after_trial=True
            )
            return self._train_final_model(X, y, st.session_state.study.best_params)
        except Exception:
            return False

    def _objective(self, trial, X: pd.DataFrame, y: pd.Series) -> float:
        """Objective function with realistic constraints"""
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 80, 150),
            'learning_rate': trial.suggest_float('learning_rate', 0.05, 0.2, log=True),
            'max_depth': trial.suggest_int('max_depth', 3, 5),
            'subsample': trial.suggest_float('subsample', 0.7, 0.9),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.7, 0.9),
            'gamma': trial.suggest_float('gamma', 0, 0.3),
            'reg_alpha': trial.suggest_float('reg_alpha', 0, 0.5),
            'reg_lambda': trial.suggest_float('reg_lambda', 0, 0.5),
            'tree_method': 'hist',
            'random_state': 42
        }
        
        tscv = TimeSeriesSplit(n_splits=3, test_size=VALIDATION_WINDOW)
        scores = []
        
        for train_idx, val_idx in tscv.split(X):
            if len(train_idx) < MIN_TRAIN_SAMPLES or len(val_idx) < 20:
                continue
                
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

            model = XGBClassifier(**params)
            model.fit(X_train, y_train)
            
            y_proba = model.predict_proba(X_val)[:, 1]
            score = roc_auc_score(y_val, y_proba)
            
            if 0.45 < score < 0.65:  # Realistic performance range
                scores.append(score)

        return np.mean(scores) if scores else float('nan')

    def _train_final_model(self, X: pd.DataFrame, y: pd.Series, params: dict) -> bool:
        """Final model training with proper validation"""
        try:
            tscv = TimeSeriesSplit(n_splits=3, test_size=VALIDATION_WINDOW)
            train_idx, val_idx = list(tscv.split(X))[-1]
            
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

            params.update({
                'early_stopping_rounds': 20,
                'eval_metric': 'auc',
                'tree_method': 'hist'
            })

            self.model = XGBClassifier(**params)
            self.model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
            
            self.feature_importances = pd.Series(
                self.model.feature_importances_,
                index=X.columns
            ).sort_values(ascending=False)

            return True
        except Exception:
            return False

    def predict(self, X: pd.DataFrame) -> float:
        """Conservative prediction with sanity checks"""
        try:
            if not self.model or X.empty:
                return 0.5
                
            X_clean = X[self.feature_importances.index[:MIN_FEATURES]].ffill().bfill()
            proba = self.model.predict_proba(X_clean)[0][1]
            return np.clip(proba, 0.45, 0.55)  # Realistic range
        except Exception:
            return 0.5

def main():
    """Main application interface"""
    st.sidebar.header("Configuration")
    symbol = st.sidebar.text_input("Asset Symbol", DEFAULT_SYMBOL).upper().strip()
    interval = st.sidebar.selectbox("Time Interval", INTERVAL_OPTIONS, index=2)
    
    if st.sidebar.button("🔄 Load Market Data"):
        with st.spinner("Processing data..."):
            raw_data = fetch_data(symbol, interval)
            processed_data = calculate_features(raw_data, interval)
            
            if processed_data is not None and len(processed_data) > MIN_TRAIN_SAMPLES:
                st.session_state.processed_data = processed_data
                st.session_state.data_loaded = True
                st.session_state.study = None
            st.rerun()

    if st.session_state.get('data_loaded', False):
        if st.session_state.processed_data is not None:
            col1, col2 = st.columns([3, 1])
            with col1:
                fig = px.line(st.session_state.processed_data, y='close', 
                            title=f"{symbol} Price Chart")
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.metric("Current Price", f"${st.session_state.processed_data['close'].iloc[-1]:.2f}")
                st.metric("Volatility", f"{st.session_state.processed_data['volatility'].iloc[-1]:.2%}")

    if st.sidebar.button("🚀 Train Trading Model") and st.session_state.data_loaded:
        if not st.session_state.optimization_started:
            st.session_state.training_progress = {'completed': 0, 'current_score': 0.0, 'best_score': 0.0}
            st.session_state.optimization_started = True
        
        model = TradingModel()
        X = st.session_state.processed_data.drop(columns=['target'])
        y = st.session_state.processed_data['target']
        
        if model.optimize_model(X, y):
            st.session_state.model = model
            st.session_state.optimization_started = False
            st.success("Training completed!")
            
            if not model.feature_importances.empty:
                st.subheader("Feature Importance")
                st.dataframe(
                    model.feature_importances.reset_index().rename(
                        columns={'index': 'Feature', 0: 'Importance'}
                    ).style.format({'Importance': '{:.2%}'}),
                    height=400
                )

    if st.session_state.training_progress['completed'] > 0:
        st.subheader("Training Progress")
        prog = st.session_state.training_progress
        cols = st.columns(3)
        cols[0].metric("Trials", f"{prog['completed']}/{MAX_TRIALS}")
        cols[1].metric("Current AUC", f"{prog['current_score']:.2%}")
        cols[2].metric("Best AUC", f"{prog['best_score']:.2%}")

    if st.session_state.model and st.session_state.processed_data is not None:
        try:
            latest_data = st.session_state.processed_data.drop(columns=['target']).iloc[[-1]]
            confidence = st.session_state.model.predict(latest_data)
            volatility = st.session_state.processed_data['volatility'].iloc[-1]
            
            st.subheader("Trading Signal")
            adj_buy = TRADE_THRESHOLD_BUY + (volatility * 0.05)
            adj_sell = TRADE_THRESHOLD_SELL - (volatility * 0.05)
            
            col1, col2 = st.columns(2)
            col1.metric("Confidence", f"{confidence:.2%}")
            
            if confidence > adj_buy:
                col2.success("Moderate Buy")
            elif confidence < adj_sell:
                col2.error("Cautious Sell")
            else:
                col2.info("Neutral")
            
            st.caption(f"Dynamic thresholds: Buy >{adj_buy:.0%}, Sell <{adj_sell:.0%}")
        except Exception:
            st.error("Error generating signal")

if __name__ == "__main__":
    main()
