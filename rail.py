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
from functools import partial
from optuna.storages import InMemoryStorage

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
VALIDATION_WINDOW = 63  # 3 months of daily data
MIN_CLASS_RATIO = 0.3
MIN_TRAIN_SAMPLES = 500

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
        'best_score': 0.0,
        'active': False
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
    """Advanced feature engineering with leakage prevention"""
    df = df.copy()
    
    # Price transformations
    df['log_price'] = np.log(df['Close'])
    df['returns'] = df['Close'].pct_change()
    df['log_returns'] = np.log1p(df['returns'])
    
    # Volatility features
    for window in [7, 21, 63]:
        df[f'volatility_{window}'] = df['returns'].rolling(window).std()
    
    # Volume features
    df['volume_zscore'] = (df['Volume'] - df['Volume'].rolling(21).mean()) / df['Volume'].rolling(21).std()
    
    # Price dynamics
    df['trend_strength'] = df['Close'].rolling(14).apply(lambda x: np.polyfit(np.arange(len(x)), x, 1)[0])
    
    # Target formulation with lookahead protection
    hold_period = max(1, HOLD_LOOKAHEAD // (24 if "d" in interval else 1))
    future_returns = df['Close'].pct_change(hold_period).shift(-hold_period)
    df['target'] = (future_returns > 0).astype(int)
    
    # Remove lookahead bias and leakage buffer
    df = df.dropna().iloc[:-hold_period-3]
    
    if len(df) < MIN_TRAIN_SAMPLES or df['target'].nunique() == 1:
        st.error("Insufficient data or no price movement detected")
        return pd.DataFrame()
    
    return df.replace([np.inf, -np.inf], np.nan).ffill().bfill()

# --------------------------
# Modeling Components
# --------------------------

class TradingModel:
    def __init__(self):
        self.model = None
        self.feature_importances = pd.DataFrame()
        self.storage = InMemoryStorage()

    def _update_progress(self, trial_number: int, current_score: float, best_score: float):
        st.session_state.training_progress = {
            'completed': trial_number,
            'current_score': current_score,
            'best_score': best_score,
            'active': True
        }
        st.experimental_rerun()

    def optimize_model(self, X: pd.DataFrame, y: pd.Series) -> bool:
        """Persistent optimization with proper state management"""
        try:
            if not st.session_state.training_progress['active']:
                self._init_study()

            study = optuna.load_study(
                study_name="trading_study",
                storage=self.storage
            )

            def optimization_callback(study, trial):
                self._update_progress(
                    trial.number + 1,
                    trial.value if trial.value else 0.0,
                    study.best_value
                )

            remaining_trials = MAX_TRIALS - len(study.trials)
            if remaining_trials > 0:
                study.optimize(
                    partial(self._objective, X=X, y=y),
                    n_trials=remaining_trials,
                    callbacks=[optimization_callback],
                    show_progress_bar=False,
                    catch=(Exception,)
            
            if len(study.trials) >= MAX_TRIALS:
                self._train_final_model(X, y, study.best_params)
                st.session_state.training_progress['active'] = False
                st.experimental_rerun()
            
            return True
            
        except Exception as e:
            st.error(f"Optimization error: {str(e)}")
            return False

    def _init_study(self):
        """Initialize new study with proper cleanup"""
        try:
            optuna.delete_study(study_name="trading_study", storage=self.storage)
        except KeyError:
            pass
        optuna.create_study(
            direction='maximize',
            study_name="trading_study",
            storage=self.storage
        )
        st.session_state.training_progress = {
            'completed': 0,
            'current_score': 0.0,
            'best_score': 0.0,
            'active': True
        }

    def _train_final_model(self, X: pd.DataFrame, y: pd.Series, params: dict) -> bool:
        """Final model training with full dataset"""
        try:
            self.model = XGBClassifier(
                **params,
                tree_method='hist',
                random_state=42,
                enable_categorical=False
            )
            self.model.fit(X, y)
            
            self.feature_importances = pd.Series(
                self.model.feature_importances_,
                index=X.columns
            ).sort_values(ascending=False)
            
            return True
        except Exception as e:
            st.error(f"Final training failed: {str(e)}")
            return False

    def _objective(self, trial, X: pd.DataFrame, y: pd.Series) -> float:
        """Optimization objective with realistic parameters"""
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
            'learning_rate': trial.suggest_float('learning_rate', 1e-4, 0.3, log=True),
            'max_depth': trial.suggest_int('max_depth', 3, 9),
            'subsample': trial.suggest_float('subsample', 0.5, 0.95),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 0.95),
            'gamma': trial.suggest_float('gamma', 0, 1.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 0, 2.0),
            'reg_lambda': trial.suggest_float('reg_lambda', 0, 2.0)
        }
        
        try:
            tscv = TimeSeriesSplit(n_splits=3, test_size=VALIDATION_WINDOW)
            scores = []
            
            for train_idx, test_idx in tscv.split(X):
                X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
                y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
                
                model = XGBClassifier(**params)
                model.fit(X_train, y_train)
                
                y_proba = model.predict_proba(X_test)[:, 1]
                scores.append(roc_auc_score(y_test, y_proba))
            
            return np.mean(scores)
        except Exception as e:
            return 0.5

    def predict(self, X: pd.DataFrame) -> float:
        """Unrestricted prediction with validation"""
        try:
            if self.model is None or X.empty:
                return 0.5
                
            X_clean = X[self.feature_importances.index[:MIN_FEATURES]].ffill().bfill()
            return self.model.predict_proba(X_clean)[0][1]
        except Exception as e:
            logging.error(f"Prediction error: {str(e)}")
            return 0.5

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
        
        # Real-time progress display
        if st.session_state.training_progress['active']:
            progress = st.progress(
                st.session_state.training_progress['completed'] / MAX_TRIALS,
                text=f"Completed {st.session_state.training_progress['completed']}/{MAX_TRIALS} trials"
            )
            cols = st.columns(2)
            cols[0].metric("Current Score", 
                          f"{st.session_state.training_progress['current_score']:.2%}")
            cols[1].metric("Best Score", 
                          f"{st.session_state.training_progress['best_score']:.2%}")
        
        # Market overview
        col1, col2 = st.columns([3, 1])
        with col1:
            st.plotly_chart(px.line(df, x=df.index, y='Close', 
                              title=f"{symbol} Price History"), 
                              use_container_width=True)
        with col2:
            st.metric("Current Price", f"${df['Close'].iloc[-1]:.2f}")
            st.metric("Market Volatility", f"{df['volatility_21'].iloc[-1]:.2%}")

    if st.sidebar.button("🚀 Start Training") and st.session_state.data_loaded:
        df = st.session_state.processed_data
        model = TradingModel()
        if not st.session_state.training_progress['active']:
            st.session_state.model = None
        st.session_state.model = model
        model.optimize_model(df.drop(columns=['target']), df['target'])

    if st.session_state.model and st.session_state.data_loaded:
        df = st.session_state.processed_data
        latest_data = df.drop(columns=['target']).iloc[[-1]]
        
        confidence = st.session_state.model.predict(latest_data)
        
        # Trading signal display
        st.subheader("Live Trading Signal")
        col1, col2 = st.columns([1, 2])
        col1.metric("Prediction Confidence", f"{confidence:.2%}")
        
        # Visual threshold analysis
        fig = px.bar(x=['Buy Threshold', 'Current', 'Sell Threshold'],
                    y=[TRADE_THRESHOLD_BUY, confidence, TRADE_THRESHOLD_SELL],
                    text_auto='.2%')
        fig.update_layout(title="Decision Threshold Analysis")
        col2.plotly_chart(fig, use_container_width=True)
        
        # Feature importance
        if not st.session_state.model.feature_importances.empty:
            with st.expander("Feature Importance Analysis"):
                st.dataframe(
                    st.session_state.model.feature_importances.reset_index().rename(
                        columns={'index': 'Feature', 0: 'Importance'}
                    ).style.format({'Importance': '{:.2%}'}),
                    height=400
                )

if __name__ == "__main__":
    main()
