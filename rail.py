# trading_system.py
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
from ta import add_all_ta_features
import talib
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
if 'study' not in st.session_state:
    st.session_state.study = None
if 'training_in_progress' not in st.session_state:
    st.session_state.training_in_progress = False

class FeatureEngineer(BaseEstimator, TransformerMixin):
    """Advanced feature engineering pipeline component"""
    def fit(self, X, y=None):
        return self
        
    def transform(self, X):
        try:
            df = X.copy()
            
            # Add all technical analysis features
            df = add_all_ta_features(
                df, 
                open="open", high="high", low="low", 
                close="close", volume="volume"
            )
            
            # Add candlestick patterns
            df['CDL2CROWS'] = talib.CDL2CROWS(df['open'], df['high'], df['low'], df['close'])
            df['CDL3STARSINSOUTH'] = talib.CDL3STARSINSOUTH(df['open'], df['high'], df['low'], df['close'])
            
            # Add volatility features
            for window in [7, 14, 21, 50]:
                df[f'volatility_{window}'] = df['close'].pct_change().rolling(window).std()
                df[f'zscore_{window}'] = (df['close'] - df['close'].rolling(window).mean()) / df['close'].rolling(window).std()
            
            # Add market regime features
            df['200d_ma'] = df['close'].rolling(200).mean()
            df['50d_ma'] = df['close'].rolling(50).mean()
            df['regime'] = np.where(df['close'] > df['200d_ma'], 1, 0)
            
            # Add lagged returns
            for lag in [1, 2, 3, 5, 8]:
                df[f'return_lag_{lag}'] = df['close'].pct_change(lag)
                
            # Add volume features
            df['volume_ma'] = df['volume'].rolling(14).mean()
            df['volume_oscillator'] = (df['volume'] - df['volume_ma']) / df['volume_ma']
            
            return df.replace([np.inf, -np.inf], np.nan).ffill().dropna()
            
        except Exception as e:
            st.error(f"Feature engineering failed: {str(e)}")
            return pd.DataFrame()

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
            if not data.empty:
                return data
            st.error(f"No data found for {symbol}")
            return pd.DataFrame()
        except (json.JSONDecodeError, Exception) as e:
            logging.error(f"Download error: {str(e)}")
            continue
    return pd.DataFrame()

@st.cache_data(ttl=300, show_spinner="Fetching market data...")
def fetch_data(symbol: str, interval: str) -> pd.DataFrame:
    """Data acquisition and preprocessing pipeline"""
    try:
        period_map = {
            '15m': '60d', '30m': '60d', 
            '1h': '730d', '1d': 'max'
        }
        
        df = safe_yf_download(
            symbol,
            period=period_map.get(interval, '60d'),
            interval=interval
        )
        
        if df.empty:
            return pd.DataFrame()

        df.columns = [col.lower().replace(' ', '_') for col in df.columns]
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        
        if not all(col in df.columns for col in required_cols):
            st.error("Missing required columns")
            return pd.DataFrame()

        return df[required_cols].ffill().dropna()
    
    except Exception as e:
        st.error(f"Data acquisition failed: {str(e)}")
        return pd.DataFrame()

def calculate_target(df: pd.DataFrame, interval: str) -> pd.Series:
    """Advanced target formulation with volatility adjustment"""
    try:
        # Calculate future returns
        future_returns = df['close'].pct_change(HOLD_LOOKAHEAD).shift(-HOLD_LOOKAHEAD)
        
        # Calculate dynamic threshold using volatility
        volatility = df['close'].pct_change().rolling(21).std()
        threshold = volatility * 1.5  # 1.5x volatility threshold
        
        # Create target variable
        target = (future_returns > threshold).astype(int)
        
        # Validate class balance
        class_ratio = target.value_counts(normalize=True)
        if min(class_ratio) < MIN_CLASS_RATIO:
            st.error(f"Class imbalance: {class_ratio.to_dict()}")
            return pd.Series()
            
        return target
    
    except Exception as e:
        st.error(f"Target calculation failed: {str(e)}")
        return pd.Series()

class TradingModel:
    """Enhanced trading model with feature selection and threshold optimization"""
    def __init__(self):
        self.pipeline = Pipeline([
            ('scaler', RobustScaler()),
            ('model', XGBClassifier(
                n_estimators=1000,
                early_stopping_rounds=50,
                eval_metric='auc',
                tree_method='hist',
                random_state=42
            ))
        ])
        self.best_threshold = 0.5
        self.feature_importances = pd.DataFrame()

    def optimize(self, X: pd.DataFrame, y: pd.Series, symbol: str, interval: str):
        """Optimization pipeline with persistent study"""
        study_name = f"{symbol}_{interval}"
        storage_name = f"sqlite:///optuna_studies/{study_name}.db"
        
        # Create storage directory
        os.makedirs(os.path.dirname(storage_name), exist_ok=True)
        
        study = optuna.create_study(
            direction='maximize',
            storage=storage_name,
            study_name=study_name,
            load_if_exists=True
        )
        
        remaining_trials = MAX_TRIALS - len(study.trials)
        if remaining_trials > 0:
            study.optimize(
                lambda trial: self.objective(trial, X, y),
                n_trials=remaining_trials,
                callbacks=[self._update_progress],
                show_progress_bar=False
            )
        
        self._train_final_model(X, y, study.best_params)
        self._optimize_threshold(X, y)

    def objective(self, trial, X: pd.DataFrame, y: pd.Series) -> float:
        """Optimization objective with feature selection"""
        params = {
            'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.3, log=True),
            'max_depth': trial.suggest_int('max_depth', 3, 9),
            'subsample': trial.suggest_float('subsample', 0.6, 0.95),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 0.95),
            'gamma': trial.suggest_float('gamma', 0, 1.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 0, 1.0),
            'reg_lambda': trial.suggest_float('reg_lambda', 0, 1.0),
        }
        
        tscv = TimeSeriesSplit(n_splits=3)
        scores = []
        
        for train_idx, val_idx in tscv.split(X):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            
            model = XGBClassifier(**params)
            model.fit(X_train, y_train)
            
            y_proba = model.predict_proba(X_val)[:, 1]
            score = roc_auc_score(y_val, y_proba)
            scores.append(score)
            
        return np.mean(scores)

    def _train_final_model(self, X: pd.DataFrame, y: pd.Series, params: dict):
        """Train final model with best parameters"""
        self.pipeline.named_steps['model'].set_params(**params)
        self.pipeline.fit(X, y)
        
        # Store feature importances
        self.feature_importances = pd.Series(
            self.pipeline.named_steps['model'].feature_importances_,
            index=X.columns
        ).sort_values(ascending=False).head(MIN_FEATURES)

    def _optimize_threshold(self, X: pd.DataFrame, y: pd.Series):
        """Optimize classification threshold using F1 score"""
        tscv = TimeSeriesSplit(n_splits=3)
        thresholds = np.linspace(0.3, 0.7, 50)
        best_score = 0
        
        for train_idx, val_idx in tscv.split(X):
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
        """Update training progress in session state"""
        st.session_state.training_progress = {
            'completed': len(study.trials),
            'current_score': trial.value,
            'best_score': study.best_value
        }
        st.rerun()

def main():
    """Main application interface"""
    st.sidebar.header("Configuration")
    symbol = st.sidebar.text_input("Asset Symbol", DEFAULT_SYMBOL).upper().strip()
    interval = st.sidebar.selectbox("Time Interval", INTERVAL_OPTIONS, index=2)
    
    if st.sidebar.button("🔄 Load Market Data"):
        with st.spinner("Processing market data..."):
            raw_data = fetch_data(symbol, interval)
            if not raw_data.empty:
                fe = FeatureEngineer()
                processed_data = fe.fit_transform(raw_data)
                target = calculate_target(processed_data, interval)
                
                if not target.empty:
                    processed_data = processed_data.iloc[:len(target)]
                    processed_data['target'] = target
                    st.session_state.processed_data = processed_data.dropna()
                    st.session_state.data_loaded = True
                    st.success("Data loaded successfully!")
                else:
                    st.error("Failed to calculate target variable")
            else:
                st.error("Failed to fetch market data")

    if st.session_state.get('data_loaded', False):
        col1, col2 = st.columns([3, 1])
        with col1:
            st.subheader(f"{symbol} Price Chart")
            fig = px.line(st.session_state.processed_data, y='close')
            st.plotly_chart(fig, use_container_width=True)
            
        with col2:
            st.metric("Current Price", f"${st.session_state.processed_data['close'].iloc[-1]:.2f}")
            st.metric("Market Volatility", 
                     f"{st.session_state.processed_data['volatility_21'].iloc[-1]:.2%}")

    if st.sidebar.button("🚀 Train Trading Model") and st.session_state.data_loaded:
        if 'processed_data' not in st.session_state:
            st.error("No data available for training")
            return
            
        X = st.session_state.processed_data.drop(columns=['target'])
        y = st.session_state.processed_data['target']
        
        model = TradingModel()
        st.session_state.training_in_progress = True
        
        with st.spinner("Optimizing trading strategy..."):
            model.optimize(X, y, symbol, interval)
            st.session_state.model = model
            st.session_state.training_in_progress = False
            st.success("Model training completed!")
            
            if not model.feature_importances.empty:
                st.subheader("Feature Importances")
                st.dataframe(
                    model.feature_importances.reset_index().rename(
                        columns={'index': 'Feature', 0: 'Importance'}
                    ).style.format({'Importance': '{:.2%}'}),
                    height=400
                )

    if st.session_state.get('model') and not st.session_state.training_in_progress:
        try:
            latest_data = st.session_state.processed_data.drop(columns=['target']).iloc[[-1]]
            confidence = st.session_state.model.pipeline.predict_proba(latest_data)[0][1]
            
            st.subheader("Trading Signal")
            col1, col2 = st.columns(2)
            col1.metric("Model Confidence", f"{confidence:.2%}")
            
            # Dynamic threshold adjustment
            adj_buy = st.session_state.model.best_threshold + 0.1
            adj_sell = st.session_state.model.best_threshold - 0.1
            
            if confidence > adj_buy:
                col2.success("🚀 Strong Buy Signal")
            elif confidence < adj_sell:
                col2.error("🔻 Strong Sell Signal")
            else:
                col2.info("🛑 Neutral Position")
                
            st.caption(f"Optimized thresholds: Buy >{adj_buy:.0%}, Sell <{adj_sell:.0%}")

        except Exception as e:
            st.error(f"Prediction error: {str(e)}")

if __name__ == "__main__":
    main()
