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
TRADE_THRESHOLD_BUY = 0.65
TRADE_THRESHOLD_SELL = 0.35
MAX_TRIALS = 50
GARCH_WINDOW = 21
MIN_FEATURES = 10
HOLD_LOOKAHEAD = 6
VALIDATION_WINDOW = 21
MIN_CLASS_RATIO = 0.35

warnings.filterwarnings("ignore", category=UserWarning)
logging.basicConfig(level=logging.INFO)

# Streamlit UI Configuration
st.set_page_config(page_title="AI Trading System", layout="wide")
st.title("🚀 Smart Crypto Trading Assistant")

# Session State Management
session_defaults = {
    'model': None,
    'processed_data': None,
    'data_loaded': False,
    'training_progress': {'completed': 0, 'current_score': 0.0, 'best_score': 0.0}
}
for key, value in session_defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value

def safe_yf_download(symbol: str, **kwargs) -> pd.DataFrame:
    """Robust data downloader with error handling and retries"""
    for _ in range(3):
        try:
            data = yf.download(symbol, progress=False, auto_adjust=True, **kwargs)
            if not data.empty: 
                return data
        except Exception as e:
            logging.error(f"Download error: {str(e)}")
    st.error(f"Failed to fetch data for {symbol}")
    return pd.DataFrame()

def normalize_columns(symbol: str, columns) -> list:
    """Normalize column names for consistency"""
    normalized_symbol = symbol.lower().replace('-', '')
    column_map = {
        'adjclose': 'close',
        'adjusted_close': 'close',
        'vol': 'volume',
        'vwap': 'close'
    }
    return [
        column_map.get(col, col)
        .lower()
        .replace('-', '')
        .replace(' ', '_')
        .replace(f"{normalized_symbol}_", "") 
        for col in columns
    ]

@st.cache_data(ttl=300, show_spinner="Fetching market data...")
def fetch_data(symbol: str, interval: str) -> pd.DataFrame:
    """Data acquisition and preprocessing pipeline"""
    try:
        period_map = {
            '15m': '60d', '30m': '60d', 
            '1h': '180d', '1d': '730d'
        }
        df = safe_yf_download(
            symbol,
            period=period_map.get(interval, '60d'),
            interval=interval
        )
        if df.empty:
            return pd.DataFrame()

        df.columns = normalize_columns(symbol, df.columns)
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        
        return df[required_cols].ffill().bfill().dropna()
    
    except Exception as e:
        st.error(f"Data processing failed: {str(e)}")
        return pd.DataFrame()

def calculate_features(df: pd.DataFrame) -> pd.DataFrame:
    """Comprehensive feature engineering with validation"""
    try:
        df = df.copy()
        
        # Price dynamics
        df['returns'] = df['close'].pct_change()
        df['log_returns'] = np.log1p(df['returns'])
        
        # Technical indicators
        for span in [12, 26]:
            df[f'ema_{span}'] = df['close'].ewm(span=span).mean()
        df['macd'] = df['ema_12'] - df['ema_26']
        df['macd_signal'] = df['macd'].ewm(span=9).mean()
        
        # Volatility metrics
        df['volatility'] = df['returns'].rolling(GARCH_WINDOW).std()
        
        # Volume analysis
        df['volume_ma'] = df['volume'].rolling(14).mean()
        df['volume_change'] = df['volume'].pct_change()
        
        # Target engineering
        future_returns = df['close'].pct_change(HOLD_LOOKAHEAD).shift(-HOLD_LOOKAHEAD)
        df['target'] = (future_returns > 0).astype(int)
        df = df.dropna()
        
        # Class balance validation
        class_ratio = df['target'].value_counts(normalize=True)
        if abs(class_ratio[0] - class_ratio[1]) > (1 - 2*MIN_CLASS_RATIO):
            st.error(f"Class imbalance detected: {class_ratio.to_dict()}")
            return pd.DataFrame()
            
        return df.replace([np.inf, -np.inf], np.nan).ffill().bfill()
    
    except Exception as e:
        st.error(f"Feature engineering failed: {str(e)}")
        return pd.DataFrame()

class TradingModel:
    def __init__(self):
        self.model = None
        self.feature_importances = pd.DataFrame()
        self.study = None

    def _update_progress(self, trial_number: int, current_score: float, best_score: float):
        st.session_state.training_progress = {
            'completed': trial_number,
            'current_score': current_score,
            'best_score': best_score
        }
        if trial_number % 5 == 0:
            st.rerun()

    def optimize_model(self, X: pd.DataFrame, y: pd.Series) -> bool:
        """Optimization pipeline with proper trial handling"""
        try:
            if X.empty or y.nunique() != 2:
                st.error("Invalid training data")
                return False

            def objective(trial):
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                    'max_depth': trial.suggest_int('max_depth', 3, 7),
                    'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                    'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                    'gamma': trial.suggest_float('gamma', 0, 0.5),
                    'reg_alpha': trial.suggest_float('reg_alpha', 0, 1),
                    'reg_lambda': trial.suggest_float('reg_lambda', 0, 1),
                    'tree_method': 'hist'
                }
                
                scores = []
                tscv = TimeSeriesSplit(n_splits=3, test_size=VALIDATION_WINDOW)
                
                for train_idx, test_idx in tscv.split(X):
                    X_train, X_val = X.iloc[train_idx], X.iloc[test_idx]
                    y_train, y_val = y.iloc[train_idx], y.iloc[test_idx]
                    
                    if y_val.nunique() < 2 or len(y_val) < 20:
                        continue
                        
                    model = XGBClassifier(**params)
                    model.fit(X_train, y_train)
                    y_proba = model.predict_proba(X_val)[:, 1]
                    scores.append(roc_auc_score(y_val, y_proba))
                
                return np.mean(scores) if scores else float('nan')

            self.study = optuna.create_study(direction='maximize')
            self.study.optimize(
                lambda trial: self._run_trial(trial, objective),
                n_trials=MAX_TRIALS,
                callbacks=[self._progress_handler],
                n_jobs=1
            )
            
            return self._train_final_model(X, y)
            
        except Exception as e:
            st.error(f"Training failed: {str(e)}")
            return False

    def _run_trial(self, trial, objective):
        try:
            return objective(trial)
        except Exception as e:
            logging.warning(f"Trial {trial.number} failed: {str(e)}")
            return float('nan')

    def _progress_handler(self, study, trial):
        self._update_progress(
            trial.number + 1,
            trial.value if trial.value else 0.0,
            study.best_value
        )

    def _train_final_model(self, X: pd.DataFrame, y: pd.Series) -> bool:
        """Final model training with validation"""
        try:
            best_params = self.study.best_params
            best_params.update({
                'early_stopping_rounds': 10,
                'eval_metric': 'auc'
            })
            
            tscv = TimeSeriesSplit(n_splits=3)
            train_idx, val_idx = list(tscv.split(X))[-1]
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

            self.model = XGBClassifier(**best_params)
            self.model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
            
            self.feature_importances = pd.Series(
                self.model.feature_importances_,
                index=X.columns
            ).sort_values(ascending=False)
            
            return True
        except Exception as e:
            st.error(f"Final training failed: {str(e)}")
            return False

    def predict(self, X: pd.DataFrame) -> float:
        """Generate prediction with confidence score"""
        try:
            if not self.model or X.empty:
                return 0.5
                
            features = self.feature_importances.index[:MIN_FEATURES]
            X_clean = X[features].ffill().bfill()
            return np.clip(self.model.predict_proba(X_clean)[0][1], 0.0, 1.0)
            
        except Exception as e:
            logging.error(f"Prediction error: {str(e)}")
            return 0.5

def main():
    """Main application interface"""
    st.sidebar.header("Configuration")
    symbol = st.sidebar.text_input("Asset Symbol", DEFAULT_SYMBOL).upper().strip()
    interval = st.sidebar.selectbox("Time Interval", INTERVAL_OPTIONS, index=2)
    
    if st.sidebar.button("🔄 Load Market Data"):
        with st.spinner("Processing data..."):
            raw_data = fetch_data(symbol, interval)
            processed_data = calculate_features(raw_data)
            if not processed_data.empty:
                st.session_state.processed_data = processed_data
                st.session_state.data_loaded = True
                st.rerun()

    if st.session_state.data_loaded and st.session_state.processed_data is not None:
        df = st.session_state.processed_data
        col1, col2 = st.columns([3, 1])
        with col1:
            st.subheader(f"{symbol} Price Chart")
            fig = px.line(df, x=df.index, y='close')
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            st.metric("Current Price", f"${df['close'].iloc[-1]:.2f}")
            st.metric("Volatility", f"{df['volatility'].iloc[-1]:.2%}")

    if st.sidebar.button("🚀 Train Model") and st.session_state.data_loaded:
        if 'target' not in st.session_state.processed_data.columns:
            st.error("Invalid training data")
            return
            
        model = TradingModel()
        X = st.session_state.processed_data.drop(columns=['target'])
        y = st.session_state.processed_data['target']
        
        st.session_state.training_progress = {'completed': 0, 'current_score': 0.0, 'best_score': 0.0}
        
        with st.spinner("Training in progress..."):
            if model.optimize_model(X, y):
                st.session_state.model = model
                st.success("Training completed successfully!")
                if not model.feature_importances.empty:
                    st.subheader("Feature Importances")
                    st.dataframe(
                        model.feature_importances.reset_index().rename(
                            columns={'index': 'Feature', 0: 'Importance'}
                        ).style.format({'Importance': '{:.2%}'}),
                        height=400
                    )

    if st.session_state.model and st.session_state.processed_data is not None:
        try:
            latest_data = st.session_state.processed_data.drop(columns=['target']).iloc[[-1]]
            confidence = st.session_state.model.predict(latest_data)
            volatility = st.session_state.processed_data['volatility'].iloc[-1]
            
            st.subheader("Trading Signal")
            col1, col2 = st.columns(2)
            col1.metric("Model Confidence", f"{confidence:.2%}")
            
            adj_buy = TRADE_THRESHOLD_BUY + (volatility * 0.15)
            adj_sell = TRADE_THRESHOLD_SELL - (volatility * 0.15)
            
            if confidence > adj_buy:
                col2.success("🚀 Strong Buy Signal")
            elif confidence < adj_sell:
                col2.error("🔻 Strong Sell Signal")
            else:
                col2.info("🛑 Market Neutral")
            
            st.caption(f"Dynamic thresholds - Buy: >{adj_buy:.0%}, Sell: <{adj_sell:.0%}")
            
        except Exception as e:
            st.error(f"Signal generation error: {str(e)}")

if __name__ == "__main__":
    main()
