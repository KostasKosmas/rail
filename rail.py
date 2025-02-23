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
from sklearn.preprocessing import RobustScaler
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
HOLD_LOOKAHEAD = 5  # Reduced for more relevant signals
VALIDATION_WINDOW = 21
MIN_CLASS_RATIO = 0.35
INITIAL_BALANCE = 10000

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
    'training_progress': {'completed': 0, 'current_score': 0.0, 'best_score': 0.0},
    'study': None,
    'scaler': RobustScaler()
}
for key, value in session_defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value

def safe_yf_download(symbol: str, **kwargs) -> pd.DataFrame:
    """Robust data downloader with error handling"""
    for _ in range(3):
        try:
            data = yf.download(symbol, progress=False, auto_adjust=True, **kwargs)
            return data[['Open', 'High', 'Low', 'Close', 'Volume']] if not data.empty else pd.DataFrame()
        except Exception as e:
            logging.error(f"Download error: {str(e)}")
    return pd.DataFrame()

def calculate_features(df: pd.DataFrame) -> pd.DataFrame:
    """Enhanced feature engineering with realistic financial features"""
    try:
        df = df.copy()
        
        # Price transformations
        df['returns'] = df['Close'].pct_change()
        df['log_returns'] = np.log1p(df['returns'])
        
        # Technical indicators
        for span in [12, 26]:
            df[f'EMA_{span}'] = df['Close'].ewm(span=span).mean()
        df['MACD'] = df['EMA_12'] - df['EMA_26']
        df['MACD_Signal'] = df['MACD'].ewm(span=9).mean()
        
        # Volatility metrics
        df['volatility'] = df['returns'].rolling(14).std()
        df['ATR'] = (df['High'] - df['Low']).rolling(14).mean()
        
        # Volume features
        df['volume_z'] = (df['Volume'] - df['Volume'].rolling(14).mean()) / df['Volume'].rolling(14).std()
        
        # Target engineering with realistic forward window
        df['target'] = (df['Close'].shift(-HOLD_LOOKAHEAD) > df['Close']).astype(int)
        df = df.dropna()
        
        # Class balance check
        class_ratio = df['target'].value_counts(normalize=True)
        if abs(class_ratio[0] - class_ratio[1]) > (1 - 2*MIN_CLASS_RATIO):
            st.error(f"Class imbalance: Long {class_ratio[1]:.1%}, Short {class_ratio[0]:.1%}")
            return pd.DataFrame()
            
        return df.replace([np.inf, -np.inf], np.nan).ffill().dropna()
    
    except Exception as e:
        st.error(f"Feature engineering failed: {str(e)}")
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
        st.experimental_rerun()

    def optimize_model(self, X: pd.DataFrame, y: pd.Series) -> bool:
        """Optimization pipeline with realistic parameter ranges"""
        try:
            if X.empty or y.nunique() != 2:
                st.error("Invalid training data")
                return False

            # Scale features
            X_scaled = st.session_state.scaler.fit_transform(X)
            X = pd.DataFrame(X_scaled, index=X.index, columns=X.columns)

            def objective(trial):
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 100, 500),
                    'learning_rate': trial.suggest_float('lr', 0.005, 0.2, log=True),
                    'max_depth': trial.suggest_int('max_depth', 3, 8),
                    'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                    'colsample_bytree': trial.suggest_float('colsample', 0.6, 1.0),
                    'gamma': trial.suggest_float('gamma', 0, 0.5),
                    'reg_alpha': trial.suggest_float('alpha', 0, 1),
                    'reg_lambda': trial.suggest_float('lambda', 0, 1),
                    'tree_method': 'hist'
                }
                
                scores = []
                tscv = TimeSeriesSplit(n_splits=3, test_size=VALIDATION_WINDOW)
                
                for train_idx, test_idx in tscv.split(X):
                    X_train, X_val = X.iloc[train_idx], X.iloc[test_idx]
                    y_train, y_val = y.iloc[train_idx], y.iloc[test_idx]
                    
                    if y_val.nunique() < 2:
                        continue
                        
                    model = XGBClassifier(**params)
                    model.fit(
                        X_train, y_train,
                        eval_set=[(X_val, y_val)],
                        early_stopping_rounds=10,
                        verbose=False
                    )
                    
                    y_proba = model.predict_proba(X_val)[:, 1]
                    scores.append(roc_auc_score(y_val, y_proba))
                
                return np.mean(scores) if scores else float('nan')

            if st.session_state.study is None:
                st.session_state.study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=42))

            st.session_state.study.optimize(
                lambda trial: self._run_trial(trial, objective),
                n_trials=MAX_TRIALS,
                callbacks=[self._progress_handler],
                show_progress_bar=False
            )
            
            return self._train_final_model(X, y)
            
        except Exception as e:
            st.error(f"Training failed: {str(e)}")
            return False

    def _run_trial(self, trial, objective):
        try:
            result = objective(trial)
            if np.isnan(result):
                trial.set_user_attr("failed", True)
            return result
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
        """Final training with walk-forward validation"""
        try:
            best_params = st.session_state.study.best_params
            best_params.update({
                'early_stopping_rounds': 15,
                'eval_metric': 'auc'
            })

            # Walk-forward validation
            tscv = TimeSeriesSplit(n_splits=3)
            scores = []
            feature_importances = []
            
            for train_idx, test_idx in tscv.split(X):
                X_train, X_val = X.iloc[train_idx], X.iloc[test_idx]
                y_train, y_val = y.iloc[train_idx], y.iloc[test_idx]
                
                model = XGBClassifier(**best_params)
                model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
                feature_importances.append(model.feature_importances_)
                scores.append(roc_auc_score(y_val, model.predict_proba(X_val)[:, 1]))
            
            self.model = XGBClassifier(**best_params).fit(X, y)
            self.feature_importances = pd.Series(
                np.mean(feature_importances, axis=0),
                index=X.columns
            ).sort_values(ascending=False)
            
            st.success(f"Final Validation AUC: {np.mean(scores):.2%}")
            return True
            
        except Exception as e:
            st.error(f"Final training failed: {str(e)}")
            return False

    def predict(self, X: pd.DataFrame) -> float:
        """Generate prediction with proper feature scaling"""
        try:
            if not self.model or X.empty:
                return 0.5
                
            X_scaled = st.session_state.scaler.transform(X)
            return self.model.predict_proba(X_scaled)[0][1]
        except Exception as e:
            logging.error(f"Prediction error: {str(e)}")
            return 0.5

def main():
    """Main application interface"""
    st.sidebar.header("Configuration")
    symbol = st.sidebar.text_input("Asset", DEFAULT_SYMBOL).upper()
    interval = st.sidebar.selectbox("Interval", INTERVAL_OPTIONS, index=2)
    
    if st.sidebar.button("🔄 Load Data"):
        with st.spinner("Processing market data..."):
            raw = safe_yf_download(symbol, period='60d' if interval == '1h' else '730d', interval=interval)
            processed = calculate_features(raw)
            if not processed.empty:
                st.session_state.processed_data = processed
                st.session_state.data_loaded = True
                st.session_state.study = None
                st.experimental_rerun()

    if st.session_state.data_loaded and st.session_state.processed_data is not None:
        df = st.session_state.processed_data
        col1, col2 = st.columns([3, 1])
        with col1:
            fig = px.line(df, x=df.index, y='Close', title=f"{symbol} Price Action")
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            st.metric("Current Price", f"${df['Close'].iloc[-1]:.2f}")
            st.metric("Market Volatility", f"{df['volatility'].iloc[-1]:.2%}")

    if st.sidebar.button("🚀 Train Model") and st.session_state.data_loaded:
        if 'target' not in st.session_state.processed_data.columns:
            st.error("Invalid training data")
            return
            
        model = TradingModel()
        X = st.session_state.processed_data.drop(columns=['target'])
        y = st.session_state.processed_data['target']
        
        st.session_state.training_progress = {'completed': 0, 'current_score': 0.0, 'best_score': 0.0}
        
        with st.spinner("Training AI Model..."):
            if model.optimize_model(X, y):
                st.session_state.model = model
                if not model.feature_importances.empty:
                    with st.expander("Model Diagnostics"):
                        st.subheader("Feature Importances")
                        st.dataframe(model.feature_importances.reset_index().rename(
                            columns={'index': 'Feature', 0: 'Importance'}
                        ).style.format({'Importance': '{:.2%}'}), height=300)
                        
                        # Backtest results
                        history = st.session_state.study.trials_dataframe()
                        st.line_chart(history[['number', 'value']].set_index('number'))

    if st.session_state.model and st.session_state.processed_data is not None:
        try:
            latest = st.session_state.processed_data.drop(columns=['target']).iloc[[-1]]
            confidence = st.session_state.model.predict(latest)
            vol = st.session_state.processed_data['volatility'].iloc[-1]
            
            st.subheader("Trading Signal")
            col1, col2 = st.columns(2)
            col1.metric("Model Confidence", f"{confidence:.2%}")
            
            # Dynamic thresholds based on market volatility
            buy_thresh = TRADE_THRESHOLD_BUY + (vol * 0.1)
            sell_thresh = TRADE_THRESHOLD_SELL - (vol * 0.1)
            
            if confidence > buy_thresh:
                col2.success("🚀 Strong Buy Signal")
            elif confidence < sell_thresh:
                col2.error("🔻 Strong Sell Signal")
            else:
                col2.info("🛑 Market Neutral")
            
            st.caption(f"Dynamic thresholds - Buy: >{buy_thresh:.0%}, Sell: <{sell_thresh:.0%}")

        except Exception as e:
            st.error(f"Signal error: {str(e)}")

if __name__ == "__main__":
    main()
