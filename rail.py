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
from sklearn.feature_selection import RFECV
from sklearn.metrics import classification_report, roc_auc_score
from imblearn.over_sampling import SMOTE
from tqdm import tqdm
import warnings
import json

# Configuration
DEFAULT_SYMBOL = 'BTC-USD'
INTERVAL_OPTIONS = ["15m", "30m", "1h", "1d"]
TRADE_THRESHOLD_BUY = 0.60
TRADE_THRESHOLD_SELL = 0.40
MAX_TRIALS = 50
GARCH_WINDOW = 21
MIN_FEATURES = 10
HOLD_LOOKAHEAD = 6
MAX_RETRIES = 3
VALIDATION_WINDOW = 21  # New: For time-based validation

warnings.filterwarnings("ignore", category=UserWarning)
logging.basicConfig(level=logging.INFO)

# Streamlit UI
st.set_page_config(page_title="AI Trading System", layout="wide")
st.title("🚀 Robust Crypto Trading System")

if 'model' not in st.session_state:
    st.session_state.model = None

def safe_yf_download(symbol: str, **kwargs) -> pd.DataFrame:
    """Wrapper for yfinance download with error handling"""
    for _ in range(MAX_RETRIES):
        try:
            data = yf.download(
                symbol,
                progress=False,
                auto_adjust=True,
                **kwargs
            )
            if not data.empty:
                return data
        except json.JSONDecodeError:
            logging.warning("JSON decode error, retrying...")
            continue
        except requests.exceptions.RequestException as e:
            logging.warning(f"Network error: {str(e)}, retrying...")
            continue
    return pd.DataFrame()

@st.cache_data(ttl=300, show_spinner="Fetching market data...")
def fetch_data(symbol: str, interval: str) -> pd.DataFrame:
    """Fetch and validate market data"""
    try:
        period_map = {
            '15m': '60d', '30m': '60d', 
            '1h': '730d', '1d': '3650d'
        }
        period = period_map.get(interval, '60d')
        
        df = safe_yf_download(
            symbol,
            period=period,
            interval=interval,
        )
        
        if df.empty:
            st.error("No data returned from Yahoo Finance")
            return pd.DataFrame()
        
        # Standardize column names
        df.columns = [
            col[0].lower().replace(' ', '_') if isinstance(col, tuple) 
            else str(col).lower().replace(' ', '_') 
            for col in df.columns
        ]
        
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            st.error(f"Missing required columns: {', '.join(missing_cols)}")
            return pd.DataFrame()
            
        return df[required_cols].ffill().dropna()
    
    except Exception as e:
        logging.error(f"Data fetch failed: {str(e)}")
        return pd.DataFrame()

def calculate_features(df: pd.DataFrame) -> pd.DataFrame:
    """Feature engineering pipeline with data sanitization"""
    try:
        if df.empty:
            return pd.DataFrame()
            
        df = df.copy()
        
        # Price features with overflow protection
        df['returns'] = df['close'].pct_change().replace([np.inf, -np.inf], np.nan).fillna(0)
        df['log_returns'] = np.log1p(df['close'].pct_change()).fillna(0)
        
        # Technical indicators
        df['ema_12'] = df['close'].ewm(span=12, adjust=False).mean()
        df['ema_26'] = df['close'].ewm(span=26, adjust=False).mean()
        df['macd'] = df['ema_12'] - df['ema_26']
        df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
        
        # Volatility with smoothing
        df['volatility'] = df['returns'].rolling(GARCH_WINDOW).std().fillna(0)
        
        # Volume features with clipping
        df['volume_ma'] = df['volume'].rolling(14).mean().fillna(0)
        df['volume_change'] = df['volume'].pct_change().replace([np.inf, -np.inf], np.nan).fillna(0)
        
        # Target engineering with validation
        future_returns = df['close'].pct_change(HOLD_LOOKAHEAD).shift(-HOLD_LOOKAHEAD)
        valid_returns = future_returns.replace([np.inf, -np.inf], np.nan).dropna()
        
        if len(valid_returns) < 10:
            return pd.DataFrame()
            
        df['target'] = pd.qcut(valid_returns, q=3, labels=[0, 1, 2], duplicates='drop')
        df['target'] = df['target'].ffill().bfill().astype(int)
        
        # Sanitize infinite values
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df.ffill(inplace=True)
        df.dropna(inplace=True)
        
        return df
    
    except Exception as e:
        logging.error(f"Feature engineering failed: {str(e)}")
        return pd.DataFrame()

class TradingModel:
    def __init__(self):
        self.model = None
        self.selected_features = []
        self.feature_importances = pd.DataFrame()

    def optimize_model(self, X: pd.DataFrame, y: pd.Series) -> bool:
        """Model training pipeline with enhanced validation"""
        try:
            if X.empty or y.nunique() < 2:
                st.error("Insufficient data for training")
                return False

            # Time-based validation split
            tscv = TimeSeriesSplit(n_splits=3, test_size=VALIDATION_WINDOW)
            
            study = optuna.create_study(direction='maximize')
            study.optimize(
                lambda trial: self._objective(trial, X, y, tscv),
                n_trials=MAX_TRIALS,
                n_jobs=-1,
                catch=(ValueError,)
            )
            
            if study.best_trial:
                self._train_final_model(X, y, study.best_params)
                return True
            return False
            
        except Exception as e:
            logging.error(f"Training failed: {str(e)}")
            return False

    def _train_final_model(self, X: pd.DataFrame, y: pd.Series, params: dict):
        """Final model training with data checks"""
        try:
            # Add data sanitization
            X = X.replace([np.inf, -np.inf], np.nan).ffill().bfill()
            
            selector = RFECV(
                XGBClassifier(**params, enable_categorical=False),
                step=1,
                cv=TimeSeriesSplit(3),
                min_features_to_select=MIN_FEATURES
            )
            selector.fit(X, y)
            self.selected_features = X.columns[selector.support_].tolist()
            
            # Add data constraints for XGBoost
            self.model = XGBClassifier(
                **params,
                missing=np.nan,
                eval_metric='auc',
                tree_method='hist'  # More stable for financial data
            ).fit(X[self.selected_features], y)
            
            self.feature_importances = pd.Series(
                self.model.feature_importances_,
                index=self.selected_features
            ).sort_values(ascending=False)
            
        except Exception as e:
            logging.error(f"Model training failed: {str(e)}")
            raise

    def _objective(self, trial, X: pd.DataFrame, y: pd.Series, tscv) -> float:
        """Optuna optimization objective with stability improvements"""
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 500),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'max_depth': trial.suggest_int('max_depth', 3, 9),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'gamma': trial.suggest_float('gamma', 0, 0.5),
            'reg_alpha': trial.suggest_float('reg_alpha', 0, 1),
            'reg_lambda': trial.suggest_float('reg_lambda', 0, 1)
        }
        
        scores = []
        for train_idx, test_idx in tscv.split(X):
            try:
                X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
                y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
                
                # Remove SMOTE for time-series data
                model = XGBClassifier(
                    **params,
                    missing=np.nan,
                    tree_method='hist'
                )
                model.fit(
                    X_train.replace([np.inf, -np.inf], np.nan).ffill().bfill(),
                    y_train
                )
                
                y_proba = model.predict_proba(X_test)
                score = roc_auc_score(y_test, y_proba, multi_class='ovo')
                scores.append(score)
                
            except Exception as e:
                scores.append(0.0)
        
        return np.nanmean(scores)

    def predict(self, X: pd.DataFrame) -> float:
        """Make predictions with data sanitization"""
        if not self.model or X.empty:
            return 0.5
            
        try:
            X_clean = X.replace([np.inf, -np.inf], np.nan).ffill().bfill()
            proba = self.model.predict_proba(X_clean[self.selected_features])[0]
            return max(0.0, min(1.0, np.max(proba)))
        except Exception as e:
            logging.error(f"Prediction failed: {str(e)}")
            return 0.5

def main():
    """Main application interface"""
    st.sidebar.header("Settings")
    symbol = st.sidebar.text_input("Crypto Symbol", DEFAULT_SYMBOL).upper()
    interval = st.sidebar.selectbox("Interval", INTERVAL_OPTIONS, index=0)
    
    raw_data = fetch_data(symbol, interval)
    processed_data = calculate_features(raw_data)
    
    if not raw_data.empty and not processed_data.empty:
        col1, col2 = st.columns([3, 1])
        with col1:
            st.subheader(f"{symbol} Price Chart")
            fig = px.line(raw_data, x=raw_data.index, y='close')
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            st.metric("Current Price", f"${raw_data['close'].iloc[-1]:.2f}")
            st.metric("Volatility", f"{processed_data['volatility'].iloc[-1]:.2%}")

    if st.sidebar.button("🚀 Train Model") and not processed_data.empty:
        model = TradingModel()
        X = processed_data.drop(columns=['target'])
        y = processed_data['target']
        
        with st.spinner("Training AI Model..."):
            if model.optimize_model(X, y):
                st.session_state.model = model
                st.success("Training completed!")
                st.write("Feature Importances:")
                st.dataframe(model.feature_importances.reset_index().rename(
                    columns={'index': 'Feature', 0: 'Importance'}
                ))

    if st.session_state.model and not processed_data.empty:
        model = st.session_state.model
        latest_data = processed_data.drop(columns=['target']).iloc[[-1]]
        
        confidence = model.predict(latest_data)
        current_vol = processed_data['volatility'].iloc[-1]
        
        st.subheader("Trading Signal")
        col1, col2 = st.columns(2)
        col1.metric("Confidence Score", f"{confidence:.2%}")
        
        adj_buy = TRADE_THRESHOLD_BUY + (current_vol * 0.1)
        adj_sell = TRADE_THRESHOLD_SELL - (current_vol * 0.1)
        
        if confidence > adj_buy:
            col2.success("🚀 Strong Buy Signal")
        elif confidence < adj_sell:
            col2.error("🔻 Strong Sell Signal")
        else:
            col2.info("🛑 Hold Position")

if __name__ == "__main__":
    main()
