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
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
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
MAX_RETRIES = 3
VALIDATION_WINDOW = 21

warnings.filterwarnings("ignore", category=UserWarning)
logging.basicConfig(level=logging.INFO)

# Streamlit UI
st.set_page_config(page_title="AI Trading System", layout="wide")
st.title("🚀 Smart Crypto Trading Assistant")

if 'model' not in st.session_state:
    st.session_state.model = None
if 'processed_data' not in st.session_state:
    st.session_state.processed_data = None

def safe_yf_download(symbol: str, **kwargs) -> pd.DataFrame:
    """Robust data downloader with retry logic"""
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
        except (json.JSONDecodeError, requests.exceptions.RequestException):
            continue
        except Exception as e:
            logging.error(f"Download error: {str(e)}")
            break
    return pd.DataFrame()

def normalize_columns(symbol: str, columns) -> list:
    """Normalize column names and remove ticker prefixes"""
    normalized_symbol = symbol.lower().replace('-', '')
    processed_cols = []
    
    for col in columns:
        if isinstance(col, tuple):
            col = '_'.join(map(str, col))
        
        col = col.lower().replace('-', '').replace(' ', '_')
        col = col.split(f"{normalized_symbol}_")[-1]
        
        col = {
            'adjclose': 'close',
            'adjusted_close': 'close',
            'vol': 'volume'
        }.get(col, col)
        
        processed_cols.append(col)
    
    return processed_cols

@st.cache_data(ttl=300, show_spinner="Fetching market data...")
def fetch_data(symbol: str, interval: str) -> pd.DataFrame:
    """Data fetcher with comprehensive preprocessing"""
    try:
        period_map = {'15m': '60d', '30m': '60d', '1h': '730d', '1d': 'max'}
        df = safe_yf_download(symbol, period=period_map.get(interval, '60d'), interval=interval)
        
        if df.empty:
            st.error("No data received from source")
            return pd.DataFrame()

        df.columns = normalize_columns(symbol, df.columns)
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        
        if missing := [col for col in required_cols if col not in df.columns]:
            st.error(f"Missing columns: {', '.join(missing)}")
            return pd.DataFrame()

        return df[required_cols].ffill().bfill().dropna()
    
    except Exception as e:
        st.error(f"Data fetch failed: {str(e)}")
        return pd.DataFrame()

def calculate_features(df: pd.DataFrame) -> pd.DataFrame:
    """Enhanced feature engineering pipeline"""
    try:
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        if df.empty or any(col not in df.columns for col in required_cols):
            return pd.DataFrame()

        df = df.copy()
        df['returns'] = df['close'].pct_change().fillna(0)
        df['log_returns'] = np.log1p(df['returns']).fillna(0)

        # Technical indicators
        for span in [12, 26]:
            df[f'ema_{span}'] = df['close'].ewm(span=span, adjust=False).mean()
        df['macd'] = df['ema_12'] - df['ema_26']
        df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()

        # Volatility features
        df['volatility'] = df['returns'].rolling(GARCH_WINDOW).std().fillna(0)
        
        # ATR calculation
        prev_close = df['close'].shift(1).bfill()
        tr = pd.DataFrame({
            'tr1': df['high'] - df['low'],
            'tr2': (df['high'] - prev_close).abs(),
            'tr3': (df['low'] - prev_close).abs()
        }).max(axis=1)
        df['atr'] = tr.rolling(14).mean().fillna(0)

        # Volume features
        df['volume_ma'] = df['volume'].rolling(14).mean().fillna(0)
        df['volume_change'] = df['volume'].pct_change().fillna(0)

        # Target engineering
        future_returns = df['close'].pct_change(HOLD_LOOKAHEAD).shift(-HOLD_LOOKAHEAD)
        if (valid_returns := future_returns.dropna()).empty:
            return pd.DataFrame()
            
        df['target'] = pd.qcut(valid_returns, q=3, labels=[0, 1, 2], duplicates='drop')
        return df.dropna().replace([np.inf, -np.inf], np.nan).ffill().bfill().dropna()

    except Exception as e:
        st.error(f"Feature engineering failed: {str(e)}")
        return pd.DataFrame()

class TradingModel:
    def __init__(self):
        self.model = None
        self.selected_features = []
        self.feature_importances = pd.DataFrame()

    def optimize_model(self, X: pd.DataFrame, y: pd.Series) -> bool:
        """Model optimization pipeline"""
        try:
            if X.empty or y.nunique() < 2:
                return False

            tscv = TimeSeriesSplit(n_splits=3, test_size=VALIDATION_WINDOW)
            study = optuna.create_study(direction='maximize')
            
            study.optimize(
                lambda trial: self._objective(trial, X, y, tscv),
                n_trials=MAX_TRIALS,
                n_jobs=-1,
                catch=(ValueError,)
            )
            
            if study.best_trial:
                return self._train_final_model(X, y, study.best_params)
            return False
            
        except Exception as e:
            st.error(f"Training failed: {str(e)}")
            return False

    def _train_final_model(self, X: pd.DataFrame, y: pd.Series, params: dict) -> bool:
        """Final model training"""
        try:
            X = X.ffill().bfill()
            selector = RFECV(
                XGBClassifier(**params),
                step=1,
                cv=TimeSeriesSplit(3),
                min_features_to_select=MIN_FEATURES
            )
            selector.fit(X, y)
            self.selected_features = X.columns[selector.support_]
            
            self.model = XGBClassifier(
                **params,
                tree_method='hist',
                eval_metric='auc'
            ).fit(X[self.selected_features], y)
            
            self.feature_importances = pd.Series(
                self.model.feature_importances_,
                index=self.selected_features
            ).sort_values(ascending=False)
            return True

        except Exception as e:
            st.error(f"Model training error: {str(e)}")
            return False

    def _objective(self, trial, X: pd.DataFrame, y: pd.Series, tscv) -> float:
        """Optimization objective function"""
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 500),
            'learning_rate': trial.suggest_float('learning_rate', 1e-3, 0.3, log=True),
            'max_depth': trial.suggest_int('max_depth', 3, 9),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'gamma': trial.suggest_float('gamma', 0, 0.5),
            'reg_alpha': trial.suggest_float('reg_alpha', 0, 1),
            'reg_lambda': trial.suggest_float('reg_lambda', 0, 1)
        }
        
        scores = []
        for train_idx, test_idx in tscv.split(X):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            
            try:
                model = XGBClassifier(**params, tree_method='hist')
                model.fit(X_train.ffill().bfill(), y_train)
                scores.append(roc_auc_score(y_test, model.predict_proba(X_test), multi_class='ovo'))
            except:
                scores.append(0.0)
        
        return np.mean(scores)

    def predict(self, X: pd.DataFrame) -> float:
        """Robust prediction method"""
        try:
            if self.model and not X.empty:
                X_clean = X[self.selected_features].ffill().bfill()
                return np.clip(np.max(self.model.predict_proba(X_clean)[0]), 0.0, 1.0)
            return 0.5
        except:
            return 0.5

def main():
    """Main application interface"""
    st.sidebar.header("Configuration")
    symbol = st.sidebar.text_input("Asset Symbol", DEFAULT_SYMBOL).upper().strip()
    interval = st.sidebar.selectbox("Time Interval", INTERVAL_OPTIONS, index=2)
    
    # Data processing
    with st.spinner("Loading market data..."):
        raw_data = fetch_data(symbol, interval)
        processed_data = calculate_features(raw_data)
        st.session_state.processed_data = processed_data if not processed_data.empty else None

    # Market overview
    if not raw_data.empty:
        col1, col2 = st.columns([3, 1])
        with col1:
            st.subheader(f"{symbol} Price Action")
            fig = px.line(raw_data, x=raw_data.index, y='close')
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.metric("Current Price", f"${raw_data['close'].iloc[-1]:.2f}")
            if processed_data is not None:
                st.metric("Volatility", f"{processed_data['volatility'].iloc[-1]:.2%}")

    # Model training
    if st.sidebar.button("🚀 Train Trading Model") and processed_data is not None:
        model = TradingModel()
        X = processed_data.drop(columns=['target'])
        y = processed_data['target']
        
        with st.spinner("Optimizing trading strategy..."):
            if model.optimize_model(X, y):
                st.session_state.model = model
                st.success("Model training completed!")
                
                if not model.feature_importances.empty:
                    st.subheader("Feature Importance")
                    st.dataframe(
                        model.feature_importances.reset_index().rename(
                            columns={'index': 'Feature', 0: 'Importance'}
                        ).style.format({'Importance': '{:.2%}'}),
                        height=400
                    )

    # Trading signals
    if st.session_state.model and st.session_state.processed_data is not None:
        try:
            processed_data = st.session_state.processed_data
            model = st.session_state.model
            latest_data = processed_data.drop(columns=['target']).iloc[[-1]]
            confidence = model.predict(latest_data)
            current_vol = processed_data['volatility'].iloc[-1]
            
            adj_buy = TRADE_THRESHOLD_BUY + (current_vol * 0.15)
            adj_sell = TRADE_THRESHOLD_SELL - (current_vol * 0.15)
            
            st.subheader("Trading Advisory")
            col1, col2 = st.columns(2)
            col1.metric("Model Confidence", f"{confidence:.2%}")
            
            if confidence > adj_buy:
                col2.success("🚀 Strong Buy Signal")
            elif confidence < adj_sell:
                col2.error("🔻 Strong Sell Signal")
            else:
                col2.info("🛑 Market Neutral")
                
            st.caption(f"Volatility-adjusted thresholds: Buy >{adj_buy:.0%}, Sell <{adj_sell:.0%}")

        except Exception as e:
            st.error(f"Signal error: {str(e)}")

if __name__ == "__main__":
    main()
