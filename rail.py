# crypto_trading_system.py (FIXED MULTIINDEX TRUTH VALUE ERROR)
import logging
import numpy as np
import pandas as pd
import yfinance as yf
import streamlit as st
import optuna
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.feature_selection import RFECV
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from imblearn.under_sampling import RandomUnderSampler
from datetime import datetime, timedelta
import warnings
import re

# Configuration
DEFAULT_SYMBOL = 'BTC-USD'
PRIMARY_INTERVAL = '15m'
TRADE_THRESHOLD_BUY = 0.58
TRADE_THRESHOLD_SELL = 0.42
MAX_TRIALS = 50
GARCH_WINDOW = 14
MIN_FEATURES = 7

warnings.filterwarnings("ignore", category=UserWarning)
logging.basicConfig(level=logging.INFO)

# Streamlit UI
st.set_page_config(page_title="AI Trading System", layout="wide")
st.title("🚀 AI-Powered Cryptocurrency Trading System")

if 'model' not in st.session_state:
    st.session_state.model = None

# Data Pipeline (Fixed MultiIndex Handling)
@st.cache_data(ttl=300, show_spinner="Fetching market data...")
def fetch_data(symbol: str, interval: str) -> pd.DataFrame:
    try:
        df = yf.download(
            symbol,
            period="60d" if interval in ['15m', '30m'] else "180d",
            interval=interval,
            progress=False,
            auto_adjust=True
        )
        
        # Convert MultiIndex to flat string columns
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [f'{col[0]}_{col[1]}' for col in df.columns]
        
        # Clean and normalize column names
        df.columns = [re.sub(r'\W+', '_', str(col)).lower().strip('_') for col in df.columns]
        
        # Required columns with fallbacks
        column_map = {
            'open': ['open', 'adj_open'],
            'high': ['high', 'adj_high'],
            'low': ['low', 'adj_low'],
            'close': ['close', 'adj_close'],
            'volume': ['volume', 'adj_volume']
        }
        
        final_cols = {}
        for standard, aliases in column_map.items():
            for alias in aliases:
                if alias in df.columns:
                    final_cols[standard] = df[alias]
                    break
            if standard not in final_cols:
                st.error(f"Missing column: {standard}")
                return pd.DataFrame()
        
        return pd.DataFrame(final_cols)[['open', 'high', 'low', 'close', 'volume']].ffill().dropna()
        
    except Exception as e:
        logging.error(f"Data fetch failed: {str(e)}", exc_info=True)
        return pd.DataFrame()

def calculate_features(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or len(df) < 100:
        return pd.DataFrame()
    
    try:
        df = df.copy()
        df['close_lag1'] = df['close'].shift(1)
        df['returns'] = df['close'].pct_change()
        
        # Feature engineering
        windows = [20, 50, 100]
        for window in windows:
            df[f'sma_{window}'] = df['close'].rolling(window).mean()
            delta = df['close'].diff()
            gain = delta.clip(lower=0)
            loss = -delta.clip(upper=0)
            rs = (gain.rolling(window).mean() / 
                 loss.rolling(window).mean().replace(0, 1))
            df[f'rsi_{window}'] = 100 - (100 / (1 + rs))
        
        df['volatility'] = df['returns'].rolling(GARCH_WINDOW).std()
        df['target'] = np.where(df['close'].pct_change().shift(-1) > 0.01, 2, 
                        np.where(df['close'].pct_change().shift(-1) < -0.01, 0, 1))
        
        return df.dropna().drop(columns=['open', 'high', 'low', 'close', 'volume', 'close_lag1'])
    
    except Exception as e:
        logging.error(f"Feature engineering failed: {str(e)}")
        return pd.DataFrame()

# Model Pipeline (Fixed Index Handling)
class TradingModel:
    def __init__(self):
        self.selected_features = []
        self.model = None

    def optimize_model(self, X: pd.DataFrame, y: pd.Series):
        try:
            tscv = TimeSeriesSplit(n_splits=3)
            
            # Feature selection with list conversion
            selector = RFECV(
                GradientBoostingClassifier(),
                step=1,
                cv=tscv,
                min_features_to_select=MIN_FEATURES
            )
            selector.fit(X, y)
            self.selected_features = X.columns[selector.get_support()].tolist()  # Convert to list
            
            # Hyperparameter tuning
            study = optuna.create_study(direction='maximize')
            study.optimize(
                lambda trial: self._objective(trial, X[self.selected_features], y, tscv),
                n_trials=MAX_TRIALS
            )
            
            self.model = GradientBoostingClassifier(**study.best_params)
            self.model.fit(X[self.selected_features], y)
            
            # Validation
            st.write("Model Performance:")
            st.text(classification_report(y, self.model.predict(X[self.selected_features])))
            
        except Exception as e:
            st.error(f"Training failed: {str(e)}")

    def _objective(self, trial, X: pd.DataFrame, y: pd.Series, cv):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 500),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0)
        }
        
        scores = []
        for train_idx, test_idx in cv.split(X):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            
            rus = RandomUnderSampler()
            X_res, y_res = rus.fit_resample(X_train, y_train)
            
            model = GradientBoostingClassifier(**params)
            model.fit(X_res, y_res)
            scores.append(f1_score(y_test, model.predict(X_test), average='weighted'))
            
        return np.mean(scores)

    def predict(self, X: pd.DataFrame) -> float:
        try:
            # Use list-based empty check
            if not self.selected_features or X.empty:
                return 0.5
                
            return self.model.predict_proba(X[self.selected_features])[0][2]
        except Exception as e:
            logging.error(f"Prediction error: {str(e)}")
            return 0.5

# Main Interface (Fixed DataFrame Checks)
def main():
    st.sidebar.header("Settings")
    symbol = st.sidebar.text_input("Cryptocurrency Symbol", DEFAULT_SYMBOL).upper()
    
    raw_data = fetch_data(symbol, PRIMARY_INTERVAL)
    processed_data = calculate_features(raw_data)
    
    # Explicit DataFrame checks
    if isinstance(raw_data, pd.DataFrame) and not raw_data.empty:
        col1, col2 = st.columns([3, 1])
        with col1:
            st.subheader(f"{symbol} Price Chart")
            st.line_chart(raw_data['close'])
        with col2:
            st.metric("Current Price", f"${raw_data['close'].iloc[-1]:.2f}")
            if 'volatility' in processed_data.columns:
                st.metric("Volatility", f"{processed_data['volatility'].iloc[-1]:.2%}")

    if isinstance(processed_data, pd.DataFrame) and not processed_data.empty:
        if st.sidebar.button("🚀 Train Model"):
            with st.spinner("Training AI model..."):
                model = TradingModel()
                X = processed_data.drop(columns=['target'])
                y = processed_data['target']
                model.optimize_model(X, y)
                st.session_state.model = model
                st.success("Model trained successfully!")

    if st.session_state.model and isinstance(processed_data, pd.DataFrame) and not processed_data.empty:
        latest_data = processed_data.drop(columns=['target']).iloc[[-1]]
        confidence = st.session_state.model.predict(latest_data)
        
        st.subheader("Trading Signal")
        col1, col2 = st.columns(2)
        col1.metric("Confidence Score", f"{confidence:.2%}")
        
        if confidence > TRADE_THRESHOLD_BUY:
            col2.success("STRONG BUY SIGNAL 🚀")
        elif confidence < TRADE_THRESHOLD_SELL:
            col2.error("STRONG SELL SIGNAL 🔻")
        else:
            col2.info("NEUTRAL POSITION 🛑")

if __name__ == "__main__":
    main()
