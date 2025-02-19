# crypto_trading_system.py (FIXED MULTIINDEX TRUTH VALUE)
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

# ======================
# CONFIGURATION
# ======================
DEFAULT_SYMBOL = 'BTC-USD'
PRIMARY_INTERVAL = '15m'
TRADE_THRESHOLD_BUY = 0.58
TRADE_THRESHOLD_SELL = 0.42
MAX_TRIALS = 50
GARCH_WINDOW = 14
MIN_FEATURES = 7

warnings.filterwarnings("ignore", category=UserWarning)
logging.basicConfig(level=logging.INFO)

# ======================
# STREAMLIT UI
# ======================
st.set_page_config(page_title="AI Trading System", layout="wide")
st.title("🚀 AI-Powered Cryptocurrency Trading System")

if 'model' not in st.session_state:
    st.session_state.model = None
if 'last_trained' not in st.session_state:
    st.session_state.last_trained = None

# ======================
# DATA PIPELINE (FIXED)
# ======================
@st.cache_data(ttl=300, show_spinner="Fetching market data...")
def fetch_data(symbol: str, interval: str) -> pd.DataFrame:
    end = datetime.now()
    lookback_days = 59 if interval in ['15m', '30m'] else 180
    
    try:
        df = yf.download(
            symbol, 
            start=end - timedelta(days=lookback_days),
            end=end,
            interval=interval,
            progress=False,
            auto_adjust=True
        )
        
        # Convert MultiIndex to flat column names
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [f'{col[0]}_{col[1]}' for col in df.columns]
        
        # Clean column names
        df.columns = [re.sub(r'\W+', '_', col).strip('_').lower() for col in df.columns]
        
        # Handle symbol-specific suffixes
        symbol_suffix = symbol.lower().replace('-', '_')
        df.columns = [col.replace(f'_{symbol_suffix}', '') for col in df.columns]
        
        # Column mapping with fallbacks
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
            else:
                st.error(f"Missing required column: {standard}")
                return pd.DataFrame()
        
        clean_df = pd.DataFrame(final_cols)[['open', 'high', 'low', 'close', 'volume']]
        return clean_df.dropna().reset_index(drop=True)
        
    except Exception as e:
        logging.error(f"Data fetch failed: {str(e)}")
        st.error(f"Failed to fetch data for {symbol}")
        return pd.DataFrame()

def calculate_features(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or len(df) < 100:
        return pd.DataFrame()
    
    try:
        df = df.copy()
        df['close_lag1'] = df['close'].shift(1)
        df['returns'] = df['close_lag1'].pct_change()
        
        windows = [20, 50, 100]
        for window in windows:
            df[f'sma_{window}'] = df['close'].rolling(window).mean()
            df[f'rsi_{window}'] = 100 - (100 / (1 + (df['close'].diff().clip(lower=0).rolling(window).mean() / 
                                         df['close'].diff().clip(upper=0).abs().rolling(window).mean())))
        
        df['volatility'] = df['returns'].rolling(GARCH_WINDOW).std()
        df['target'] = np.where(df['close'].pct_change().shift(-1) > 0, 1, 0)
        
        return df.dropna().drop(columns=['open', 'high', 'low', 'close', 'volume', 'close_lag1'])
    
    except Exception as e:
        logging.error(f"Feature engineering failed: {str(e)}")
        return pd.DataFrame()

# ======================
# MODEL PIPELINE (FIXED INDEX HANDLING)
# ======================
class TradingModel:
    def __init__(self):
        self.selected_features = []
        self.model = None
        self.selector = None

    def optimize_model(self, X: pd.DataFrame, y: pd.Series):
        try:
            tscv = TimeSeriesSplit(n_splits=3)
            
            # Convert columns to list to avoid Index issues
            self.selector = RFECV(
                GradientBoostingClassifier(),
                step=1,
                cv=tscv,
                min_features_to_select=MIN_FEATURES
            )
            self.selector.fit(X, y)
            
            # Store features as list of strings
            self.selected_features = X.columns[self.selector.get_support()].tolist()
            
            study = optuna.create_study(direction='maximize')
            study.optimize(
                lambda trial: self._objective(trial, X[self.selected_features], y, tscv),
                n_trials=MAX_TRIALS
            )
            
            self.model = GradientBoostingClassifier(**study.best_params)
            self.model.fit(X[self.selected_features], y)
            
            st.write("Model Validation Report:")
            st.text(classification_report(y, self.model.predict(X[self.selected_features])))
            
        except Exception as e:
            st.error(f"Training failed: {str(e)}")

    def _objective(self, trial, X: pd.DataFrame, y: pd.Series, cv):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 300),
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
            scores.append(f1_score(y_test, model.predict(X_test)))
            
        return np.mean(scores)

    def predict(self, X: pd.DataFrame) -> float:
        try:
            # Use explicit empty check for list
            if not self.selected_features or X.empty:
                return 0.5
                
            return self.model.predict_proba(X[self.selected_features])[0][1]
        except Exception as e:
            logging.error(f"Prediction error: {str(e)}")
            return 0.5

# ======================
# MAIN INTERFACE
# ======================
def main():
    st.sidebar.header("Settings")
    symbol = st.sidebar.text_input("Cryptocurrency Symbol", DEFAULT_SYMBOL).upper()
    
    raw_data = fetch_data(symbol, PRIMARY_INTERVAL)
    processed_data = calculate_features(raw_data)
    
    if not processed_data.empty:
        st.line_chart(raw_data['close'])
        st.write(f"Latest Price: {raw_data['close'].iloc[-1]:.2f}")
    
    if st.sidebar.button("Train Model") and not processed_data.empty:
        with st.spinner("Training model..."):
            model = TradingModel()
            X = processed_data.drop(columns=['target'])
            y = processed_data['target']
            model.optimize_model(X, y)
            st.session_state.model = model
            st.success("Model trained!")
    
    if st.session_state.model and not processed_data.empty:
        latest = processed_data.drop(columns=['target']).iloc[[-1]]
        proba = st.session_state.model.predict(latest)
        
        st.subheader("Trading Signal")
        col1, col2 = st.columns(2)
        col1.metric("Confidence", f"{proba:.2%}")
        
        if proba > TRADE_THRESHOLD_BUY:
            col2.success("Strong Buy Signal")
        elif proba < TRADE_THRESHOLD_SELL:
            col2.error("Strong Sell Signal")
        else:
            col2.info("Hold Position")

if __name__ == "__main__":
    main()
