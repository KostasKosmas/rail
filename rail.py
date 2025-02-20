# crypto_trading_system.py (FIXED TUPLE ERROR)
import logging
import numpy as np
import pandas as pd
import yfinance as yf
import streamlit as st
import optuna
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline as ImbPipeline
from datetime import datetime, timedelta
import warnings
import re

# Configuration
DEFAULT_SYMBOL = 'BTC-USD'
PRIMARY_INTERVAL = '15m'
TRADE_THRESHOLD = 0.65
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

# Fixed Data Pipeline with Tuple Handling
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
        
        # Convert all column names to strings
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = ['_'.join(map(str, col)).strip() for col in df.columns.values]
        else:
            df.columns = [str(col).strip() for col in df.columns]
        
        # Clean column names safely
        def clean_column_name(col: str) -> str:
            # Convert tuple to string if needed
            col_str = '_'.join(col) if isinstance(col, tuple) else str(col)
            # Remove special characters
            return re.sub(r'[^a-zA-Z0-9_]+', '_', col_str).lower().strip('_')
        
        df.columns = [clean_column_name(col) for col in df.columns]
        
        # Validate required columns
        required = ['open', 'high', 'low', 'close', 'volume']
        missing = [col for col in required if col not in df.columns]
        if missing:
            st.error(f"Missing columns: {missing}\nAvailable: {list(df.columns)}")
            return pd.DataFrame()
            
        return df[required].ffill().dropna()
        
    except Exception as e:
        logging.error(f"Data fetch failed: {str(e)}", exc_info=True)
        return pd.DataFrame()

def calculate_features(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or len(df) < 100:
        return pd.DataFrame()
    
    try:
        df = df.copy()
        df['returns'] = df['close'].pct_change()
        df['volatility'] = df['returns'].rolling(GARCH_WINDOW).std() * np.sqrt(365)
        
        # Target generation
        future_returns = df['close'].pct_change().shift(-1)
        df['target'] = np.where(
            future_returns > df['volatility'] * 1.5, 2,
            np.where(
                future_returns < -df['volatility'] * 1.5, 0,
                1
            )
        )
        
        # Technical indicators
        windows = [20, 50, 100]
        for window in windows:
            df[f'sma_{window}'] = df['close'].rolling(window).mean()
            df[f'ema_{window}'] = df['close'].ewm(span=window).mean()
            df[f'roc_{window}'] = df['close'].pct_change(window)
            
        return df.dropna().drop(columns=['open', 'high', 'low', 'close', 'volume'])
    
    except Exception as e:
        logging.error(f"Feature engineering failed: {str(e)}")
        return pd.DataFrame()

class TradingModel:
    def __init__(self):
        self.model = None
        self.classes_ = None

    def optimize_model(self, X: pd.DataFrame, y: pd.Series):
        try:
            sampler = ImbPipeline([
                ('under', RandomUnderSampler(sampling_strategy={1: 0.5})),
                ('over', SMOTE(sampling_strategy={0: 0.3, 2: 0.3}))
            ])
            
            study = optuna.create_study(direction='maximize')
            study.optimize(
                lambda trial: self._objective(trial, X, y, sampler),
                n_trials=MAX_TRIALS
            )
            
            self.model = GradientBoostingClassifier(**study.best_params)
            X_res, y_res = sampler.fit_resample(X, y)
            self.model.fit(X_res, y_res)
            self.classes_ = self.model.classes_
            
            # Validation
            st.subheader("Validation Report")
            y_pred = self.model.predict(X)
            st.text(classification_report(y, y_pred))
            st.write("Confusion Matrix:")
            st.dataframe(confusion_matrix(y, y_pred))
            
        except Exception as e:
            logging.error(f"Training failed: {str(e)}")
            st.error("Model training failed. Check logs for details.")

    def _objective(self, trial, X, y, sampler):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 500),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'max_depth': trial.suggest_int('max_depth', 3, 15),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0)
        }
        
        X_res, y_res = sampler.fit_resample(X, y)
        model = GradientBoostingClassifier(**params)
        model.fit(X_res, y_res)
        return f1_score(y, model.predict(X), average='weighted')

    def predict(self, X: pd.DataFrame) -> tuple:
        try:
            if not self.model or X.empty:
                return 0.0, "Hold"
                
            proba = self.model.predict_proba(X)[0]
            class_idx = np.argmax(proba)
            confidence = proba[class_idx]
            labels = {0: "Sell", 1: "Hold", 2: "Buy"}
            return confidence, labels.get(class_idx, "Hold")
            
        except Exception as e:
            logging.error(f"Prediction error: {str(e)}")
            return 0.0, "Hold"

def main():
    st.sidebar.header("Settings")
    symbol = st.sidebar.text_input("Cryptocurrency Symbol", DEFAULT_SYMBOL).upper()
    
    raw_data = fetch_data(symbol, PRIMARY_INTERVAL)
    processed_data = calculate_features(raw_data)
    
    if not raw_data.empty:
        col1, col2 = st.columns([3, 1])
        with col1:
            st.subheader(f"{symbol} Price Chart")
            st.line_chart(raw_data['close'])
        with col2:
            current_price = raw_data['close'].iloc[-1]
            current_vol = processed_data['volatility'].iloc[-1] if not processed_data.empty else 0
            st.metric("Current Price", f"${current_price:.2f}")
            st.metric("Volatility", f"{current_vol:.2%}")

    if st.sidebar.button("🚀 Train Model") and not processed_data.empty:
        with st.spinner("Training AI model..."):
            model = TradingModel()
            X = processed_data.drop(columns=['target'])
            y = processed_data['target']
            model.optimize_model(X, y)
            st.session_state.model = model
            st.success("Model trained successfully!")

    if st.session_state.model and not processed_data.empty:
        latest_data = processed_data.drop(columns=['target']).iloc[[-1]]
        confidence, signal = st.session_state.model.predict(latest_data)
        
        st.subheader("Trading Signal")
        col1, col2 = st.columns(2)
        col1.metric("Confidence Score", f"{confidence:.2%}")
        col2.markdown(
            f"<h2 style='color:{'green' if signal == 'Buy' else 'red' if signal == 'Sell' else 'blue'}'>"
            f"{signal} Signal</h2>",
            unsafe_allow_html=True
        )

if __name__ == "__main__":
    main()
