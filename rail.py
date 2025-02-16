# crypto_trading_system.py
import logging
import time
import numpy as np
import pandas as pd
import yfinance as yf
import streamlit as st
import optuna
from arch import arch_model
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.feature_selection import SelectFromModel
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import precision_score
from sklearn.utils.class_weight import compute_sample_weight
from imblearn.over_sampling import SMOTE
from datetime import datetime, timedelta
import warnings

# ======================
# CONFIGURATION
# ======================
SYMBOL = 'BTC-USD'
PRIMARY_INTERVAL = '15m'
FALLBACK_INTERVAL = '60m'
TRADE_THRESHOLD_BUY = 0.65
TRADE_THRESHOLD_SELL = 0.35
MAX_TRIALS = 12
CV_SPLITS = 3
MAX_RETRIES = 3

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning)
logging.basicConfig(level=logging.INFO)

# ======================
# STREAMLIT UI
# ======================
st.set_page_config(page_title="AI Trading System", layout="wide")
st.title("🚀 AI-Powered Cryptocurrency Trading System")
st.markdown("Real-time trading signals with machine learning and volatility modeling")

if 'model' not in st.session_state:
    st.session_state.model = None
if 'training_progress' not in st.session_state:
    st.session_state.training_progress = None
if 'data' not in st.session_state:
    st.session_state.data = None
if 'data_warning' not in st.session_state:
    st.session_state.data_warning = None

# ======================
# ROBUST DATA PIPELINE
# ======================
def safe_download(symbol, interval, days_back):
    """Safe data download with retries and validation"""
    end = datetime.now()
    start = end - timedelta(days=days_back)
    
    for _ in range(MAX_RETRIES):
        try:
            df = yf.download(
                symbol,
                start=start,
                end=end,
                interval=interval,
                progress=False,
                auto_adjust=True
            ).ffill()
            
            if not df.empty and len(df) > 100:
                return df
        except Exception as e:
            logging.warning(f"Download failed: {str(e)}")
            time.sleep(1)
    
    return pd.DataFrame()

@st.cache_data(ttl=300, show_spinner="Fetching market data...")
def get_data() -> pd.DataFrame:
    """Robust data fetching with fallback mechanism"""
    try:
        # Try primary interval with 59 days lookback
        df = safe_download(SYMBOL, PRIMARY_INTERVAL, 59)
        if not df.empty:
            st.session_state.data_warning = None
            return process_data(df)
            
        # Fallback to 60m interval with 90 days lookback
        st.session_state.data_warning = (
            f"Using {FALLBACK_INTERVAL} data due to {PRIMARY_INTERVAL} availability limits"
        )
        df = safe_download(SYMBOL, FALLBACK_INTERVAL, 90)
        return process_data(df) if not df.empty else pd.DataFrame()
        
    except Exception as e:
        st.error(f"Data download failed: {str(e)}")
        return pd.DataFrame()

def process_data(df):
    """Safe data processing with validation"""
    try:
        df = df.copy()
        df['Returns'] = df['Close'].pct_change()
        
        # Technical indicators with validation
        if len(df) > 20:
            df['SMA_20'] = df['Close'].rolling(20).mean()
            df['RSI'] = 100 - (100 / (1 + (
                df['Close'].diff().clip(lower=0).rolling(14).mean() / 
                df['Close'].diff().clip(upper=0).abs().rolling(14).mean()
            ))
            df['MACD'] = df['Close'].ewm(span=12).mean() - df['Close'].ewm(span=26).mean()
            df['Volatility'] = df['Returns'].rolling(14).std() * np.sqrt(365)
            df['Target'] = (df['Returns'].shift(-1) > 0).astype(int)
            
            return df.dropna()
        return pd.DataFrame()
    except Exception as e:
        logging.error(f"Data processing failed: {str(e)}")
        return pd.DataFrame()

# ======================
# MODEL PIPELINE
# ======================
class TradingModel:
    def __init__(self):
        self.models = {}
        self.feature_selector = None
        self.selected_features = []
        self.smote = SMOTE(random_state=42)
        self.best_score = 0
        self.calibrators = {}

    def _feature_selection(self, X, y):
        """Safe feature selection"""
        try:
            self.feature_selector = SelectFromModel(
                RandomForestClassifier(n_estimators=100, random_state=42),
                threshold="1.25*median"
            )
            self.feature_selector.fit(X, y)
            self.selected_features = X.columns[self.feature_selector.get_support()]
            return X[self.selected_features]
        except Exception as e:
            logging.error(f"Feature selection failed: {str(e)}")
            return X

    def optimize_model(self, X, y, model_type):
        """Optimization with validation"""
        if X.empty or len(y) == 0:
            raise ValueError("Invalid data for optimization")
            
        def objective(trial):
            model = self._create_model(trial, model_type)
            tscv = TimeSeriesSplit(CV_SPLITS)
            scores = []
            
            for train_idx, test_idx in tscv.split(X):
                try:
                    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
                    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
                    
                    X_res, y_res = self.smote.fit_resample(X_train, y_train)
                    model.fit(X_res, y_res)
                    preds = model.predict(X_test)
                    scores.append(precision_score(y_test, preds))
                except Exception as e:
                    logging.warning(f"CV fold failed: {str(e)}")
                    
            return np.mean(scores) if scores else 0.0
        
        study = optuna.create_study(direction='maximize')
        study.optimize(
            objective, 
            n_trials=MAX_TRIALS,
            timeout=600,
            callbacks=[self._progress_callback(model_type)]
        )
        
        if study.best_value > self.best_score:
            self.best_score = study.best_value
            self.models[model_type] = study.best_params
            self._update_calibrator(X, y, model_type)

    # Rest of the model class remains the same as previous version
    # (keep the same methods for _create_model, _update_calibrator, 
    # _create_final_model, _progress_callback, and predict)

# ======================
# STREAMLIT INTERFACE
# ======================
def main():
    df = get_data()
    
    if st.session_state.data_warning:
        st.warning(st.session_state.data_warning)
    
    if not df.empty:
        col1, col2 = st.columns([3, 1])
        with col1:
            st.subheader("Price Chart")
            st.line_chart(df['Close'])
            
        with col2:
            try:
                current_price = df['Close'].iloc[-1] if len(df) > 0 else 0
                current_vol = df['Volatility'].iloc[-1] if len(df) > 0 else 0
                st.metric("Current Price", f"${current_price:.2f}")
                st.metric("Volatility", f"{current_vol:.2%}")
            except Exception as e:
                st.error(f"Data display error: {str(e)}")

    # Model controls
    if st.sidebar.button("🚀 Train Model"):
        if df.empty or 'Target' not in df.columns:
            st.error("Invalid or insufficient data for training")
            return
            
        try:
            st.session_state.training_progress = st.progress(0, text="Initializing...")
            model = TradingModel()
            
            X = df.drop(['Target', 'Returns'], axis=1, errors='ignore')
            y = df['Target']
            
            with st.spinner("Selecting important features..."):
                X_sel = model._feature_selection(X, y)
            
            if not X_sel.empty:
                with st.spinner("Optimizing Random Forest..."):
                    model.optimize_model(X_sel, y, 'rf')
                    
                with st.spinner("Optimizing Gradient Boosting..."):
                    model.optimize_model(X_sel, y, 'gb')
                
                st.session_state.model = model
                st.success(f"Model trained successfully! Best Precision: {model.best_score:.2%}")
            else:
                st.error("Feature selection failed - no features available")
            
        except Exception as e:
            st.error(f"Training failed: {str(e)}")
        finally:
            time.sleep(1)
            st.session_state.training_progress = None

    # Trading signals
    if st.session_state.model and not df.empty:
        try:
            X = df.drop(['Target', 'Returns'], axis=1, errors='ignore')
            if not X.empty and len(st.session_state.model.selected_features) > 0:
                X_sel = X[st.session_state.model.selected_features]
                confidence = st.session_state.model.predict(X_sel.iloc[[-1]])[0]
                
                st.subheader("Trading Signal")
                col1, col2, col3 = st.columns(3)
                col1.metric("Confidence", f"{confidence:.2%}")
                
                if confidence > TRADE_THRESHOLD_BUY:
                    col2.success("🚀 Strong Buy Signal")
                    col3.write(f"Threshold: >{TRADE_THRESHOLD_BUY:.0%}")
                elif confidence < TRADE_THRESHOLD_SELL:
                    col2.error("🔻 Strong Sell Signal")
                    col3.write(f"Threshold: <{TRADE_THRESHOLD_SELL:.0%}")
                else:
                    col2.info("🛑 Hold Position")
                    col3.write("No clear market signal")
            
        except Exception as e:
            st.error(f"Prediction error: {str(e)}")

if __name__ == "__main__":
    main()
