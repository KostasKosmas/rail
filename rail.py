# crypto_trading_system.py (FIXED CLASS IMBALANCE & LEAKAGE)
import logging
import numpy as np
import pandas as pd
import yfinance as yf
import streamlit as st
import optuna
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline as ImbPipeline
from datetime import datetime, timedelta
import warnings
import re
from sklearn.model_selection import TimeSeriesSplit

# Configuration
DEFAULT_SYMBOL = 'BTC-USD'
PRIMARY_INTERVAL = '15m'
TRADE_THRESHOLD = 0.65  # Unified threshold for clearer signals
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

# Enhanced Data Pipeline
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
        
        # Column normalization
        df.columns = [re.sub(r'\W+', '_', col).lower().strip('_') for col in df.columns]
        
        # Validate required columns
        required = ['open', 'high', 'low', 'close', 'volume']
        missing = [col for col in required if col not in df.columns]
        if missing:
            st.error(f"Missing columns: {missing}")
            return pd.DataFrame()
            
        return df[required].ffill().dropna()
        
    except Exception as e:
        logging.error(f"Data fetch failed: {str(e)}")
        return pd.DataFrame()

def calculate_features(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or len(df) < 100:
        return pd.DataFrame()
    
    try:
        df = df.copy()
        # Percentage-based volatility calculation
        df['returns'] = df['close'].pct_change()
        df['volatility'] = df['returns'].rolling(GARCH_WINDOW).std() * np.sqrt(365)
        
        # Dynamic target creation
        future_returns = df['close'].pct_change().shift(-1)
        df['target'] = np.where(
            future_returns > df['volatility'] * 1.5, 2,  # Buy
            np.where(
                future_returns < -df['volatility'] * 1.5, 0,  # Sell
                1  # Hold
            )
        )
        
        # Technical indicators
        windows = [20, 50, 100]
        for window in windows:
            df[f'sma_{window}'] = df['close'].rolling(window).mean()
            df[f'ema_{window}'] = df['close'].ewm(span=window).mean()
            df[f'returns_{window}'] = df['close'].pct_change(window)
            
        # Cleanup
        return df.dropna().drop(columns=['open', 'high', 'low', 'close', 'volume'])
    
    except Exception as e:
        logging.error(f"Feature engineering failed: {str(e)}")
        return pd.DataFrame()

class PurgedTimeSeriesSplit:
    """Adapted from Advances in Financial Machine Learning"""
    def __init__(self, n_splits=5, purge_days=3):
        self.n_splits = n_splits
        self.purge_days = purge_days

    def split(self, X, y=None, groups=None):
        indices = np.arange(len(X))
        test_starts = [(i[0], i[-1]) for i in np.array_split(indices, self.n_splits)]
        
        for i, j in test_starts:
            test_start = i
            test_end = j
            train_end = test_start - self.purge_days
            train_indices = indices[0:train_end]
            test_indices = indices[test_start:test_end]
            yield train_indices, test_indices

# Model Pipeline with Enhanced Validation
class TradingModel:
    def __init__(self):
        self.selected_features = []
        self.model = None
        self.classes_ = None

    def optimize_model(self, X: pd.DataFrame, y: pd.Series):
        try:
            cv = PurgedTimeSeriesSplit(n_splits=3, purge_days=2)
            
            # Class-balanced pipeline
            sampler = ImbPipeline([
                ('under', RandomUnderSampler(sampling_strategy={1: 0.5})),
                ('over', SMOTE(sampling_strategy={0: 0.3, 2: 0.3}))
            ])
            
            study = optuna.create_study(direction='maximize')
            study.optimize(
                lambda trial: self._objective(trial, X, y, cv, sampler),
                n_trials=MAX_TRIALS
            )
            
            # Final model with best params
            self.model = GradientBoostingClassifier(**study.best_params)
            X_res, y_res = sampler.fit_resample(X, y)
            self.model.fit(X_res, y_res)
            self.classes_ = self.model.classes_
            
            # Proper validation
            st.subheader("Validation Metrics")
            val_report, conf_matrix = self._cross_validate(X, y, cv, sampler)
            st.text(val_report)
            st.write("Confusion Matrix:")
            st.dataframe(conf_matrix)
            
        except Exception as e:
            logging.error(f"Training failed: {str(e)}")
            st.error("Model training failed. Check logs for details.")

    def _objective(self, trial, X, y, cv, sampler):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 500),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'max_depth': trial.suggest_int('max_depth', 3, 15),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0)
        }
        
        scores = []
        for train_idx, test_idx in cv.split(X):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            
            # Balanced sampling
            X_res, y_res = sampler.fit_resample(X_train, y_train)
            
            model = GradientBoostingClassifier(**params)
            model.fit(X_res, y_res)
            preds = model.predict(X_test)
            scores.append(f1_score(y_test, preds, average='weighted'))
            
        return np.mean(scores)

    def _cross_validate(self, X, y, cv, sampler):
        reports = []
        matrices = []
        
        for train_idx, test_idx in cv.split(X):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            
            X_res, y_res = sampler.fit_resample(X_train, y_train)
            self.model.fit(X_res, y_res)
            preds = self.model.predict(X_test)
            
            reports.append(classification_report(y_test, preds))
            matrices.append(confusion_matrix(y_test, preds))
            
        avg_report = "\n".join(
            f"Fold {i+1}:\n{report}" 
            for i, report in enumerate(reports)
        )
        avg_matrix = np.mean(matrices, axis=0).astype(int)
        
        return avg_report, pd.DataFrame(avg_matrix)

    def predict(self, X: pd.DataFrame) -> tuple:
        try:
            if not self.model or X.empty:
                return 0.0, "Hold"
                
            proba = self.model.predict_proba(X)[0]
            class_idx = np.argmax(proba)
            confidence = proba[class_idx]
            
            # Map classes to labels
            labels = {0: "Sell", 1: "Hold", 2: "Buy"}
            return confidence, labels.get(class_idx, "Hold")
            
        except Exception as e:
            logging.error(f"Prediction error: {str(e)}")
            return 0.0, "Hold"

# Main Interface with Enhanced Visualization
def main():
    st.sidebar.header("Settings")
    symbol = st.sidebar.text_input("Cryptocurrency Symbol", DEFAULT_SYMBOL).upper()
    
    raw_data = fetch_data(symbol, PRIMARY_INTERVAL)
    processed_data = calculate_features(raw_data)
    
    if not raw_data.empty and not processed_data.empty:
        col1, col2 = st.columns([3, 1])
        with col1:
            st.subheader(f"{symbol} Price Chart")
            st.line_chart(raw_data['close'])
        with col2:
            current_price = raw_data['close'].iloc[-1]
            current_vol = processed_data['volatility'].iloc[-1]
            st.metric("Current Price", f"${current_price:.2f}")
            st.metric("Volatility", f"{current_vol:.2%}")

    if st.sidebar.button("🚀 Train Model") and not processed_data.empty:
        with st.spinner("Training AI model..."):
            model = TradingModel()
            X = processed_data.drop(columns=['target'])
            y = processed_data['target']
            
            # Show class distribution
            st.write("Class Distribution:", y.value_counts().to_dict())
            
            model.optimize_model(X, y)
            st.session_state.model = model
            st.success("Model trained successfully!")

    if st.session_state.model and not processed_data.empty:
        latest_data = processed_data.drop(columns=['target']).iloc[[-1]]
        confidence, signal = st.session_state.model.predict(latest_data)
        
        st.subheader("Trading Signal")
        col1, col2 = st.columns(2)
        col1.metric("Confidence Score", f"{confidence:.2%}")
        
        signal_color = {
            "Buy": "green",
            "Sell": "red",
            "Hold": "blue"
        }
        col2.markdown(
            f"<h2 style='color:{signal_color[signal]}'>{signal} Signal</h2>",
            unsafe_allow_html=True
        )

if __name__ == "__main__":
    main()
