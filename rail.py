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
TRADE_THRESHOLD_BUY = 0.65  # Higher threshold for fewer but more accurate signals
TRADE_THRESHOLD_SELL = 0.35
MAX_TRIALS = 12  # Optimized trial count
CV_SPLITS = 3
FEATURE_SELECTION_THRESHOLD = "1.25*median"

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

# ======================
# OPTIMIZED DATA PIPELINE
# ======================
@st.cache_data(ttl=300, show_spinner="Fetching market data...")
def get_data() -> pd.DataFrame:
    """Optimized data fetching with essential features only"""
    end = datetime.now()
    df = yf.download(
        SYMBOL, 
        start=end - timedelta(days=90),
        end=end,
        interval=PRIMARY_INTERVAL,
        progress=False
    ).ffill()
    
    # Feature engineering
    df['Returns'] = df['Close'].pct_change()
    df['SMA_20'] = df['Close'].rolling(20).mean()
    df['RSI'] = 100 - (100 / (1 + df['Close'].diff().clip(lower=0).rolling(14).mean() / 
                           df['Close'].diff().clip(upper=0).abs().rolling(14).mean()))
    df['MACD'] = df['Close'].ewm(span=12).mean() - df['Close'].ewm(span=26).mean()
    df['Volatility'] = df['Returns'].rolling(14).std() * np.sqrt(365)
    df['Target'] = (df['Returns'].shift(-1) > 0).astype(int)
    
    return df.dropna()

# ======================
# EFFICIENT MODEL PIPELINE
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
        """Stable feature selection using model-based approach"""
        self.feature_selector = SelectFromModel(
            RandomForestClassifier(n_estimators=100, random_state=42),
            threshold=FEATURE_SELECTION_THRESHOLD
        )
        self.feature_selector.fit(X, y)
        self.selected_features = X.columns[self.feature_selector.get_support()]
        return X[self.selected_features]

    def optimize_model(self, X, y, model_type):
        """Optimization with early stopping and time-series validation"""
        def objective(trial):
            model = self._create_model(trial, model_type)
            tscv = TimeSeriesSplit(CV_SPLITS)
            scores = []
            
            for train_idx, test_idx in tscv.split(X):
                X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
                y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
                
                # Handle class imbalance
                X_res, y_res = self.smote.fit_resample(X_train, y_train)
                
                model.fit(X_res, y_res)
                preds = model.predict(X_test)
                scores.append(precision_score(y_test, preds))
                
            return np.mean(scores)
        
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

    def _create_model(self, trial, model_type):
        """Dynamic model creation with optimized parameter ranges"""
        if model_type == 'rf':
            return RandomForestClassifier(
                n_estimators=trial.suggest_int('n_estimators', 200, 600),
                max_depth=trial.suggest_int('max_depth', 10, 30),
                min_samples_split=trial.suggest_int('min_samples_split', 5, 20),
                class_weight='balanced',
                random_state=42
            )
        return GradientBoostingClassifier(
            n_estimators=trial.suggest_int('n_estimators', 200, 600),
            learning_rate=trial.suggest_float('learning_rate', 0.05, 0.3),
            max_depth=trial.suggest_int('max_depth', 3, 10),
            subsample=trial.suggest_float('subsample', 0.7, 1.0),
            random_state=42
        )

    def _update_calibrator(self, X, y, model_type):
        """Calibrate model on hold-out set"""
        X_train, X_cal, y_train, y_cal = train_test_split(
            X, y, test_size=0.2, shuffle=False
        )
        model = self._create_final_model(model_type)
        model.fit(X_train, y_train)
        
        self.calibrators[model_type] = CalibratedClassifierCV(
            model, 
            method='isotonic', 
            cv='prefit'
        ).fit(X_cal, y_cal)

    def _create_final_model(self, model_type):
        """Create final model instance with best parameters"""
        params = self.models[model_type]
        if model_type == 'rf':
            return RandomForestClassifier(**params)
        return GradientBoostingClassifier(**params)

    def _progress_callback(self, model_type):
        """Real-time progress updates with proper completion handling"""
        def callback(study, trial):
            if st.session_state.training_progress:
                progress = (trial.number + 1) / MAX_TRIALS
                text = (f"{model_type.upper()} Trial {trial.number + 1}/{MAX_TRIALS} - "
                        f"Best Precision: {study.best_value:.2%}")
                st.session_state.training_progress.progress(progress, text=text)
                
                if trial.number == MAX_TRIALS - 1:
                    time.sleep(1)
                    st.session_state.training_progress.progress(1.0, 
                        f"{model_type.upper()} Optimization Complete!")
        return callback

    def predict(self, X):
        """Ensemble prediction from both models"""
        rf_proba = self.calibrators['rf'].predict_proba(X)[:, 1]
        gb_proba = self.calibrators['gb'].predict_proba(X)[:, 1]
        return 0.6 * rf_proba + 0.4 * gb_proba

# ======================
# STREAMLIT INTERFACE
# ======================
def main():
    df = get_data()
    
    # Display market data
    col1, col2 = st.columns([3, 1])
    with col1:
        st.subheader("Price Chart")
        st.line_chart(df['Close'])
    with col2:
        st.metric("Current Price", f"${df['Close'].iloc[-1]:.2f}")
        st.metric("Volatility", f"{df['Volatility'].iloc[-1]:.2%}")

    # Model controls
    if st.sidebar.button("🚀 Train Model"):
        if len(df) < 100:
            st.error("Insufficient data for training")
            return
            
        try:
            st.session_state.training_progress = st.progress(0, text="Initializing...")
            model = TradingModel()
            
            X = df.drop(['Target', 'Returns'], axis=1)
            y = df['Target']
            
            # Feature selection
            with st.spinner("Selecting important features..."):
                X_sel = model._feature_selection(X, y)
            
            # Optimize models
            with st.spinner("Optimizing Random Forest..."):
                model.optimize_model(X_sel, y, 'rf')
                
            with st.spinner("Optimizing Gradient Boosting..."):
                model.optimize_model(X_sel, y, 'gb')
            
            st.session_state.model = model
            st.success(f"Model trained successfully! Best Precision: {model.best_score:.2%}")
            
        except Exception as e:
            st.error(f"Training failed: {str(e)}")
        finally:
            time.sleep(1)
            st.session_state.training_progress = None

    # Trading signals
    if st.session_state.model:
        try:
            X = df.drop(['Target', 'Returns'], axis=1)
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
