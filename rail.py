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
from sklearn.metrics import f1_score
from sklearn.utils.class_weight import compute_sample_weight
from imblearn.over_sampling import SMOTE
from datetime import datetime, timedelta
import warnings

# ======================
# CONFIGURATION
# ======================
DEFAULT_SYMBOL = 'BTC-USD'
PRIMARY_INTERVAL = '15m'
FALLBACK_INTERVAL = '60m'
TRADE_THRESHOLD_BUY = 0.58
TRADE_THRESHOLD_SELL = 0.42
MAX_TRIALS = 15
GARCH_WINDOW = 14
DATA_RETRIES = 3
MIN_FEATURES = 5  # Minimum number of features to select

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning)
logging.basicConfig(level=logging.INFO)

# ======================
# STREAMLIT UI
# ======================
st.set_page_config(page_title="AI Trading System", layout="wide")
st.title("🚀 AI-Powered Cryptocurrency Trading System")
st.markdown("Real-time trading signals with machine learning and volatility modeling")

# Initialize session state
if 'model' not in st.session_state:
    st.session_state.model = None
if 'last_trained' not in st.session_state:
    st.session_state.last_trained = None
if 'training_progress' not in st.session_state:
    st.session_state.training_progress = None
if 'data_warning' not in st.session_state:
    st.session_state.data_warning = None

# ======================
# ROBUST DATA PIPELINE
# ======================
@st.cache_data(ttl=300, show_spinner="Fetching market data...")
def fetch_data(symbol: str, interval: str) -> pd.DataFrame:
    """Smart data fetching with dynamic fallback mechanism"""
    end = datetime.now()
    lookback_days = 59 if interval in ['15m', '30m'] else 180
    
    for _ in range(DATA_RETRIES):
        try:
            df = yf.download(
                symbol, 
                start=end - timedelta(days=lookback_days),
                end=end,
                interval=interval,
                progress=False,
                auto_adjust=True
            )
            if not df.empty and len(df) > 100:
                st.session_state.data_warning = None
                return df
        except Exception as e:
            logging.warning(f"Download attempt failed: {str(e)}")
            time.sleep(1)
    
    try:  # Fallback to daily data
        df = yf.download(
            symbol,
            start=end - timedelta(days=365),
            end=end,
            interval=FALLBACK_INTERVAL,
            progress=False,
            auto_adjust=True
        )
        st.session_state.data_warning = (
            f"Using {FALLBACK_INTERVAL} data due to {interval} availability limits"
        )
        return df[df.index >= end - timedelta(days=365)] if not df.empty else pd.DataFrame()
    except Exception as e:
        st.error(f"Data unavailable for {symbol}")
        return pd.DataFrame()

def calculate_features(df: pd.DataFrame) -> pd.DataFrame:
    """Feature engineering with enhanced validation"""
    if df.empty or len(df) < 100:
        return pd.DataFrame()
        
    df = df.copy()
    try:
        # Price features
        df['Returns'] = df['Close'].pct_change()
        
        # Technical indicators
        windows = [20, 50, 200]
        for window in windows:
            df[f'SMA_{window}'] = df['Close'].rolling(window).mean()
        
        # RSI with error handling
        delta = df['Close'].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.rolling(14).mean()
        avg_loss = loss.rolling(14).mean().replace(0, 1e-6)
        df['RSI'] = 100 - (100 / (1 + (avg_gain / avg_loss)))
        df['RSI'] = df['RSI'].clip(0, 100)
        
        # MACD
        df['MACD'] = df['Close'].ewm(span=12).mean() - df['Close'].ewm(span=26).mean()
        df['MACD_Signal'] = df['MACD'].ewm(span=9).mean()
        
        # Volatility with dynamic scaling
        returns = df['Returns'].dropna()
        if not returns.empty:
            scale_factor = 100 / (returns.std() or 1e-6)
            scaled_returns = returns * scale_factor
            
            try:
                garch = arch_model(scaled_returns, vol='Garch', p=1, q=1, rescale=False)
                garch_fit = garch.fit(disp='off', options={'maxiter': 500})
                df['Volatility'] = garch_fit.conditional_volatility / scale_factor
            except Exception:
                df['Volatility'] = returns.rolling(GARCH_WINDOW).std()
        
        # Target encoding
        df['Target'] = (df['Returns'].shift(-1) > 0).astype(int)
        
        return df.dropna()
    except Exception as e:
        logging.error(f"Feature engineering failed: {str(e)}")
        return pd.DataFrame()

# ======================
# ROBUST MODEL PIPELINE
# ======================
class TradingModel:
    def __init__(self):
        self.feature_selector = None
        self.selected_features = []
        self.model_rf = RandomForestClassifier(class_weight='balanced', random_state=42)
        self.model_gb = GradientBoostingClassifier(random_state=42)
        self.calibrated_rf = None
        self.calibrated_gb = None
        self.smote = SMOTE(random_state=42, k_neighbors=3)  # Reduced neighbors
        self.study = None
        self.best_score = 0

    def _progress_callback(self, study, trial):
        """Progress tracking with error handling"""
        if st.session_state.training_progress:
            progress = (trial.number + 1) / MAX_TRIALS
            text = (f"Trial {trial.number + 1}/{MAX_TRIALS} - "
                    f"Best F1: {study.best_value:.2%}")
            st.session_state.training_progress.progress(progress, text=text)

    def _safe_feature_selection(self, X, y):
        """Robust feature selection with fallbacks"""
        # First attempt with default threshold
        self.feature_selector = SelectFromModel(
            RandomForestClassifier(n_estimators=100),
            threshold="1.25*median"
        )
        self.feature_selector.fit(X, y)
        self.selected_features = X.columns[self.feature_selector.get_support()]
        
        # Fallback to lower threshold if no features selected
        if len(self.selected_features) < MIN_FEATURES:
            logging.warning("Initial feature selection failed, trying lower threshold")
            self.feature_selector.threshold = "median"
            self.feature_selector.fit(X, y)
            self.selected_features = X.columns[self.feature_selector.get_support()]
            
        # Final fallback to top 5 features
        if len(self.selected_features) < MIN_FEATURES:
            logging.warning("Using top 5 features as fallback")
            self.feature_selector = SelectFromModel(
                RandomForestClassifier(n_estimators=100),
                max_features=5
            )
            self.feature_selector.fit(X, y)
            self.selected_features = X.columns[self.feature_selector.get_support()]
        
        return X[self.selected_features]

    def optimize_models(self, X: pd.DataFrame, y: pd.Series):
        """Optimization pipeline with feature validation"""
        try:
            # Robust feature selection
            X_sel = self._safe_feature_selection(X, y)
            
            # Validate features
            if X_sel.shape[1] == 0:
                raise ValueError("No features available after selection")
                
            # Optimization study
            self.study = optuna.create_study(direction='maximize')
            self.study.optimize(
                lambda trial: self._objective(trial, X_sel, y),
                n_trials=MAX_TRIALS,
                callbacks=[self._progress_callback],
                timeout=1200
            )
        except Exception as e:
            logging.error(f"Optimization failed: {str(e)}")
            raise

    def _objective(self, trial, X, y):
        """Objective function with error handling"""
        try:
            # Random Forest parameters
            rf_params = {
                'n_estimators': trial.suggest_int('rf_n_estimators', 200, 800),
                'max_depth': trial.suggest_int('rf_max_depth', 10, 30),
                'min_samples_split': trial.suggest_int('rf_min_samples_split', 5, 20),
                'max_features': trial.suggest_categorical('rf_max_features', ['sqrt', 'log2'])
            }
            
            # Gradient Boosting parameters
            gb_params = {
                'n_estimators': trial.suggest_int('gb_n_estimators', 200, 800),
                'learning_rate': trial.suggest_float('gb_learning_rate', 0.01, 0.3, log=True),
                'max_depth': trial.suggest_int('gb_max_depth', 3, 10),
                'subsample': trial.suggest_float('gb_subsample', 0.7, 1.0)
            }
            
            tscv = TimeSeriesSplit(3)
            scores = []
            
            for train_idx, test_idx in tscv.split(X):
                X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
                y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
                
                # Handle class imbalance with validation
                if len(np.unique(y_train)) < 2:
                    raise ValueError("Insufficient class variety for SMOTE")
                    
                X_res, y_res = self.smote.fit_resample(X_train, y_train)
                
                # Train models
                rf = RandomForestClassifier(**rf_params).fit(X_res, y_res)
                gb = GradientBoostingClassifier(**gb_params).fit(X_res, y_res)
                
                # Ensemble predictions
                preds = (0.6 * rf.predict_proba(X_test)[:, 1] + 
                        0.4 * gb.predict_proba(X_test)[:, 1])
                scores.append(f1_score(y_test, (preds >= 0.5).astype(int)))
                
            return np.mean(scores)
        except Exception as e:
            logging.warning(f"Trial failed: {str(e)}")
            return 0.0  # Return minimum score for failed trials

    def train(self, X: pd.DataFrame, y: pd.Series):
        """Training pipeline with validation"""
        try:
            X_sel = X[self.selected_features]
            
            # Final SMOTE validation
            if len(np.unique(y)) < 2:
                raise ValueError("Insufficient class distribution for training")
                
            X_res, y_res = self.smote.fit_resample(X_sel, y)
            
            # Get best parameters
            best_params = self.study.best_params
            rf_params = {k[3:]: v for k, v in best_params.items() if k.startswith('rf_')}
            gb_params = {k[3:]: v for k, v in best_params.items() if k.startswith('gb_')}
            
            # Train final models
            self.model_rf.set_params(**rf_params).fit(X_res, y_res)
            self.model_gb.set_params(**gb_params).fit(X_res, y_res)
            
            # Calibration
            self.calibrated_rf = CalibratedClassifierCV(self.model_rf, cv=TimeSeriesSplit(3))
            self.calibrated_gb = CalibratedClassifierCV(self.model_gb, cv=TimeSeriesSplit(3))
            self.calibrated_rf.fit(X_sel, y)
            self.calibrated_gb.fit(X_sel, y)
            
        except Exception as e:
            logging.error(f"Training failed: {str(e)}")
            raise

    def predict(self, X: pd.DataFrame) -> float:
        """Safe prediction method"""
        try:
            X_sel = X[self.selected_features]
            prob_rf = self.calibrated_rf.predict_proba(X_sel)[:, 1]
            prob_gb = self.calibrated_gb.predict_proba(X_sel)[:, 1]
            return 0.6 * prob_rf + 0.4 * prob_gb
        except Exception as e:
            logging.error(f"Prediction failed: {str(e)}")
            return 0.5  # Neutral prediction on failure

# ======================
# STREAMLIT INTERFACE
# ======================
def main():
    # Symbol selection
    st.sidebar.header("Settings")
    symbol = st.sidebar.text_input("Cryptocurrency Symbol", DEFAULT_SYMBOL).upper()
    
    # Data section
    raw_data = fetch_data(symbol, PRIMARY_INTERVAL)
    processed_data = calculate_features(raw_data)
    
    if st.session_state.data_warning:
        st.warning(st.session_state.data_warning)
    
    if not processed_data.empty:
        col1, col2 = st.columns([3, 1])
        with col1:
            st.subheader(f"{symbol} Price Chart")
            st.line_chart(processed_data['Close'])
            
        with col2:
            try:
                current_price = processed_data['Close'].iloc[-1].item()
                current_vol = processed_data['Volatility'].iloc[-1].item()
                st.metric("Current Price", f"${current_price:,.2f}")
                st.metric("Volatility", f"{current_vol:.2%}")
            except Exception as e:
                st.error(f"Display error: {str(e)}")

    # Model training
    st.sidebar.header("Model Controls")
    if st.sidebar.button("🚀 Train Model"):
        if processed_data.empty or 'Target' not in processed_data.columns:
            st.error("Insufficient data for training")
            return
            
        try:
            st.session_state.training_progress = st.progress(0, text="Initializing...")
            model = TradingModel()
            
            X = processed_data.drop(['Target', 'Returns'], axis=1, errors='ignore')
            y = processed_data['Target']
            
            with st.spinner("Optimizing trading models..."):
                model.optimize_models(X, y)
                model.train(X, y)
            
            st.session_state.model = model
            st.session_state.last_trained = datetime.now()
            st.success(f"Model trained successfully! Best F1: {model.study.best_value:.2%}")
            
        except Exception as e:
            st.error(f"Training failed: {str(e)}")
        finally:
            time.sleep(1)
            if st.session_state.training_progress:
                st.session_state.training_progress = None

    # Trading signals
    if st.session_state.model and not processed_data.empty:
        try:
            latest_data = processed_data[st.session_state.model.selected_features].iloc[[-1]]
            confidence = st.session_state.model.predict(latest_data)
            
            st.subheader("Trading Signal")
            col1, col2, col3 = st.columns(3)
            col1.metric("Confidence Score", f"{confidence:.2%}")
            
            if confidence > TRADE_THRESHOLD_BUY:
                col2.success("🚀 Strong Buy Signal")
                col3.write(f"Threshold: >{TRADE_THRESHOLD_BUY:.0%}")
            elif confidence < TRADE_THRESHOLD_SELL:
                col2.error("🔻 Strong Sell Signal")
                col3.write(f"Threshold: <{TRADE_THRESHOLD_SELL:.0%}")
            else:
                col2.info("🛑 Hold Position")
                col3.write("No clear market signal")
            
            if st.session_state.last_trained:
                st.caption(f"Last trained: {st.session_state.last_trained.strftime('%Y-%m-%d %H:%M')}")
                
        except Exception as e:
            st.error(f"Prediction error: {str(e)}")

if __name__ == "__main__":
    main()
