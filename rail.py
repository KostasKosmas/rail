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
GARCH_WINDOW = 7
MAX_TRIALS = 20
MAX_RETRIES = 3

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
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
# DATA PIPELINE
# ======================
@st.cache_data(ttl=300, show_spinner="Fetching market data...")
def fetch_data(symbol: str, interval: str) -> pd.DataFrame:
    """Smart data fetching with fallback mechanism"""
    end = datetime.now()
    max_days = 60 if interval in ['15m', '30m'] else 730
    lookback = max_days - 2  # Add buffer for timezone issues
    
    for attempt in range(MAX_RETRIES):
        try:
            df = yf.download(
                symbol, 
                start=end - timedelta(days=lookback),
                end=end,
                interval=interval,
                progress=False,
                auto_adjust=True
            )
            if not df.empty:
                st.session_state.data_warning = None
                return df
        except Exception as e:
            logging.warning(f"Attempt {attempt+1} failed: {str(e)}")
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
        return df
    except Exception as fallback_error:
        logging.error(f"Fallback failed: {str(fallback_error)}")
        st.error(f"Data unavailable for {symbol}")
        return pd.DataFrame()

def calculate_features(df: pd.DataFrame) -> pd.DataFrame:
    """Feature engineering with validation"""
    if df.empty or len(df) < 20:
        return pd.DataFrame()
        
    df = df.copy()
    try:
        # Price features
        df['Returns'] = df['Close'].pct_change()
        
        # Technical indicators
        for window in [20, 50, 200]:
            df[f'SMA_{window}'] = df['Close'].rolling(window).mean()
        
        # RSI Calculation
        delta = df['Close'].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        rs = (gain.rolling(14).mean() / 
              loss.rolling(14).mean()).replace(np.inf, 100)
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # MACD
        df['MACD'] = df['Close'].ewm(span=12).mean() - df['Close'].ewm(span=26).mean()
        df['MACD_Signal'] = df['MACD'].ewm(span=9).mean()
        
        # Bollinger Bands
        df['BB_MA20'] = df['Close'].rolling(20).mean()
        df['BB_STD20'] = df['Close'].rolling(20).std()
        df['BB_Upper'] = df['BB_MA20'] + (df['BB_STD20'] * 2)
        df['BB_Lower'] = df['BB_MA20'] - (df['BB_STD20'] * 2)
        
        # Volume
        df['OBV'] = (np.sign(df['Close'].diff()) * df['Volume']).cumsum()
        df['Volume_MA_20'] = df['Volume'].rolling(20).mean()
        
        # Volatility
        returns = df['Returns'].dropna()
        if len(returns) > 100:
            try:
                model = arch_model(returns * 100, vol='Garch', p=1, q=1)
                res = model.fit(disp='off')
                df['Volatility'] = res.conditional_volatility / 100
            except:
                df['Volatility'] = returns.rolling(GARCH_WINDOW).std()
        else:
            df['Volatility'] = returns.rolling(GARCH_WINDOW).std()
        
        # Target
        df['Target'] = (df['Returns'].shift(-1) > 0).astype(int)
        
        return df.dropna()
    except Exception as e:
        logging.error(f"Feature engineering failed: {str(e)}")
        return pd.DataFrame()

# ======================
# MODEL PIPELINE
# ======================
class TradingModel:
    def __init__(self):
        self.feature_selector = None
        self.selected_features = []
        self.model_rf = RandomForestClassifier(class_weight='balanced', random_state=42)
        self.model_gb = GradientBoostingClassifier(random_state=42)
        self.calibrated_rf = None
        self.calibrated_gb = None
        self.smote = SMOTE(random_state=42)
        self.study_rf = None
        self.study_gb = None

    def _progress_callback(self, study, trial):
        if st.session_state.training_progress:
            model_type = "RF" if study == self.study_rf else "GB"
            progress = (trial.number + 1) / MAX_TRIALS
            text = (f"{model_type} Trial {trial.number+1}/{MAX_TRIALS} - "
                    f"Best F1: {study.best_value:.2%}")
            st.session_state.training_progress.progress(progress, text=text)

    def optimize_hyperparameters(self, X: pd.DataFrame, y: pd.Series):
        # Feature selection
        self.feature_selector = SelectFromModel(
            RandomForestClassifier(n_estimators=100),
            threshold="median"
        )
        X_sel = self.feature_selector.fit_transform(X, y)
        self.selected_features = X.columns[self.feature_selector.get_support()]
        
        # RF optimization
        self.study_rf = optuna.create_study(direction='maximize')
        self.study_rf.optimize(
            lambda trial: self._objective(trial, X_sel, y, 'rf'), 
            n_trials=MAX_TRIALS,
            callbacks=[self._progress_callback]
        )
        self.model_rf.set_params(**self.study_rf.best_params)
        
        # GB optimization
        self.study_gb = optuna.create_study(direction='maximize')
        self.study_gb.optimize(
            lambda trial: self._objective(trial, X_sel, y, 'gb'), 
            n_trials=MAX_TRIALS,
            callbacks=[self._progress_callback]
        )
        self.model_gb.set_params(**self.study_gb.best_params)

    def _objective(self, trial, X, y, model_type):
        if model_type == 'rf':
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 200, 800),
                'max_depth': trial.suggest_int('max_depth', 10, 30),
                'min_samples_split': trial.suggest_int('min_samples_split', 5, 20),
                'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2'])
            }
            model = RandomForestClassifier(**params)
        else:
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 200, 800),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'subsample': trial.suggest_float('subsample', 0.7, 1.0)
            }
            model = GradientBoostingClassifier(**params)
        
        tscv = TimeSeriesSplit(3)
        scores = []
        for train_idx, test_idx in tscv.split(X):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            
            X_res, y_res = self.smote.fit_resample(X_train, y_train)
            model.fit(X_res, y_res)
            scores.append(f1_score(y_test, model.predict(X_test)))
        
        return np.mean(scores)

    def train(self, X: pd.DataFrame, y: pd.Series):
        X_sel = X[self.selected_features]
        X_res, y_res = self.smote.fit_resample(X_sel, y)
        
        self.model_rf.fit(X_res, y_res)
        self.model_gb.fit(X_res, y_res)
        
        # Calibration
        self.calibrated_rf = CalibratedClassifierCV(self.model_rf, cv=TimeSeriesSplit(3))
        self.calibrated_gb = CalibratedClassifierCV(self.model_gb, cv=TimeSeriesSplit(3))
        self.calibrated_rf.fit(X_sel, y)
        self.calibrated_gb.fit(X_sel, y)

    def predict(self, X: pd.DataFrame) -> float:
        X_sel = X[self.selected_features]
        prob_rf = self.calibrated_rf.predict_proba(X_sel)[:, 1]
        prob_gb = self.calibrated_gb.predict_proba(X_sel)[:, 1]
        return 0.6 * prob_rf + 0.4 * prob_gb

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
                current_price = processed_data['Close'].iloc[-1]
                current_vol = processed_data['Volatility'].iloc[-1]
                st.metric("Current Price", f"${current_price:,.2f}")
                st.metric("Volatility", f"{current_vol:.2%}")
            except IndexError:
                st.error("Insufficient data points")
            except Exception as e:
                st.error(f"Data display error: {str(e)}")

    # Model training
    st.sidebar.header("Model Controls")
    if st.sidebar.button("🚀 Train Model"):
        if processed_data.empty or 'Target' not in processed_data.columns:
            st.error("Invalid data for training")
            return
            
        try:
            st.session_state.training_progress = st.progress(0, text="Initializing...")
            model = TradingModel()
            
            X = processed_data.drop(['Target', 'Returns'], axis=1, errors='ignore')
            y = processed_data['Target']
            
            with st.spinner("Optimizing Models..."):
                model.optimize_hyperparameters(X, y)
                model.train(X, y)
            
            st.session_state.model = model
            st.session_state.last_trained = datetime.now()
            st.success(f"Model trained! Best F1: {max(model.study_rf.best_value, model.study_gb.best_value):.2%}")
            
        except Exception as e:
            st.error(f"Training failed: {str(e)}")
        finally:
            time.sleep(1)
            st.session_state.training_progress = None

    # Trading signals
    if st.session_state.model and not processed_data.empty:
        try:
            latest_data = processed_data[st.session_state.model.selected_features].iloc[[-1]]
            confidence = st.session_state.model.predict(latest_data)[0]
            
            st.subheader("Trading Signal")
            col1, col2, col3 = st.columns(3)
            col1.metric("Confidence", f"{confidence:.2%}")
            
            if confidence > TRADE_THRESHOLD_BUY:
                col2.success("🚀 Strong Buy")
                col3.write(f"Threshold: >{TRADE_THRESHOLD_BUY:.0%}")
            elif confidence < TRADE_THRESHOLD_SELL:
                col2.error("🔻 Strong Sell")
                col3.write(f"Threshold: <{TRADE_THRESHOLD_SELL:.0%}")
            else:
                col2.info("🛑 Hold")
                col3.write("No clear signal")
            
            if st.session_state.last_trained:
                st.caption(f"Last trained: {st.session_state.last_trained.strftime('%Y-%m-%d %H:%M')}")
                
        except Exception as e:
            st.error(f"Prediction error: {str(e)}")

if __name__ == "__main__":
    main()
