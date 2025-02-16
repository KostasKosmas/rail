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
from sklearn.metrics import accuracy_score
from sklearn.utils.class_weight import compute_sample_weight
from datetime import datetime, timedelta
import warnings

# ======================
# CONFIGURATION
# ======================
SYMBOL = 'BTC-USD'
INTERVAL = '15m'  # Primary interval
FALLBACK_INTERVAL = '60m'  # Fallback if 15m fails
LOOKBACK_DAYS = 59  # Max for 15m (Yahoo limit: 60 days)
FALLBACK_LOOKBACK_DAYS = 180  # For 60m data
TRADE_THRESHOLD_BUY = 0.55
TRADE_THRESHOLD_SELL = 0.45
GARCH_WINDOW = 7
MAX_TRIALS = 15

# Interval validation mapping
INTERVAL_LIMITS = {
    '15m': 60,
    '30m': 60,
    '60m': 730,
    '1d': 365*5
}

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning, module="arch")
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
    
    # Validate interval and lookback days
    max_days = INTERVAL_LIMITS.get(interval, 60)
    lookback = min(LOOKBACK_DAYS, max_days-1)
    start = end - timedelta(days=lookback)
    
    try:
        df = yf.download(
            symbol, 
            start=start, 
            end=end, 
            interval=interval, 
            progress=False,
            auto_adjust=False
        )
        
        if df.empty:
            raise ValueError("Empty dataframe received")
            
        st.session_state.data_warning = None
        return df
        
    except Exception as e:
        logging.warning(f"Primary interval failed: {str(e)}")
        
        # Fallback to longer interval
        fallback_days = FALLBACK_LOOKBACK_DAYS
        start_fallback = end - timedelta(days=fallback_days)
        
        try:
            df = yf.download(
                symbol,
                start=start_fallback,
                end=end,
                interval=FALLBACK_INTERVAL,
                progress=False,
                auto_adjust=False
            )
            
            st.session_state.data_warning = (
                f"Using {FALLBACK_INTERVAL} data (last {fallback_days} days) "
                f"due to {interval} availability limits"
            )
            return df
            
        except Exception as fallback_error:
            logging.error(f"Fallback failed: {str(fallback_error)}")
            st.error("Data unavailable for all intervals")
            return pd.DataFrame()

def calculate_features(df: pd.DataFrame) -> pd.DataFrame:
    """Adaptive feature engineering"""
    df = df.copy()
    try:
        # Price features
        df['SMA_50'] = df['Close'].rolling(50).mean()
        df['SMA_200'] = df['Close'].rolling(200).mean()
        df['Returns'] = df['Close'].pct_change()
        
        # Momentum indicators
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = np.where(loss != 0, gain / loss, 1)
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # Volatility calculation
        returns = df['Returns'].dropna()
        if len(returns) > 100:
            try:
                scaled_returns = returns * 1000
                garch = arch_model(scaled_returns, vol='Garch', p=1, q=1, rescale=False)
                garch_fit = garch.fit(disp='off', options={'maxiter': 500})
                df['Volatility'] = garch_fit.conditional_volatility / 1000 * 100
            except Exception:
                df['Volatility'] = returns.rolling(GARCH_WINDOW).std() * 100
        else:
            df['Volatility'] = returns.rolling(GARCH_WINDOW).std() * 100
        
        # Target encoding
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
        self.model_rf = RandomForestClassifier(
            warm_start=True,
            class_weight='balanced',
            random_state=42
        )
        self.model_gb = GradientBoostingClassifier(random_state=42)
        self.calibrated_rf = None
        self.calibrated_gb = None
        self.progress = None

    def _progress_callback(self, study, trial):
        if self.progress:
            self.progress.progress(
                (trial.number + 1) / MAX_TRIALS,
                text=f"Trial {trial.number + 1}/{MAX_TRIALS} - Best: {study.best_value:.2%}"
            )

    def optimize_hyperparameters(self, X: pd.DataFrame, y: pd.Series):
        def objective_rf(trial):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 200, 800),
                'max_depth': trial.suggest_int('max_depth', 10, 40),
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 15),
                'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2'])
            }
            return self._cross_val_score(RandomForestClassifier(**params), X, y)

        def objective_gb(trial):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 200, 800),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2),
                'max_depth': trial.suggest_int('max_depth', 3, 15),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0)
            }
            return self._cross_val_score(GradientBoostingClassifier(**params), X, y)

        study_rf = optuna.create_study(direction='maximize')
        study_rf.optimize(objective_rf, n_trials=MAX_TRIALS, 
                        callbacks=[self._progress_callback])
        self.model_rf.set_params(**study_rf.best_params)

        study_gb = optuna.create_study(direction='maximize')
        study_gb.optimize(objective_gb, n_trials=MAX_TRIALS,
                        callbacks=[self._progress_callback])
        self.model_gb.set_params(**study_gb.best_params)

    def _cross_val_score(self, model, X, y) -> float:
        tscv = TimeSeriesSplit(n_splits=3)
        scores = []
        
        for train_idx, test_idx in tscv.split(X):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            
            if self.feature_selector is None:
                self.feature_selector = SelectFromModel(
                    RandomForestClassifier(n_estimators=100),
                    threshold="median"
                )
                self.feature_selector.fit(X_train, y_train)
            
            X_train_sel = self.feature_selector.transform(X_train)
            X_test_sel = self.feature_selector.transform(X_test)
            
            sample_weights = compute_sample_weight('balanced', y_train)
            
            model.fit(X_train_sel, y_train, sample_weight=sample_weights)
            scores.append(accuracy_score(y_test, model.predict(X_test_sel)))
            
        return np.mean(scores)

    def train(self, X: pd.DataFrame, y: pd.Series):
        try:
            self.feature_selector = SelectFromModel(
                RandomForestClassifier(n_estimators=100),
                threshold="median"
            )
            self.feature_selector.fit(X, y)
            X_sel = self.feature_selector.transform(X)

            sample_weights = compute_sample_weight('balanced', y)
            self.model_rf.fit(X_sel, y, sample_weight=sample_weights)
            self.model_gb.fit(X_sel, y, sample_weight=sample_weights)

            self.calibrated_rf = CalibratedClassifierCV(
                self.model_rf,
                method='isotonic',
                cv=TimeSeriesSplit(3)
            )
            self.calibrated_gb = CalibratedClassifierCV(
                self.model_gb,
                method='sigmoid',
                cv=TimeSeriesSplit(3)
            )
            self.calibrated_rf.fit(X_sel, y)
            self.calibrated_gb.fit(X_sel, y)
            
        except Exception as e:
            logging.error(f"Training failed: {str(e)}")
            raise

    def predict(self, X: pd.DataFrame) -> float:
        X_sel = self.feature_selector.transform(X)
        prob_rf = self.calibrated_rf.predict_proba(X_sel)[:, 1]
        prob_gb = self.calibrated_gb.predict_proba(X_sel)[:, 1]
        return 0.6 * prob_rf + 0.4 * prob_gb

# ======================
# STREAMLIT INTERFACE
# ======================
def main():
    # Data section
    raw_data = fetch_data(SYMBOL, INTERVAL)
    processed_data = calculate_features(raw_data)
    
    if st.session_state.data_warning:
        st.warning(st.session_state.data_warning)
    
    if not processed_data.empty:
        col1, col2 = st.columns([3, 1])
        with col1:
            st.subheader("Price Chart")
            st.line_chart(processed_data['Close'])
            
        with col2:
            try:
                current_price = processed_data['Close'].iloc[-1].item()
                current_vol = processed_data['Volatility'].iloc[-1].item()
                st.metric("Current Price", f"${current_price:,.2f}")
                st.metric("24h Volatility", f"{current_vol:.2f}%")
            except (IndexError, KeyError, ValueError) as e:
                st.error(f"Data display error: {str(e)}")

    # Model training
    st.sidebar.header("Model Controls")
    if st.sidebar.button("🚀 Train New Model"):
        if processed_data.empty:
            st.error("No data available for training!")
            return
            
        try:
            st.session_state.training_progress = st.progress(0, text="Initializing...")
            model = TradingModel()
            model.progress = st.session_state.training_progress
            
            X = processed_data.drop(['Target', 'Returns'], axis=1, errors='ignore')
            y = processed_data['Target']
            
            with st.spinner("Optimizing Random Forest..."):
                model.optimize_hyperparameters(X, y)
                
            with st.spinner("Finalizing Calibration..."):
                model.train(X, y)
                
            st.session_state.model = model
            st.session_state.last_trained = datetime.now()
            st.session_state.training_progress.progress(1.0, "Training complete!")
            time.sleep(1)
            st.session_state.training_progress = None
            st.success("Model trained successfully!")
            
        except Exception as e:
            st.error(f"Training failed: {str(e)}")
            if st.session_state.training_progress:
                st.session_state.training_progress = None

    # Trading signals
    if st.session_state.model and not processed_data.empty:
        latest_features = processed_data.drop(
            ['Target', 'Returns'], axis=1, errors='ignore'
        ).iloc[[-1]]
        
        try:
            prediction = st.session_state.model.predict(latest_features)[0]
            st.subheader("Trading Signal")
            
            col1, col2, col3 = st.columns(3)
            col1.metric("Confidence Score", f"{prediction:.2%}")
            
            if prediction > TRADE_THRESHOLD_BUY:
                col2.success("🚦 Strong Buy Signal")
                col3.write(f"Threshold: >{TRADE_THRESHOLD_BUY:.0%}")
            elif prediction < TRADE_THRESHOLD_SELL:
                col2.error("🚦 Strong Sell Signal")
                col3.write(f"Threshold: <{TRADE_THRESHOLD_SELL:.0%}")
            else:
                col2.info("🛑 Hold Position")
                col3.write("No clear market signal")
                
            st.caption(f"Last trained: {st.session_state.last_trained.strftime('%Y-%m-%d %H:%M')}")
            
        except Exception as e:
            st.error(f"Prediction failed: {str(e)}")

if __name__ == "__main__":
    main()
