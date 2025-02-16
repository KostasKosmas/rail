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
from sklearn.metrics import accuracy_score, f1_score
from sklearn.utils.class_weight import compute_sample_weight
from imblearn.over_sampling import SMOTE
from datetime import datetime, timedelta
import warnings
import talib

# ======================
# CONFIGURATION
# ======================
SYMBOL = 'BTC-USD'
PRIMARY_INTERVAL = '15m'
FALLBACK_INTERVAL = '60m'
LOOKBACK_DAYS = 59  # Max for 15m data
FALLBACK_LOOKBACK = 180  # For 60m data
TRADE_THRESHOLD_BUY = 0.58
TRADE_THRESHOLD_SELL = 0.42
GARCH_WINDOW = 7
MAX_TRIALS = 20
INITIAL_FEATURES = [
    'SMA_20', 'SMA_50', 'SMA_200', 'RSI', 'MACD', 'MACD_Signal',
    'BB_upper', 'BB_middle', 'BB_lower', 'ATR', 'VWAP', 'Volume_MA_20',
    'Volatility', 'OBV'
]

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
    max_days = 60 if interval in ['15m', '30m'] else 730
    lookback = min(LOOKBACK_DAYS, max_days-1)
    
    try:
        df = yf.download(
            symbol, 
            start=end - timedelta(days=lookback),
            end=end,
            interval=interval,
            progress=False,
            auto_adjust=False
        )
        if df.empty: raise ValueError("Empty dataframe")
        st.session_state.data_warning = None
        return df
    except Exception as e:
        logging.warning(f"Primary interval failed: {str(e)}")
        try:
            df = yf.download(
                symbol,
                start=end - timedelta(days=FALLBACK_LOOKBACK),
                end=end,
                interval=FALLBACK_INTERVAL,
                progress=False,
                auto_adjust=False
            )
            st.session_state.data_warning = (
                f"Using {FALLBACK_INTERVAL} data (last {FALLBACK_LOOKBACK} days) "
                f"due to {interval} availability limits"
            )
            return df
        except Exception as fallback_error:
            logging.error(f"Fallback failed: {str(fallback_error)}")
            st.error("Data unavailable for all intervals")
            return pd.DataFrame()

def calculate_features(df: pd.DataFrame) -> pd.DataFrame:
    """Advanced feature engineering with technical indicators"""
    df = df.copy()
    try:
        # Price features
        df['SMA_20'] = df['Close'].rolling(20).mean()
        df['SMA_50'] = df['Close'].rolling(50).mean()
        df['SMA_200'] = df['Close'].rolling(200).mean()
        df['Returns'] = df['Close'].pct_change()
        
        # Bollinger Bands
        df['BB_upper'], df['BB_middle'], df['BB_lower'] = talib.BBANDS(
            df['Close'], timeperiod=20
        )
        
        # Momentum indicators
        df['RSI'] = talib.RSI(df['Close'], timeperiod=14)
        macd, macdsignal, _ = talib.MACD(
            df['Close'], fastperiod=12, slowperiod=26, signalperiod=9
        )
        df['MACD'] = macd
        df['MACD_Signal'] = macdsignal
        
        # Volatility features
        df['ATR'] = talib.ATR(
            df['High'], df['Low'], df['Close'], timeperiod=14
        )
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
        
        # Volume features
        df['VWAP'] = (df['Volume'] * (df['High'] + df['Low'] + df['Close']) / 3).cumsum() / df['Volume'].cumsum()
        df['OBV'] = talib.OBV(df['Close'], df['Volume'])
        df['Volume_MA_20'] = df['Volume'].rolling(20).mean()
        
        # Target encoding
        df['Target'] = (df['Returns'].shift(-1) > 0).astype(int)
        
        return df[INITIAL_FEATURES + ['Target']].dropna()
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
        self.model_rf = RandomForestClassifier(
            warm_start=True,
            class_weight='balanced',
            random_state=42
        )
        self.model_gb = GradientBoostingClassifier(random_state=42)
        self.calibrated_rf = None
        self.calibrated_gb = None
        self.progress = None
        self.smote = SMOTE(random_state=42)

    def _progress_callback(self, study, trial):
        if self.progress:
            self.progress.progress(
                (trial.number + 1) / MAX_TRIALS,
                text=f"Trial {trial.number + 1}/{MAX_TRIALS} - Best: {study.best_value:.2%}"
            )

    def optimize_hyperparameters(self, X: pd.DataFrame, y: pd.Series):
        def objective_rf(trial):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 200, 1000),
                'max_depth': trial.suggest_int('max_depth', 10, 50),
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
                'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),
                'class_weight': trial.suggest_categorical('class_weight', ['balanced', 'balanced_subsample'])
            }
            return self._cross_val_score(RandomForestClassifier(**params), X, y)

        def objective_gb(trial):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 200, 1000),
                'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.3, log=True),
                'max_depth': trial.suggest_int('max_depth', 3, 20),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 20)
            }
            return self._cross_val_score(GradientBoostingClassifier(**params), X, y)

        # Optimize RF
        study_rf = optuna.create_study(direction='maximize')
        study_rf.optimize(objective_rf, n_trials=MAX_TRIALS, 
                        callbacks=[self._progress_callback])
        self.model_rf.set_params(**study_rf.best_params)

        # Optimize GB
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
            
            # Feature selection and SMOTE only on training data
            if not self.selected_features:
                self.feature_selector = SelectFromModel(
                    RandomForestClassifier(n_estimators=100),
                    threshold="median"
                )
                X_train_sel = self.feature_selector.fit_transform(X_train, y_train)
                self.selected_features = X_train.columns[self.feature_selector.get_support()]
            else:
                X_train_sel = X_train[self.selected_features]
            
            # Handle class imbalance
            X_res, y_res = self.smote.fit_resample(X_train_sel, y_train)
            
            model.fit(X_res, y_res)
            X_test_sel = X_test[self.selected_features]
            scores.append(f1_score(y_test, model.predict(X_test_sel)))
            
        return np.mean(scores)

    def train(self, X: pd.DataFrame, y: pd.Series):
        try:
            # Final feature selection
            self.feature_selector = SelectFromModel(
                RandomForestClassifier(n_estimators=100),
                threshold="median"
            )
            X_sel = self.feature_selector.fit_transform(X, y)
            self.selected_features = X.columns[self.feature_selector.get_support()]
            
            # Handle class imbalance
            X_res, y_res = self.smote.fit_resample(X_sel, y)
            
            # Train models
            self.model_rf.fit(X_res, y_res)
            self.model_gb.fit(X_res, y_res)

            # Probability calibration
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
            self.calibrated_rf.fit(X_res, y_res)
            self.calibrated_gb.fit(X_res, y_res)
            
        except Exception as e:
            logging.error(f"Training failed: {str(e)}")
            raise

    def predict(self, X: pd.DataFrame) -> float:
        try:
            X_sel = X[self.selected_features]
            prob_rf = self.calibrated_rf.predict_proba(X_sel)[:, 1]
            prob_gb = self.calibrated_gb.predict_proba(X_sel)[:, 1]
            return 0.6 * prob_rf + 0.4 * prob_gb
        except KeyError as e:
            missing = list(set(self.selected_features) - set(X.columns))
            logging.error(f"Missing features: {missing}")
            raise ValueError(f"Required features missing: {missing}")

# ======================
# STREAMLIT INTERFACE
# ======================
def main():
    # Data section
    raw_data = fetch_data(SYMBOL, PRIMARY_INTERVAL)
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
            except Exception as e:
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
            
            X = processed_data.drop(['Target'], axis=1, errors='ignore')
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
        try:
            required_features = st.session_state.model.selected_features
            latest_features = processed_data[required_features].iloc[[-1]]
            
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
            st.error(f"Prediction error: {str(e)}")

if __name__ == "__main__":
    main()
