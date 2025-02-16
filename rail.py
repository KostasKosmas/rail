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
from arch import ConvergenceWarning

# ======================
# CONFIGURATION
# ======================
SYMBOL = 'BTC-USD'
INTERVAL = '5m'
LOOKBACK_DAYS = 60
TRADE_THRESHOLD_BUY = 0.65
TRADE_THRESHOLD_SELL = 0.35
GARCH_WINDOW = 21

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning, module="arch")
warnings.filterwarnings("ignore", category=ConvergenceWarning)
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

# ======================
# DATA PIPELINE
# ======================
@st.cache_data(ttl=300, show_spinner="Fetching market data...")
def fetch_data(symbol: str, interval: str) -> pd.DataFrame:
    """Optimized data fetching with yfinance"""
    end = datetime.now()
    start = end - timedelta(days=LOOKBACK_DAYS)
    return yf.download(symbol, start=start, end=end, interval=interval, progress=False)

def calculate_features(df: pd.DataFrame) -> pd.DataFrame:
    """Robust feature engineering with fallback mechanisms"""
    df = df.copy()
    try:
        # Price features
        df['SMA_20'] = df['Close'].rolling(20).mean()
        df['SMA_50'] = df['Close'].rolling(50).mean()
        df['Returns'] = df['Close'].pct_change()
        
        # Momentum indicators
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = np.where(loss != 0, gain / loss, 1)
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # Volume features
        df['OBV'] = (np.sign(df['Close'].diff()) * df['Volume']).fillna(0).cumsum()
        df['Volume_MA_20'] = df['Volume'].rolling(20).mean()
        
        # MACD
        ema12 = df['Close'].ewm(span=12, adjust=False).mean()
        ema26 = df['Close'].ewm(span=26, adjust=False).mean()
        df['MACD'] = ema12 - ema26
        df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
        
        # Volatility with GARCH fallback
        returns = df['Returns'].dropna()
        if len(returns) > 100:
            try:
                scaled_returns = returns * 1000  # Fix scaling issues
                garch = arch_model(scaled_returns, vol='Garch', p=1, q=1, rescale=False)
                garch_fit = garch.fit(disp='off', options={'maxiter': 1000})
                df['Volatility'] = garch_fit.conditional_volatility / 1000
            except Exception:
                df['Volatility'] = returns.rolling(GARCH_WINDOW).std()
        else:
            df['Volatility'] = returns.rolling(GARCH_WINDOW).std()
        
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

    def optimize_hyperparameters(self, X: pd.DataFrame, y: pd.Series):
        """Bayesian optimization with Optuna"""
        def objective_rf(trial):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 100, 500),
                'max_depth': trial.suggest_int('max_depth', 5, 30),
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 10),
            }
            return self._cross_val_score(RandomForestClassifier(**params), X, y)

        def objective_gb(trial):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 100, 500),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                'max_depth': trial.suggest_int('max_depth', 3, 10),
            }
            return self._cross_val_score(GradientBoostingClassifier(**params), X, y)

        # Optimize RF
        study_rf = optuna.create_study(direction='maximize')
        study_rf.optimize(objective_rf, n_trials=20)
        self.model_rf.set_params(**study_rf.best_params)

        # Optimize GB
        study_gb = optuna.create_study(direction='maximize')
        study_gb.optimize(objective_gb, n_trials=20)
        self.model_gb.set_params(**study_gb.best_params)

    def _cross_val_score(self, model, X, y) -> float:
        """Time-series aware cross-validation"""
        tscv = TimeSeriesSplit(n_splits=3)
        scores = []
        
        for train_idx, test_idx in tscv.split(X):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            
            # Feature selection
            if self.feature_selector is None:
                self.feature_selector = SelectFromModel(
                    RandomForestClassifier(n_estimators=100),
                    threshold="median"
                )
                self.feature_selector.fit(X_train, y_train)
            
            X_train_sel = self.feature_selector.transform(X_train)
            X_test_sel = self.feature_selector.transform(X_test)
            
            # Balanced weighting
            sample_weights = compute_sample_weight('balanced', y_train)
            
            model.fit(X_train_sel, y_train, sample_weight=sample_weights)
            scores.append(accuracy_score(y_test, model.predict(X_test_sel)))
            
        return np.mean(scores)

    def train(self, X: pd.DataFrame, y: pd.Series):
        """Full training pipeline with calibration"""
        try:
            # Feature selection
            self.feature_selector = SelectFromModel(
                RandomForestClassifier(n_estimators=100),
                threshold="median"
            )
            self.feature_selector.fit(X, y)
            X_sel = self.feature_selector.transform(X)

            # Train models
            sample_weights = compute_sample_weight('balanced', y)
            self.model_rf.fit(X_sel, y, sample_weight=sample_weights)
            self.model_gb.fit(X_sel, y, sample_weight=sample_weights)

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
            self.calibrated_rf.fit(X_sel, y)
            self.calibrated_gb.fit(X_sel, y)
            
        except Exception as e:
            logging.error(f"Training failed: {str(e)}")
            raise

    def predict(self, X: pd.DataFrame) -> float:
        """Ensemble prediction with dynamic weighting"""
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
    
    if not processed_data.empty:
        col1, col2 = st.columns([3, 1])
        with col1:
            st.subheader("Price Chart")
            st.line_chart(processed_data['Close'])
            
        with col2:
            st.metric("Current Price", f"${processed_data['Close'].iloc[-1]:.2f}")
            st.metric("24h Volatility", f"{processed_data['Volatility'].iloc[-1]*100:.2f}%")

    # Model training
    st.sidebar.header("Model Controls")
    if st.sidebar.button("🚀 Train New Model"):
        if processed_data.empty:
            st.error("No data available for training!")
            return
            
        with st.spinner("Training AI model (this may take a few minutes)..."):
            try:
                X = processed_data.drop(['Target', 'Returns'], axis=1, errors='ignore')
                y = processed_data['Target']
                
                if st.session_state.model is None:
                    st.session_state.model = TradingModel()
                
                st.session_state.model.optimize_hyperparameters(X, y)
                st.session_state.model.train(X, y)
                st.session_state.last_trained = datetime.now()
                st.success("Model trained successfully!")
                
            except Exception as e:
                st.error(f"Training failed: {str(e)}")

    # Trading signals
    if st.session_state.model and not processed_data.empty:
        latest_features = processed_data.drop(
            ['Target', 'Returns'], axis=1, errors='ignore'
        ).iloc[[-1]]
        
        try:
            prediction = st.session_state.model.predict(latest_features)[0]
            st.subheader("Trading Signal")
            
            col1, col2, col3 = st.columns(3)
            col1.metric("Confidence Score", f"{prediction:.2%}", 
                       help="Model's confidence in the prediction")
            
            if prediction > TRADE_THRESHOLD_BUY:
                col2.success("🚦 Strong Buy Signal")
                col3.write(f"Threshold: >{TRADE_THRESHOLD_BUY:.0%}")
            elif prediction < TRADE_THRESHOLD_SELL:
                col2.error("🚦 Strong Sell Signal")
                col3.write(f"Threshold: <{TRADE_THRESHOLD_SELL:.0%}")
            else:
                col2.info("🛑 Hold Position")
                col3.write("No clear market signal")
                
            st.session_state.last_trained = st.session_state.last_trained or datetime.now()
            st.caption(f"Last trained: {st.session_state.last_trained.strftime('%Y-%m-%d %H:%M')}")
            
        except Exception as e:
            st.error(f"Prediction failed: {str(e)}")

if __name__ == "__main__":
    main()
