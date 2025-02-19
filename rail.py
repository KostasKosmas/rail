# crypto_trading_system.py (FINAL CORRECTED VERSION)
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
# DATA PIPELINE
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
        return df.reset_index(drop=True) if not df.empty else pd.DataFrame()
    except Exception as e:
        logging.error(f"Data fetch failed: {str(e)}")
        st.error(f"Failed to fetch data for {symbol}")
        return pd.DataFrame()

def calculate_features(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or len(df) < 100:
        return pd.DataFrame()
    
    df = df.copy()
    try:
        # Validate required columns
        required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        if not all(col in df.columns for col in required_cols):
            raise ValueError("Missing required price columns in DataFrame")

        # Calculate lagged price changes
        df['Close_Lag1'] = df['Close'].shift(1)
        df['Returns'] = df['Close_Lag1'].pct_change()
        df['Log_Returns'] = np.log(df['Close_Lag1']).diff()
        
        # Technical Indicators
        windows = [20, 50, 100, 200]
        for window in windows:
            # Simple Moving Averages
            df[f'SMA_{window}'] = df['Close_Lag1'].rolling(window).mean()
            df[f'STD_{window}'] = df['Close_Lag1'].rolling(window).std()
            
            # RSI Calculation (fixed dimensionality)
            delta = df['Close_Lag1'].diff()
            gain = delta.clip(lower=0)
            loss = -delta.clip(upper=0)
            
            avg_gain = gain.rolling(window, min_periods=1).mean().values
            avg_loss = loss.rolling(window, min_periods=1).mean().values
            
            # Handle division by zero and ensure 1D array
            with np.errstate(divide='ignore', invalid='ignore'):
                rs = np.divide(avg_gain, avg_loss, where=avg_loss!=0)
                rs[avg_loss == 0] = 1  # Handle zero division cases
                
            rsi = 100 - (100 / (1 + rs))
            df[f'RSI_{window}'] = rsi.astype(np.float64)

        # Volatility Features
        df['Volatility'] = df['Log_Returns'].rolling(GARCH_WINDOW).std()
        
        # Momentum Features
        for period in [3, 7, 14]:
            df[f'Momentum_{period}'] = df['Close_Lag1'].pct_change(period)
        
        # Target Engineering
        future_returns = df['Close'].pct_change().shift(-1)
        df['Target'] = pd.cut(
            future_returns,
            bins=[-np.inf, -0.01, 0.01, np.inf],
            labels=[0, 1, 2],
            ordered=False
        )
        
        # Cleanup
        df = df.drop(columns=required_cols + ['Close_Lag1'])
        return df.dropna().reset_index(drop=True)
        
    except Exception as e:
        logging.error(f"Feature engineering failed: {str(e)}", exc_info=True)
        return pd.DataFrame()

# ======================
# MODEL PIPELINE
# ======================
class TradingModel:
    def __init__(self):
        self.selected_features = []
        self.model = None
        self.feature_selector = None

    def _validate_leakage(self, X: pd.DataFrame, y: pd.Series):
        """Ensure no future data in features"""
        for col in X.columns:
            if any(X[col].diff().shift(-1).fillna(0) != 0):
                raise ValueError(f"Leakage detected in {col}")

    def optimize_model(self, X: pd.DataFrame, y: pd.Series):
        try:
            tscv = TimeSeriesSplit(n_splits=3)
            self._validate_leakage(X, y)
            
            # Feature Selection
            self.feature_selector = RFECV(
                estimator=GradientBoostingClassifier(),
                step=1,
                cv=tscv,
                min_features_to_select=MIN_FEATURES
            )
            self.feature_selector.fit(X, y)
            self.selected_features = X.columns[self.feature_selector.get_support()]
            
            # Hyperparameter Optimization
            study = optuna.create_study(direction='maximize')
            study.optimize(
                lambda trial: self._objective(trial, X[self.selected_features], y, tscv),
                n_trials=MAX_TRIALS
            )
            
            # Final Model Training
            self.model = GradientBoostingClassifier(**study.best_params)
            self.model.fit(X[self.selected_features], y)
            
            # Validation Report
            y_pred = self.model.predict(X[self.selected_features])
            st.subheader("Model Validation")
            st.text(classification_report(y, y_pred))
            st.write("Confusion Matrix:")
            st.dataframe(confusion_matrix(y, y_pred))
            
        except Exception as e:
            logging.error(f"Model training failed: {str(e)}", exc_info=True)
            st.error("Model training failed. Check logs for details.")

    def _objective(self, trial, X: pd.DataFrame, y: pd.Series, cv):
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
            
            rus = RandomUnderSampler(random_state=42)
            X_res, y_res = rus.fit_resample(X_train, y_train)
            
            model = GradientBoostingClassifier(**params)
            model.fit(X_res, y_res)
            scores.append(f1_score(y_test, model.predict(X_test), average='weighted'))
            
        return np.mean(scores)

    def predict(self, X: pd.DataFrame) -> float:
        if not self.selected_features or X.empty:
            return 0.5
        return self.model.predict_proba(X[self.selected_features])[0][2]

# ======================
# MAIN INTERFACE
# ======================
def main():
    st.sidebar.header("Settings")
    symbol = st.sidebar.text_input("Cryptocurrency Symbol", DEFAULT_SYMBOL).upper()
    
    raw_data = fetch_data(symbol, PRIMARY_INTERVAL)
    processed_data = calculate_features(raw_data)
    
    if not processed_data.empty:
        col1, col2 = st.columns([3, 1])
        with col1:
            st.subheader(f"{symbol} Price Chart")
            st.line_chart(raw_data['Close'])
            
        with col2:
            current_price = raw_data['Close'].iloc[-1].item()
            current_vol = processed_data['Volatility'].iloc[-1].item() if 'Volatility' in processed_data else 0
            st.metric("Current Price", f"${current_price:,.2f}")
            st.metric("Volatility", f"{current_vol:.2%}")

    if st.sidebar.button("🚀 Train Model") and not processed_data.empty:
        try:
            model = TradingModel()
            X = processed_data.drop(columns=['Target'])
            y = processed_data['Target']
            
            with st.spinner("Training AI model..."):
                model.optimize_model(X, y)
                st.session_state.model = model
                st.session_state.last_trained = datetime.now()
                st.success("Model trained successfully!")
                
        except Exception as e:
            st.error(f"Training failed: {str(e)}")

    if st.session_state.model and not processed_data.empty:
        latest_data = processed_data.drop(columns=['Target']).iloc[[-1]]
        confidence = st.session_state.model.predict(latest_data)
        
        st.subheader("Trading Signal")
        col1, col2 = st.columns(2)
        col1.metric("Confidence Score", f"{confidence:.2%}")
        
        current_vol = processed_data['Volatility'].iloc[-1].item()
        adj_buy = TRADE_THRESHOLD_BUY + (current_vol * 0.1)
        adj_sell = TRADE_THRESHOLD_SELL - (current_vol * 0.1)
        
        if confidence > adj_buy:
            col2.success("🚀 Strong Buy Signal")
        elif confidence < adj_sell:
            col2.error("🔻 Strong Sell Signal")
        else:
            col2.info("🛑 Hold Position")

if __name__ == "__main__":
    main()
