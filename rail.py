# crypto_trading_system.py (FINAL FIXED VERSION)
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
        # Ensure flat column index
        df.columns = df.columns.str.replace(' ', '_')
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
        required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        if not all(col in df.columns for col in required_cols):
            raise ValueError("Missing required price columns")

        # Feature calculations
        df['Close_Lag1'] = df['Close'].shift(1)
        df['Returns'] = df['Close_Lag1'].pct_change()
        df['Log_Returns'] = np.log(df['Close_Lag1']).diff()
        
        windows = [20, 50, 100, 200]
        for window in windows:
            df[f'SMA_{window}'] = df['Close_Lag1'].rolling(window).mean()
            df[f'STD_{window}'] = df['Close_Lag1'].rolling(window).std()
            
            delta = df['Close_Lag1'].diff()
            gain = delta.clip(lower=0)
            loss = -delta.clip(upper=0)
            
            avg_gain = gain.rolling(window, min_periods=1).mean()
            avg_loss = loss.rolling(window, min_periods=1).mean()
            
            with np.errstate(divide='ignore', invalid='ignore'):
                rs = avg_gain / avg_loss.replace(0, 1)
            rsi = 100 - (100 / (1 + rs))
            df[f'RSI_{window}'] = rsi

        df['Volatility'] = df['Log_Returns'].rolling(GARCH_WINDOW).std()
        for period in [3, 7, 14]:
            df[f'Momentum_{period}'] = df['Close_Lag1'].pct_change(period)
        
        future_returns = df['Close'].pct_change().shift(-1).to_numpy().ravel()
        df['Target'] = pd.cut(
            future_returns,
            bins=[-np.inf, -0.01, 0.01, np.inf],
            labels=[0, 1, 2],
            ordered=False
        )
        
        # Ensure flat column names
        df = df.drop(columns=required_cols + ['Close_Lag1'])
        df.columns = [str(col) for col in df.columns]
        return df.dropna().reset_index(drop=True)
        
    except Exception as e:
        logging.error(f"Feature engineering failed: {str(e)}", exc_info=True)
        return pd.DataFrame()

# ======================
# MODEL PIPELINE (FIXED)
# ======================
class TradingModel:
    def __init__(self):
        self.selected_features = []
        self.model = None
        self.feature_selector = None

    def _validate_leakage(self, X: pd.DataFrame):
        leakage_found = False
        for col in X.columns:
            shifted = X[col].shift(1)
            current = X[col].iloc[1:]
            if not current.equals(shifted.dropna()):
                forward_shifted = X[col].shift(-1).iloc[:-1]
                if X[col].iloc[:-1].equals(forward_shifted):
                    st.error(f"⚠️ Potential leakage in {col}")
                    leakage_found = True
        if leakage_found:
            raise ValueError("Data leakage detected in features")

    def optimize_model(self, X: pd.DataFrame, y: pd.Series):
        try:
            tscv = TimeSeriesSplit(n_splits=3)
            self._validate_leakage(X)
            
            # Convert to flat column names
            X.columns = [str(col) for col in X.columns]
            
            self.feature_selector = RFECV(
                estimator=GradientBoostingClassifier(),
                step=1,
                cv=tscv,
                min_features_to_select=MIN_FEATURES
            )
            self.feature_selector.fit(X, y)
            self.selected_features = X.columns[self.feature_selector.get_support()].tolist()
            
            study = optuna.create_study(direction='maximize')
            study.optimize(
                lambda trial: self._objective(trial, X[self.selected_features], y, tscv),
                n_trials=MAX_TRIALS
            )
            
            self.model = GradientBoostingClassifier(**study.best_params)
            self.model.fit(X[self.selected_features], y)
            
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
        """Robust prediction with proper type handling"""
        try:
            # Convert to flat column names
            X.columns = [str(col) for col in X.columns]
            
            if not self.selected_features or X.empty:
                return 0.5
                
            if not hasattr(self, 'model') or self.model is None:
                return 0.5
                
            missing_features = [f for f in self.selected_features if f not in X.columns]
            
            if len(missing_features) > 0:
                logging.error(f"Missing features: {missing_features}")
                return 0.5
                
            return self.model.predict_proba(X[self.selected_features])[0][2]
            
        except Exception as e:
            logging.error(f"Prediction failed: {str(e)}")
            return 0.5

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
