# crypto_trading_system.py (FIXED MULTIINDEX & COLUMN CLEANING)
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
import re

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
# ENHANCED DATA PIPELINE
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
        
        # Handle MultiIndex columns
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [f"{col[0]}_{col[1]}" for col in df.columns.values]
        
        # Clean column names and remove symbol suffix
        symbol_clean = symbol.lower().replace("-", "_")
        new_columns = []
        for col in df.columns:
            # Clean special characters and normalize
            col_clean = re.sub(r'[^a-zA-Z0-9]', '_', str(col)).strip('_').lower()
            
            # Remove symbol suffix using regex
            col_clean = re.sub(rf'_{symbol_clean}$', '', col_clean)
            
            # Remove any remaining symbol parts
            col_clean = col_clean.replace(symbol_clean, '').strip('_')
            
            new_columns.append(col_clean)
        
        df.columns = new_columns
        
        # Column mapping with priority for standard names
        column_mapping = {
            'open': ['open', 'opening_price', 'price_open', 'o'],
            'high': ['high', 'highest_price', 'price_high', 'h'],
            'low': ['low', 'lowest_price', 'price_low', 'l'],
            'close': ['close', 'closing_price', 'price_close', 'c', 'adj_close'],
            'volume': ['volume', 'vol', 'v', 'adj_volume']
        }
        
        # Validate and map columns
        final_columns = {}
        for standard_name, aliases in column_mapping.items():
            found = False
            for alias in aliases:
                if alias in df.columns:
                    final_columns[standard_name] = df[alias]
                    found = True
                    break
            
            if not found:
                available = '\n'.join(df.columns)
                st.error(f"""Missing required column: {standard_name}
                          Tried: {aliases}
                          Available columns:\n{available}""")
                return pd.DataFrame()

        # Create clean dataframe with verified columns
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        clean_df = pd.DataFrame(final_columns)[required_cols]
        
        # Forward-fill missing values for crypto markets
        clean_df = clean_df.ffill().dropna()
        
        if clean_df.empty:
            st.error("No data available after cleaning")
            return pd.DataFrame()
            
        return clean_df.reset_index(drop=True)
        
    except Exception as e:
        logging.error(f"Data fetch failed: {str(e)}")
        st.error(f"Failed to fetch data for {symbol}")
        return pd.DataFrame()

def calculate_features(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or len(df) < 100:
        return pd.DataFrame()
    
    df = df.copy()
    try:
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        missing = [col for col in required_cols if col not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        # Feature engineering calculations
        df['close_lag1'] = df['close'].shift(1)
        df['returns'] = df['close_lag1'].pct_change().fillna(0)
        df['log_returns'] = np.log(df['close_lag1']).diff().fillna(0)
        
        windows = [20, 50, 100, 200]
        for window in windows:
            df[f'sma_{window}'] = df['close_lag1'].rolling(window).mean()
            df[f'ema_{window}'] = df['close_lag1'].ewm(span=window, adjust=False).mean()
            df[f'std_{window}'] = df['close_lag1'].rolling(window).std()
            
            delta = df['close_lag1'].diff()
            gain = delta.clip(lower=0)
            loss = -delta.clip(upper=0)
            
            avg_gain = gain.rolling(window, min_periods=1).mean()
            avg_loss = loss.rolling(window, min_periods=1).mean()
            
            with np.errstate(divide='ignore', invalid='ignore'):
                rs = avg_gain / avg_loss.replace(0, 1)
            df[f'rsi_{window}'] = 100 - (100 / (1 + rs))

        df['volatility'] = df['log_returns'].rolling(GARCH_WINDOW).std()
        for period in [3, 7, 14]:
            df[f'momentum_{period}'] = df['close_lag1'].pct_change(period)
            df[f'roc_{period}'] = (df['close_lag1'] / df['close_lag1'].shift(period) - 1)
        
        future_returns = df['close'].pct_change().shift(-1).to_numpy().ravel()
        df['target'] = pd.cut(
            future_returns,
            bins=[-np.inf, -0.01, 0.01, np.inf],
            labels=[0, 1, 2],
            ordered=False
        )
        
        df = df.drop(columns=required_cols + ['close_lag1'])
        return df.dropna().reset_index(drop=True)
        
    except Exception as e:
        logging.error(f"Feature engineering failed: {str(e)}", exc_info=True)
        return pd.DataFrame()

# ======================
# MODEL PIPELINE (FIXED INDEX HANDLING)
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
        try:
            if not self.selected_features or X.empty:
                return 0.5
                
            if not hasattr(self, 'model') or self.model is None:
                return 0.5
                
            missing = [f for f in self.selected_features if f not in X.columns]
            if missing:
                logging.error(f"Missing features: {missing}")
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
            st.line_chart(raw_data['close'])
            
        with col2:
            current_price = raw_data['close'].iloc[-1].item()
            current_vol = processed_data['volatility'].iloc[-1].item() if 'volatility' in processed_data else 0
            st.metric("Current Price", f"${current_price:,.2f}")
            st.metric("Volatility", f"{current_vol:.2%}")

    if st.sidebar.button("🚀 Train Model") and not processed_data.empty:
        try:
            model = TradingModel()
            X = processed_data.drop(columns=['target'])
            y = processed_data['target']
            
            with st.spinner("Training AI model..."):
                model.optimize_model(X, y)
                st.session_state.model = model
                st.session_state.last_trained = datetime.now()
                st.success("Model trained successfully!")
                
        except Exception as e:
            st.error(f"Training failed: {str(e)}")

    if st.session_state.model and not processed_data.empty:
        latest_data = processed_data.drop(columns=['target']).iloc[[-1]]
        confidence = st.session_state.model.predict(latest_data)
        
        st.subheader("Trading Signal")
        col1, col2 = st.columns(2)
        col1.metric("Confidence Score", f"{confidence:.2%}")
        
        current_vol = processed_data['volatility'].iloc[-1].item()
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
