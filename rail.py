import logging
import numpy as np
import pandas as pd
import yfinance as yf
import streamlit as st
import optuna
from xgboost import XGBClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.feature_selection import RFECV
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import label_binarize
import warnings
import re
from tqdm import tqdm

# Check if xgboost is installed
try:
    from xgboost import XGBClassifier
except ImportError:
    st.error("The 'xgboost' library is not installed. Please install it using `pip install xgboost`.")
    st.stop()

# Configuration
DEFAULT_SYMBOL = 'BTC-USD'
PRIMARY_INTERVAL = '15m'
TRADE_THRESHOLD_BUY = 0.58
TRADE_THRESHOLD_SELL = 0.42
MAX_TRIALS = 20
GARCH_WINDOW = 14
MIN_FEATURES = 7

warnings.filterwarnings("ignore", category=UserWarning)
logging.basicConfig(level=logging.INFO)

# Streamlit UI
st.set_page_config(page_title="AI Trading System", layout="wide")
st.title("🚀 AI-Powered Cryptocurrency Trading System")

if 'model' not in st.session_state:
    st.session_state.model = None

@st.cache_data(ttl=300, show_spinner="Fetching market data...")
def fetch_data(symbol: str, interval: str) -> pd.DataFrame:
    try:
        df = yf.download(
            symbol,
            period="30d",
            interval=interval,
            progress=False,
            auto_adjust=True
        )
        
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = ['_'.join(map(str, col)).strip() for col in df.columns.values]
        
        def clean_column_name(col: str) -> str:
            col = re.sub(r'([a-z])([A-Z])', r'\1_\2', str(col))
            col = re.sub(r'[^a-zA-Z0-9]+', '_', col)
            return col.lower().strip('_')
        
        df.columns = [clean_column_name(col) for col in df.columns]
        symbol_clean = symbol.lower().replace('-', '_')
        df.columns = [col.replace(f'_{symbol_clean}', '') for col in df.columns]
        
        column_map = {
            'open': ['open', 'adj_open', 'adjusted_open', 'opening_price'],
            'high': ['high', 'adj_high', 'adjusted_high', 'highest_price'],
            'low': ['low', 'adj_low', 'adjusted_low', 'lowest_price'],
            'close': ['close', 'adj_close', 'adjusted_close', 'closing_price'],
            'volume': ['volume', 'adj_volume', 'adjusted_volume', 'vol']
        }
        
        final_cols = {}
        for standard, aliases in column_map.items():
            found = False
            for alias in aliases:
                if alias in df.columns:
                    final_cols[standard] = df[alias]
                    found = True
                    break
            if not found:
                st.error(f"Missing required column: {standard.upper()}")
                return pd.DataFrame()
        
        clean_df = pd.DataFrame(final_cols)[['open', 'high', 'low', 'close', 'volume']]
        return clean_df.ffill().dropna()
        
    except Exception as e:
        logging.error(f"Data fetch failed: {str(e)}", exc_info=True)
        return pd.DataFrame()

@st.cache_data
def calculate_features(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or len(df) < 100:
        return pd.DataFrame()
    
    try:
        df = df.copy()
        df['close_lag1'] = df['close'].shift(1)
        df['returns'] = df['close'].pct_change()
        
        windows = [20, 50, 100]
        for window in windows:
            df[f'sma_{window}'] = df['close'].rolling(window).mean()
            
            delta = df['close'].diff()
            gain = delta.clip(lower=0)
            loss = -delta.clip(upper=0)
            avg_gain = gain.rolling(window, min_periods=1).mean()
            avg_loss = loss.rolling(window, min_periods=1).mean()
            rs = avg_gain / avg_loss.replace(0, 1)
            df[f'rsi_{window}'] = 100 - (100 / (1 + rs))
        
        df['volatility'] = df['returns'].rolling(GARCH_WINDOW).std()
        df['ema_12'] = df['close'].ewm(span=12, adjust=False).mean()
        df['ema_26'] = df['close'].ewm(span=26, adjust=False).mean()
        df['macd'] = df['ema_12'] - df['ema_26']
        df['signal'] = df['macd'].ewm(span=9, adjust=False).mean()
        
        future_returns = df['close'].pct_change().shift(-1)
        dynamic_threshold = df['volatility'].rolling(window=14).mean()
        df['target'] = np.select(
            [future_returns > dynamic_threshold, future_returns < -dynamic_threshold],
            [2, 0],
            default=1
        )
        
        return df.dropna().drop(columns=['open', 'high', 'low', 'close', 'volume', 'close_lag1'])
    
    except Exception as e:
        logging.error(f"Feature engineering failed: {str(e)}", exc_info=True)
        return pd.DataFrame()

class TradingModel:
    def __init__(self):
        self.selected_features = []
        self.model = None

    def optimize_model(self, X: pd.DataFrame, y: pd.Series):
        try:
            tscv = TimeSeriesSplit(n_splits=3)
            smote = SMOTE(random_state=42)
            X_res, y_res = smote.fit_resample(X, y)
            
            selector = RFECV(
                XGBClassifier(),
                step=1,
                cv=tscv,
                min_features_to_select=MIN_FEATURES
            )
            selector.fit(X_res, y_res)
            self.selected_features = X.columns[selector.get_support()].tolist()
            
            study = optuna.create_study(direction='maximize')
            study.optimize(
                lambda trial: self._objective(trial, X_res[self.selected_features], y_res, tscv),
                n_trials=MAX_TRIALS,
                n_jobs=-1
            )
            
            self.model = XGBClassifier(**study.best_params)
            self.model.fit(X_res[self.selected_features], y_res)
            
            st.subheader("Model Validation")
            y_pred = self.model.predict(X[self.selected_features])
            st.text(classification_report(y, y_pred))
            st.write("Confusion Matrix:")
            st.dataframe(confusion_matrix(y, y_pred))
            
            y_proba = self.model.predict_proba(X[self.selected_features])
            roc_auc = roc_auc_score(y, y_proba, multi_class='ovr')
            st.metric("ROC-AUC Score", f"{roc_auc:.2f}")
            
        except Exception as e:
            logging.error(f"Model training failed: {str(e)}", exc_info=True)
            st.error("Model training failed. Check logs for details.")

    def _objective(self, trial, X: pd.DataFrame, y: pd.Series, cv):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 500),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'max_depth': trial.suggest_int('max_depth', 3, 15),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'gamma': trial.suggest_float('gamma', 0, 1)
        }
        
        scores = []
        for train_idx, test_idx in tqdm(cv.split(X), desc="CV Progress"):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            
            smote = SMOTE(random_state=42)
            X_res, y_res = smote.fit_resample(X_train, y_train)
            
            model = XGBClassifier(**params)
            model.fit(X_res, y_res)
            
            y_proba = model.predict_proba(X_test)
            labels = np.unique(y_res)
            
            try:
                y_test_bin = label_binarize(y_test, classes=labels)
                n_classes = y_test_bin.shape[1]
                
                auc_scores = []
                for i in range(n_classes):
                    if np.sum(y_test_bin[:, i]) >= 1:
                        auc = roc_auc_score(y_test_bin[:, i], y_proba[:, i])
                        auc_scores.append(auc)
                
                if len(auc_scores) > 0:
                    scores.append(np.mean(auc_scores))
                else:
                    scores.append(0.0)
            except Exception as e:
                scores.append(0.0)
        
        return np.mean(scores)

    def predict(self, X: pd.DataFrame) -> float:
        try:
            if not self.selected_features or X.empty:
                return 0.5
                
            missing = [f for f in self.selected_features if f not in X.columns]
            if missing:
                logging.error(f"Missing features: {missing}")
                return 0.5
                
            return self.model.predict_proba(X[self.selected_features])[0][2]
        except Exception as e:
            logging.error(f"Prediction failed: {str(e)}")
            return 0.5

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
        try:
            model = TradingModel()
            X = processed_data.drop(columns=['target'])
            y = processed_data['target']
            
            with st.spinner("Training AI model..."):
                model.optimize_model(X, y)
                st.session_state.model = model
                st.success("Model trained successfully!")
                
        except Exception as e:
            st.error(f"Training failed: {str(e)}")

    if st.session_state.model and not processed_data.empty:
        latest_data = processed_data.drop(columns=['target']).iloc[[-1]]
        confidence = st.session_state.model.predict(latest_data)
        
        st.subheader("Trading Signal")
        col1, col2 = st.columns(2)
        col1.metric("Confidence Score", f"{confidence:.2%}")
        
        current_vol = processed_data['volatility'].iloc[-1]
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
