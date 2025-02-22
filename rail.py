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

# Configuration
DEFAULT_SYMBOL = 'BTC-USD'
PRIMARY_INTERVAL = '15m'
TRADE_THRESHOLD_BUY = 0.58
TRADE_THRESHOLD_SELL = 0.42
MAX_TRIALS = 20
GARCH_WINDOW = 14
MIN_FEATURES = 7
MIN_SAMPLES_PER_CLASS = 50  # Increased minimum samples
REQUIRED_FEATURES = ['ema_12', 'ema_26', 'macd', 'signal']

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
            period="60d",  # Increased data period
            interval=interval,
            progress=False,
            auto_adjust=True
        )
        
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = ['_'.join(col).strip() for col in df.columns.values]
        
        def clean_column_name(col: str) -> str:
            col = re.sub(r'([a-z])([A-Z])', r'\1_\2', str(col))
            col = re.sub(r'[^a-zA-Z0-9]+', '_', col)
            return col.lower().strip('_')
        
        df.columns = [clean_column_name(col) for col in df.columns]
        symbol_clean = symbol.lower().replace('-', '_')
        df.columns = [col.replace(f'_{symbol_clean}', '') for col in df.columns]
        
        column_map = {
            'open': ['open', 'adj_open'],
            'high': ['high', 'adj_high'],
            'low': ['low', 'adj_low'],
            'close': ['close', 'adj_close'],
            'volume': ['volume', 'adj_volume']
        }
        
        final_cols = {}
        for standard, aliases in column_map.items():
            for alias in aliases:
                if alias in df.columns:
                    final_cols[standard] = df[alias]
                    break
            if standard not in final_cols:
                st.error(f"Missing required column: {standard}")
                return pd.DataFrame()
        
        return pd.DataFrame(final_cols)[['open', 'high', 'low', 'close', 'volume']].ffill().dropna()
    
    except Exception as e:
        logging.error(f"Data fetch failed: {str(e)}", exc_info=True)
        return pd.DataFrame()

@st.cache_data
def calculate_features(df: pd.DataFrame) -> pd.DataFrame:
    try:
        df = df.copy()
        df['returns'] = df['close'].pct_change()
        
        # Technical indicators
        df['ema_12'] = df['close'].ewm(span=12, adjust=False).mean()
        df['ema_26'] = df['close'].ewm(span=26, adjust=False).mean()
        df['macd'] = df['ema_12'] - df['ema_26']
        df['signal'] = df['macd'].ewm(span=9, adjust=False).mean()
        
        for window in [20, 50, 100]:
            df[f'sma_{window}'] = df['close'].rolling(window).mean()
            delta = df['close'].diff()
            gain = delta.where(delta > 0, 0)
            loss = -delta.where(delta < 0, 0)
            avg_gain = gain.rolling(window).mean()
            avg_loss = loss.rolling(window).mean()
            rs = avg_gain / avg_loss.replace(0, 1)
            df[f'rsi_{window}'] = 100 - (100 / (1 + rs))
        
        df['volatility'] = df['returns'].rolling(GARCH_WINDOW).std()
        
        # Improved target engineering
        future_returns = df['close'].pct_change().shift(-2)  # Look 2 periods ahead
        df['target'] = pd.qcut(future_returns, 
                             q=[0, 0.3, 0.7, 1], 
                             labels=[0, 1, 2]).astype(int)
        
        return df.dropna().drop(columns=['open', 'high', 'low', 'close', 'volume'])
    
    except Exception as e:
        logging.error(f"Feature engineering failed: {str(e)}", exc_info=True)
        return pd.DataFrame()

class TradingModel:
    def __init__(self):
        self.selected_features = []
        self.model = None
        self.required_features = REQUIRED_FEATURES
        self.is_trained = False
        self.classes_ = None

    def optimize_model(self, X: pd.DataFrame, y: pd.Series) -> bool:
        try:
            self._reset_state()
            if not self._validate_inputs(X, y):
                return False

            tscv = TimeSeriesSplit(n_splits=3)
            study = optuna.create_study(direction='maximize')
            
            # Prevent data leakage in optimization
            study.optimize(
                lambda trial: self._objective(trial, X, y, tscv),
                n_trials=MAX_TRIALS,
                n_jobs=-1,
                catch=(ValueError,)
            )
            
            if not study.best_trial:
                st.error("No successful trials completed")
                return False
                
            # Final training with best params
            self._train_final_model(X, y, study.best_params)
            
            self.is_trained = True
            return True
            
        except Exception as e:
            logging.error(f"Training failed: {str(e)}", exc_info=True)
            st.error("Model training failed. Check logs for details.")
            return False

    def _reset_state(self):
        self.selected_features = []
        self.model = None
        self.is_trained = False
        self.classes_ = None

    def _validate_inputs(self, X: pd.DataFrame, y: pd.Series) -> bool:
        missing_features = [f for f in self.required_features if f not in X.columns]
        if missing_features:
            st.error(f"Missing required features: {missing_features}")
            return False

        class_counts = y.value_counts()
        if len(class_counts) < 3 or any(class_counts < MIN_SAMPLES_PER_CLASS):
            st.error(f"Insufficient class samples: {class_counts.to_dict()}")
            return False
            
        return True

    def _train_final_model(self, X: pd.DataFrame, y: pd.Series, best_params: dict):
        # Feature selection with time-series split
        tscv = TimeSeriesSplit(n_splits=3)
        selector = RFECV(
            XGBClassifier(**best_params),
            step=1,
            cv=tscv,
            min_features_to_select=MIN_FEATURES
        )
        selector.fit(X, y)
        self.selected_features = X.columns[selector.support_].tolist()
        
        # Ensure required features
        for feat in self.required_features:
            if feat not in self.selected_features:
                self.selected_features.append(feat)

        # Train final model with selected features
        self.model = XGBClassifier(**best_params)
        self.model.fit(X[self.selected_features], y)
        self.classes_ = self.model.classes_

        # Validation with walk-forward split
        st.subheader("Validation Performance")
        X_sorted = X.sort_index()
        y_sorted = y.sort_index()
        
        tscv = TimeSeriesSplit(n_splits=3)
        reports = []
        for fold, (train_idx, test_idx) in enumerate(tscv.split(X_sorted)):
            X_train, X_test = X_sorted.iloc[train_idx], X_sorted.iloc[test_idx]
            y_train, y_test = y_sorted.iloc[train_idx], y_sorted.iloc[test_idx]
            
            # Apply SMOTE only to training data
            sm = SMOTE(random_state=42)
            X_res, y_res = sm.fit_resample(X_train, y_train)
            
            model = XGBClassifier(**best_params)
            model.fit(X_res[self.selected_features], y_res)
            
            y_pred = model.predict(X_test[self.selected_features])
            reports.append(classification_report(y_test, y_pred, output_dict=True))
            
        # Aggregate results
        avg_precision = np.mean([r['weighted avg']['precision'] for r in reports])
        avg_recall = np.mean([r['weighted avg']['recall'] for r in reports])
        avg_f1 = np.mean([r['weighted avg']['f1-score'] for r in reports])
        
        st.write(f"Average Validation Performance (3-fold time series split):")
        st.write(f"Precision: {avg_precision:.2f}, Recall: {avg_recall:.2f}, F1: {avg_f1:.2f}")

    def _objective(self, trial, X: pd.DataFrame, y: pd.Series, tscv) -> float:
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 500),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'max_depth': trial.suggest_int('max_depth', 3, 9),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'gamma': trial.suggest_float('gamma', 0, 0.5),
            'enable_categorical': True
        }
        
        scores = []
        
        for train_idx, test_idx in tqdm(tscv.split(X), desc="CV Progress"):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            
            try:
                # Apply SMOTE only to training fold
                sm = SMOTE(random_state=42)
                X_res, y_res = sm.fit_resample(X_train, y_train)
                
                model = XGBClassifier(**params)
                model.fit(X_res, y_res)
                
                y_proba = model.predict_proba(X_test)
                y_test_bin = label_binarize(y_test, classes=model.classes_)
                
                fold_scores = []
                for class_idx in range(y_test_bin.shape[1]):
                    if y_test_bin[:, class_idx].sum() > 0:
                        fold_scores.append(
                            roc_auc_score(
                                y_test_bin[:, class_idx], 
                                y_proba[:, class_idx]
                            )
                        )
                
                scores.append(np.mean(fold_scores) if fold_scores else 0.0)
                    
            except Exception as e:
                scores.append(0.0)
        
        return np.nanmean(scores) if scores else 0.0

    def predict(self, X: pd.DataFrame) -> float:
        if not self.is_trained or self.model is None or X.empty:
            return 0.5
            
        try:
            missing = [f for f in self.selected_features if f not in X.columns]
            if missing:
                logging.error(f"Missing features: {missing}")
                return 0.5
                
            # Get probability for class 2 (buy signal)
            proba = self.model.predict_proba(X[self.selected_features])[0][2]
            return max(0.0, min(1.0, proba))  # Clamp between 0-1
        except Exception as e:
            logging.error(f"Prediction failed: {str(e)}")
            return 0.5

def main():
    st.sidebar.header("Settings")
    symbol = st.sidebar.text_input("Crypto Symbol", DEFAULT_SYMBOL).upper()
    
    raw_data = fetch_data(symbol, PRIMARY_INTERVAL)
    processed_data = calculate_features(raw_data)
    
    if not raw_data.empty and not processed_data.empty:
        missing_features = [f for f in REQUIRED_FEATURES if f not in processed_data.columns]
        if missing_features:
            st.error(f"Missing critical features: {missing_features}")
            st.stop()
            
        col1, col2 = st.columns([3, 1])
        with col1:
            st.subheader(f"{symbol} Price Chart")
            st.line_chart(raw_data['close'])
        with col2:
            st.metric("Current Price", f"${raw_data['close'].iloc[-1]:.2f}")
            st.metric("Volatility", f"{processed_data['volatility'].iloc[-1]:.2%}")

    if st.sidebar.button("🚀 Train Model") and not processed_data.empty:
        try:
            model = TradingModel()
            X = processed_data.drop(columns=['target'])
            y = processed_data['target']
            
            with st.spinner("Training AI Model..."):
                if model.optimize_model(X, y):
                    st.session_state.model = model
                    st.success("Training completed!")
                else:
                    st.error("Model training failed. See errors above.")
        except Exception as e:
            st.error(f"Training error: {str(e)}")

    if st.session_state.model and not processed_data.empty:
        model = st.session_state.model
        latest_data = processed_data.drop(columns=['target']).iloc[[-1]]
        
        missing = [f for f in model.selected_features if f not in latest_data.columns]
        if missing:
            st.error(f"Missing features in latest data: {missing}")
            st.stop()
            
        confidence = model.predict(latest_data)
        
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
