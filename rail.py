# crypto_trading_system.py
import logging
import time
import numpy as np
import pandas as pd
import yfinance as yf
import streamlit as st
import optuna
import warnings
import os
from threading import Lock
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.feature_selection import SelectFromModel
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import f1_score
from imblearn.over_sampling import SMOTE
from datetime import datetime, timedelta
from sklearn.cluster import KMeans
from concurrent.futures import ThreadPoolExecutor

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
VOLATILITY_CLUSTERS = 3
MIN_SAMPLES_FOR_SMOTE = 10
STUDY_DIR = "optuna_studies"
STUDY_NAME = "main_study"
STUDY_STORAGE = f"sqlite:///{STUDY_DIR}/trading_studies.db"
YF_MAX_RETRIES = 5
YF_RETRY_DELAY = 3
MIN_DATA_POINTS = 100

warnings.filterwarnings("ignore", category=UserWarning)
logging.basicConfig(level=logging.INFO)
progress_lock = Lock()

# ======================
# STREAMLIT UI
# ======================
st.set_page_config(page_title="AI Trading System", layout="wide")
st.title("🚀 AI-Powered Cryptocurrency Trading System")

class TrainingProgress:
    def __init__(self):
        self.current_trial = 0
        self.best_score = 0.0
        self.status = "Initializing..."
        self.params = {}
        self.start_time = time.time()
        self.latest_score = 0.0
        self.last_update = time.time()
        self._trials_completed = 0

    @property
    def trials_completed(self):
        with progress_lock:
            return self._trials_completed

    @trials_completed.setter
    def trials_completed(self, value):
        with progress_lock:
            self._trials_completed = value

if 'model' not in st.session_state:
    st.session_state.model = None
if 'training_progress' not in st.session_state:
    st.session_state.training_progress = TrainingProgress()

# ======================
# DATA PIPELINE
# ======================
@st.cache_data(ttl=300, show_spinner="Fetching market data...")
def fetch_data(symbol: str, interval: str) -> pd.DataFrame:
    params = {
        'tickers': symbol,
        'interval': interval,
        'progress': False,
        'auto_adjust': True,
        'ignore_tz': True,
        'actions': False,
        'timeout': 10
    }
    
    # Set period based on interval
    interval_period_map = {
        '1m': '7d', '5m': '60d', '15m': '60d', '30m': '60d',
        '1h': '730d', '1d': 'max', '1wk': 'max', '1mo': 'max'
    }
    params['period'] = interval_period_map.get(interval, '60d')

    df = pd.DataFrame()
    for attempt in range(YF_MAX_RETRIES):
        try:
            df = yf.download(**params)
            if not df.empty and len(df) >= MIN_DATA_POINTS:
                break
            time.sleep(YF_RETRY_DELAY * (attempt + 1))
        except Exception as e:
            logging.warning(f"Attempt {attempt+1} failed: {str(e)}")
            time.sleep(YF_RETRY_DELAY * (attempt + 1))
    
    if df.empty:
        st.error("""
        🔴 Data fetch failed. Possible solutions:
        1. Check internet connection
        2. Verify cryptocurrency symbol (e.g., BTC-USD)
        3. Try a different time interval
        4. Wait 5 minutes and reload
        """)
        return pd.DataFrame()

    try:
        df = df.reset_index()
        df.columns = ['Date' if 'Date' in str(col) else col for col in df.columns]
        numeric_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce')
        df = df.dropna().reset_index(drop=True)
        return df[(df['Close'] > 0) & (df['Volume'] >= 0)]
    except Exception as e:
        logging.error(f"Data processing failed: {str(e)}")
        return pd.DataFrame()

def calculate_features(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or len(df) < MIN_DATA_POINTS:
        return pd.DataFrame()
    
    try:
        df = df.copy()
        df['Returns'] = df['Close'].pct_change()
        df['Log_Returns'] = np.log(df['Close']).diff()
        
        # Dynamic window calculation
        max_window = min(200, len(df)//2)
        windows = [w for w in [20, 50, 100, 200] if w <= max_window]
        
        for window in windows:
            df[f'SMA_{window}'] = df['Close'].rolling(window).mean()
            df[f'STD_{window}'] = df['Close'].rolling(window).std()
            df[f'RSI_{window}'] = 100 - (100 / (1 + (
                df['Close'].diff().clip(lower=0).rolling(window).mean() / 
                df['Close'].diff().clip(upper=0).abs().rolling(window).mean()
            )))

        df['Volatility'] = df['Log_Returns'].rolling(GARCH_WINDOW).std()
        
        if len(df) >= VOLATILITY_CLUSTERS * 2:
            kmeans = KMeans(n_clusters=VOLATILITY_CLUSTERS)
            df['Vol_Cluster'] = kmeans.fit_predict(df[['Volatility']].fillna(0))
        
        for period in [3, 7, 14]:
            df[f'Momentum_{period}'] = df['Close'].pct_change(period)
        
        df['Target'] = pd.cut(df['Returns'].shift(-1), 
                            bins=[-np.inf, -0.01, 0.01, np.inf],
                            labels=[0, 1, 2])
        
        return df.dropna().reset_index(drop=True)
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
        self.study = None

    def _safe_feature_selection(self, X, y):
        try:
            self.feature_selector = SelectFromModel(
                GradientBoostingClassifier(n_estimators=100),
                threshold="1.25*median"
            )
            self.feature_selector.fit(X, y)
            self.selected_features = X.columns[self.feature_selector.get_support()].tolist()
            if len(self.selected_features) < MIN_FEATURES:
                self.selected_features = X.columns[:MIN_FEATURES].tolist()
            return X[self.selected_features]
        except Exception as e:
            logging.error(f"Feature selection failed: {str(e)}")
            return X.iloc[:, :MIN_FEATURES]

    def optimize_models(self, X: pd.DataFrame, y: pd.Series):
        try:
            os.makedirs(STUDY_DIR, exist_ok=True)
            tscv = TimeSeriesSplit(n_splits=3)
            
            with progress_lock:
                st.session_state.training_progress.status = "Performing feature selection..."
            
            X_sel = self._safe_feature_selection(X, y)
            
            try:
                self.study = optuna.load_study(
                    study_name=STUDY_NAME,
                    storage=STUDY_STORAGE,
                    sampler=optuna.samplers.TPESampler(),
                    pruner=optuna.pruners.MedianPruner()
                )
            except:
                self.study = optuna.create_study(
                    study_name=STUDY_NAME,
                    storage=STUDY_STORAGE,
                    direction='maximize',
                    sampler=optuna.samplers.TPESampler(),
                    pruner=optuna.pruners.MedianPruner()
                )

            def trial_callback(study, trial):
                with progress_lock:
                    st.session_state.training_progress.current_trial = trial.number + 1
                    st.session_state.training_progress.best_score = study.best_value
                    st.session_state.training_progress.params = trial.params
                    st.session_state.training_progress.latest_score = trial.value
                    st.session_state.training_progress.trials_completed += 1
                    st.session_state.training_progress.status = f"Trial {trial.number + 1}/{MAX_TRIALS}"
                time.sleep(0.1)

            self.study.optimize(
                lambda trial: self._objective(trial, X_sel, y, tscv),
                n_trials=MAX_TRIALS,
                callbacks=[trial_callback],
                show_progress_bar=False
            )
            
            with progress_lock:
                st.session_state.training_progress.status = "Optimization complete"
                
        except Exception as e:
            with progress_lock:
                st.session_state.training_progress.status = f"Failed: {str(e)}"
            raise

    def _objective(self, trial, X, y, tscv):
        rf_params = {
            'n_estimators': trial.suggest_int('rf_n_estimators', 300, 1000),
            'max_depth': trial.suggest_int('rf_max_depth', 15, 40),
            'min_samples_split': trial.suggest_int('rf_min_samples_split', 10, 30)
        }
        
        gb_params = {
            'n_estimators': trial.suggest_int('gb_n_estimators', 300, 1000),
            'learning_rate': trial.suggest_float('gb_learning_rate', 0.05, 0.3, log=True),
            'max_depth': trial.suggest_int('gb_max_depth', 5, 15)
        }
        
        scores = []
        for train_idx, test_idx in tscv.split(X):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            
            if len(X_train) < MIN_SAMPLES_FOR_SMOTE:
                continue
                
            class_counts = y_train.value_counts()
            if len(class_counts) < 1:
                continue
                
            minority_class_count = class_counts.min()
            safe_k_neighbors = min(5, minority_class_count - 1)
            if safe_k_neighbors < 1:
                continue
                
            try:
                smote = SMOTE(
                    random_state=42,
                    k_neighbors=safe_k_neighbors
                )
                X_res, y_res = smote.fit_resample(X_train, y_train)
            except:
                continue
            
            rf = RandomForestClassifier(**rf_params).fit(X_res, y_res)
            gb = GradientBoostingClassifier(**gb_params).fit(X_res, y_res)
            
            preds = 0.6 * rf.predict_proba(X_test) + 0.4 * gb.predict_proba(X_test)
            scores.append(f1_score(y_test, np.argmax(preds, axis=1), average='weighted'))
            
        return np.mean(scores) if scores else 0.0

    def train(self, X: pd.DataFrame, y: pd.Series):
        try:
            X_sel = X[self.selected_features]
            
            smote = SMOTE(
                random_state=42,
                k_neighbors=min(5, min(y.value_counts()) - 1)
            )
            X_res, y_res = smote.fit_resample(X_sel, y)
            
            best_params = self.study.best_params
            rf_params = {k[3:]: v for k, v in best_params.items() if k.startswith('rf_')}
            gb_params = {k[3:]: v for k, v in best_params.items() if k.startswith('gb_')}
            
            self.model_rf.set_params(**rf_params).fit(X_res, y_res)
            self.model_gb.set_params(**gb_params).fit(X_res, y_res)
            
            self.calibrated_rf = CalibratedClassifierCV(self.model_rf, cv=TimeSeriesSplit(3))
            self.calibrated_gb = CalibratedClassifierCV(self.model_gb, cv=TimeSeriesSplit(3))
            self.calibrated_rf.fit(X_sel, y)
            self.calibrated_gb.fit(X_sel, y)
        except Exception as e:
            logging.error(f"Training failed: {str(e)}")
            raise

    def predict(self, X: pd.DataFrame) -> float:
        try:
            if (not self.selected_features or 
                X.empty or 
                self.calibrated_rf is None or 
                self.calibrated_gb is None):
                return 0.5
                
            X_sel = X[self.selected_features].reset_index(drop=True)
            if X_sel.empty:
                return 0.5
                
            prob_rf = self.calibrated_rf.predict_proba(X_sel)
            prob_gb = self.calibrated_gb.predict_proba(X_sel)
            return (0.6 * prob_rf + 0.4 * prob_gb)[0][2]
        except:
            return 0.5

# ======================
# MAIN INTERFACE
# ======================
def main():
    if 'training_progress' not in st.session_state:
        st.session_state.training_progress = TrainingProgress()
    
    st.sidebar.header("Settings")
    symbol = st.sidebar.text_input("Cryptocurrency Symbol", DEFAULT_SYMBOL).upper()
    
    with st.spinner("Loading market data..."):
        raw_data = fetch_data(symbol, PRIMARY_INTERVAL)
        processed_data = calculate_features(raw_data)
    
    if processed_data.empty:
        st.warning("""
        ⚠️ Insufficient data for analysis. This could be because:
        - The selected cryptocurrency isn't supported
        - Yahoo Finance is experiencing temporary issues
        - The selected time interval has no trading history
        """)
        return

    col1, col2 = st.columns([3, 1])
    with col1:
        st.subheader(f"{symbol} Price Chart")
        if 'Date' in processed_data:
            st.line_chart(processed_data.set_index('Date')['Close'])
        else:
            st.line_chart(processed_data['Close'])
            
    with col2:
        current_price = processed_data['Close'].iloc[-1].item()
        current_vol = processed_data['Volatility'].iloc[-1].item() if 'Volatility' in processed_data else 0
        st.metric("Current Price", f"${current_price:,.2f}")
        st.metric("Volatility", f"{current_vol:.2%}")

    train_disabled = processed_data.empty or len(processed_data) < MIN_DATA_POINTS
    if st.sidebar.button("🚀 Train Model", disabled=train_disabled):
        try:
            st.session_state.training_progress = TrainingProgress()
            model = TradingModel()
            X = processed_data.drop(['Target'], axis=1)
            y = processed_data['Target']
            
            progress_placeholder = st.empty()
            start_time = time.time()
            
            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(model.optimize_models, X, y)
                
                while not future.done():
                    current_time = time.time()
                    progress = st.session_state.training_progress
                    elapsed = current_time - start_time
                    
                    with progress_placeholder.container():
                        st.caption(f"Status: {progress.status}")
                        st.progress(
                            progress.current_trial/MAX_TRIALS,
                            text=f"Trial {progress.current_trial}/{MAX_TRIALS} (Completed: {progress.trials_completed})"
                        )
                        cols = st.columns(2)
                        cols[0].metric("Best F1 Score", f"{progress.best_score:.2%}")
                        cols[1].metric("Latest F1 Score", f"{progress.latest_score:.2%}")
                        st.metric("Elapsed Time", f"{elapsed:.1f}s")
                        if progress.params:
                            st.write("Current Parameters:", progress.params)
                    
                    time.sleep(0.1)
                
                future.result()
                model.train(X, y)
                st.session_state.model = model
                st.success(f"Training completed in {time.time()-start_time:.1f}s")
                st.session_state.training_progress = None
                
        except Exception as e:
            st.error(f"Training failed: {str(e)}")
            st.session_state.training_progress = None
            st.session_state.model = None

    if st.session_state.model and not processed_data.empty:
        latest_data = processed_data.drop(columns=['Target']).iloc[[-1]].reset_index(drop=True)
        confidence = st.session_state.model.predict(latest_data)
        
        st.subheader("Trading Signal")
        col1, col2, col3 = st.columns(3)
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
