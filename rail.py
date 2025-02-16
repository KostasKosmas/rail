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
from sklearn.feature_selection import SelectFromModel, SelectKBest, f_classif
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import f1_score
from sklearn.utils.class_weight import compute_sample_weight
from imblearn.over_sampling import SMOTE
from datetime import datetime, timedelta
import warnings
from sklearn.cluster import KMeans  # Added import for KMeans

# ======================
# CONFIGURATION
# ======================
DEFAULT_SYMBOL = 'BTC-USD'
PRIMARY_INTERVAL = '15m'
FALLBACK_INTERVAL = '60m'
TRADE_THRESHOLD_BUY = 0.58
TRADE_THRESHOLD_SELL = 0.42
MAX_TRIALS = 50
GARCH_WINDOW = 14
DATA_RETRIES = 3
MIN_FEATURES = 7
VOLATILITY_CLUSTERS = 3

warnings.filterwarnings("ignore", category=UserWarning)
logging.basicConfig(level=logging.INFO)

# ======================
# STREAMLIT UI
# ======================
st.set_page_config(page_title="AI Trading System", layout="wide")
st.title("🚀 AI-Powered Cryptocurrency Trading System")
st.markdown("Real-time trading signals with machine learning and volatility modeling")

if 'model' not in st.session_state:
    st.session_state.model = None
if 'last_trained' not in st.session_state:
    st.session_state.last_trained = None
if 'training_progress' not in st.session_state:
    st.session_state.training_progress = None
if 'data_warning' not in st.session_state:
    st.session_state.data_warning = None

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
        ).reset_index()
        
        if not df.empty and len(df) > 100:
            st.session_state.data_warning = None
            return df.set_index('Datetime')
    except Exception as e:
        logging.warning(f"Primary data fetch failed: {str(e)}")
    
    try:
        df = pd.read_json(f"https://api.binance.com/api/v3/klines?symbol={symbol.replace('-', '')}&interval={interval}&limit=1000")
        df = df.iloc[:, :6]
        df.columns = ['Open', 'High', 'Low', 'Close', 'Volume', 'Close time']
        df['Date'] = pd.to_datetime(df['Close time'], unit='ms')
        st.session_state.data_warning = f"Using Binance {interval} data"
        return df.set_index('Date')
    except Exception as e:
        st.error(f"Data unavailable for {symbol}")
        return pd.DataFrame()

def calculate_features(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or len(df) < 100:
        return pd.DataFrame()
        
    df = df.copy().reset_index()
    try:
        # Core price features
        df['Returns'] = df['Close'].pct_change()
        df['Log_Returns'] = np.log(df['Close']).diff()
        
        # Enhanced technical indicators
        windows = [20, 50, 100, 200]
        for window in windows:
            df[f'SMA_{window}'] = df['Close'].rolling(window).mean()
            df[f'STD_{window}'] = df['Close'].rolling(window).std()
            df[f'RSI_{window}'] = 100 - (100 / (1 + (df['Close'].diff().clip(lower=0).rolling(window).mean() / 
                                                   df['Close'].diff().clip(upper=0).abs().rolling(window).mean())))
        
        # Volume features
        df['VWAP'] = (df['Volume'] * (df['High'] + df['Low'] + df['Close']) / 3).cumsum() / df['Volume'].cumsum()
        df['Volume_Change'] = df['Volume'].pct_change()
        
        # Advanced volatility features
        returns = df['Log_Returns'].dropna()
        df['Volatility'] = returns.rolling(GARCH_WINDOW).std()
        
        # Volatility clustering with proper index alignment
        vol_series = df['Volatility'].dropna()
        if len(vol_series) >= VOLATILITY_CLUSTERS:
            kmeans = KMeans(n_clusters=VOLATILITY_CLUSTERS)
            clusters = kmeans.fit_predict(vol_series.values.reshape(-1, 1))
            df['Vol_Cluster'] = pd.Series(clusters, index=vol_series.index)
        
        # Momentum features
        for period in [3, 7, 14]:
            df[f'Momentum_{period}'] = df['Close'].pct_change(period)
        
        # Target engineering
        df['Target'] = pd.cut(df['Returns'].shift(-1), 
                            bins=[-np.inf, -0.01, 0.01, np.inf],
                            labels=[0, 1, 2])
        
        return df.dropna().set_index('Datetime') if 'Datetime' in df.columns else df.dropna()
    except Exception as e:
        logging.error(f"Feature engineering failed: {str(e)}")
        return pd.DataFrame()

# ======================
# ADVANCED MODEL PIPELINE
# ======================
class TradingModel:
    def __init__(self):
        self.feature_selector = None
        self.selected_features = []
        self.model_rf = RandomForestClassifier(class_weight='balanced', random_state=42)
        self.model_gb = GradientBoostingClassifier(random_state=42)
        self.calibrated_rf = None
        self.calibrated_gb = None
        self.smote = SMOTE(random_state=42, k_neighbors=5)
        self.study = None
        self.best_score = 0

    def _progress_callback(self, study, trial):
        if st.session_state.training_progress:
            progress = (trial.number + 1) / MAX_TRIALS
            text = (f"Trial {trial.number + 1}/{MAX_TRIALS} - "
                    f"Best F1: {study.best_value:.2%}")
            st.session_state.training_progress.progress(progress, text=text)

    def _safe_feature_selection(self, X, y):
        try:
            if not isinstance(X, pd.DataFrame):
                X = pd.DataFrame(X)
            X.columns = [str(col) for col in X.columns]
            
            self.feature_selector = SelectFromModel(
                GradientBoostingClassifier(n_estimators=100),
                threshold="1.25*median"
            )
            self.feature_selector.fit(X, y)
            self.selected_features = X.columns[self.feature_selector.get_support()]
            
            if len(self.selected_features) < MIN_FEATURES:
                self.selected_features = X.columns[:MIN_FEATURES]
            
            return X[self.selected_features]
        except Exception as e:
            logging.error(f"Feature selection failed: {str(e)}")
            return X.iloc[:, :MIN_FEATURES]

    def optimize_models(self, X: pd.DataFrame, y: pd.Series):
        try:
            X_sel = self._safe_feature_selection(X, y)
            
            self.study = optuna.create_study(
                direction='maximize',
                sampler=optuna.samplers.TPESampler(n_startup_trials=20)
            )
            self.study.optimize(
                lambda trial: self._objective(trial, X_sel, y),
                n_trials=MAX_TRIALS,
                timeout=3600,
                callbacks=[self._progress_callback]
            )
        except Exception as e:
            logging.error(f"Optimization failed: {str(e)}")
            raise

    def _objective(self, trial, X, y):
        try:
            rf_params = {
                'n_estimators': trial.suggest_int('rf_n_estimators', 300, 1000),
                'max_depth': trial.suggest_int('rf_max_depth', 15, 40),
                'min_samples_split': trial.suggest_int('rf_min_samples_split', 10, 30),
                'max_features': trial.suggest_categorical('rf_max_features', ['sqrt', 'log2'])
            }
            
            gb_params = {
                'n_estimators': trial.suggest_int('gb_n_estimators', 300, 1000),
                'learning_rate': trial.suggest_float('gb_learning_rate', 0.05, 0.3, log=True),
                'max_depth': trial.suggest_int('gb_max_depth', 5, 15),
                'subsample': trial.suggest_float('gb_subsample', 0.8, 1.0)
            }
            
            tscv = TimeSeriesSplit(n_splits=3)
            scores = []
            
            for train_idx, test_idx in tscv.split(X):
                X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
                y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
                
                X_res, y_res = self.smote.fit_resample(X_train, y_train)
                
                rf = RandomForestClassifier(**rf_params).fit(X_res, y_res)
                gb = GradientBoostingClassifier(**gb_params).fit(X_res, y_res)
                
                preds = 0.6*rf.predict_proba(X_test) + 0.4*gb.predict_proba(X_test)
                f1 = f1_score(y_test, np.argmax(preds, axis=1), average='weighted')
                scores.append(f1)
                
            return np.mean(scores)
        except Exception as e:
            logging.warning(f"Trial failed: {str(e)}")
            return 0.0

    def train(self, X: pd.DataFrame, y: pd.Series):
        try:
            X_sel = X[self.selected_features]
            X_res, y_res = self.smote.fit_resample(X_sel, y)
            
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
            if not self.selected_features:
                raise ValueError("No features selected for prediction")
                
            X_sel = X[self.selected_features]
            prob_rf = self.calibrated_rf.predict_proba(X_sel)
            prob_gb = self.calibrated_gb.predict_proba(X_sel)
            ensemble_probs = 0.6*prob_rf + 0.4*prob_gb
            return ensemble_probs[:, 2]
        except Exception as e:
            logging.error(f"Prediction failed: {str(e)}")
            return np.zeros(len(X))

# ======================
# STREAMLIT INTERFACE
# ======================
def main():
    st.sidebar.header("Settings")
    symbol = st.sidebar.text_input("Cryptocurrency Symbol", DEFAULT_SYMBOL).upper()
    
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
            except Exception as e:
                st.error(f"Display error: {str(e)}")

    st.sidebar.header("Model Controls")
    if st.sidebar.button("🚀 Train Model"):
        if processed_data.empty or 'Target' not in processed_data.columns:
            st.error("Insufficient data for training")
            return
            
        try:
            st.session_state.training_progress = st.progress(0, text="Initializing...")
            model = TradingModel()
            
            X = processed_data.drop(['Target'], axis=1, errors='ignore')
            y = processed_data['Target']
            
            with st.spinner("Optimizing trading models..."):
                model.optimize_models(X, y)
                model.train(X, y)
            
            st.session_state.model = model
            st.session_state.last_trained = datetime.now()
            st.success(f"Model trained successfully! Best F1: {model.study.best_value:.2%}")
            
        except Exception as e:
            st.error(f"Training failed: {str(e)}")
        finally:
            time.sleep(1)
            st.session_state.training_progress = None

    if st.session_state.model and not processed_data.empty:
        try:
            latest_data = processed_data[st.session_state.model.selected_features].iloc[[-1]]
            confidence = st.session_state.model.predict(latest_data)[0]
            
            st.subheader("Trading Signal")
            col1, col2, col3 = st.columns(3)
            col1.metric("Confidence Score", f"{confidence:.2%}")
            
            current_vol = processed_data['Volatility'].iloc[-1]
            adj_buy = TRADE_THRESHOLD_BUY + (current_vol * 0.1)
            adj_sell = TRADE_THRESHOLD_SELL - (current_vol * 0.1)
            
            if confidence > adj_buy:
                col2.success("🚀 Strong Buy Signal")
                col3.write(f"Volatility-adjusted threshold: >{adj_buy:.0%}")
            elif confidence < adj_sell:
                col2.error("🔻 Strong Sell Signal")
                col3.write(f"Volatility-adjusted threshold: <{adj_sell:.0%}")
            else:
                col2.info("🛑 Hold Position")
                col3.write("No clear market signal")
            
            if st.session_state.last_trained:
                st.caption(f"Last trained: {st.session_state.last_trained.strftime('%Y-%m-%d %H:%M')}")
                
        except Exception as e:
            st.error(f"Prediction error: {str(e)}")

if __name__ == "__main__":
    main()
