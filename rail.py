# trading_system.py
import logging
import numpy as np
import pandas as pd
import yfinance as yf
import streamlit as st
import plotly.express as px
import optuna
import requests
from xgboost import XGBClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.feature_selection import RFECV
from sklearn.metrics import roc_auc_score
import warnings
import json

# Configuration
DEFAULT_SYMBOL = 'BTC-USD'
INTERVAL_OPTIONS = ["15m", "30m", "1h", "1d"]
TRADE_THRESHOLD_BUY = 0.65
TRADE_THRESHOLD_SELL = 0.35
MAX_TRIALS = 50
GARCH_WINDOW = 21
MIN_FEATURES = 10
HOLD_LOOKAHEAD = 6
MAX_RETRIES = 3
VALIDATION_WINDOW = 21

warnings.filterwarnings("ignore", category=UserWarning)
logging.basicConfig(level=logging.INFO)

# Streamlit UI Configuration
st.set_page_config(page_title="AI Trading System", layout="wide")
st.title("🚀 Smart Crypto Trading Assistant")

# Session State Management
if 'model' not in st.session_state:
    st.session_state.update({
        'model': None,
        'processed_data': None,
        'training_progress': {
            'completed': 0,
            'current_score': 0.0,
            'best_score': 0.0
        },
        'data_loaded': False
    })

def safe_yf_download(symbol: str, **kwargs) -> pd.DataFrame:
    """Robust data downloader with error handling and retries"""
    for _ in range(MAX_RETRIES):
        try:
            data = yf.download(
                symbol,
                group_by='ticker',
                progress=False,
                auto_adjust=True,
                **kwargs
            )
            if not data.empty:
                return data
            st.error(f"No data found for {symbol}")
            return pd.DataFrame()
        except (json.JSONDecodeError, requests.exceptions.RequestException):
            continue
        except Exception as e:
            logging.error(f"Critical download error: {str(e)}")
            break
    return pd.DataFrame()

def normalize_columns(symbol: str, columns) -> list:
    """Normalize column names and remove ticker prefixes"""
    normalized_symbol = symbol.lower().replace('-', '')
    processed_cols = []
    
    for col in columns:
        if isinstance(col, tuple):
            col = '_'.join(map(str, col))
        
        col = str(col).lower() \
                      .replace('-', '') \
                      .replace(' ', '_') \
                      .replace(f"{normalized_symbol}_", "")
        
        # Handle common column variations
        col = {
            'adjclose': 'close',
            'adjusted_close': 'close',
            'vol': 'volume',
            'vwap': 'close'
        }.get(col, col)
        
        processed_cols.append(col)
    
    return processed_cols

@st.cache_data(ttl=300, show_spinner="Fetching market data...")
def fetch_data(symbol: str, interval: str) -> pd.DataFrame:
    """Data acquisition and preprocessing pipeline"""
    try:
        period_map = {
            '15m': '60d', '30m': '60d', 
            '1h': '730d', '1d': 'max'
        }
        
        df = safe_yf_download(
            symbol,
            period=period_map.get(interval, '60d'),
            interval=interval
        )
        
        if df.empty:
            st.error("Data source returned empty dataset")
            return pd.DataFrame()

        # Column normalization
        df.columns = normalize_columns(symbol, df.columns)
        
        # Validate required columns
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            st.error(f"Missing critical columns: {', '.join(missing_cols)}")
            return pd.DataFrame()

        # Clean and validate data
        df = df[required_cols] \
            .replace([np.inf, -np.inf], np.nan) \
            .ffill() \
            .bfill() \
            .dropna()
        
        return df if not df.empty else pd.DataFrame()
    
    except Exception as e:
        st.error(f"Data acquisition failed: {str(e)}")
        return pd.DataFrame()

def calculate_features(df: pd.DataFrame) -> pd.DataFrame:
    """Comprehensive feature engineering pipeline"""
    try:
        if df.empty:
            st.error("Empty DataFrame received for feature engineering")
            return pd.DataFrame()

        required_cols = ['open', 'high', 'low', 'close', 'volume']
        if any(col not in df.columns for col in required_cols):
            st.error("Missing required columns for feature engineering")
            return pd.DataFrame()

        df = df.copy()
        
        # Price dynamics features
        df['returns'] = df['close'].pct_change().fillna(0)
        df['log_returns'] = np.log1p(df['returns']).fillna(0)

        # Technical indicators
        for span in [12, 26]:
            df[f'ema_{span}'] = df['close'].ewm(span=span, adjust=False).mean()
        df['macd'] = df['ema_12'] - df['ema_26']
        df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()

        # Volatility metrics
        df['volatility'] = df['returns'].rolling(GARCH_WINDOW).std().fillna(0)
        
        # ATR calculation with vectorized operations
        try:
            prev_close = df['close'].shift(1).bfill()
            tr = pd.DataFrame({
                'high_low': df['high'] - df['low'],
                'high_prev_close': (df['high'] - prev_close).abs(),
                'low_prev_close': (df['low'] - prev_close).abs()
            }).max(axis=1)
            df['atr'] = tr.rolling(14).mean().fillna(0)
        except KeyError as e:
            st.error(f"Missing column for ATR calculation: {str(e)}")
            return pd.DataFrame()

        # Volume analysis
        df['volume_ma'] = df['volume'].rolling(14).mean().fillna(0)
        df['volume_change'] = df['volume'].pct_change().fillna(0)

        # Target engineering
        try:
            future_returns = df['close'].pct_change(HOLD_LOOKAHEAD).shift(-HOLD_LOOKAHEAD)
            valid_returns = future_returns.dropna()
            
            if len(valid_returns) < 100:
                st.error("Insufficient data for target calculation")
                return pd.DataFrame()
                
            df['target'] = pd.qcut(valid_returns, q=3, labels=[0, 1, 2], duplicates='drop')
            df = df.dropna()
            
            if df.empty:
                st.error("No data remaining after target engineering")
                return pd.DataFrame()

            return df.replace([np.inf, -np.inf], np.nan).ffill().bfill().dropna()
        
        except Exception as e:
            st.error(f"Target calculation failed: {str(e)}")
            return pd.DataFrame()

    except Exception as e:
        st.error(f"Feature engineering failed: {str(e)}")
        return pd.DataFrame()

class TradingModel:
    def __init__(self):
        self.model = None
        self.selected_features = []
        self.feature_importances = pd.DataFrame()
        self.study = None

    def _update_progress(self, trial_number: int, current_score: float, best_score: float):
        st.session_state.training_progress = {
            'completed': trial_number,
            'current_score': current_score,
            'best_score': best_score
        }
        st.experimental_rerun()

    def optimize_model(self, X: pd.DataFrame, y: pd.Series) -> bool:
        """Optimization pipeline with real-time progress tracking"""
        try:
            if X.empty or y.nunique() < 2:
                st.error("Insufficient training data")
                return False

            class ProgressTracker:
                def __init__(self, outer):
                    self.outer = outer
                    self.trial_count = 0

                def __call__(self, study, trial):
                    self.trial_count += 1
                    self.outer._update_progress(
                        self.trial_count,
                        trial.value if trial.value else 0.0,
                        study.best_value
                    )

            tscv = TimeSeriesSplit(n_splits=3, test_size=VALIDATION_WINDOW)
            
            self.study = optuna.create_study(direction='maximize')
            self.study.optimize(
                lambda trial: self._objective(trial, X, y, tscv),
                n_trials=MAX_TRIALS,
                n_jobs=1,  # Single job for better UI updates
                callbacks=[ProgressTracker(self)],
                show_progress_bar=False
            )

            if self.study.best_trial:
                return self._train_final_model(X, y, self.study.best_params)
            return False
            
        except Exception as e:
            st.error(f"Training process failed: {str(e)}")
            return False

    def _train_final_model(self, X: pd.DataFrame, y: pd.Series, params: dict) -> bool:
        """Final model training with feature selection"""
        try:
            X = X.ffill().bfill()
            
            selector = RFECV(
                XGBClassifier(**params),
                step=1,
                cv=TimeSeriesSplit(3),
                min_features_to_select=MIN_FEATURES
            )
            selector.fit(X, y)
            self.selected_features = X.columns[selector.support_]
            
            if len(self.selected_features) < MIN_FEATURES:
                raise ValueError("Insufficient features selected for modeling")

            self.model = XGBClassifier(
                **params,
                tree_method='hist',
                eval_metric='auc',
                enable_categorical=False
            ).fit(X[self.selected_features], y)
            
            self.feature_importances = pd.Series(
                self.model.feature_importances_,
                index=self.selected_features
            ).sort_values(ascending=False)

            return True

        except Exception as e:
            st.error(f"Model training error: {str(e)}")
            return False

    def _objective(self, trial, X: pd.DataFrame, y: pd.Series, tscv) -> float:
        """Objective function for Optuna optimization"""
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 500),
            'learning_rate': trial.suggest_float('learning_rate', 1e-3, 0.3, log=True),
            'max_depth': trial.suggest_int('max_depth', 3, 9),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'gamma': trial.suggest_float('gamma', 0, 0.5),
            'reg_alpha': trial.suggest_float('reg_alpha', 0, 1),
            'reg_lambda': trial.suggest_float('reg_lambda', 0, 1)
        }
        
        scores = []
        for train_idx, test_idx in tscv.split(X):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            
            try:
                model = XGBClassifier(**params)
                model.fit(X_train, y_train)
                y_proba = model.predict_proba(X_test)
                scores.append(roc_auc_score(y_test, y_proba, multi_class='ovo'))
            except Exception as e:
                scores.append(0.0)
                logging.warning(f"Trial failed: {str(e)}")
        
        return np.mean(scores)

    def predict(self, X: pd.DataFrame) -> float:
        """Generate prediction with confidence score"""
        try:
            if not self.model or X.empty:
                return 0.5
                
            X_clean = X[self.selected_features].ffill().bfill()
            if X_clean.isnull().values.any():
                raise ValueError("NaN values in prediction data")
                
            proba = self.model.predict_proba(X_clean)[0]
            return np.clip(np.max(proba), 0.0, 1.0)
        except Exception as e:
            logging.error(f"Prediction failed: {str(e)}")
            return 0.5

def main():
    """Main application interface"""
    st.sidebar.header("Configuration")
    symbol = st.sidebar.text_input("Asset Symbol", DEFAULT_SYMBOL).upper().strip()
    interval = st.sidebar.selectbox("Time Interval", INTERVAL_OPTIONS, index=2)
    
    # Data loading section
    if st.sidebar.button("🔄 Load Market Data"):
        with st.spinner("Fetching and processing data..."):
            raw_data = fetch_data(symbol, interval)
            processed_data = calculate_features(raw_data)
            st.session_state.processed_data = processed_data if not processed_data.empty else None
            st.session_state.data_loaded = True
            st.experimental_rerun()

    # Display market data
    if st.session_state.get('data_loaded', False):
        if st.session_state.processed_data is not None:
            raw_data = fetch_data(symbol, interval)
            
            col1, col2 = st.columns([3, 1])
            with col1:
                st.subheader(f"{symbol} Price Action")
                fig = px.line(raw_data, x=raw_data.index, y='close', 
                            title=f"{symbol} Price Chart ({interval})")
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.metric("Current Price", f"${raw_data['close'].iloc[-1]:.2f}")
                st.metric("Market Volatility", 
                         f"{st.session_state.processed_data['volatility'].iloc[-1]:.2%}",
                         help="21-day rolling volatility")

    # Training progress display
    if st.session_state.training_progress['completed'] > 0:
        st.subheader("Training Progress")
        prog = st.session_state.training_progress
        cols = st.columns(3)
        cols[0].metric("Completed Trials", f"{prog['completed']}/{MAX_TRIALS}")
        cols[1].metric("Current Score", f"{prog['current_score']:.2%}")
        cols[2].metric("Best Score", f"{prog['best_score']:.2%}")

    # Model training section
    if st.sidebar.button("🚀 Train Trading Model") and st.session_state.data_loaded:
        if st.session_state.processed_data is None:
            st.error("No valid data available for training")
            return
            
        model = TradingModel()
        X = st.session_state.processed_data.drop(columns=['target'])
        y = st.session_state.processed_data['target']
        
        st.session_state.training_progress = {'completed': 0, 'current_score': 0.0, 'best_score': 0.0}
        
        with st.spinner("Optimizing trading strategy..."):
            if model.optimize_model(X, y):
                st.session_state.model = model
                st.success("Model training completed!")
                
                if not model.feature_importances.empty:
                    st.subheader("Model Insights")
                    st.dataframe(
                        model.feature_importances.reset_index().rename(
                            columns={'index': 'Feature', 0: 'Importance'}
                        ).style.format({'Importance': '{:.2%}'}),
                        height=400,
                        use_container_width=True
                    )

    # Trading signals section
    if st.session_state.model and st.session_state.processed_data is not None:
        try:
            processed_data = st.session_state.processed_data
            model = st.session_state.model
            latest_data = processed_data.drop(columns=['target']).iloc[[-1]]
            
            confidence = model.predict(latest_data)
            current_vol = processed_data['volatility'].iloc[-1]
            
            st.subheader("Trading Advisory")
            col1, col2 = st.columns(2)
            col1.metric("Model Confidence", f"{confidence:.2%}")
            
            adj_buy = TRADE_THRESHOLD_BUY + (current_vol * 0.15)
            adj_sell = TRADE_THRESHOLD_SELL - (current_vol * 0.15)
            
            if confidence > adj_buy:
                col2.success("🚀 Strong Buy Signal")
            elif confidence < adj_sell:
                col2.error("🔻 Strong Sell Signal")
            else:
                col2.info("🛑 Market Neutral")
            
            st.caption(f"Volatility-adjusted thresholds: Buy >{adj_buy:.0%}, Sell <{adj_sell:.0%}")

        except Exception as e:
            st.error(f"Signal generation error: {str(e)}")

if __name__ == "__main__":
    main()
