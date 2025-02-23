# trading_system.py
import logging
import numpy as np
import pandas as pd
import yfinance as yf
import streamlit as st
import plotly.express as px
import optuna
from xgboost import XGBClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.feature_selection import RFECV
from sklearn.metrics import classification_report, roc_auc_score
from imblearn.over_sampling import SMOTE
from tqdm import tqdm
import warnings

# Configuration
DEFAULT_SYMBOL = 'BTC-USD'
PRIMARY_INTERVAL = '15m'
TRADE_THRESHOLD_BUY = 0.60
TRADE_THRESHOLD_SELL = 0.40
MAX_TRIALS = 50
GARCH_WINDOW = 21
MIN_FEATURES = 10
HOLD_LOOKAHEAD = 6
MAX_RETURN = 2.0  # 200% maximum daily return
EPSILON = 1e-6

warnings.filterwarnings("ignore", category=UserWarning)
logging.basicConfig(level=logging.INFO)

# Streamlit UI Configuration
st.set_page_config(page_title="AI Trading System", layout="wide")
st.title("🚀 Cryptocurrency Trading System")

if 'model' not in st.session_state:
    st.session_state.model = None

# Data Processing Functions
def clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize column names from Yahoo Finance response"""
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = ['_'.join(col).lower().replace(' ', '_') for col in df.columns]
    else:
        df.columns = [str(col).lower().replace(' ', '_') for col in df.columns]
    return df

def handle_missing_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and impute missing values"""
    df = df.replace([np.inf, -np.inf], np.nan)
    numeric_cols = df.select_dtypes(include=np.number).columns
    df[numeric_cols] = df[numeric_cols].ffill().bfill()
    return df.dropna()

@st.cache_data(ttl=300, show_spinner="Fetching market data...")
def fetch_data(symbol: str, interval: str) -> pd.DataFrame:
    """Fetch and validate market data from Yahoo Finance"""
    try:
        period_map = {
            '1m': '7d', '2m': '60d', '5m': '60d', 
            '15m': '60d', '30m': '60d', '60m': '730d', '1d': '3650d'
        }
        period = period_map.get(interval, '60d')
        
        df = yf.download(
            symbol,
            period=period,
            interval=interval,
            progress=False,
            auto_adjust=True
        )
        
        if df.empty:
            st.error("No data returned from Yahoo Finance")
            return pd.DataFrame()
            
        df = clean_column_names(df)
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        missing = [col for col in required_cols if col not in df.columns]
        
        if missing:
            st.error(f"Missing columns: {', '.join(missing)}")
            return pd.DataFrame()
            
        return handle_missing_data(df[required_cols])
    
    except Exception as e:
        logging.error(f"Data fetch failed: {str(e)}")
        return pd.DataFrame()

def calculate_features(df: pd.DataFrame) -> pd.DataFrame:
    """Feature engineering pipeline with safeguards"""
    try:
        if df.empty:
            return pd.DataFrame()
            
        df = df.copy()
        
        # Price transformations
        df['returns'] = df['close'].pct_change()
        df['returns'] = df['returns'].replace([np.inf, -np.inf], np.nan)
        df['returns'] = df['returns'].fillna(0).clip(-MAX_RETURN, MAX_RETURN)
        
        # Volatility features
        df['volatility'] = df['returns'].rolling(
            GARCH_WINDOW, min_periods=5
        ).std().fillna(0)
        
        # Technical indicators
        df['ema_12'] = df['close'].ewm(span=12, adjust=False).mean()
        df['ema_26'] = df['close'].ewm(span=26, adjust=False).mean()
        df['macd'] = df['ema_12'] - df['ema_26']
        df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
        
        # Volume features
        df['volume'] = df['volume'].replace(0, EPSILON)
        df['volume_ma'] = df['volume'].rolling(14).mean().fillna(EPSILON)
        df['volume_change'] = df['volume'].pct_change().fillna(0)
        
        # Target engineering
        future_returns = df['close'].pct_change().shift(-HOLD_LOOKAHEAD)
        future_returns = future_returns.clip(-MAX_RETURN, MAX_RETURN)
        df['target'] = pd.qcut(future_returns, q=3, labels=[0, 1, 2], duplicates='drop')
        
        return handle_missing_data(df.dropna())
        
    except Exception as e:
        logging.error(f"Feature engineering failed: {str(e)}")
        return pd.DataFrame()

# Machine Learning Model
class TradingModel:
    def __init__(self):
        self.model = None
        self.selected_features = []
        self.feature_importances = pd.DataFrame()

    def optimize_model(self, X: pd.DataFrame, y: pd.Series) -> bool:
        """Hyperparameter optimization with Optuna"""
        try:
            if X.empty or y.empty or len(y.unique()) < 2:
                st.error("Insufficient data for training")
                return False

            tscv = TimeSeriesSplit(n_splits=3)
            study = optuna.create_study(direction='maximize')
            
            study.optimize(
                lambda trial: self._objective(trial, X, y, tscv),
                n_trials=MAX_TRIALS,
                n_jobs=-1,
                catch=(ValueError,)
            )
            
            if study.best_trial:
                self._train_final_model(X, y, study.best_params)
                return True
            return False
            
        except Exception as e:
            logging.error(f"Training failed: {str(e)}")
            return False

    def _train_final_model(self, X: pd.DataFrame, y: pd.Series, params: dict):
        """Train final model with feature selection"""
        try:
            params['missing'] = np.nan  # Handle missing values
            
            selector = RFECV(
                XGBClassifier(**params),
                step=1,
                cv=TimeSeriesSplit(3),
                min_features_to_select=MIN_FEATURES
            )
            selector.fit(X, y)
            self.selected_features = X.columns[selector.support_].tolist()
            
            self.model = XGBClassifier(**params).fit(
                X[self.selected_features], 
                y,
                eval_set=[(X[self.selected_features], y)],
                early_stopping_rounds=10,
                verbose=False
            )
            
            self.feature_importances = pd.Series(
                self.model.feature_importances_,
                index=self.selected_features
            ).sort_values(ascending=False)
            
        except Exception as e:
            logging.error(f"Model training failed: {str(e)}")
            raise

    def _objective(self, trial, X: pd.DataFrame, y: pd.Series, tscv) -> float:
        """Optuna objective function for hyperparameter tuning"""
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 500),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'max_depth': trial.suggest_int('max_depth', 3, 9),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'gamma': trial.suggest_float('gamma', 0, 0.5)
        }
        
        scores = []
        
        for train_idx, test_idx in tscv.split(X):
            try:
                X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
                y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
                
                if len(y_train.unique()) < 2:
                    scores.append(0.0)
                    continue
                
                sm = SMOTE(random_state=42)
                X_res, y_res = sm.fit_resample(X_train, y_train)
                
                model = XGBClassifier(**params, missing=np.nan)
                model.fit(X_res, y_res)
                
                y_proba = model.predict_proba(X_test)
                score = roc_auc_score(y_test, y_proba, multi_class='ovo')
                scores.append(score)
                
            except Exception as e:
                scores.append(0.0)
        
        return np.nanmean(scores)

    def predict(self, X: pd.DataFrame) -> float:
        """Make prediction with confidence score"""
        if not self.model or X.empty:
            return 0.5
            
        try:
            if not all(feat in X.columns for feat in self.selected_features):
                raise ValueError("Missing features for prediction")
                
            proba = self.model.predict_proba(X[self.selected_features])[0]
            return max(0.0, min(1.0, np.max(proba)))
        except Exception as e:
            logging.error(f"Prediction failed: {str(e)}")
            return 0.5

# Main Application
def main():
    """Streamlit main application interface"""
    st.sidebar.header("Settings")
    symbol = st.sidebar.text_input("Crypto Symbol", DEFAULT_SYMBOL).upper()
    interval = st.sidebar.selectbox("Interval", ["15m", "30m", "1h", "1d"], index=0)
    
    with st.spinner("Loading market data..."):
        raw_data = fetch_data(symbol, interval)
        processed_data = calculate_features(raw_data)
    
    if not raw_data.empty and not processed_data.empty:
        col1, col2 = st.columns([3, 1])
        with col1:
            st.subheader(f"{symbol} Price Chart")
            fig = px.line(raw_data, x=raw_data.index, y='close')
            st.plotly_chart(fig, use_container_width=True)
            
        with col2:
            st.metric("Current Price", f"${raw_data['close'].iloc[-1]:.2f}")
            st.metric("24h Volatility", f"{processed_data['volatility'].iloc[-1]:.2%}")
            st.metric("Volume", f"{raw_data['volume'].iloc[-1]:,.2f}")

    if st.sidebar.button("🚀 Train Model") and not processed_data.empty:
        model = TradingModel()
        X = processed_data.drop(columns=['target'])
        y = processed_data['target']
        
        with st.spinner("Training AI Model..."):
            if model.optimize_model(X, y):
                st.session_state.model = model
                st.success("Model training completed!")
                
                st.subheader("Model Insights")
                col1, col2 = st.columns(2)
                with col1:
                    st.write("Top Predictive Features:")
                    st.dataframe(model.feature_importances.head(10))
                with col2:
                    st.write("Model Configuration:")
                    st.json(model.model.get_params())
            else:
                st.error("Model training failed. Check data quality.")

    if st.session_state.model and not processed_data.empty:
        model = st.session_state.model
        latest_data = processed_data.drop(columns=['target']).iloc[[-1]]
        
        confidence = model.predict(latest_data)
        current_vol = processed_data['volatility'].iloc[-1]
        
        st.subheader("Trading Signal")
        col1, col2 = st.columns(2)
        col1.metric("Model Confidence", f"{confidence:.2%}")
        
        adj_buy = TRADE_THRESHOLD_BUY + (current_vol * 0.15)
        adj_sell = TRADE_THRESHOLD_SELL - (current_vol * 0.15)
        
        if confidence > adj_buy:
            col2.success("🚀 Strong Buy Signal")
        elif confidence < adj_sell:
            col2.error("🔻 Strong Sell Signal")
        else:
            col2.info("🛑 Hold Position")

if __name__ == "__main__":
    main()
