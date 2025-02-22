# trading_system.py
import logging
import numpy as np
import pandas as pd
import yfinance as yf
import streamlit as st
import plotly.express as px
import optuna
import pandas_ta as ta
from xgboost import XGBClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.feature_selection import RFECV
from sklearn.metrics import classification_report, roc_auc_score
from imblearn.over_sampling import SMOTE
from arch import arch_model
from tqdm import tqdm
import warnings
import shap
from sklearn.preprocessing import label_binarize

# Configuration
DEFAULT_SYMBOL = 'BTC-USD'
PRIMARY_INTERVAL = '15m'
TRADE_THRESHOLD_BUY = 0.60
TRADE_THRESHOLD_SELL = 0.40
MAX_TRIALS = 50
GARCH_WINDOW = 21
MIN_FEATURES = 10
HOLD_LOOKAHEAD = 6

warnings.filterwarnings("ignore", category=UserWarning)
logging.basicConfig(level=logging.INFO)

# Streamlit UI
st.set_page_config(page_title="AI Trading System", layout="wide")
st.title("🚀 Pandas-TA Trading System")

if 'model' not in st.session_state:
    st.session_state.model = None

@st.cache_data(ttl=300, show_spinner="Fetching market data...")
def fetch_data(symbol: str, interval: str) -> pd.DataFrame:
    try:
        df = yf.download(
            symbol,
            period="180d",
            interval=interval,
            progress=False,
            auto_adjust=True
        )
        df.columns = [col.lower() for col in df.columns]
        return df[['open', 'high', 'low', 'close', 'volume']].ffill().dropna()
    except Exception as e:
        logging.error(f"Data fetch failed: {str(e)}")
        return pd.DataFrame()

def calculate_features(df: pd.DataFrame) -> pd.DataFrame:
    try:
        df = df.copy()
        
        # Price transformations
        df['returns'] = df['close'].pct_change()
        df['log_returns'] = np.log(df['close']/df['close'].shift(1))
        
        # Momentum indicators
        df['rsi_14'] = ta.rsi(df['close'], length=14)
        macd = ta.macd(df['close'], fast=12, slow=26, signal=9)
        df['macd'] = macd['MACD_12_26_9']
        df['macd_signal'] = macd['MACDs_12_26_9']
        df['adx_14'] = ta.adx(df['high'], df['low'], df['close'], length=14)['ADX_14']
        
        # Volatility features
        df['atr_14'] = ta.atr(df['high'], df['low'], df['close'], length=14)
        df['volatility_21'] = df['returns'].rolling(21).std()
        
        # Volume analysis
        df['volume_ma_21'] = ta.sma(df['volume'], length=21)
        df['obv'] = ta.obv(df['close'], df['volume'])
        
        # Pattern recognition
        df['doji'] = ta.cdl_pattern(df['open'], df['high'], df['low'], df['close'], name="doji")
        
        # Target engineering
        future_returns = df['close'].pct_change().shift(-HOLD_LOOKAHEAD)
        df['target'] = pd.qcut(future_returns, q=3, labels=[0, 1, 2], duplicates='drop')
        
        return df.dropna()
    except Exception as e:
        logging.error(f"Feature engineering failed: {str(e)}")
        return pd.DataFrame()

class TradingModel:
    def __init__(self):
        self.model = None
        self.selected_features = []
        self.explainer = None
        self.feature_importances = pd.DataFrame()

    def optimize_model(self, X: pd.DataFrame, y: pd.Series) -> bool:
        try:
            tscv = TimeSeriesSplit(n_splits=5)
            study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=42))
            
            study.optimize(
                lambda trial: self._objective(trial, X, y, tscv),
                n_trials=MAX_TRIALS,
                n_jobs=-1
            )
            
            if study.best_trial:
                self._train_final_model(X, y, study.best_params)
                self._create_explainer(X)
                return True
            return False
        except Exception as e:
            logging.error(f"Training failed: {str(e)}")
            return False

    def _create_explainer(self, X: pd.DataFrame):
        try:
            self.explainer = shap.TreeExplainer(self.model)
            shap_values = self.explainer.shap_values(X[self.selected_features])
            self.feature_importances = pd.DataFrame({
                'feature': X[self.selected_features].columns,
                'importance': np.abs(shap_values).mean(0)
            }).sort_values('importance', ascending=False)
        except Exception as e:
            logging.error(f"SHAP error: {str(e)}")

    def _train_final_model(self, X: pd.DataFrame, y: pd.Series, params: dict):
        try:
            selector = RFECV(
                XGBClassifier(**params),
                step=1,
                cv=TimeSeriesSplit(3),
                min_features_to_select=MIN_FEATURES
            )
            selector.fit(X, y)
            self.selected_features = X.columns[selector.support_].tolist()
            self.model = XGBClassifier(**params).fit(X[self.selected_features], y)
        except Exception as e:
            logging.error(f"Training error: {str(e)}")
            raise

    def _objective(self, trial, X: pd.DataFrame, y: pd.Series, tscv) -> float:
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
            'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.3, log=True),
            'max_depth': trial.suggest_int('max_depth', 3, 9),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.4, 1.0),
            'gamma': trial.suggest_float('gamma', 0, 1.0)
        }
        
        scores = []
        for train_idx, test_idx in tqdm(tscv.split(X), desc="Optimizing"):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            
            try:
                sm = SMOTE(sampling_strategy='not majority')
                X_res, y_res = sm.fit_resample(X_train, y_train)
                model = XGBClassifier(**params).fit(X_res, y_res)
                y_proba = model.predict_proba(X_test)
                scores.append(roc_auc_score(label_binarize(y_test, classes=model.classes_), y_proba, multi_class='ovo'))
            except:
                scores.append(0.0)
        
        return np.mean(scores)

    def predict(self, X: pd.DataFrame) -> tuple:
        try:
            proba = self.model.predict_proba(X[self.selected_features])[0]
            explanation = self.explainer.shap_values(X[self.selected_features])[0]
            return np.max(proba), explanation
        except:
            return 0.5, None

def main():
    st.sidebar.header("Settings")
    symbol = st.sidebar.text_input("Crypto Symbol", DEFAULT_SYMBOL).upper()
    
    with st.spinner("Loading market data..."):
        raw_data = fetch_data(symbol, PRIMARY_INTERVAL)
        processed_data = calculate_features(raw_data)
    
    if not raw_data.empty:
        st.subheader(f"{symbol} Market Analysis")
        col1, col2 = st.columns([3, 1])
        with col1:
            fig = px.line(raw_data, y='close', title="Price Chart")
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            st.metric("Current Price", f"${raw_data['close'].iloc[-1]:.2f}")
            st.metric("Volatility", f"{processed_data['volatility_21'].iloc[-1]:.2%}")

    if st.sidebar.button("🚀 Train Model") and not processed_data.empty:
        model = TradingModel()
        X = processed_data.drop(columns=['target'])
        y = processed_data['target']
        
        with st.spinner("Training AI Model..."):
            if model.optimize_model(X, y):
                st.session_state.model = model
                st.success("Model trained successfully!")
                
                st.subheader("Model Insights")
                col1, col2 = st.columns(2)
                with col1:
                    st.write("Top Features:")
                    st.dataframe(model.feature_importances.head(10))
                with col2:
                    st.write("Model Configuration:")
                    st.json(model.model.get_params())

    if st.session_state.model and not processed_data.empty:
        model = st.session_state.model
        latest_data = processed_data.drop(columns=['target']).iloc[[-1]]
        
        confidence, explanation = model.predict(latest_data)
        current_vol = processed_data['volatility_21'].iloc[-1]
        
        st.subheader("Trading Signal")
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Confidence Score", f"{confidence:.2%}")
            if explanation is not None:
                st.write("Feature Impact:")
                fig = px.bar(
                    x=model.selected_features,
                    y=explanation,
                    labels={'x': 'Features', 'y': 'Impact'},
                    height=300
                )
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            buy_thresh = TRADE_THRESHOLD_BUY + (current_vol * 0.15)
            sell_thresh = TRADE_THRESHOLD_SELL - (current_vol * 0.15)
            
            if confidence > buy_thresh:
                st.success("🚀 Strong Buy Signal")
            elif confidence < sell_thresh:
                st.error("🔻 Strong Sell Signal")
            else:
                st.info("🛑 Hold Position")

if __name__ == "__main__":
    main()
