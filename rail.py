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
from pykalman import KalmanFilter
from arch import arch_model
from tqdm import tqdm
import warnings
import re

# Configuration
DEFAULT_SYMBOL = 'BTC-USD'
PRIMARY_INTERVAL = '15m'
TRADE_THRESHOLD_BUY = 0.58
TRADE_THRESHOLD_SELL = 0.42
MAX_TRIALS = 20
GARCH_WINDOW = 14
MIN_FEATURES = 7
MIN_SAMPLES_PER_CLASS = 50
REQUIRED_FEATURES = ['ema_12', 'ema_26', 'macd', 'signal', 'volatility', 'liquidity_ratio', 'price_pressure']

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
            period="60d",
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
        return df[['open', 'high', 'low', 'close', 'volume']].ffill().dropna()
    
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
        
        # SMAs and RSI
        for window in [20, 50, 100]:
            df[f'sma_{window}'] = df['close'].rolling(window).mean()
            delta = df['close'].diff()
            gain = delta.where(delta > 0, 0)
            loss = -delta.where(delta < 0, 0)
            avg_gain = gain.rolling(window).mean()
            avg_loss = loss.rolling(window).mean()
            rs = avg_gain / avg_loss.replace(0, 1)
            df[f'rsi_{window}'] = 100 - (100 / (1 + rs))
        
        # Volatility features
        df['volatility'] = df['returns'].rolling(GARCH_WINDOW).std()
        garch = arch_model(df['returns'].dropna(), vol='GARCH', dist='normal')
        garch_fit = garch.fit(disp='off')
        df['garch_vol'] = garch_fit.conditional_volatility
        
        # Triple Barrier Labeling
        future_returns = df['close'].pct_change().shift(-4)
        df['target'] = pd.qcut(future_returns, q=3, labels=[0, 1, 2], duplicates='drop')
        
        # Market microstructure features
        df['liquidity_ratio'] = df['volume'] / df['volatility'].replace(0, 1e-6)
        df['price_pressure'] = (df['high'] - df['close']) / (df['close'] - df['low'])
        
        return df.dropna().drop(columns=['open', 'high', 'low', 'close', 'volume'])
    
    except Exception as e:
        logging.error(f"Feature engineering failed: {str(e)}", exc_info=True)
        return pd.DataFrame()

class AdvancedTradingModel:
    def __init__(self):
        self.selected_features = []
        self.model = None
        self.required_features = REQUIRED_FEATURES
        self.is_trained = False
        self.regimes = None
        self.feature_importances_ = None
        self.classes_ = None

    def optimize_model(self, X: pd.DataFrame, y: pd.Series) -> bool:
        try:
            self._reset_state()
            if not self._validate_inputs(X, y):
                return False

            tscv = TimeSeriesSplit(n_splits=3)
            study = optuna.create_study(direction='maximize')
            
            study.optimize(
                lambda trial: self._objective(trial, X, y, tscv),
                n_trials=MAX_TRIALS,
                n_jobs=-1,
                catch=(ValueError,)
            )
            
            if not study.best_trial:
                st.error("No successful trials completed")
                return False
                
            self._train_final_model(X, y, study.best_params)
            self._detect_regimes(X)
            
            self.is_trained = True
            return True
            
        except Exception as e:
            logging.error(f"Training failed: {str(e)}", exc_info=True)
            st.error("Model training failed. Check logs for details.")
            return False

    def _detect_regimes(self, X: pd.DataFrame):
        """Market regime detection using Kalman Filter"""
        try:
            kf = KalmanFilter(
                initial_state_mean=X.iloc[0],
                n_dim_obs=X.shape[1],
                em_vars=['transition_covariance', 'observation_covariance']
            )
            self.regimes, _ = kf.em(X).smooth(X.values)
        except Exception as e:
            logging.error(f"Regime detection failed: {str(e)}")
            self.regimes = np.zeros(len(X))

    def _reset_state(self):
        self.selected_features = []
        self.model = None
        self.is_trained = False
        self.regimes = None
        self.feature_importances_ = None
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
        try:
            tscv = TimeSeriesSplit(n_splits=3)
            selector = RFECV(
                XGBClassifier(**best_params),
                step=1,
                cv=tscv,
                min_features_to_select=MIN_FEATURES
            )
            selector.fit(X, y)
            self.selected_features = X.columns[selector.support_].tolist()
            
            for feat in self.required_features:
                if feat not in self.selected_features:
                    self.selected_features.append(feat)

            self.model = XGBClassifier(**best_params)
            self.model.fit(X[self.selected_features], y)
            self.classes_ = self.model.classes_
            self.feature_importances_ = pd.Series(
                self.model.feature_importances_,
                index=self.selected_features
            ).sort_values(ascending=False)

            # Validation reporting
            st.subheader("Model Performance")
            y_pred = self.model.predict(X[self.selected_features])
            st.text(classification_report(y, y_pred))
            
        except Exception as e:
            logging.error(f"Model training failed: {str(e)}")
            raise

    def _objective(self, trial, X: pd.DataFrame, y: pd.Series, tscv) -> float:
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 500),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'max_depth': trial.suggest_int('max_depth', 3, 9),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'gamma': trial.suggest_float('gamma', 0, 0.5)
        }
        
        scores = []
        
        for train_idx, test_idx in tqdm(tscv.split(X), desc="CV Progress"):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            
            try:
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
                
            proba = self.model.predict_proba(X[self.selected_features])[0][2]
            return max(0.0, min(1.0, proba))
        except Exception as e:
            logging.error(f"Prediction failed: {str(e)}")
            return 0.5

    def probabilistic_forecast(self, X: pd.DataFrame, steps: int = 24):
        """Generate probabilistic price paths using Monte Carlo simulations"""
        scenarios = []
        current_state = X.iloc[-1][self.selected_features].values.reshape(1, -1)
        
        for _ in range(100):
            scenario = []
            state = current_state.copy()
            for _ in range(steps):
                proba = self.model.predict_proba(state)
                scenario.append(proba[0][2])  # Probability of buy signal
                # Simulate state evolution with noise
                state = state * 0.9 + np.random.normal(0, 0.1, state.shape)
            scenarios.append(scenario)
            
        return pd.DataFrame(scenarios)

    def calculate_position_size(self, confidence: float, volatility: float) -> float:
        """Adaptive position sizing using modified Kelly Criterion"""
        win_prob = confidence
        win_loss_ratio = 1.5  # Conservative estimate for crypto
        kelly_fraction = (win_prob * (win_loss_ratio + 1) - 1) / win_loss_ratio
        return min(max(kelly_fraction * (1 / (volatility + 1e-6)), 0.0), 0.1)

def show_advanced_ui(model: AdvancedTradingModel, data: pd.DataFrame):
    """Enhanced Streamlit visualization interface"""
    tab1, tab2, tab3, tab4 = st.tabs(["Forecast", "Regimes", "Features", "Trading"])
    
    with tab1:
        st.subheader("Probabilistic Price Forecast")
        forecast = model.probabilistic_forecast(data)
        fig = px.line(forecast.T.quantile([0.05, 0.5, 0.95], axis=1), 
                      labels={'value': 'Buy Probability', 'variable': 'Quantile'},
                      title="24-Step Buy Signal Probability Forecast")
        st.plotly_chart(fig)
    
    with tab2:
        st.subheader("Market Regime Analysis")
        if model.regimes is not None:
            regime_data = data.copy()
            regime_data['regime'] = model.regimes[:, 0]  # First component
            fig = px.scatter(regime_data, x='returns', y='volatility',
                           color='regime', hover_data=['volatility'],
                           color_discrete_sequence=['red', 'green', 'blue'])
            st.plotly_chart(fig)
    
    with tab3:
        st.subheader("Feature Analysis")
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Feature Importances**")
            fig = px.bar(model.feature_importances_, 
                        labels={'index': 'Feature', 'value': 'Importance'},
                        height=400)
            st.plotly_chart(fig)
            
        with col2:
            st.write("**Feature Correlation**")
            corr_matrix = data[model.selected_features].corr()
            fig = px.imshow(corr_matrix, text_auto=".2f", aspect="auto")
            st.plotly_chart(fig)
    
    with tab4:
        st.subheader("Trading Strategy")
        if not data.empty:
            current_vol = data['volatility'].iloc[-1]
            confidence = model.predict(data.drop(columns=['target']).iloc[[-1]])
            position_size = model.calculate_position_size(confidence, current_vol)
            
            col1, col2 = st.columns(2)
            col1.metric("Current Confidence", f"{confidence:.2%}")
            col2.metric("Recommended Position", f"{position_size:.1%}")
            
            st.write("**Strategy Rules**")
            st.progress(position_size / 0.1)
            st.caption("Max position size capped at 10% of capital")

def main():
    st.sidebar.header("Settings")
    symbol = st.sidebar.text_input("Crypto Symbol", DEFAULT_SYMBOL).upper()
    
    raw_data = fetch_data(symbol, PRIMARY_INTERVAL)
    processed_data = calculate_features(raw_data)
    
    if not raw_data.empty and not processed_data.empty:
        col1, col2 = st.columns([3, 1])
        with col1:
            st.subheader(f"{symbol} Price Chart")
            st.line_chart(raw_data['close'])
        with col2:
            st.metric("Current Price", f"${raw_data['close'].iloc[-1]:.2f}")
            st.metric("Volatility", f"{processed_data['volatility'].iloc[-1]:.2%}")

    if st.sidebar.button("🚀 Train Model") and not processed_data.empty:
        try:
            model = AdvancedTradingModel()
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
        show_advanced_ui(model, processed_data)

if __name__ == "__main__":
    main()
