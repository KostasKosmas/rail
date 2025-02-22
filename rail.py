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
from arch import arch_model
from tqdm import tqdm
import warnings
import talib
from sklearn.preprocessing import label_binarize
import shap

# Configuration
DEFAULT_SYMBOL = 'BTC-USD'
PRIMARY_INTERVAL = '5m'  # Increased resolution
TRADE_THRESHOLD_BUY = 0.60  # Stricter thresholds
TRADE_THRESHOLD_SELL = 0.40
MAX_TRIALS = 50  # More thorough optimization
GARCH_WINDOW = 21
MIN_FEATURES = 10
MIN_SAMPLES_PER_CLASS = 100
HOLD_LOOKAHEAD = 6  # Increased prediction window

warnings.filterwarnings("ignore", category=UserWarning)
logging.basicConfig(level=logging.INFO)

# Streamlit UI
st.set_page_config(page_title="AI Trading System", layout="wide")
st.title("🚀 Enhanced Cryptocurrency Trading System")

if 'model' not in st.session_state:
    st.session_state.model = None

@st.cache_data(ttl=300, show_spinner="Fetching market data...")
def fetch_data(symbol: str, interval: str) -> pd.DataFrame:
    try:
        df = yf.download(
            symbol,
            period="180d",  # Longer history
            interval=interval,
            progress=False,
            auto_adjust=True
        )
        
        # Process column names
        new_columns = []
        for col in df.columns:
            if isinstance(col, tuple):
                col_name = col[0].lower()
            else:
                col_name = str(col).lower()
            new_columns.append(col_name.replace(' ', '_'))
        
        df.columns = new_columns
        required_cols = {'open', 'high', 'low', 'close', 'volume'}
        
        if not required_cols.issubset(df.columns):
            missing = required_cols - set(df.columns)
            st.error(f"Missing columns: {', '.join(missing)}")
            return pd.DataFrame()
            
        return df[list(required_cols)].ffill().dropna()
    
    except Exception as e:
        logging.error(f"Data fetch failed: {str(e)}", exc_info=True)
        return pd.DataFrame()

def calculate_advanced_features(df: pd.DataFrame) -> pd.DataFrame:
    try:
        df = df.copy()
        
        # Price Transformations
        df['returns'] = df['close'].pct_change()
        df['log_returns'] = np.log(df['close']/df['close'].shift(1))
        
        # Volatility Features
        df['volatility_21'] = df['returns'].rolling(21).std()
        df['atr_14'] = talib.ATR(df['high'], df['low'], df['close'], timeperiod=14)
        df['garch_vol'] = df['returns'].rolling(21).var()  # Simplified GARCH
        
        # Momentum Indicators
        df['rsi_14'] = talib.RSI(df['close'], timeperiod=14)
        df['macd'], df['macd_signal'], _ = talib.MACD(df['close'])
        df['adx_14'] = talib.ADX(df['high'], df['low'], df['close'], timeperiod=14)
        
        # Volume Features
        df['volume_ma_21'] = df['volume'].rolling(21).mean()
        df['volume_change'] = df['volume'].pct_change()
        
        # Cycle Features
        df['ht_dcperiod'] = talib.HT_DCPERIOD(df['close'])
        
        # Pattern Recognition
        df['cdl_doji'] = talib.CDLDOJI(df['open'], df['high'], df['low'], df['close'])
        
        # Target Engineering
        future_returns = df['close'].pct_change().shift(-HOLD_LOOKAHEAD)
        df['target'] = pd.qcut(future_returns, q=5, labels=[0, 1, 2, 3, 4], duplicates='drop')
        
        return df.dropna()
    
    except Exception as e:
        logging.error(f"Feature engineering failed: {str(e)}", exc_info=True)
        return pd.DataFrame()

class EnhancedTradingModel:
    def __init__(self):
        self.model = None
        self.selected_features = []
        self.explainer = None
        self.feature_importances = pd.DataFrame()
        self.is_trained = False

    def optimize_model(self, X: pd.DataFrame, y: pd.Series) -> bool:
        try:
            self._reset_state()
            
            if not self._validate_inputs(X, y):
                return False

            tscv = TimeSeriesSplit(n_splits=5)  # More validation folds
            study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=42))
            
            study.optimize(
                lambda trial: self._objective(trial, X, y, tscv),
                n_trials=MAX_TRIALS,
                n_jobs=-1,
                timeout=3600,
                catch=(ValueError,)
            )
            
            if not study.best_trial:
                st.error("Optimization failed - no successful trials")
                return False
                
            self._train_final_model(X, y, study.best_params)
            self._create_explainer(X)
            self.is_trained = True
            return True
            
        except Exception as e:
            logging.error(f"Training failed: {str(e)}", exc_info=True)
            st.error("Model training failed. Check logs for details.")
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
            logging.error(f"SHAP explanation failed: {str(e)}")

    def _train_final_model(self, X: pd.DataFrame, y: pd.Series, params: dict):
        try:
            selector = RFECV(
                XGBClassifier(**params, use_label_encoder=False),
                step=1,
                cv=TimeSeriesSplit(3),
                min_features_to_select=MIN_FEATURES,
                scoring='roc_auc_ovo'
            )
            selector.fit(X, y)
            self.selected_features = X.columns[selector.support_].tolist()
            
            self.model = XGBClassifier(
                **params,
                eval_metric='mlogloss',
                use_label_encoder=False
            )
            self.model.fit(X[self.selected_features], y)
            
        except Exception as e:
            logging.error(f"Final training failed: {str(e)}")
            raise

    def _objective(self, trial, X: pd.DataFrame, y: pd.Series, tscv) -> float:
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
            'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.3, log=True),
            'max_depth': trial.suggest_int('max_depth', 3, 9),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.4, 1.0),
            'gamma': trial.suggest_float('gamma', 0, 1.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 0, 1.0),
            'reg_lambda': trial.suggest_float('reg_lambda', 0, 1.0)
        }
        
        scores = []
        
        for train_idx, test_idx in tqdm(tscv.split(X), desc="Optimization Progress"):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            
            try:
                sm = SMOTE(sampling_strategy='not majority', random_state=42)
                X_res, y_res = sm.fit_resample(X_train, y_train)
                
                model = XGBClassifier(**params, use_label_encoder=False)
                model.fit(
                    X_res, y_res,
                    eval_set=[(X_test, y_test)],
                    early_stopping_rounds=20,
                    verbose=False
                )
                
                y_proba = model.predict_proba(X_test)
                y_test_bin = label_binarize(y_test, classes=model.classes_)
                
                fold_scores = []
                for class_idx in range(y_test_bin.shape[1]):
                    if y_test_bin[:, class_idx].sum() > 0:
                        fold_scores.append(roc_auc_score(
                            y_test_bin[:, class_idx], 
                            y_proba[:, class_idx]
                        ))
                
                scores.append(np.nanmean(fold_scores) if fold_scores else 0.0)
                    
            except Exception as e:
                scores.append(0.0)
        
        return np.nanmean(scores)

    def predict(self, X: pd.DataFrame) -> tuple:
        if not self.is_trained or self.model is None or X.empty:
            return 0.5, None
            
        try:
            proba = self.model.predict_proba(X[self.selected_features])[0]
            explanation = self.explainer.shap_values(X[self.selected_features])[0]
            return np.max(proba), explanation
            
        except Exception as e:
            logging.error(f"Prediction failed: {str(e)}")
            return 0.5, None

def main():
    st.sidebar.header("Configuration")
    symbol = st.sidebar.text_input("Asset Symbol", DEFAULT_SYMBOL).upper()
    
    raw_data = fetch_data(symbol, PRIMARY_INTERVAL)
    processed_data = calculate_advanced_features(raw_data)
    
    if not raw_data.empty and not processed_data.empty:
        col1, col2 = st.columns([3, 1])
        with col1:
            st.subheader(f"{symbol} Market Overview")
            fig = px.line(raw_data, y='close', title="Price Chart")
            st.plotly_chart(fig, use_container_width=True)
            
        with col2:
            st.metric("Current Price", f"${raw_data['close'].iloc[-1]:.2f}")
            st.metric("24h Volatility", f"{processed_data['volatility_21'].iloc[-1]:.2%}")
            st.metric("Market Sentiment", 
                      processed_data['cdl_doji'].iloc[-1] and "Neutral" or "Directional")

    if st.sidebar.button("🚀 Train Model") and not processed_data.empty:
        with st.spinner("Training Next-Gen Trading Model..."):
            model = EnhancedTradingModel()
            X = processed_data.drop(columns=['target'])
            y = processed_data['target']
            
            if model.optimize_model(X, y):
                st.session_state.model = model
                st.success("Model Training Successful!")
                
                st.subheader("Model Insights")
                col1, col2 = st.columns(2)
                with col1:
                    st.write("Top Predictive Features:")
                    st.dataframe(model.feature_importances.head(10))
                
                with col2:
                    st.write("Hyperparameter Configuration:")
                    st.json(model.model.get_params())
            else:
                st.error("Model training failed - check data quality")

    if st.session_state.model and not processed_data.empty:
        model = st.session_state.model
        latest_data = processed_data.drop(columns=['target']).iloc[[-1]]
        
        confidence, explanation = model.predict(latest_data)
        current_vol = processed_data['volatility_21'].iloc[-1]
        
        st.subheader("Trading Intelligence")
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Model Confidence", f"{confidence:.2%}")
            st.write("Feature Impact Analysis:")
            if explanation is not None:
                fig = px.bar(
                    x=model.selected_features,
                    y=explanation,
                    labels={'x': 'Features', 'y': 'SHAP Value'},
                    height=300
                )
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            dynamic_buy = TRADE_THRESHOLD_BUY + (current_vol * 0.15)
            dynamic_sell = TRADE_THRESHOLD_SELL - (current_vol * 0.15)
            
            if confidence > dynamic_buy:
                st.success("""
                🚀 Strong Buy Signal
                - High confidence in upward movement
                - Confirmed by volume and momentum indicators
                """)
            elif confidence < dynamic_sell:
                st.error("""
                🔻 Strong Sell Signal
                - High confidence in downward trend
                - Supported by volatility and pattern analysis
                """)
            else:
                st.info("""
                🛑 Hold Position
                - Market conditions uncertain
                - Await stronger signals
                """)

if __name__ == "__main__":
    main()
