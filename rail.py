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
EPSILON = 1e-6  # Small value to prevent division by zero

warnings.filterwarnings("ignore", category=UserWarning)
logging.basicConfig(level=logging.INFO)

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Handle infinite values and data validation"""
    # Replace infinities with NaNs then fill
    df = df.replace([np.inf, -np.inf], np.nan)
    
    # Clip extreme values
    numeric_cols = df.select_dtypes(include=np.number).columns
    df[numeric_cols] = df[numeric_cols].clip(-1e6, 1e6)
    
    return df.ffill().dropna()

def calculate_features(df: pd.DataFrame) -> pd.DataFrame:
    try:
        if df.empty:
            return pd.DataFrame()
            
        df = df.copy()
        
        # Price transformations with safeguards
        df['returns'] = df['close'].pct_change().replace([np.inf, -np.inf], np.nan).fillna(0)
        df['returns'] = df['returns'].clip(-1, 1)  # Cap returns at ±100%
        
        # Volatility with minimum observations
        df['volatility'] = df['returns'].rolling(
            GARCH_WINDOW, min_periods=5
        ).std().replace([np.inf, -np.inf], np.nan).fillna(0)
        
        # Technical indicators with smoothing
        df['ema_12'] = df['close'].ewm(span=12, adjust=False).mean()
        df['ema_26'] = df['close'].ewm(span=26, adjust=False).mean()
        df['macd'] = (df['ema_12'] - df['ema_26']).replace([np.inf, -np.inf], np.nan).fillna(0)
        df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
        
        # Volume features with zero protection
        df['volume'] = df['volume'].replace(0, EPSILON)
        df['volume_change'] = (df['volume'].pct_change()
                              .replace([np.inf, -np.inf], np.nan)
                              .fillna(0))
        df['volume_ma'] = df['volume'].rolling(14).mean().fillna(EPSILON)
        
        # Target engineering with validation
        future_returns = (df['close'].pct_change().shift(-HOLD_LOOKAHEAD)
                        .replace([np.inf, -np.inf], np.nan)
                        .fillna(0))
        future_returns = future_returns.clip(-1, 1)
        df['target'] = pd.qcut(future_returns, q=3, labels=[0, 1, 2], duplicates='drop')
        
        # Final data cleaning
        df = clean_data(df)
        return df.dropna()
        
    except Exception as e:
        logging.error(f"Feature engineering failed: {str(e)}")
        return pd.DataFrame()

class TradingModel:
    def __init__(self):
        self.model = None
        self.selected_features = []
        self.feature_importances = pd.DataFrame()

    def optimize_model(self, X: pd.DataFrame, y: pd.Series) -> bool:
        try:
            # Final data validation
            X = clean_data(X)
            y = clean_data(y.to_frame()).squeeze()
            
            if X.empty or y.empty:
                st.error("No valid data available for training")
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
        try:
            # Add missing value handling to XGBoost
            params['missing'] = np.nan
            
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
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 500),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'max_depth': trial.suggest_int('max_depth', 3, 9),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'gamma': trial.suggest_float('gamma', 0, 0.5),
            'missing': np.nan  # Explicit NaN handling
        }
        
        scores = []
        
        for train_idx, test_idx in tscv.split(X):
            try:
                X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
                y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
                
                # Final data check before training
                X_train = clean_data(X_train)
                X_test = clean_data(X_test)
                y_train = clean_data(y_train.to_frame()).squeeze()
                y_test = clean_data(y_test.to_frame()).squeeze()
                
                sm = SMOTE(random_state=42)
                X_res, y_res = sm.fit_resample(X_train, y_train)
                
                model = XGBClassifier(**params)
                model.fit(
                    X_res, y_res,
                    eval_set=[(X_test, y_test)],
                    early_stopping_rounds=10,
                    verbose=False
                )
                
                y_proba = model.predict_proba(X_test)
                score = roc_auc_score(y_test, y_proba, multi_class='ovo')
                scores.append(score)
                
            except Exception as e:
                scores.append(0.0)
        
        return np.mean(scores)

    def predict(self, X: pd.DataFrame) -> float:
        if not self.model or X.empty:
            return 0.5
            
        try:
            proba = self.model.predict_proba(X[self.selected_features])[0]
            return max(0.0, min(1.0, np.max(proba)))
        except Exception as e:
            logging.error(f"Prediction failed: {str(e)}")
            return 0.5

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
        model = TradingModel()
        X = processed_data.drop(columns=['target'])
        y = processed_data['target']
        
        with st.spinner("Training AI Model..."):
            if model.optimize_model(X, y):
                st.session_state.model = model
                st.success("Training completed!")
                st.write("Feature Importances:")
                st.dataframe(model.feature_importances.reset_index().rename(columns={'index': 'Feature', 0: 'Importance'}))
            else:
                st.error("Model training failed")

    if st.session_state.model and not processed_data.empty:
        model = st.session_state.model
        latest_data = processed_data.drop(columns=['target']).iloc[[-1]]
        
        confidence = model.predict(latest_data)
        current_vol = processed_data['volatility'].iloc[-1]
        
        st.subheader("Trading Signal")
        col1, col2 = st.columns(2)
        col1.metric("Confidence Score", f"{confidence:.2%}")
        
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
