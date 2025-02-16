import logging
import time
import numpy as np
import pandas as pd
import yfinance as yf
from arch import arch_model
import optuna
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.feature_selection import SelectFromModel
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score
from sklearn.utils.class_weight import compute_sample_weight
from datetime import datetime, timedelta

# ======================
# CONFIGURATION
# ======================
SYMBOL = 'BTC-USD'
INTERVAL = '5m'
LOOKBACK_DAYS = 30
TRADE_THRESHOLD_BUY = 0.65
TRADE_THRESHOLD_SELL = 0.35

# ======================
# LOGGING SETUP
# ======================
logging.basicConfig(
    filename='crypto_trading.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def safe_execute(func):
    """Decorator for centralized error handling"""
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logging.error(f"Error in {func.__name__}: {str(e)}", exc_info=True)
            return None
    return wrapper

# ======================
# DATA PIPELINE
# ======================
@safe_execute
def fetch_data(symbol: str, interval: str) -> pd.DataFrame:
    """Fetch historical data with yfinance"""
    end = datetime.now()
    start = end - timedelta(days=LOOKBACK_DAYS)
    return yf.download(symbol, start=start, end=end, interval=interval)

def calculate_features(df: pd.DataFrame) -> pd.DataFrame:
    """Feature engineering with technical indicators"""
    df = df.copy()
    
    # Price-based features
    df['SMA_20'] = df['Close'].rolling(20).mean()
    df['SMA_50'] = df['Close'].rolling(50).mean()
    df['Returns'] = df['Close'].pct_change()
    
    # Momentum indicators
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # Volume-based features
    df['OBV'] = (np.sign(df['Close'].diff()) * df['Volume']).cumsum()
    df['Volume_MA_20'] = df['Volume'].rolling(20).mean()
    
    # MACD
    ema12 = df['Close'].ewm(span=12, adjust=False).mean()
    ema26 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = ema12 - ema26
    df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    
    # Volatility (GARCH)
    returns = df['Returns'].dropna()
    if len(returns) > 10:  # Minimum data check
        garch = arch_model(returns, vol='Garch', p=1, q=1)
        garch_fit = garch.fit(disp='off')
        df['Volatility'] = garch_fit.conditional_volatility
    
    # Target: 1 if next period's return is positive
    df['Target'] = (df['Returns'].shift(-1) > 0).astype(int)
    
    return df.dropna()

# ======================
# MODEL PIPELINE
# ======================
class TradingModel:
    def __init__(self):
        self.feature_selector = None
        self.model_rf = RandomForestClassifier(
            warm_start=True,
            class_weight='balanced',
            random_state=42
        )
        self.model_gb = GradientBoostingClassifier(random_state=42)
        self.calibrated_rf = None
        self.calibrated_gb = None

    def optimize_hyperparameters(self, X, y):
        """Optuna-based hyperparameter tuning"""
        # Random Forest optimization
        def objective_rf(trial):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 100, 500),
                'max_depth': trial.suggest_int('max_depth', 5, 30),
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 10),
            }
            model = RandomForestClassifier(**params, warm_start=True)
            return self._cross_val_score(model, X, y)

        # Gradient Boosting optimization
        def objective_gb(trial):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 100, 500),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                'max_depth': trial.suggest_int('max_depth', 3, 10),
            }
            model = GradientBoostingClassifier(**params)
            return self._cross_val_score(model, X, y)

        study_rf = optuna.create_study(direction='maximize')
        study_rf.optimize(objective_rf, n_trials=20)
        
        study_gb = optuna.create_study(direction='maximize')
        study_gb.optimize(objective_gb, n_trials=20)

        self.model_rf.set_params(**study_rf.best_params)
        self.model_gb.set_params(**study_gb.best_params)

    def _cross_val_score(self, model, X, y):
        """Time-series aware cross-validation"""
        tscv = TimeSeriesSplit(n_splits=3)
        scores = []
        for train_idx, test_idx in tscv.split(X):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            
            if self.feature_selector is None:
                self.feature_selector = SelectFromModel(
                    RandomForestClassifier(n_estimators=100),
                    threshold="median"
                )
                self.feature_selector.fit(X_train, y_train)
            
            X_train_sel = self.feature_selector.transform(X_train)
            X_test_sel = self.feature_selector.transform(X_test)
            
            model.fit(X_train_sel, y_train)
            scores.append(accuracy_score(y_test, model.predict(X_test_sel)))
        return np.mean(scores)

    def train(self, X, y):
        """Full training pipeline"""
        # Feature selection
        self.feature_selector = SelectFromModel(
            RandomForestClassifier(n_estimators=100),
            threshold="median"
        )
        self.feature_selector.fit(X, y)
        X_sel = self.feature_selector.transform(X)

        # Dynamic class weighting
        sample_weights = compute_sample_weight('balanced', y)

        # Train models
        self.model_rf.fit(X_sel, y, sample_weight=sample_weights)
        self.model_gb.fit(X_sel, y, sample_weight=sample_weights)

        # Calibration
        self.calibrated_rf = CalibratedClassifierCV(
            self.model_rf, method='isotonic', cv=TimeSeriesSplit(3)
        self.calibrated_gb = CalibratedClassifierCV(
            self.model_gb, method='sigmoid', cv=TimeSeriesSplit(3))
        
        self.calibrated_rf.fit(X_sel, y)
        self.calibrated_gb.fit(X_sel, y)

    def predict(self, X):
        """Ensemble prediction with dynamic weighting"""
        if self.feature_selector is None:
            raise ValueError("Model not trained yet")
            
        X_sel = self.feature_selector.transform(X)
        prob_rf = self.calibrated_rf.predict_proba(X_sel)[:, 1]
        prob_gb = self.calibrated_gb.predict_proba(X_sel)[:, 1]
        return 0.6 * prob_rf + 0.4 * prob_gb

# ======================
# TRADING EXECUTION
# ======================
@safe_execute
def execute_trade(signal: str):
    """Placeholder for actual trading logic"""
    logging.info(f"Executing {signal} order")
    # Implement your exchange API integration here
    print(f"{datetime.now()}: {signal} signal triggered")

# ======================
# MAIN LOOP
# ======================
def trading_loop():
    model = TradingModel()
    last_trained = datetime.now()
    
    while True:
        try:
            # Fetch and prepare data
            raw_data = fetch_data(SYMBOL, INTERVAL)
            processed_data = calculate_features(raw_data)
            
            if len(processed_data) < 100:  # Minimum data check
                time.sleep(300)
                continue

            # Prepare features and target
            X = processed_data.drop(['Target', 'Returns'], axis=1, errors='ignore')
            y = processed_data['Target']
            
            # Retrain models periodically
            if (datetime.now() - last_trained).hours > 6:
                model.optimize_hyperparameters(X, y)
                model.train(X, y)
                last_trained = datetime.now()
                logging.info("Model retrained successfully")

            # Make prediction
            latest_features = X.iloc[[-1]]
            prediction = model.predict(latest_features)[0]
            
            # Trading logic
            if prediction > TRADE_THRESHOLD_BUY:
                execute_trade('BUY')
            elif prediction < TRADE_THRESHOLD_SELL:
                execute_trade('SELL')

            # Sleep until next interval
            time.sleep(300 - (time.time() % 300))  # Align to 5m intervals

        except KeyboardInterrupt:
            logging.info("Trading stopped by user")
            break
        except Exception as e:
            logging.error(f"Critical error in main loop: {str(e)}")
            time.sleep(600)  # Backoff on critical errors

if __name__ == "__main__":
    trading_loop()
