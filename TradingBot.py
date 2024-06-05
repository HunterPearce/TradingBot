import yfinance as yf
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, roc_curve
import numpy as np
import matplotlib.pyplot as plt
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()

# Data Collection
def get_historical_data(ticker, start_date, end_date):
    data = yf.download(ticker, start=start_date, end=end_date)
    return data

# Data Preprocessing
def preprocess_data(data):
    data = data.dropna()
    scaler = MinMaxScaler()
    data[['Open', 'High', 'Low', 'Close', 'Volume']] = scaler.fit_transform(data[['Open', 'High', 'Low', 'Close', 'Volume']])
    return data

# Feature Engineering
def compute_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    rsi[rs == 0] = 100  # Handle division by zero
    return rsi

def compute_macd(data):
    ShortEMA = data['Close'].ewm(span=12, adjust=False).mean()
    LongEMA = data['Close'].ewm(span=26, adjust=False).mean()
    MACD = ShortEMA - LongEMA
    signal = MACD.ewm(span=9, adjust=False).mean()
    data.loc[:, 'MACD'] = MACD
    data.loc[:, 'Signal Line'] = signal
    return data

def compute_bollinger_bands(data, window=20, no_of_stds=2):
    rolling_mean = data['Close'].rolling(window=window).mean()
    rolling_std = data['Close'].rolling(window=window).std()
    data.loc[:, 'Bollinger High'] = rolling_mean + (rolling_std * no_of_stds)
    data.loc[:, 'Bollinger Low'] = rolling_mean - (rolling_std * no_of_stds)
    return data

def create_features(data):
    data = compute_macd(data)
    data = compute_bollinger_bands(data)
    data['MA50'] = data['Close'].rolling(window=50).mean()
    data['MA200'] = data['Close'].rolling(window=200).mean()
    data['RSI'] = compute_rsi(data['Close'])
    data['MA50-200'] = data['MA50'] - data['MA200']
    data['Returns'] = data['Close'].pct_change()
    data = data.replace([np.inf, -np.inf], np.nan).dropna()  # Replace infinities and drop NaNs
    return data

# Model Training
def train_model(data):
    X = data[['MA50', 'MA200', 'RSI', 'MA50-200', 'Returns', 'MACD', 'Signal Line', 'Bollinger High', 'Bollinger Low']].dropna()
    y = (data['Close'].shift(-1) > data['Close']).astype(int).dropna()
    X, y = X.align(y, join='inner', axis=0)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    rf_param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [10, 20, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }
    rf_model = RandomForestClassifier(random_state=42)
    rf_grid_search = GridSearchCV(rf_model, rf_param_grid, cv=5, scoring='accuracy')
    rf_grid_search.fit(X_train, y_train)
    best_rf_model = rf_grid_search.best_estimator_
    gb_param_grid = {
        'n_estimators': [100, 200, 300],
        'learning_rate': [0.01, 0.1, 0.2],
        'max_depth': [3, 5, 7]
    }
    gb_model = GradientBoostingClassifier(random_state=42)
    gb_grid_search = GridSearchCV(gb_model, gb_param_grid, cv=5, scoring='accuracy')
    gb_grid_search.fit(X_train, y_train)
    best_gb_model = gb_grid_search.best_estimator_

    # Ensemble Model
    ensemble_model = VotingClassifier(estimators=[
        ('rf', best_rf_model),
        ('gb', best_gb_model)
    ], voting='soft')
    ensemble_model.fit(X_train, y_train)
    return ensemble_model, X_test, y_test

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]  # Probability estimates for the positive class
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    auc_roc = roc_auc_score(y_test, y_pred_proba)

    # Plotting AUC-ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % auc_roc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.show()

    return {
        'accuracy': accuracy,
        'classification_report': report,
        'confusion_matrix': conf_matrix,
        'AUC_ROC': auc_roc
    }

# Economic Performance Metrics
def calculate_sharpe_ratio(returns, risk_free_rate=0.01):
    excess_returns = returns - risk_free_rate / 252
    sharpe_ratio = excess_returns.mean() / excess_returns.std() * np.sqrt(252)
    return sharpe_ratio

def calculate_maximum_drawdown(cumulative_returns):
    logger.info("Starting maximum drawdown calculation.")
    if cumulative_returns.isnull().any():
        logger.warning("NaN values present in cumulative returns.")
        cumulative_returns = cumulative_returns.ffill().bfill()
    running_max = np.maximum.accumulate(cumulative_returns)
    drawdown = (cumulative_returns - running_max) / running_max
    drawdown[running_max == 0] = 0  # Adjust for zero division
    max_drawdown = drawdown.min()
    logger.info(f"Maximum drawdown calculated: {max_drawdown}")
    return max_drawdown

def calculate_total_trades(signals):
    trades = signals.diff().fillna(0) != 0
    total_trades = trades.sum()
    return total_trades

# Backtesting
def backtest(data, model):
    data = data.copy()
    data.loc[:, 'Predicted'] = model.predict(data[['MA50', 'MA200', 'RSI', 'MA50-200', 'Returns', 'MACD', 'Signal Line', 'Bollinger High', 'Bollinger Low']])
    data.loc[:, 'Strategy Returns'] = data['Predicted'].shift(1) * data['Close'].pct_change()
    cumulative_returns = data['Strategy Returns'].cumsum() + 1
    sharpe_ratio = calculate_sharpe_ratio(data['Strategy Returns'])
    max_drawdown = calculate_maximum_drawdown(cumulative_returns)
    total_trades = calculate_total_trades(data['Predicted'])
    return cumulative_returns, sharpe_ratio, max_drawdown, total_trades

# Main
def main():
    data = get_historical_data('AAPL', '2023-01-01', '2024-01-01')
    preprocessed_data = preprocess_data(data)
    feature_data = create_features(preprocessed_data)
    model, X_test, y_test = train_model(feature_data)
    evaluation_results = evaluate_model(model, X_test, y_test)
    print("Model Evaluation:")
    print(f"Accuracy: {evaluation_results['accuracy']}")
    print(f"Classification Report:\n{evaluation_results['classification_report']}")
    print(f"Confusion Matrix:\n{evaluation_results['confusion_matrix']}")
    print(f"AUC-ROC: {evaluation_results['AUC_ROC']}")

    backtest_results, sharpe_ratio, max_drawdown, total_trades = backtest(feature_data, model)
    final_amount = 1000 * backtest_results.iloc[-1]
    percentage_return = (backtest_results.iloc[-1] - 1) * 100

    print("Backtesting Results:")
    print(f"Sharpe Ratio: {sharpe_ratio:.2f}")
    print(f"Maximum Drawdown: {max_drawdown:.2f}")
    print(f"Total Number of Trades: {total_trades}")
    print(f"Final Amount: ${final_amount:.2f}")
    print(f"Percentage Return: {percentage_return:.2f}%")

if __name__ == "__main__":
    main()
