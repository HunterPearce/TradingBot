import yfinance as yf
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, roc_curve
import numpy as np
import matplotlib.pyplot as plt
import logging
from datetime import datetime, timedelta
from xgboost import XGBClassifier
import smtplib

# Configure logging
logging.basicConfig(filename='trading_bot.log', level=logging.INFO, 
                    format='%(asctime)s:%(levelname)s:%(message)s')

# Data Collection
def get_historical_data(ticker, start_date, end_date):
    try:
        data = yf.download(ticker, start=start_date, end=end_date)
        logging.info(f"Data retrieval successful, shape: {data.shape}")
        return data
    except Exception as e:
        logging.error(f"Error in data retrieval: {e}")
        return pd.DataFrame()

# Data Preprocessing
def preprocess_data(data):
    try:
        data = data.dropna()
        scaler = MinMaxScaler()
        data[['Open', 'High', 'Low', 'Close', 'Volume']] = scaler.fit_transform(data[['Open', 'High', 'Low', 'Close', 'Volume']])
        logging.info(f"Data preprocessing successful, shape: {data.shape}")
        return data
    except Exception as e:
        logging.error(f"Error in data preprocessing: {e}")
        return pd.DataFrame()

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
    try:
        data = compute_macd(data)
        data = compute_bollinger_bands(data)
        data['MA50'] = data['Close'].rolling(window=50).mean()
        data['MA200'] = data['Close'].rolling(window=200).mean()
        data['RSI'] = compute_rsi(data['Close'])
        data['MA50-200'] = data['MA50'] - data['MA200']
        data['Returns'] = data['Close'].pct_change()
        data['Log Returns'] = np.log(data['Close'] / data['Close'].shift(1))
        data['Volatility'] = data['Log Returns'].rolling(window=21).std()
        data['Momentum'] = data['Close'] - data['Close'].shift(5)
        data = data.replace([np.inf, -np.inf], np.nan).dropna()  # Replace infinities and drop NaNs
        logging.info(f"Feature engineering successful, shape: {data.shape}")
        return data
    except Exception as e:
        logging.error(f"Error in feature engineering: {e}")
        return pd.DataFrame()

# Model Training
def train_model(data):
    try:
        X = data[['MA50', 'MA200', 'RSI', 'MA50-200', 'Returns', 'MACD', 'Signal Line', 'Bollinger High', 'Bollinger Low', 'Log Returns', 'Volatility', 'Momentum']].dropna()
        y = (data['Close'].shift(-1) > data['Close']).astype(int).dropna()
        X, y = X.align(y, join='inner', axis=0)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        rf_param_grid = {
            'n_estimators': [100, 200, 300, 500],
            'max_depth': [10, 20, 30, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
        rf_model = RandomForestClassifier(random_state=42)
        rf_grid_search = RandomizedSearchCV(rf_model, rf_param_grid, cv=5, scoring='accuracy', n_iter=50, n_jobs=-1, random_state=42)
        rf_grid_search.fit(X_train, y_train)
        best_rf_model = rf_grid_search.best_estimator_

        gb_param_grid = {
            'n_estimators': [100, 200, 300],
            'learning_rate': [0.01, 0.05, 0.1, 0.2],
            'max_depth': [3, 5, 7]
        }
        gb_model = GradientBoostingClassifier(random_state=42)
        gb_grid_search = RandomizedSearchCV(gb_model, gb_param_grid, cv=5, scoring='accuracy', n_iter=50, n_jobs=-1, random_state=42)
        gb_grid_search.fit(X_train, y_train)
        best_gb_model = gb_grid_search.best_estimator_

        xgb_model = XGBClassifier(random_state=42)

        ensemble_model = VotingClassifier(estimators=[
            ('rf', best_rf_model),
            ('gb', best_gb_model),
            ('xgb', xgb_model)
        ], voting='soft')
        ensemble_model.fit(X_train, y_train)

        logging.info(f"Model training successful, X_train shape: {X_train.shape}, X_test shape: {X_test.shape}")
        return ensemble_model, X_test, y_test, best_rf_model, best_gb_model, xgb_model, X_train
    except Exception as e:
        logging.error(f"Error in model training: {e}")
        return None, None, None, None, None, None, None

def evaluate_model(model, X_test, y_test):
    try:
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

        logging.info("Model evaluation successful")
        return {
            'accuracy': accuracy,
            'classification_report': report,
            'confusion_matrix': conf_matrix,
            'AUC_ROC': auc_roc
        }
    except Exception as e:
        logging.error(f"Error in model evaluation: {e}")
        return {}

def simulate_trading(model, data):
    try:
        initial_balance = 1000
        balance = initial_balance
        shares = 0
        num_trades = 0
        position_size = 100  # Fixed position size of $100
        transaction_cost = 0.001  # Example transaction cost: 0.1%
        open_position = False  # Track if there is an open position

        trade_log = []  # List to store trade details
        buy_price = 0  # Variable to store the price at which shares were bought

        feature_columns = ['MA50', 'MA200', 'RSI', 'MA50-200', 'Returns', 'MACD', 'Signal Line', 'Bollinger High', 'Bollinger Low', 'Log Returns', 'Volatility', 'Momentum']

        for index, row in data.iterrows():
            try:
                features = pd.DataFrame([[
                    row['MA50'], row['MA200'], row['RSI'], row['MA50-200'], 
                    row['Returns'], row['MACD'], row['Signal Line'], 
                    row['Bollinger High'], row['Bollinger Low'], row['Log Returns'], row['Volatility'], row['Momentum']
                ]], columns=feature_columns)
                
                prediction = model.predict(features)[0]
            except Exception as e:
                logging.error(f"Error predicting at {index}: {e}")
                continue

            if prediction == 1 and not open_position:  # Buy signal and no open position
                if balance >= position_size:  # Buy if balance is sufficient
                    shares_to_buy = (position_size / row['Close']) * (1 - transaction_cost)
                    cost = shares_to_buy * row['Close']
                    if cost <= balance:
                        balance -= cost  # Subtract the cost from the balance
                        shares += shares_to_buy
                        buy_price = row['Close']
                        open_position = True
                        num_trades += 1
                        trade_log.append([index, 'Buy', shares, balance, ""])
            elif prediction == 0 and open_position:  # Sell signal and an open position
                if shares > 0:  # Sell if holding shares
                    sell_revenue = shares * row['Close'] * (1 - transaction_cost)
                    balance += sell_revenue
                    profit = sell_revenue - (shares * buy_price) * (1 + transaction_cost)
                    shares = 0
                    open_position = False
                    num_trades += 1
                    trade_log.append([index, 'Sell', shares, balance, profit])

        final_balance = balance + shares * data.iloc[-1]['Close'] * (1 - transaction_cost)
        profit = final_balance - initial_balance
        percentage_change = (profit / initial_balance) * 100

        print(f"Initial Balance: ${initial_balance:.2f}")
        print(f"Final Balance: ${final_balance:.2f}")
        print(f"Profit: ${profit:.2f}")
        print(f"Percentage Change: {percentage_change:.2f}%")
        print(f"Number of Trades: {num_trades}")

        trade_df = pd.DataFrame(trade_log, columns=['Date', 'Action', 'Shares', 'Balance', 'Profit'])
        trade_df.to_csv('trade_log.csv', index=False)

        logging.info("Trading simulation successful")
    except Exception as e:
        logging.error(f"Error in trading simulation: {e}")

def main():
    today = datetime.today()
    start_date = (today - timedelta(days=3*365)).strftime('%Y-%m-%d')
    end_date = today.strftime('%Y-%m-%d')

    data = get_historical_data('BTC-AUD', start_date, end_date)
    if data.empty:
        print("No data retrieved. Exiting.")
        return

    preprocessed_data = preprocess_data(data)
    if preprocessed_data.empty:
        print("Data preprocessing failed. Exiting.")
        return

    feature_data = create_features(preprocessed_data)
    if feature_data.empty:
        print("Feature engineering failed. Exiting.")
        return

    model, X_test, y_test, best_rf_model, best_gb_model, xgb_model, X_train = train_model(feature_data)
    if model is None:
        print("Model training failed. Exiting.")
        return

    evaluation_results = evaluate_model(model, X_test, y_test)
    print("Model Evaluation:")
    print(f"Accuracy: {evaluation_results.get('accuracy', 'N/A')}")
    print(f"Classification Report:\n{evaluation_results.get('classification_report', 'N/A')}")
    print(f"Confusion Matrix:\n{evaluation_results.get('confusion_matrix', 'N/A')}")
    print(f"AUC-ROC: {evaluation_results.get('AUC_ROC', 'N/A')}")

    # Simulate trading
    simulate_trading(model, feature_data)

if __name__ == "__main__":
    main()
