import os
import requests
import yfinance as yf
import pandas as pd
import pytz
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from skopt import BayesSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
import numpy as np
import matplotlib.pyplot as plt
import logging
from datetime import datetime, timedelta
from xgboost import XGBClassifier
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(filename='trading_bot.log', level=logging.INFO, 
                    format='%(asctime)s:%(levelname)s:%(message)s')

# Function to send Telegram messages
def send_telegram_message(message):
    try:
        bot_token = os.getenv('TELEGRAM_BOT_TOKEN')
        chat_id = os.getenv('TELEGRAM_CHAT_ID')

        if not bot_token or not chat_id:
            raise ValueError("TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID environment variables are not set")

        url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
        payload = {'chat_id': chat_id, 'text': message}

        response = requests.post(url, data=payload)
        response.raise_for_status()
        logging.info(f"Telegram message sent successfully: {message}")
    except Exception as e:
        logging.error(f"Error sending Telegram message: {e}")

# Data Collection
def get_historical_data(ticker, start_date, end_date, csv_file='historical_data.csv'):
    try:
        if os.path.exists(csv_file):
            existing_data = pd.read_csv(csv_file, index_col='Date', parse_dates=True)
            last_date = existing_data.index[-1].strftime('%Y-%m-%d')
            if last_date >= end_date:
                logging.info(f"No new data needed, up to date with data ending {last_date}")
                return existing_data
            new_data = yf.download(ticker, start=last_date, end=end_date)
            if not new_data.empty:
                new_data = new_data.iloc[1:]  # Remove the overlapping last row
                combined_data = pd.concat([existing_data, new_data])
                combined_data.to_csv(csv_file)
                logging.info(f"New data appended, combined data shape: {combined_data.shape}")
                return combined_data
            else:
                logging.info("No new data retrieved")
                return existing_data
        else:
            data = yf.download(ticker, start=start_date, end=end_date)
            data.to_csv(csv_file)
            logging.info(f"Data retrieval and CSV storage successful, shape: {data.shape}")
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
    data['MACD'] = MACD
    data['Signal Line'] = signal
    return data

def compute_bollinger_bands(data, window=20, no_of_stds=2):
    rolling_mean = data['Close'].rolling(window=window).mean()
    rolling_std = data['Close'].rolling(window=window).std()
    data['Bollinger High'] = rolling_mean + (rolling_std * no_of_stds)
    data['Bollinger Low'] = rolling_mean - (rolling_std * no_of_stds)
    return data

def compute_ema(data, span):
    return data['Close'].ewm(span=span, adjust=False).mean()

def compute_stochastic_oscillator(data, window=14):
    data['L14'] = data['Low'].rolling(window=window).min()
    data['H14'] = data['High'].rolling(window=window).max()
    data['%K'] = 100 * ((data['Close'] - data['L14']) / (data['H14'] - data['L14']))
    return data

def compute_obv(data):
    data['OBV'] = (np.sign(data['Close'].diff()) * data['Volume']).fillna(0).cumsum()
    return data

def create_features(data):
    try:
        data = compute_macd(data)
        data = compute_bollinger_bands(data)
        data = compute_stochastic_oscillator(data)
        data = compute_obv(data)
        data['EMA12'] = compute_ema(data, 12)
        data['EMA26'] = compute_ema(data, 26)
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
        feature_columns = ['MA50', 'MA200', 'RSI', 'MA50-200', 'Returns', 'MACD', 'Signal Line', 'Bollinger High', 'Bollinger Low', 'Log Returns', 'Volatility', 'Momentum', 'EMA12', 'EMA26', '%K', 'OBV']
        X = data[feature_columns].dropna()
        y = (data['Close'].shift(-1) > data['Close']).astype(int).dropna()
        X, y = X.align(y, join='inner', axis=0)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        rf_model = RandomForestClassifier(random_state=42)
        rf_search = BayesSearchCV(
            rf_model,
            {
                'n_estimators': (100, 1000),
                'max_depth': (10, 50),
                'min_samples_split': (2, 10),
                'min_samples_leaf': (1, 4)
            },
            n_iter=50,
            cv=5,
            random_state=42
        )
        rf_search.fit(X_train, y_train)
        best_rf_model = rf_search.best_estimator_

        gb_model = GradientBoostingClassifier(random_state=42)
        gb_search = BayesSearchCV(
            gb_model,
            {
                'n_estimators': (100, 1000),
                'learning_rate': (0.01, 0.2, 'log-uniform'),
                'max_depth': (3, 10)
            },
            n_iter=50,
            cv=5,
            random_state=42
        )
        gb_search.fit(X_train, y_train)
        best_gb_model = gb_search.best_estimator_

        xgb_model = XGBClassifier(random_state=42)

        ensemble_model = VotingClassifier(estimators=[
            ('rf', best_rf_model),
            ('gb', best_gb_model),
            ('xgb', xgb_model)
        ], voting='soft')
        ensemble_model.fit(X_train, y_train)

        logging.info(f"Model training successful, X_train shape: {X_train.shape}, X_test shape: {X_test.shape}")
        return ensemble_model, X_test, y_test
    except Exception as e:
        logging.error(f"Error in model training: {e}")
        return None, None, None

def evaluate_model(model, X_test, y_test):
    try:
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]  # Probability estimates for the positive class
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred)
        conf_matrix = confusion_matrix(y_test, y_pred)
        auc_roc = roc_auc_score(y_test, y_pred_proba)

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

def handle_buy_signal(index, row, balance, shares, position_size, transaction_cost, trade_log):
    shares_to_buy = (position_size / row['Close']) * (1 - transaction_cost)
    cost = shares_to_buy * row['Close']
    if cost <= balance:
        balance -= cost
        shares += shares_to_buy
        trade_log.append([index.strftime('%Y-%m-%d %H:%M:%S'), 'Buy', shares, balance, ""])
        logging.info(f"Buy signal executed at {index}")
    return balance, shares

def handle_sell_signal(index, row, balance, shares, buy_price, transaction_cost, trade_log):
    sell_revenue = shares * row['Close'] * (1 - transaction_cost)
    balance += sell_revenue
    profit = sell_revenue - (shares * buy_price) * (1 + transaction_cost)
    shares = 0
    trade_log.append([index.strftime('%Y-%m-%d %H:%M:%S'), 'Sell', shares, balance, profit])
    logging.info(f"Sell signal executed at {index}")
    return balance, shares

def simulate_trading(model, data):
    try:
        initial_balance = 1000
        balance = initial_balance
        shares = 0
        num_trades = 0
        buy_signals = 0
        sell_signals = 0
        position_size = 100  # Fixed position size of $100
        transaction_cost = 0.001  # Example transaction cost: 0.1%
        open_position = False  # Track if there is an open position

        trade_log = []  # List to store trade details
        buy_price = 0  # Variable to store the price at which shares were bought

        feature_columns = ['MA50', 'MA200', 'RSI', 'MA50-200', 'Returns', 'MACD', 'Signal Line', 'Bollinger High', 'Bollinger Low', 'Log Returns', 'Volatility', 'Momentum', 'EMA12', 'EMA26', '%K', 'OBV']

        logging.info("Starting trading simulation")

        for index, row in data.iterrows():
            logging.debug(f"Processing row: {index}")
            try:
                features = pd.DataFrame([[
                    row['MA50'], row['MA200'], row['RSI'], row['MA50-200'], 
                    row['Returns'], row['MACD'], row['Signal Line'], 
                    row['Bollinger High'], row['Bollinger Low'], row['Log Returns'], row['Volatility'], row['Momentum'], row['EMA12'], row['EMA26'], row['%K'], row['OBV']
                ]], columns=feature_columns)
                
                prediction = model.predict(features)[0]
                logging.debug(f"Prediction at {index}: {prediction}")
            except Exception as e:
                logging.error(f"Error predicting at {index}: {e}")
                continue

            if prediction == 1 and not open_position:  # Buy signal and no open position
                logging.debug(f"Buy signal at {index}")
                if balance >= position_size:  # Buy if balance is sufficient
                    balance, shares = handle_buy_signal(index, row, balance, shares, position_size, transaction_cost, trade_log)
                    buy_price = row['Close']
                    open_position = True
                    num_trades += 1
                    buy_signals += 1

            elif prediction == 0 and open_position:  # Sell signal and an open position
                logging.debug(f"Sell signal at {index}")
                if shares > 0:  # Sell if holding shares
                    balance, shares = handle_sell_signal(index, row, balance, shares, buy_price, transaction_cost, trade_log)
                    open_position = False
                    num_trades += 1
                    sell_signals += 1

        if num_trades == 0:
            logging.info("No trades executed. Exiting.")
            print("No trades executed. Exiting.")
            return

        if buy_signals == 0:
            logging.info("No buy signals executed.")
            print("No buy signals executed.")

        if sell_signals == 0:
            logging.info("No sell signals executed.")
            print("No sell signals executed.")

        final_balance = balance + shares * data.iloc[-1]['Close'] * (1 - transaction_cost)
        profit = final_balance - initial_balance
        percentage_change = (profit / initial_balance) * 100

        print(f"Initial Balance: ${initial_balance:.2f}")
        print(f"Final Balance: ${final_balance:.2f}")
        print(f"Profit: ${profit:.2f}")
        print(f"Percentage Change: {percentage_change:.2f}%")
        print(f"Number of Trades: {num_trades}")

        # Save the trade log to a CSV file
        trade_df = pd.DataFrame(trade_log, columns=['Date', 'Action', 'Shares', 'Balance', 'Profit'])
        trade_df.to_csv('trade_log.csv', index=False)

        # Create a DataFrame for simulation results for each trade
        simulation_results = trade_df.copy()
        simulation_results['Initial Balance'] = initial_balance
        simulation_results['Final Balance'] = final_balance
        simulation_results['Total Profit'] = profit
        simulation_results['Percentage Change'] = percentage_change
        simulation_results['Number of Trades'] = num_trades
        simulation_results['Buy Signals'] = buy_signals
        simulation_results['Sell Signals'] = sell_signals

        # Save the simulation results to a CSV file
        simulation_results.to_csv('simulation_results.csv', index=False)

        # Send the most recent trade to Telegram only if it occurred in the last day
        if not trade_df.empty:
            last_trade = trade_df.iloc[-1]
            last_trade_date = datetime.strptime(last_trade['Date'], '%Y-%m-%d %H:%M:%S')
            if (datetime.now() - last_trade_date).days <= 1:
                trade_log_msg = f"Trade Alert! \nAction: {last_trade['Action']} \nDate: {last_trade['Date']},"
                send_telegram_message(trade_log_msg)

        logging.info("Trading simulation successful")
    except Exception as e:
        logging.error(f"Error in trading simulation: {e}")

def main():
    today = datetime.today()
    start_date = (today - timedelta(days=3*365)).strftime('%Y-%m-%d')
    end_date = today.strftime('%Y-%m-%d')

    data = get_historical_data('BTC-USD', start_date, end_date)
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

    model, X_test, y_test = train_model(feature_data)
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
