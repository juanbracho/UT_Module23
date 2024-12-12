from flask import Flask, render_template, request
from sklearn.model_selection import train_test_split
import pandas as pd
import joblib
import sqlite3
import os
from keras.models import Sequential, load_model
from keras.layers import LSTM, Dense
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import plotly.graph_objects as go
import numpy as np
from flask import jsonify
import yfinance as yf

app = Flask(__name__)

# Paths
DB_PATH = 'database/stocks_data.db'
MODELS_PATH = 'models/'

# Default ticker and model type
DEFAULT_TICKER = 'XOM'

# Helpter functions

# Fetch and Process Data
def fetch_and_process_data(ticker):
    """
    Fetches and processes real-time data for a given ticker.
    """
    try:
        data = yf.download(ticker, start="2000-01-01", end="2023-12-31")
        if data.empty:
            return None

        # Add calculated columns (moving averages, lags, volatility)
        data['7-day MA'] = data['Adj Close'].rolling(window=7).mean()
        data['14-day MA'] = data['Adj Close'].rolling(window=14).mean()
        data['Volatility'] = data['Adj Close'].rolling(window=14).std()
        data['Lag_1'] = data['Adj Close'].shift(1)
        data['Lag_2'] = data['Adj Close'].shift(2)

        # Drop rows with NaN values
        data.dropna(inplace=True)

        return data.reset_index()
    except Exception as e:
        print(f"Error fetching data for {ticker}: {e}")
        return None

# Prepare Features and Labels
def prepare_features_and_labels(data):
    features = ['7-day MA', '14-day MA', 'Volatility', 'Lag_1', 'Lag_2']
    labels = data['Adj Close'].values
    X = data[features].values
    return X, labels

# Load or train model
def load_or_train_model(ticker, model_type, features, labels):
    model_path = os.path.join(MODELS_PATH, f"model_{ticker}_{model_type}.pkl" if model_type in ['rf', 'lr'] else f"model_{ticker}_{model_type}.h5")
    scaler_path = os.path.join(MODELS_PATH, f"scaler_{ticker}.pkl")
    scaler = StandardScaler()

    # Train new model if it doesn't exist
    if not os.path.exists(model_path) or not os.path.exists(scaler_path):
        print(f"Training a new {model_type} model for {ticker}...")
        X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        if model_type == 'rf':  # Random Forest
            model = RandomForestRegressor(n_estimators=100, random_state=42)
            model.fit(X_train_scaled, y_train)
        elif model_type == 'lr':  # Linear Regression
            model = LinearRegression()
            model.fit(X_train_scaled, y_train)
        elif model_type == 'lstm':  # LSTM
            model = Sequential([
                LSTM(50, activation='relu', input_shape=(1, X_train_scaled.shape[1])),
                Dense(1)
            ])
            model.compile(optimizer='adam', loss='mse')
            X_train_scaled = X_train_scaled.reshape((X_train_scaled.shape[0], 1, X_train_scaled.shape[1]))
            X_test_scaled = X_test_scaled.reshape((X_test_scaled.shape[0], 1, X_test_scaled.shape[1]))
            model.fit(X_train_scaled, y_train, epochs=10, batch_size=32, verbose=1)

        # Save model and scaler
        if model_type == 'lstm':
            model.save(model_path)
        else:
            joblib.dump(model, model_path)
        joblib.dump(scaler, scaler_path)
    else:
        print(f"Loading existing {model_type} model for {ticker}...")
        if model_type == 'lstm':
            model = load_model(model_path)
        else:
            model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)

    return model, scaler

# Generate Visualizations
def generate_visualizations(data, y_actual, y_pred):
    """
    Generates visualizations for actual vs. predicted prices and residuals.
    """
    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(x=data['Date'], y=y_actual, mode='lines', name='Actual Prices'))
    fig1.add_trace(go.Scatter(x=data['Date'], y=y_pred, mode='lines', name='Predicted Prices'))
    graph1 = fig1.to_html(full_html=False)

    residuals = y_actual - y_pred
    fig2 = go.Figure()
    fig2.add_trace(go.Histogram(x=residuals, nbinsx=30, name='Residuals'))
    fig2.update_layout(title='Residuals Distribution', xaxis_title='Residuals', yaxis_title='Frequency')
    graph2 = fig2.to_html(full_html=False)

    fig3 = go.Figure()
    fig3.add_trace(go.Scatter(x=y_pred, y=residuals, mode='markers', name='Residuals'))
    fig3.update_layout(title='Residuals vs Predicted Prices', xaxis_title='Predicted Prices', yaxis_title='Residuals')
    graph3 = fig3.to_html(full_html=False)

    return {
        'graph1': graph1,
        'graph2': graph2,
        'graph3': graph3
    }

@app.route('/')
def index():
    """
    Renders the home page with a list of tickers and model selection.
    """
    with sqlite3.connect(DB_PATH) as conn:
        tickers = pd.read_sql("SELECT DISTINCT Ticker FROM processed_stocks", conn)['Ticker'].tolist()
    return render_template('index.html', tickers=tickers)

@app.route('/results', methods=['POST'])
def results():
    """
    Handles requests to display results for a selected ticker and model type.
    """
    ticker = request.form.get('ticker', DEFAULT_TICKER)
    model_type = request.form.get('model_type', 'lstm')

    # Fetch real-time data for the selected ticker
    data = fetch_and_process_data(ticker)

    if data is None or data.empty:
        return render_template('error.html', message=f"No data available for ticker {ticker}.")

    # Prepare features and labels
    features = ['7-day MA', '14-day MA', 'Volatility', 'Lag_1', 'Lag_2']
    X, y_actual = prepare_features_and_labels(data, features)

    # Load or train model and scaler
    model, scaler = load_or_train_model(ticker, model_type, X, y_actual)

    # Scale and reshape features based on model type
    if model_type == 'lstm':
        X_scaled = scaler.transform(X).reshape((X.shape[0], 1, X.shape[1]))
    else:
        X_scaled = scaler.transform(X)

    # Make predictions
    y_pred = model.predict(X_scaled)
    if model_type in ['lstm', 'rf']:  # Flatten predictions if necessary
        y_pred = y_pred.flatten()

    # Calculate metrics
    metrics = calculate_metrics(y_actual, y_pred)

    # Generate visualizations
    graphs = generate_visualizations(data, y_actual, y_pred)

    # Render the results page
    return render_template(
        'results.html',
        ticker=ticker,
        model_type=model_type,
        metrics=metrics,
        **graphs
    )

def train_ticker_model(ticker, model_type):
    """
    Train and save an LSTM, Linear Regression, or Random Forest model with scaler for the specified ticker.
    """
    with sqlite3.connect(DB_PATH) as conn:
        query = f"SELECT * FROM processed_stocks WHERE Ticker = '{ticker}'"
        data = pd.read_sql(query, conn)

    features = ['7-day MA', '14-day MA', 'Volatility', 'Lag_1', 'Lag_2']
    target = 'Adj Close'
    X = data[features]
    y = data[target]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    if model_type == 'lstm':
        scaler = MinMaxScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        X_train_scaled = X_train_scaled.reshape(X_train_scaled.shape[0], 1, X_train_scaled.shape[1])
        X_test_scaled = X_test_scaled.reshape(X_test_scaled.shape[0], 1, X_test_scaled.shape[1])

        model = Sequential()
        model.add(LSTM(264, input_shape=(1, X_train_scaled.shape[2]), activation='relu'))
        model.add(Dense(1))
        model.compile(optimizer='adam', loss='mean_squared_error')
        model.fit(X_train_scaled, y_train, epochs=50, batch_size=32, validation_data=(X_test_scaled, y_test))

        model_path = os.path.join(MODELS_PATH, f"model_{ticker}_lstm.h5")
        scaler_path = os.path.join(MODELS_PATH, f"scaler_{ticker}_lstm.pkl")
        model.save(model_path)

    elif model_type == 'rf':
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train_scaled, y_train)

        model_path = os.path.join(MODELS_PATH, f"model_{ticker}_rf.pkl")
        scaler_path = os.path.join(MODELS_PATH, f"scaler_{ticker}_rf.pkl")
        joblib.dump(model, model_path)

    else:
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        model = LinearRegression()
        model.fit(X_train_scaled, y_train)

        model_path = os.path.join(MODELS_PATH, f"model_{ticker}_linear.pkl")
        scaler_path = os.path.join(MODELS_PATH, f"scaler_{ticker}_linear.pkl")
        joblib.dump(model, model_path)

    joblib.dump(scaler, scaler_path)
    print(f"{model_type.upper()} model and scaler saved for {ticker}.")
    return model, scaler

@app.route('/about')
def about():
    """
    Renders the About page.
    """
    return render_template('about.html')

@app.route('/api/tickers', methods=['GET'])
def get_tickers():
    """
    API endpoint to fetch all tickers from the database.
    """
    try:
        with sqlite3.connect(DB_PATH) as conn:
            query = "SELECT Symbol, Company FROM tickers"
            cursor = conn.execute(query)
            results = cursor.fetchall()

        tickers = [{"symbol": row[0], "company": row[1]} for row in results]
        return jsonify(tickers)
    except Exception as e:
        return jsonify({"error": f"Failed to fetch tickers: {str(e)}"}), 500

if __name__ == '__main__':
    if not os.path.exists(MODELS_PATH):
        os.makedirs(MODELS_PATH)
    app.run(debug=True)