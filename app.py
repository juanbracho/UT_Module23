from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import sqlite3
import os
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import plotly.graph_objects as go
import joblib

app = Flask(__name__)

# Paths
DB_PATH = 'database/stocks_data.db'
MODELS_PATH = 'models/'

# Default ticker and model type
DEFAULT_TICKER = 'XOM'

@app.route('/')
def index():
    """
    Renders the home page with a list of tickers and model selection.
    """
    with sqlite3.connect(DB_PATH) as conn:
        tickers = pd.read_sql("SELECT DISTINCT Ticker FROM processed_stocks", conn)['Ticker'].tolist()
    return render_template('index.html', tickers=tickers, default_ticker=DEFAULT_TICKER)

@app.route('/results', methods=['POST'])
def results():
    """
    Handles requests to display results for a selected ticker and model type.
    """
    ticker = request.form.get('ticker', DEFAULT_TICKER)
    model_type = request.form.get('model_type', 'lstm')

    # Define paths for the model and scaler
    model_path = os.path.join(MODELS_PATH, f"model_{ticker}_lstm.h5")
    scaler_path = os.path.join(MODELS_PATH, f"scaler_{ticker}_lstm.pkl")

    # Load the model and scaler
    if not os.path.exists(model_path) or not os.path.exists(scaler_path):
        return f"Model or scaler for ticker '{ticker}' does not exist. Train it first."

    model = load_model(model_path, compile=False)
    scaler = joblib.load(scaler_path)

    # Fetch data for the selected ticker
    with sqlite3.connect(DB_PATH) as conn:
        query = f"SELECT * FROM processed_stocks WHERE Ticker = '{ticker}'"
        data = pd.read_sql(query, conn)

    # Feature Engineering: Recreate features used during training
    data['Lag_1'] = data['Adj Close'].shift(1)
    data['Lag_2'] = data['Adj Close'].shift(2)
    data['Lag_3'] = data['Adj Close'].shift(3)
    data['Volatility'] = data['Adj Close'].rolling(window=7).std()
    data['Momentum'] = data['Adj Close'].pct_change(periods=3)

    # Drop rows with NaN values introduced by feature engineering
    data = data.dropna()

    # Define features and target
    features = scaler.feature_names_in_
    target = 'Adj Close'

    # Align features with training order
    X_raw = data[features]
    y_actual = data[target]

    # Normalize the features
    X_scaled = scaler.transform(X_raw)

    # Reshape the features for LSTM (samples, timesteps, features)
    X_scaled = X_scaled.reshape(X_scaled.shape[0], 1, X_scaled.shape[1])

    # Make Predictions
    y_pred = model.predict(X_scaled)

    # Calculate evaluation metrics
    metrics = {
        'mae': mean_absolute_error(y_actual, y_pred),
        'mse': mean_squared_error(y_actual, y_pred),
        'r2': r2_score(y_actual, y_pred)
    }

    # Generate visualizations
    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(x=data['Date'], y=y_actual, mode='lines', name='Actual Prices'))
    fig1.add_trace(go.Scatter(x=data['Date'], y=y_pred.flatten(), mode='lines', name='Predicted Prices'))
    graph1 = fig1.to_html(full_html=False)

    residuals = y_actual - y_pred.flatten()
    fig2 = go.Figure()
    fig2.add_trace(go.Histogram(x=residuals, nbinsx=30, name='Residuals'))
    fig2.update_layout(title='Residuals Distribution', xaxis_title='Residuals', yaxis_title='Frequency')
    graph2 = fig2.to_html(full_html=False)

    fig3 = go.Figure()
    fig3.add_trace(go.Scatter(x=y_pred.flatten(), y=residuals, mode='markers', name='Residuals'))
    fig3.update_layout(title='Residuals vs Predicted Prices', xaxis_title='Predicted Prices', yaxis_title='Residuals')
    graph3 = fig3.to_html(full_html=False)

    return render_template(
        'results.html',
        ticker=ticker,
        model_type=model_type,
        metrics=metrics,
        graph1=graph1,
        graph2=graph2,
        graph3=graph3
    )

@app.route('/about')
def about():
    """
    Renders the About page.
    """
    return render_template('about.html')

if __name__ == '__main__':
    if not os.path.exists(MODELS_PATH):
        os.makedirs(MODELS_PATH)
    app.run(debug=True)