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
    model_path = os.path.join(MODELS_PATH, f"model_{ticker}_{model_type}.pkl" if model_type == 'rf' else f"model_{ticker}_{model_type}.h5")
    scaler_path = os.path.join(MODELS_PATH, f"scaler_{ticker}_{model_type}.pkl")

    # Load or train the selected model
    if not os.path.exists(model_path) or not os.path.exists(scaler_path):
        model, scaler = train_ticker_model(ticker, model_type)
    else:
        if model_type == 'lstm':
            model = load_model(model_path)
        else:
            model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)

    # Fetch and preprocess data
    with sqlite3.connect(DB_PATH) as conn:
        query = f"SELECT * FROM processed_stocks WHERE Ticker = '{ticker}'"
        data = pd.read_sql(query, conn)

    features = ['7-day MA', '14-day MA', 'Volatility', 'Lag_1', 'Lag_2']
    X = data[features].values
    X_scaled = scaler.transform(X)

    if model_type == 'lstm':
        X_scaled = X_scaled.reshape(X_scaled.shape[0], 1, X_scaled.shape[1])

    y_actual = data['Adj Close'].values
    y_pred = model.predict(X_scaled)
    if model_type == 'lstm' or model_type == 'rf':
        y_pred = y_pred.flatten()

    # Calculate evaluation metrics
    metrics = {
        'mae': mean_absolute_error(y_actual, y_pred),
        'mse': mean_squared_error(y_actual, y_pred),
        'r2': r2_score(y_actual, y_pred)
    }

    # Generate visualizations
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

    return render_template(
        'results.html',
        ticker=ticker,
        model_type=model_type,
        metrics=metrics,
        graph1=graph1,
        graph2=graph2,
        graph3=graph3
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

if __name__ == '__main__':
    if not os.path.exists(MODELS_PATH):
        os.makedirs(MODELS_PATH)
    app.run(debug=True)