from flask import Flask, render_template, request
import pandas as pd
import joblib
import sqlite3
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import os

app = Flask(__name__)

# Path to SQLite database
DB_PATH = 'database/stocks_data.db'
MODELS_PATH = 'models/'

# Default ticker
DEFAULT_TICKER = 'XOM'

@app.route('/')
def index():
    # Fetch available tickers from the database
    with sqlite3.connect(DB_PATH) as conn:
        tickers = pd.read_sql("SELECT DISTINCT Ticker FROM processed_stocks", conn)['Ticker'].tolist()
    
    return render_template('index.html', tickers=tickers, default_ticker=DEFAULT_TICKER)

@app.route('/results', methods=['POST'])
def results():
    # Get the selected ticker
    ticker = request.form.get('ticker', DEFAULT_TICKER)

    # Check if the model and scaler exist
    model_path = os.path.join(MODELS_PATH, f"model_{ticker}.pkl")
    scaler_path = os.path.join(MODELS_PATH, f"scaler_{ticker}.pkl")

    if not os.path.exists(model_path) or not os.path.exists(scaler_path):
        # Create the model and scaler dynamically
        model, scaler = train_ticker_model(ticker)
    else:
        # Load existing model and scaler
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)

    # Fetch data for the selected ticker
    with sqlite3.connect(DB_PATH) as conn:
        query = f"SELECT * FROM processed_stocks WHERE Ticker = '{ticker}'"
        data = pd.read_sql(query, conn)

    # Extract features and scale them
    features = ['7-day MA', '14-day MA', 'Volatility', 'Lag_1', 'Lag_2']
    X = scaler.transform(data[features])
    y = data['Adj Close']

    # Predict
    y_pred = model.predict(X)

    # Calculate evaluation metrics
    mae = mean_absolute_error(y, y_pred)
    mse = mean_squared_error(y, y_pred)
    r2 = r2_score(y, y_pred)

    # Generate visualizations using Plotly
    # Visualization 1: Actual vs Predicted Prices
    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(x=data['Date'], y=data['Adj Close'], mode='lines', name='Actual Prices'))
    fig1.add_trace(go.Scatter(x=data['Date'], y=y_pred, mode='lines', name='Predicted Prices'))
    graph1 = fig1.to_html(full_html=False)

    # Visualization 2: Residuals Distribution
    residuals = y - y_pred
    fig2 = go.Figure()
    fig2.add_trace(go.Histogram(x=residuals, nbinsx=30, name='Residuals'))
    fig2.update_layout(title='Residuals Distribution', xaxis_title='Residuals', yaxis_title='Frequency')
    graph2 = fig2.to_html(full_html=False)

    # Visualization 3: Residuals vs Predicted Prices
    fig3 = go.Figure()
    fig3.add_trace(go.Scatter(x=y_pred, y=residuals, mode='markers', name='Residuals'))
    fig3.update_layout(title='Residuals vs Predicted Prices', xaxis_title='Predicted Prices', yaxis_title='Residuals')
    graph3 = fig3.to_html(full_html=False)

    return render_template(
        'results.html',
        ticker=ticker,
        mae=mae,
        mse=mse,
        r2=r2,
        graph1=graph1,
        graph2=graph2,
        graph3=graph3
    )

def train_ticker_model(ticker):
    """
    Train a model and scaler for the specified ticker and save them.
    """
    # Fetch data for the ticker
    with sqlite3.connect(DB_PATH) as conn:
        query = f"SELECT * FROM processed_stocks WHERE Ticker = '{ticker}'"
        data = pd.read_sql(query, conn)

    # Define features and target
    features = ['7-day MA', '14-day MA', 'Volatility', 'Lag_1', 'Lag_2']
    target = 'Adj Close'

    X = data[features]
    y = data[target]

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Normalize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train the model
    model = LinearRegression()
    model.fit(X_train_scaled, y_train)

    # Evaluate the model
    y_pred = model.predict(X_test_scaled)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"Trained model for {ticker}: MSE={mse:.2f}, R2={r2:.2f}")

    # Save the model and scaler
    model_path = os.path.join(MODELS_PATH, f"model_{ticker}.pkl")
    scaler_path = os.path.join(MODELS_PATH, f"scaler_{ticker}.pkl")

    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)

    print(f"Model and scaler saved for {ticker}.")

    return model, scaler

@app.route('/about')
def about():
    """
    Renders the About page (about.html).
    """
    return render_template('about.html')


# Ensure the app runs properly
if __name__ == '__main__':
    if not os.path.exists(MODELS_PATH):
        os.makedirs(MODELS_PATH)
    app.run(debug=True)