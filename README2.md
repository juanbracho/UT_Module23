# Stock Price Prediction Using Machine Learning

## Overview
This project aims to predict stock prices for multiple companies using machine learning models. It includes data fetching, preprocessing, model training, evaluation, and a user-friendly Flask web application for interactive exploration of results.

The project is structured into four core components:
1. **Data Collection**: Fetching stock data using the `yfinance` library.
2. **Data Preprocessing**: Engineering features for machine learning models.
3. **Model Training and Evaluation**: Training individual models for each ticker.
4. **Flask Web App**: Providing an interactive platform to visualize predictions.

## Project Features
- **Dynamic Predictions**: Train and generate predictions for individual stock tickers dynamically.
- **Interactive Web App**: Users can select tickers and visualize prediction results via a Flask-powered interface.
- **Documentation**: Each step of the project is well-documented in Jupyter notebooks.

## Technologies Used
- **Python Libraries**: `pandas`, `sqlite3`, `scikit-learn`, `joblib`, `yfinance`, `plotly`, `flask`
- **Database**: SQLite for storing raw and processed stock data.
- **Visualization**: Plotly for interactive charts.
- **Framework**: Flask for web app development.

---

## Project Structure
The project consists of the following notebooks and files:
1. **Notebook 1 - Data Fetching**:
   - Fetches stock data using `yfinance`.
   - Stores raw data in an SQLite database.

2. **Notebook 2 - Data Preprocessing**:
   - Calculates features like moving averages, volatility, and lagged prices.
   - Stores processed data in the database.

3. **Notebook 3 - ML Model Training**:
   - Trains a single model for all tickers.
   - Saves the model and scaler for reuse.

4. **Notebook 3.1 - ML Model Training for Individual Tickers**:
   - Trains separate models for each ticker.
   - Saves individual models and scalers in `pkl` files.

5. **Notebook 4 - Model Evaluation**:
   - Evaluates the aggregated model with visualizations.

6. **Notebook 4.1 - Model Evaluation for Individual Tickers**:
   - Evaluates individual ticker models dynamically.

7. **Flask App (`app.py`)**:
   - Provides an interface to select tickers and view predictions.

---

## Installation Instructions
### Prerequisites
- Python 3.8 or above installed on your machine.
- Recommended IDE: Jupyter Notebook or VS Code.

### Required Libraries
Install the necessary libraries using pip:

```bash
pip install pandas sqlite3 scikit-learn joblib yfinance plotly flask