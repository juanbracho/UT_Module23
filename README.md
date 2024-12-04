# Machine Learning Project: Oil and Telecom Sector Stock Analysis

## Overview
The **Stock Price Predictor** is a Flask-based web application that utilizes machine learning to predict stock prices for selected tickers. This project demonstrates the integration of Python, Flask, SQL databases, and interactive visualizations to build a functional web application for data analytics. It was developed as part of the **University of Texas at Austin Data Analytics Bootcamp**.

---

## Features
1. **Stock Price Prediction**:
   - Predicts stock prices based on a linear regression model.
   - Supports multiple stock tickers fetched from a database.

2. **Interactive Visualizations**:
   - **Actual vs. Predicted Prices**: Line chart showing the model's accuracy.
   - **Residuals Distribution**: Histogram to assess the model's error distribution.
   - **Residuals vs. Predicted Prices**: Scatter plot to check bias in predictions.

3. **Dynamic Dropdown for Ticker Selection**:
   - Allows users to select a stock ticker and view predictions for the selected ticker.

4. **Responsive Design**:
   - Optimized layout that adapts to different screen sizes, including a grid for graphs.

5. **Pre-trained and On-the-Fly Models**:
   - Models and scalers for specific tickers are pre-trained and saved for quick predictions.
   - Automatically trains and saves models for new tickers when needed.

---

## Technologies and Skills
### Technologies
- **Python**:
  - Core programming language used for data processing, machine learning, and server-side scripting.
- **Flask**:
  - Web framework to manage routes and serve dynamic content.
- **SQLite**:
  - SQL-based lightweight database for storing stock data.
- **Plotly**:
  - For creating interactive visualizations.
- **Bootstrap**:
  - Front-end framework for responsive design and styling.

### Skills
- Data Preprocessing: Using `pandas` for cleaning and transforming data.
- Machine Learning:
  - **Linear Regression** (via `scikit-learn`): Predicting stock prices.
  - Performance Metrics: MAE, MSE, and R².
- Database Management: SQL queries to interact with `SQLite` for dynamic ticker handling.
- Web Development:
  - HTML, CSS, and JavaScript for user interaction.
  - Flask integration for back-end logic.
- Visualization: Plotly for creating interactive charts.

---

## Requirements
### Functional Requirements
1. Use machine learning to predict stock prices.
2. Dataset with at least 100 records.
3. Use at least 2 of the following:
   - Python Pandas
   - Python Matplotlib/Plotly
   - SQL Database
   - Flask for web integration

### Implementation
- Linear regression models predict stock prices based on historical data.
- Data sourced from Yahoo Finance, processed, and stored in an SQLite database.

---

## Instructions
### Prerequisites
1. Install Python (3.8 or higher) and the following libraries:
   ```bash
   pip install flask pandas joblib scikit-learn plotly

2. Ensure the following project structure

* 📂 project-root/
* ├── 📁 templates/      # HTML templates (Flask)
* │   ├── index.html     # Home page
* │   ├── results.html   # Results page
* │   └── about.html     # About page
* ├── 📁 static/
* │   ├── 📁 css/
* │   │   └── styles.css # CSS styles for the project
* │   ├── 📁 js/
* │   │   └── script.js  # JavaScript functionality (e.g., spinner)
* ├── app.py             # Main Flask app
* ├── stocks_data.db     # SQLite database
* └── README.md          # Project documentation


## How to Run
1. Clone this repository and navigate to the project folder.
2. Ensure the database is populated with the stock data (`stocks_data.db`).
3. Run the Flask app:
   ```bash
   python app.py
4. Open your browser and go to http://127.0.0.1:500

## Resutls and Visualizations
* The application displays:
    1. Model Evaluation Metris: MAE, MSE and R².
    2. Interactive Visualizations:
        * Actual vs. Predicted Prices
        * Residuals Distribution
        * Residuals vs. Predicted Prices
* These visualizations provide insights into the model's performance and prediction accuracy

## Project Team
* Juan Avila
* Kendall Burkett
* Patricia Taylor
* Cassio Sperb
* Rahmeen Zindani
