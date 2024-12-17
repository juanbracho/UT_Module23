
# Stock Price Prediction and Analysis

## Table of Contents
- [Project Overview](#project-overview)
- [Background](#background)
- [Questions We Hope to Answer](#questions-we-hope-to-answer)
- [Approach](#approach)
- [Data Exploration Phase](#data-exploration-phase)
- [Analysis Phase](#analysis-phase)
- [Technologies](#technologies)
- [Summary of Model Performance](#summary-of-model-performance)

## Project Overview
This project uses machine learning models to analyze and predict stock prices for various companies. The focus is on creating robust pipelines for data fetching, preprocessing, model training, and evaluation. The project includes multiple approaches, such as LSTM, Linear Regression, and Random Forest models, to assess prediction accuracy and efficiency.

## Background
The stock market is inherently volatile, and predicting stock prices has always been a challenge. This project aims to bridge the gap between data-driven insights and actionable predictions by leveraging machine learning. By exploring multiple algorithms and data features, this project seeks to:

1. Provide investors with tools to better anticipate stock price movements.

2. Showcase the effectiveness of different machine learning approaches, including time-series models like LSTM and traditional models like Random Forest and Linear Regression.

3. Highlight trends and anomalies in historical stock data that can inform investment decisions.


The selected companies include industry leaders, ensuring diverse representation and meaningful insights across sectors. The project is designed to evolve into a robust prediction tool, offering scalability and adaptability for new data and models.

## Setup Instructions

1. Clone this repository or download the project files.

2. Install the necessary dependancies:

	```bash 
	
	pip install -r requirements.txt

3. Open the Jupyter Notebook (e.g. data_fetching.ipynb) to follow along with the code and visualizations.


## Questions We Hope to Answer:

1. How accurately can stock prices be predicted using different machine learning models?

2. What are the key features that influence stock price predictions?

3. How do models like LSTM, Random Forest, and Linear Regression compare in performance for stock price forecasting?

4. Can engineered features such as moving averages and volatility improve prediction accuracy?

5. What trends or anomalies can be observed in historical stock data for specific companies?

6. What are the residual patterns and how can they inform improvements in model performance?

7. How does the prediction accuracy vary for different stock tickers and industries?

## Approach
To address these questions, the project follows a structured methodology, combining data collection, preprocessing, model development, and evaluation:

### 1. Data Collection

Source: 

• Historical stock price data is fetched from Yahoo Finance using the yfinance library.
Scope: 

• The dataset includes features like adjusted closing prices, volume, and market indicators for selected companies.
Storage: 

• Data is stored in an SQLite database for ease of access and future processing.

### 2. Data Preprocessing

Cleaning: 

• Handles missing values and filters out inconsistencies.

Feature Engineering: 

• Constructs new features such as moving averages (7-day, 14-day), volatility, and lagged values.

Normalization: 

• Scales features using MinMaxScaler or StandardScaler to prepare data for model input.

### 3. Model Training

LSTM Models:

• Trains long short-term memory (LSTM) models for time-series predictions.

• Leverages sequential data to capture trends and patterns over time.

Linear Regression Models:

• Implements traditional linear regression for baseline comparisons.

• Trains models on engineered features to assess their contribution.

Random Forest Models:

• Utilizes ensemble learning to improve prediction stability and accuracy.

• Captures non-linear relationships in the dataset.

### 4. Model Evaluation

Metrics:

• Mean Squared Error (MSE):

Definition: 

• Measures the average squared difference between actual and predicted values.

Significance:

• Penalizes larger errors more heavily, making it useful for identifying models that struggle with significant deviations.

• Lower MSE indicates better predictive accuracy.

• Root Mean Squared Error (RMSE):

Definition: 

• The square root of the Mean Squared Error, providing an error measure in the same units as the target variable.

Significance:

• More interpretable than MSE because it is expressed in the same units as stock prices.

• Lower RMSE reflects higher accuracy and minimal large errors.

• Mean Absolute Error (MAE):

Definition: 

• The average absolute difference between actual and predicted values.

Significance:

• Provides a linear scale of errors, giving equal weight to all deviations regardless of their magnitude.

• Lower MAE indicates the model’s ability to provide consistent predictions close to the actual values.

• R² (Coefficient of Determination):

Definition: 

• Represents the proportion of variance in the target variable that is explained by the model.

Significance:

• Values range from 0 to 1, where higher values indicate better model fit.

• A value close to 1 implies that the model explains a significant portion of the variability in stock prices.

Visualizations:

• Actual vs. Predicted stock prices.

• Residual distributions and scatter plots.

Comparison:

• Evaluates and compares models to identify strengths and weaknesses.

### 5. Iterative Refinement

• Incorporates findings from evaluations to refine feature engineering and hyperparameters for improved performance.

• Saves trained models and scalers for deployment or further testing.

## Data Exploration Phase
### Data Retrieval
Sources:

• Historical stock price data is retrieved from Yahoo Finance using the yfinance Python library.

• Stocks included in the analysis are selected for their market relevance, such as Exxon Mobil (XOM), Chevron (CVX), and others.

### 1. Methods 

• A Python script automates data fetching, using API calls to Yahoo Finance for daily stock price details, including Adj Close, Volume, and other financial metrics.

 • Data is downloaded in batches for each ticker and company, ensuring completeness and consistency.

### 2. Database Design

Database Type: 

• SQLite database was chosen for its simplicity and efficient handling of structured data.

Schema: 

• The database contains a single table for raw stock data with the following columns:

Date: 

• Date of the stock price.

Ticker: 

• Stock ticker symbol (e.g., XOM for Exxon Mobil).

Adj Close: 

• Adjusted closing price of the stock.

Volume: 

• Number of shares traded.

Company: 

• Full name of the company.

• Another table, processed_stocks, stores cleaned and preprocessed data, ready for analysis.

• Advantages: The SQLite database enables rapid queries and facilitates seamless integration with data analysis workflows.

### 3. Data Processing

Data Cleaning:

• Missing values in critical fields (e.g., Adj Close) are removed.

• Non-numeric values in price columns are coerced into numeric types, with errors handled gracefully.

Feature Engineering:

• Added features include:	Moving Averages:

* 7-day and 14-day moving averages to smooth out price fluctuations.

Volatility: 

• Captures daily price variations.

Lagged Features: 

• Previous day’s (Lag_1, Lag_2, etc.) stock prices are included for predictive modeling.

Storage: 

• The cleaned and engineered dataset is saved back into the SQLite database under a new table, processed_stocks.

## Analysis Phase
### Machine Learning Models
### 1. Machine Learning Models

• LSTM (Long Short-Term Memory)

Purpose: 

• Time-series model designed to capture sequential dependencies in stock price movements.

Key Features: 

• Moving Averages (7-day, 14-day)

• Volatility

• Lagged values (Lag_1, Lag_2, etc.) i.e. prices from previous day, prices from two days ago, etc.

Model Details: 

• Optimized architecture with multiple layers of LSTM, dropout layers for regularization, and dense layers for output predictions.

• Predicts future stock prices using sequential data.

• Linear Regression

Purpose: 

• Baseline model to predict stock prices based on engineered features.

Key Features: 

• Moving Averages

• Volatility

• Lagged values

Model Details: 

• Simple linear regression to establish a benchmark for predictive accuracy.

• Random Forest

Purpose: 

• Ensemble model to capture complex, non-linear relationships between features.

Key Features: 

• Moving Averages

• Volatility

• Lagged values

Model Details:

• Ensemble learning with 100 decision trees for improved prediction stability.

• Feature importance analysis to identify key drivers of stock prices.

### 2. Visualizations

#### Prediction Accuracy

Actual vs. Predicted Prices:

• Line charts comparing actual stock prices with model predictions for each ticker.

• Example: Visualization of Exxon Mobil (XOM) stock prices using LSTM predictions.

• Scatter Plot: Plots of actual vs. predicted values to assess the distribution of predictions.

#### Residual Analysis

Histogram of Residuals:	
	
• Displays the error distribution (difference between actual and predicted prices).

Residuals vs. Predicted: 

• Scatter plots showing patterns in residual errors, useful for identifying biases in model predictions.

## Technologies
This project leverages the following tools, programming languages, and platforms:

1. Programming Languages

	• Python: Primary language for data analysis, machine learning, and automation.

2. Libraries and Frameworks

    Data Handling:

	• pandas: For data manipulation and analysis.

	• numpy: For numerical operations.

	• sqlite3: For database management.

3. Visualization:

	• matplotlib: For creating static, interactive, and dynamic visualizations.

	• seaborn: For statistical data visualization (where applicable).

4. Machine Learning:

	• scikit-learn: For implementing Linear Regression and Random Forest models.

	•  keras and tensorflow: For building and training LSTM models.

5. Feature Engineering:

	• scikit-learn: For scaling and transforming data.

6. Miscellaneous:

	• joblib: For saving and loading models and scalers.

	• yfinance: For fetching stock price data.

7. Platforms and Tools

	Development:

	• Jupyter Notebook: Interactive environment for creating and sharing code and outputs.

	Data Storage:

	• SQLite: Lightweight, file-based database for structured data.

	Version Control:

	• Git and GitHub: For managing source code and collaboration.

## Summary of Model Performance
This project explored three machine learning models—LSTM, Linear Regression, and Random Forest—to predict stock prices. Each model was evaluated based on its accuracy, robustness, and ability to generalize to unseen data. Below is a detailed comparison:

### 1. Linear Regression

Strengths:

• Provides a baseline performance for stock price predictions.

• Simplicity and interpretability make it suitable for quick comparisons.

Weaknesses:

• Struggles with non-linear relationships inherent in stock price data.

• Limited ability to capture temporal dependencies or complex patterns.

Performance:

• Achieved moderate accuracy on engineered features like moving averages and lagged values.

Metrics:

• RMSE: Higher compared to the other models.

• R²: Moderate fit, highlighting its limitations in this context.

### 2. Random Forest

Strengths:

• Handles non-linear relationships effectively and performs feature importance analysis.

• Robust to overfitting due to ensemble learning.

Weaknesses:

• Computationally intensive compared to Linear Regression.

• Limited in capturing sequential dependencies in time-series data.

Performance:

• Outperformed Linear Regression in predicting stock prices.

Metrics:

• RMSE: Lower than Linear Regression but higher than LSTM.

• R²: High fit, with good generalization to unseen data.


### 3. LSTM (Long Short-Term Memory)

Strengths:

• Specifically designed for time-series data, capturing sequential dependencies and trends.

• Handles long-term dependencies effectively, making it ideal for stock price prediction.

Weaknesses:

• Computationally expensive to train and tune.

• Requires extensive preprocessing and normalization for optimal performance.

Performance:

• Outperformed both Linear Regression and Random Forest models.

Metrics:

• RMSE: Lowest among all models, indicating high accuracy.

• R²: Near-perfect fit for most stock tickers, showcasing its predictive power.

Overall Performance

• Most Effective Model: LSTM

• The LSTM model demonstrated superior performance due to its ability to model temporal dependencies and capture patterns in sequential data.

Implications:

• LSTM models are highly effective for stock price predictions, particularly when historical trends and sequential data are critical.

• While they require more computational resources and preprocessing, the trade-off is justified by their accuracy and robustness.

• Investors and financial analysts could leverage LSTM models to develop predictive tools for decision-making and risk management. 
