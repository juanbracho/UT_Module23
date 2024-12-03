# Machine Learning Project: Oil and Telecom Sector Stock Analysis

## Overview

This project is designed to provide a comprehensive data visualization of stock market trends for three major companies in both the oil and telecom sectors: Exxon, ConocoPhillips, Chevron (Oil) and Verizon, T-Mobile, AT&T (Telecom). The visualizations display critical stock data such as adjusted close prices, stock volume, opening prices, high and low prices, percentage changes in stock prices, quarterly volumes, quarterly returns on investment (ROI), and projected ROI for a hypothetical $1500 investment. 

The purpose of this project is to analyze and compare these two sectors to understand which industry offered better investment opportunities based on stock performance during the year 2021. The project utilizes data fetched from external APIs and processed to provide clean, understandable, and interactive visualizations.

## Instructions

### Running the Application

1. Clone the repository to your local machine.
   ```bash
   git clone <repository-url>
2. Install the necessary dependencies by running the following:
    ```bash
    pip install -r requirements.txt
3. The project includes a Flask application. You can start the Flask app by running:
    ```bash
    python app.py
4. Once the Flask server is running, navigate to http://127.0.0.1:5000/ in your web browser to view the visualizations.
5. The visualizations include interactive plots built using Plotly. The user can toggle between:
* Oil Sector Data
* Telecom Sector Data
* Combined Data (Oil & Telecom)
6. A theme switch button is provided to toggle between a light and dark theme, making it easier for users to customize the view based on their preference.
7. You can also view the static HTML version without Flask by opening the index_no_flask.html file directly in your browser. This version requires no backend but limits interactivity and real-time data fetching.
## Features
* Toggle between Oil, Telecom, and Combined sector visualizations.
* Dropdown options allow users to filter by company and metric (e.g., Adjusted Close, Volume, ROI, etc.).
* Responsive design to ensure readability on different screen sizes.
* Light/Dark theme toggle.
* Interactive descriptions and annotations to explain the significance of the metrics.
* Footer with project team members and university information.
## Ethical Considerations
Ethical concerns were considered throughout the project development process:
1. Accuracy of Data: The data used for analysis was sourced from verified APIs (Yahoo Finance API via yfinance library). Care was taken to ensure that the data was not manipulated beyond typical cleaning steps, such as filling missing values or removing duplicates.
2. Privacy: The project does not use any personal or private data. All stock market data is publicly available through APIs and is used solely for educational and analysis purposes.
3. Financial Responsibility: The visualizations and analyses provided are meant for educational purposes and do not constitute financial advice. The data is historical, and stock performance should not be taken as an indicator of future results. Users should exercise caution and perform their own research before making any investment decisions.
## Data Sources
* Yahoo Finance API via the yfinance Python library: All stock data for the oil and telecom companies is fetched using this API.
## Code References
* Plotly: Interactive plotting library used for rendering charts and graphs.
* Flask: Used to run the web application backend.
* Pandas: Data manipulation and analysis library used to clean and process stock data.
* yfinance: Python library used to fetch stock data from Yahoo Finance.
## Project Team
* Juan Avila
* Kendall Burkett
* Patricia Taylor
* Rahmeen Zindani
* Cassio Sperb
