# Stock Price Forecasting 

This application predicts stock prices using a pre-trained machine learning model and provides relevant stock news and sentiment analysis. The stock market predictor is built with **Streamlit**, **Keras**, **yfinance**, and **Alpha Vantage** for news sentiment.

## Features

- **Stock Price Prediction:** Uses a pre-trained Keras model to predict stock prices.
- **Historical Data Visualization:** Plots price trends with Moving Averages (MA100, MA200).
- **Future Price Prediction:** Generates future trend predictions for 100 days.
- **Model Performance Metrics:** Displays MAE, MSE, and RMSE performance metrics.
- **Stock News and Sentiment:** Fetches top stock-related news articles and sentiment scores using Alpha Vantage API.

## Technologies Used

- **Frontend:** Streamlit for the UI.
- **Backend:** 
  - **Keras:** Pre-trained model for price predictions.
  - **yfinance:** For fetching historical stock data.
  - **Alpha Vantage:** For stock news and sentiment analysis.
- **Other Libraries:**
  - `numpy`, `pandas`, `requests`, `matplotlib`, `scikit-learn`


