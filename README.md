# Stock Market Predictor

A machine learning application to analyze and predict stock prices using historical data and sentiment analysis of related news articles. The app is built with Streamlit for interactive visualizations and integrates a Keras model and BERT-based NLP models for analysis.

## Features

- **Historical Data Visualization**: 
   - Fetches historical stock data and displays it for the past 10 years.
   - Presents stock data such as company details, market cap, and sector.

- **Moving Averages & Trend Analysis**:
   - Computes and plots moving averages (50, 100, and 200-day) alongside the stock’s closing prices for trend analysis.

- **Price Prediction**:
   - Predicts future stock prices using a pre-trained model.
   - Displays original vs. predicted prices and generates a future trend based on the model's forecast.

- **Sentiment Analysis**:
   - Uses BERT to analyze the sentiment of the latest news articles about the stock.
   - Scores the relevance and sentiment of each article and presents them in an easy-to-read format.

## Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/stock-market-predictor.git
   cd stock-market-predictor

2. **Install the dependencies**:
   ```bash
   pip install -r requirements.txt

3. **Set up Environment Variables**:
      Create a .env file with your Alpha Vantage API key
   ```bash
   KEY=your_alpha_vantage_api_key
4. **Run the Application**:
   ```bash
   streamlit run app.py


