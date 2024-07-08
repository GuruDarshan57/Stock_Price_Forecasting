import numpy as np
import pandas as pd
import yfinance as yf
import requests
from keras.models import load_model
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from datetime import datetime, timedelta

# Load the pre-trained model
model = load_model('Stock Predictions Model.keras')

# Streamlit header
st.header('Stock Market Predictor')

# User input for stock symbol
stock = st.text_input('Enter Stock Symbol', 'GOOG')
# Get current date and calculate the date 10 years ago
end_date = datetime.today().strftime('%Y-%m-%d')
start_date = (datetime.today() - timedelta(days=365*10)).strftime('%Y-%m-%d')

# Fetch stock data
data = yf.download(stock, start, end)

# Display stock data
st.subheader('Stock Data')
st.write(data)

# Fetch stock details
stock_info = yf.Ticker(stock).info
stock_details = {
    'Company Name': stock_info.get('longName', 'N/A'),
    'Sector': stock_info.get('sector', 'N/A'),
    'Industry': stock_info.get('industry', 'N/A'),
    'Country': stock_info.get('country', 'N/A'),
    'Market Cap': stock_info.get('marketCap', 'N/A'),
    'Website': stock_info.get('website', 'N/A')
}

# Display stock details in table format
st.subheader('Stock Details')
details_df = pd.DataFrame(stock_details.items(), columns=['Attribute', 'Value'])
st.table(details_df)

# Data preparation
data_train = pd.DataFrame(data.Close[0: int(len(data) * 0.80)])
data_test = pd.DataFrame(data.Close[int(len(data) * 0.80): len(data)])

scaler = MinMaxScaler(feature_range=(0, 1))

pas_100_days = data_train.tail(100)
data_test = pd.concat([pas_100_days, data_test], ignore_index=True)
data_test_scale = scaler.fit_transform(data_test)

# Moving averages
ma_50_days = data.Close.rolling(50).mean()
ma_100_days = data.Close.rolling(100).mean()
ma_200_days = data.Close.rolling(200).mean()

# Plot Price vs MA100 vs MA200
st.subheader('Price vs MA100 vs MA200')
fig3 = plt.figure(figsize=(8, 6))
plt.plot(ma_100_days, 'r', label='MA100')
plt.plot(ma_200_days, 'b', label='MA200')
plt.plot(data.Close, 'g', label='Close Price')
plt.legend()
plt.show()
st.pyplot(fig3)

# Prepare data for prediction
x = []
y = []

for i in range(100, data_test_scale.shape[0]):
    x.append(data_test_scale[i-100:i])
    y.append(data_test_scale[i, 0])

x, y = np.array(x), np.array(y)

# Predict prices
predict = model.predict(x)
scale = 1 / scaler.scale_
predict = predict * scale
y = y * scale

# Generate future predictions
future_x = data_test_scale[-100:]
future_x = np.expand_dims(future_x, axis=0)
future_predict = []

for _ in range(100):
    pred = model.predict(future_x)
    future_predict.append(pred[0, 0])
    future_x = np.append(future_x[:, 1:, :], np.expand_dims(pred, axis=2), axis=1)

future_predict = np.array(future_predict) * scale

# Plot original price vs predicted price with future trend
st.subheader('Original Price vs Predicted Price with Future Trend')
fig4 = plt.figure(figsize=(8, 6))
plt.plot(predict, 'r', label='Predicted Price')
plt.plot(y, 'g', label='Original Price')
plt.plot(np.arange(len(y), len(y) + len(future_predict)), future_predict, 'b--', label='Future Trend')
plt.xlabel('Days')
plt.ylabel('Price')
plt.legend()
plt.show()
st.pyplot(fig4)

# Calculate performance metrics
mae = mean_absolute_error(y, predict)
mse = mean_squared_error(y, predict)
rmse = np.sqrt(mse)

# Display performance metrics
st.subheader('Model Performance Metrics')
performance_metrics = pd.DataFrame({
    'Metric': ['Mean Absolute Error (MAE)', 'Mean Squared Error (MSE)', 'Root Mean Squared Error (RMSE)'],
    'Value': [mae, mse, rmse]
})
st.table(performance_metrics)

# Streamlit header
st.header('Stock News and Sentiment')

# API endpoint for fetching news sentiment
url = f'https://www.alphavantage.co/query?function=NEWS_SENTIMENT&tickers={stock}&limit=20&sort=RELEVANCE&apikey=#'

# Fetch data from API
response = requests.get(url)
data = response.json()

# Check if data fetch was successful
if 'feed' in data:
    feed = data['feed'][:10]  # Fetching top 10 articles
    articles = []
    total_relevance = 0
    weighted_sentiment_sum = 0

    # Prepare data for display
    for item in feed:
        title = item['title']
        url = item['url']
        ticker_sentiment = item['ticker_sentiment'][0]  # Assuming there's only one ticker sentiment per article
        relevance_score = float(ticker_sentiment['relevance_score'])
        sentiment_score = float(ticker_sentiment['ticker_sentiment_score'])
        sentiment_label = ticker_sentiment['ticker_sentiment_label']

        total_relevance += relevance_score
        weighted_sentiment_sum += relevance_score * sentiment_score

        articles.append({
            'Title': title,
            'URL': url,
            f'{stock} Relevance Score': relevance_score,
            f'{stock} Sentiment Score': sentiment_score,
            f'{stock} Sentiment Label': sentiment_label
        })

    # Calculate overall sentiment
    overall_sentiment = weighted_sentiment_sum / total_relevance if total_relevance != 0 else 0

    # Display articles in a table format
    st.subheader('Top News Articles')
    df = pd.DataFrame(articles)
    st.table(df[['Title', 'URL', f'{stock} Relevance Score', f'{stock} Sentiment Score', f'{stock} Sentiment Label']])

    # Display overall sentiment
    st.subheader('Overall Sentiment')
    sentiment_df = pd.DataFrame({
        'Attribute': ['Overall Sentiment Score'],
        'Value': [overall_sentiment]
    })
    st.table(sentiment_df)
else:
    st.write("No news articles found for the chosen stock symbol.")
