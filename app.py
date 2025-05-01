import streamlit as st
import yfinance as yf
import pandas as pd
import cufflinks as cf
import datetime
from datetime import date
from prophet import Prophet
from prophet.plot import plot_plotly
from plotly import graph_objs as go
import requests
from bs4 import BeautifulSoup

# 🎯 App Title 
st.markdown("""
# 📈 Stock Analysis & Prediction App  
Analyze stock performance, and predict future trends!  

**Features:**  
✔ Stock information (logo, summary, industry, market cap)  
✔ Historical stock price charts
✔ Stock price prediction using 
✔ Compare two stocks  
✔ Download stock data  
""")
st.write("---")

# 🎯 Sidebar for user input
st.sidebar.subheader('📊 Query Parameters')
start_date = st.sidebar.date_input("📅 Start Date", datetime.date(2019, 1, 1))
end_date = st.sidebar.date_input("📅 End Date", datetime.date(2021, 1, 31))

# 🎯 Load ticker symbols
ticker_list = pd.read_csv('https://raw.githubusercontent.com/dataprofessor/s-and-p-500-companies/master/data/constituents_symbols.txt')

# Inspect the columns to confirm the name of the column containing the stock symbols
stocks = ticker_list.iloc[:, 0].tolist()  # Assuming the first column contains the stock symbols

# 🎯 Select stock from dropdown menu (Only one dropdown here)
selected_stock = st.sidebar.selectbox("📌 Choose Stock Ticker", stocks)

# 🎯 Fetch stock data
tickerData = yf.Ticker(selected_stock)
tickerDf = tickerData.history(period='1d', start=start_date, end=end_date)

# 🎯 Display company information and logo
st.header(f'**📊 {tickerData.info.get("longName", "Company Name Not Available")}**')

# Fetch company logo from Yahoo Finance or Clearbit API
logo_url = tickerData.info.get('logo_url', None)
if not logo_url:
    company_domain = tickerData.info.get('website', '').replace('http://', '').replace('https://', '').strip('/')
    if company_domain:
        logo_url = f"https://logo.clearbit.com/{company_domain}"

# Display the logo and company description
if logo_url:
    st.image(logo_url, width=150)
else:
    st.warning("⚠️ No logo available for this company.")
    
# Display company summary
string_summary = tickerData.info.get('longBusinessSummary', 'No company summary available.')
st.info(string_summary)

# 🎯 Display Additional Stock Data
st.subheader("📊 Stock Overview")

# Company Details
sector = tickerData.info.get('sector', 'N/A')
industry = tickerData.info.get('industry', 'N/A')
market_cap = tickerData.info.get('marketCap', 'N/A')
st.write(f"**Sector:** {sector} | **Industry:** {industry}")
st.write(f"**Market Cap:** {market_cap:,}")

# Price Details
current_price = tickerData.info.get('currentPrice', 'N/A')
previous_close = tickerData.info.get('previousClose', 'N/A')
st.write(f"**Current Price:** {current_price} | **Previous Close:** {previous_close}")

# 52-Week High & Low
week_high = tickerData.info.get('fiftyTwoWeekHigh', 'N/A')
week_low = tickerData.info.get('fiftyTwoWeekLow', 'N/A')
st.write(f"**52-Week High:** {week_high} | **52-Week Low:** {week_low}")

# P/E Ratio & Dividend
pe_ratio = tickerData.info.get('trailingPE', 'N/A')
dividend_yield = tickerData.info.get('dividendYield', 'N/A')
st.write(f"**P/E Ratio:** {pe_ratio}")
st.write(f"**Dividend Yield:** {dividend_yield:.2%}" if isinstance(dividend_yield, (int, float)) and dividend_yield else "**Dividend Yield:** N/A")

# Stock Movement & Technicals
beta = tickerData.info.get('beta', 'N/A')
volume = tickerData.info.get('volume', 'N/A')
avg_volume = tickerData.info.get('averageVolume', 'N/A')
ma_50 = tickerData.info.get('fiftyDayAverage', 'N/A')
ma_200 = tickerData.info.get('twoHundredDayAverage', 'N/A')

st.write(f"**Beta (Volatility Indicator):** {beta}")
st.write(f"**Current Volume:** {volume:,} | **Average Volume:** {avg_volume:,}")
st.write(f"**50-Day Moving Average:** {ma_50} | **200-Day Moving Average:** {ma_200}")


def fetch_stock_news_from_api(ticker, api_key):
    """Fetches news articles for a given stock using the News API."""
    url = f'https://newsapi.org/v2/everything?q={ticker}&apiKey=047ad87e36534422b4bf4491b9ac6a71'
    response = requests.get(url)
    
    if response.status_code == 200:
        news_data = response.json()
        articles = news_data.get('articles', [])
        
        return [{'title': article['title'], 'link': article['url'], 'source': article['source']['name']} for article in articles]
    else:
        return []

# Replace with your own API key from NewsAPI
news_api_key = '047ad87e36534422b4bf4491b9ac6a71'

# Integrate into Streamlit for news
st.subheader("📰 Latest News")

news = fetch_stock_news_from_api(selected_stock, news_api_key)

# Display the news
if news:
    for article in news[:5]:  # Display only the top 5 news articles
        title = article.get("title", "No Title Available")
        link = article.get("link", "#")
        source = article.get("source", "Unknown Source")
        
        st.markdown(f"**[{title}]({link})**")
        st.write(f"🔗 Source: {source}")
        st.write("---")  # Separator for better readability
else:
    st.warning("⚠️ No news available for this stock.")



# Set up for stock prediction with Prophet
START = "2015-01-01"
TODAY = date.today().strftime("%Y-%m-%d")

# 🎯 Sidebar for Years of Prediction
n_years = st.sidebar.slider("📅 Years of Prediction", 1, 4)
period = n_years * 365  # Number of days to predict

# Caching function to load stock data efficiently
@st.cache_data
def load_data(ticker):
    """Fetch stock data using yfinance and clean it."""
    data = yf.download(ticker, START, TODAY)
    data.reset_index(inplace=True)  # Reset index to make "Date" a column
    data.columns = [col[0] if isinstance(col, tuple) else col for col in data.columns]  # Clean column names
    return data

# 🎯 Load stock data and display raw data
with st.spinner("📥 Fetching data..."):
    data = load_data(selected_stock)

# Display raw stock data
st.subheader("📊 Raw Stock Data")
st.write(data.tail())

# 🎯 Plotting Stock Data
def plot_raw_data(data):
    """Plot the stock's open and close prices."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Open'], mode='lines', name='Stock Open'))
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], mode='lines', name='Stock Close'))
    fig.update_layout(
        title="📈 Stock Price Movement",
        xaxis_title="Date",
        yaxis_title="Price",
        xaxis_rangeslider_visible=True,
        template="plotly_dark"  # Optional: Makes the chart look better
    )
    st.plotly_chart(fig)

# Plot stock price movement
plot_raw_data(data)

# 🎯 Forecasting Stock Prices using Prophet
st.subheader("🔮 Stock Price Prediction")

# Prepare the data for forecasting
df_train = data[['Date', 'Close']]
df_train = df_train.rename(columns={"Date": "ds", "Close": "y"})
df_train['ds'] = df_train['ds'].dt.tz_localize(None)  # Remove timezone information

# Initialize and fit the Prophet model
m = Prophet()
m.fit(df_train)

# Create future dates and make a prediction
future = m.make_future_dataframe(periods=period)
forecast = m.predict(future)

# Display forecasted data
st.write("📈 Forecast Data")
st.write(forecast.tail())

# 🎯 Plot Forecasted Data
fig1 = plot_plotly(m, forecast)
st.plotly_chart(fig1)

# 🎯 Display Forecast Components (Trend, Weekly, Yearly)
st.write("📊 Forecast Components")
fig2 = m.plot_components(forecast)
st.write(fig2)
