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

# 🎯 App Title 
st.markdown("""
# 📈 Stock Analysis & Prediction App  
Analyze stock performance, and predict future trends!  

**Features:**  
✔ Stock information (logo, summary, industry, market cap)  
✔ Historical stock price charts  
✔ Stock price prediction using Prophet  
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
stocks = ticker_list.iloc[:, 0].tolist()

# 🎯 Select stock from dropdown menu
selected_stock = st.sidebar.selectbox("📌 Choose Stock Ticker", stocks)

# 🎯 Fetch stock data using yf.download() with caching
@st.cache_data(ttl=3600)
def get_stock_data(ticker, start, end):
    return yf.download(ticker, start=start, end=end)

# 🎯 Display company information using Ticker object (for meta only)
@st.cache_data(ttl=3600)
def get_ticker_info(ticker):
    return yf.Ticker(ticker).info

ticker_info = get_ticker_info(selected_stock)
tickerDf = get_stock_data(selected_stock, start_date, end_date)

# 🎯 Display company information and logo
st.header(f'**📊 {ticker_info.get("longName", "Company Name Not Available")}**')

# Logo
logo_url = ticker_info.get('logo_url')
if not logo_url:
    company_domain = ticker_info.get('website', '').replace('http://', '').replace('https://', '').strip('/')
    if company_domain:
        logo_url = f"https://logo.clearbit.com/{company_domain}"

if logo_url:
    st.image(logo_url, width=150)
else:
    st.warning("⚠️ No logo available for this company.")

# Summary
st.info(ticker_info.get('longBusinessSummary', 'No summary available.'))

# 🎯 Stock Overview
st.subheader("📊 Stock Overview")
st.write(f"**Sector:** {ticker_info.get('sector', 'N/A')} | **Industry:** {ticker_info.get('industry', 'N/A')}")
st.write(f"**Market Cap:** {ticker_info.get('marketCap', 'N/A'):,}")
st.write(f"**Current Price:** {ticker_info.get('currentPrice', 'N/A')} | **Previous Close:** {ticker_info.get('previousClose', 'N/A')}")
st.write(f"**52-Week High:** {ticker_info.get('fiftyTwoWeekHigh', 'N/A')} | **52-Week Low:** {ticker_info.get('fiftyTwoWeekLow', 'N/A')}")
st.write(f"**P/E Ratio:** {ticker_info.get('trailingPE', 'N/A')}")
div_yield = ticker_info.get('dividendYield', 'N/A')
st.write(f"**Dividend Yield:** {div_yield:.2%}" if isinstance(div_yield, (int, float)) else "**Dividend Yield:** N/A")
st.write(f"**Beta:** {ticker_info.get('beta', 'N/A')}")
st.write(f"**Volume:** {ticker_info.get('volume', 0):,} | **Average Volume:** {ticker_info.get('averageVolume', 0):,}")
st.write(f"**50-Day Avg:** {ticker_info.get('fiftyDayAverage', 'N/A')} | **200-Day Avg:** {ticker_info.get('twoHundredDayAverage', 'N/A')}")

# 🎯 Fetch news
def fetch_stock_news_from_api(ticker, api_key):
    url = f'https://newsapi.org/v2/everything?q={ticker}&apiKey={api_key}'
    response = requests.get(url)
    if response.status_code == 200:
        return [{'title': a['title'], 'link': a['url'], 'source': a['source']['name']} for a in response.json().get('articles', [])]
    return []

st.subheader("📰 Latest News")
news_api_key = '047ad87e36534422b4bf4491b9ac6a71'
news = fetch_stock_news_from_api(selected_stock, news_api_key)

if news:
    for article in news[:5]:
        st.markdown(f"**[{article['title']}]({article['link']})**")
        st.write(f"🔗 Source: {article['source']}")
        st.write("---")
else:
    st.warning("⚠️ No news available.")

# 🎯 Plot stock data
st.subheader("📊 Raw Stock Data")
st.write(tickerDf.tail())

def plot_raw_data(df):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df['Open'], name='Open'))
    fig.add_trace(go.Scatter(x=df.index, y=df['Close'], name='Close'))
    fig.update_layout(title="📈 Stock Price Movement", xaxis_title="Date", yaxis_title="Price", template="plotly_dark", xaxis_rangeslider_visible=True)
    st.plotly_chart(fig)

plot_raw_data(tickerDf)

# 🎯 Forecast with Prophet
st.subheader("🔮 Stock Price Prediction")
n_years = st.sidebar.slider("📅 Years of Prediction", 1, 4)
period = n_years * 365

df_train = tickerDf[['Close']].reset_index()
df_train.rename(columns={"Date": "ds", "Close": "y"}, inplace=True)
df_train['ds'] = pd.to_datetime(df_train['ds'])

m = Prophet()
m.fit(df_train)
future = m.make_future_dataframe(periods=period)
forecast = m.predict(future)

st.write("📈 Forecast Data")
st.write(forecast.tail())

fig1 = plot_plotly(m, forecast)
st.plotly_chart(fig1)

st.write("📊 Forecast Components")
fig2 = m.plot_components(forecast)
st.write(fig2)
