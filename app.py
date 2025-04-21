import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from transformers import pipeline
from pytube import YouTube
import requests

# Retrieve the API key from Streamlit secrets
key = st.secrets["api_keys"]["youtube"]

# Initialize sentiment analysis model
pipe = pipeline("text-classification", model="mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis")

# Define functions
def yt_id(url):
    yt = YouTube(url)
    return yt.channel_id

def get_channel_data(key, id):
    channel_data = []
    res = f'https://www.googleapis.com/youtube/v3/search?key={key}&channelId={id}&part=snippet,id&order=date&maxResults=50'
    r = requests.get(res)
    data = r.json()

    for item in data['items']:
        channel_data.append({
            'Publish Date': item['snippet']['publishedAt'],
            'Title': item['snippet']['title'],
            'Description': item['snippet']['description'],
            'Channel Name': item['snippet']['channelTitle']
        })
    return pd.DataFrame(channel_data)

def find_stock_mentions(text, stock_symbol, company_name):
    return stock_symbol.lower() in text.lower() or company_name.lower() in text.lower()

# Streamlit application
st.title('YouTube Channel Sentiment Analysis')

# Input for API key
st.write("This app uses the YouTube API to analyze recent videos from specific channels.")
st.write("Ensure that the API key is set in Streamlit secrets.")

# List of video URLs (you might want to make this dynamic)
videos = [
    'https://www.youtube.com/watch?v=RKFxWzJuQTw',
    'https://www.youtube.com/watch?v=Xa5cc8mgczc',
    'https://www.youtube.com/watch?v=EP6JqpjtUjM',
    'https://www.youtube.com/watch?v=3FnQmDld9gA'
]

channel_ids = [yt_id(video) for video in videos]

yahoo = get_channel_data(key, channel_ids[0])
cnbc = get_channel_data(key, channel_ids[1])
bloomberg = get_channel_data(key, channel_ids[2])

df = pd.concat([yahoo, cnbc, bloomberg]).reset_index(drop=True)

# Define stock dictionary
stock_dict = {
    'AAPL': 'Apple',
    'TSLA': 'Tesla',
    'GOOGL': 'Google',
    'MSFT': 'Microsoft',
    'AMZN': 'Amazon',
    'META': 'Meta Platforms',
    'NFLX': 'Netflix',
    'NVDA': 'NVIDIA',
    'BRK-B': 'Berkshire Hathaway',
    'JPM': 'JPMorgan Chase',
    'V': 'Visa',
    'MA': 'Mastercard',
    'DIS': 'Walt Disney',
    'BA': 'Boeing',
    'HD': 'Home Depot',
    'IBM': 'IBM',
    'PFE': 'Pfizer',
    'CSCO': 'Cisco Systems',
    'ADBE': 'Adobe',
    'INTC': 'Intel',
    'ORCL': 'Oracle',
    'COST': 'Costco',
    'WMT': 'Walmart',
    'T': 'AT&T',
    'KO': 'Coca-Cola',
    'XOM': 'ExxonMobil',
    'CVX': 'Chevron',
    'LMT': 'Lockheed Martin',
    'MCD': 'McDonald\'s',
    'NKE': 'Nike',
    'UNH': 'UnitedHealth Group',
    'MDT': 'Medtronic',
    'GILD': 'Gilead Sciences',
    'MRK': 'Merck',
    'ABBV': 'AbbVie',
    'BMY': 'Bristol-Myers Squibb',
    'TXN': 'Texas Instruments',
    'SBUX': 'Starbucks',
    'GS': 'Goldman Sachs',
    'USB': 'U.S. Bancorp',
    'SCHW': 'Charles Schwab',
    'AMT': 'American Tower',
    'DHR': 'Danaher',
    'UNP': 'Union Pacific',
    'CAT': 'Caterpillar',
    'UPS': 'United Parcel Service',
    'TMO': 'Thermo Fisher Scientific',
    'CME': 'CME Group',
    'TGT': 'Target',
    'DE': 'Deere & Co',
    'CVS': 'CVS Health',
    'AON': 'Aon',
    'AIG': 'American International Group',
    'BNS': 'Bank of Nova Scotia',
    'RBLX': 'Roblox',
    'SHOP': 'Shopify',
    'TWTR': 'Twitter',
    'SPOT': 'Spotify',
    'SNOW': 'Snowflake',
    'BYND': 'Beyond Meat',
    'PINS': 'Pinterest',
    'SQ': 'Square',
    'PLTR': 'Palantir Technologies'
}


reversed_stock_dict = {v.lower(): k for k, v in stock_dict.items()}

def get_stock_info(user_input):
    user_input = user_input.lower()
    return reversed_stock_dict.get(user_input, None) or stock_dict.get(user_input.upper(), None)

# User input for stock symbol or name
user_input = st.text_input('Enter stock symbol or name:')

if user_input:
    stock_symbol = get_stock_info(user_input)
    company_name = stock_dict.get(stock_symbol, None)
    
    if company_name:
        df['Stock Mentioned'] = df.apply(lambda row: find_stock_mentions(row['Title'] + row['Description'], stock_symbol, company_name), axis=1)
        df_filtered = df[df['Stock Mentioned']]
        
        if df_filtered.empty:
            st.write(f"No mentions of {stock_symbol} ({company_name}) found.")
        else:
            df_filtered['Sentiment Label'] = df_filtered.apply(lambda row: pipe(row['Title'] + ' ' + row['Description'])[0]['label'], axis=1)
            df_filtered['Sentiment Score'] = df_filtered.apply(lambda row: pipe(row['Title'] + ' ' + row['Description'])[0]['score'], axis=1)
            
            st.write(df_filtered[['Publish Date', 'Title', 'Sentiment Label', 'Sentiment Score']].head(10))
            
            sentiment_counts = df_filtered['Sentiment Label'].value_counts()
            
            fig, ax = plt.subplots()
            ax.pie(sentiment_counts, labels=sentiment_counts.index, autopct='%1.1f%%')
            ax.set_title(f'Sentiment Details for {stock_symbol} ({company_name})')
            st.pyplot(fig)
    else:
        st.write(f"No stock data available for the input: {user_input}")
