import streamlit as st
import pandas as pd
import io
import contextlib
import os
from datetime import datetime
from src.data_fetch.news_fetcher import NewsFetcher

st.set_page_config(
    page_title="News Activation",
    layout="wide"
)

st.title("News Activation")
st.write("Fetch and activate external news data for sentiment analysis and feature engineering.")

# --- Configuration ---
st.header("1. Configure News Fetching")
col1, col2 = st.columns(2)

with col1:
    ticker_list_str = st.text_area(
        "Enter stock tickers (comma-separated)",
        value="AAPL, MSFT, GOOGL"
    )
    tickers = [t.strip().upper() for t in ticker_list_str.split(',') if t.strip()]

with col2:
    days_to_fetch = st.slider("Days to fetch (back from today)", 1, 30, 7)

# --- Actions ---
st.header("2. Execute Fetching")

if st.button("Fetch News Articles", use_container_width=True):
    fetcher = NewsFetcher()
    log_stream = io.StringIO()
    
    with st.spinner(f"Fetching news for {', '.join(tickers)}..."):
        with contextlib.redirect_stdout(log_stream):
            # Using the fetch_news method from our new class
            articles = fetcher.fetch_news(tickers, days=days_to_fetch)
    
    if articles:
        st.success(f"Successfully fetched {len(articles)} articles!")
        df = pd.DataFrame(articles)
        
        # Display sample
        st.subheader("Data Preview")
        st.dataframe(df.head(10), use_container_width=True)
        
        # Save Option
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"news_fetch_{timestamp}.ods"
        
        if st.button("Save to ODS"):
            path = fetcher.save_to_ods(articles, filename)
            st.info(f"Data saved to `{path}`")
    else:
        st.warning("No articles found for the selected criteria. Try increasing the day range or checking your API key.")

st.sidebar.info("""
**Tip:** The news data will be stored in `data/fetched/news/`. 
You can later process these articles using the sentiment analysis pipeline.
""")
