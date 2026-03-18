import streamlit as st
import pandas as pd
import os

# Import paths from the backend config
from src.config import FEATURES_PATH, RAW_FILINGS_PATH

st.set_page_config(
    page_title="Filing Inspector",
    page_icon="🔍",
    layout="wide"
)

st.title("🔍 Filing Inspector")
st.write("A tool to 'trust but verify' the data for any specific filing. Use this to understand why the model might have made a certain prediction.")

# --- Load Data ---
if not os.path.exists(FEATURES_PATH) or not os.path.exists(RAW_FILINGS_PATH):
    st.error("Data files not found. Please run the Data Pipeline first.")
    st.stop()

features_df = pd.read_excel(FEATURES_PATH)
raw_df = pd.read_excel(RAW_FILINGS_PATH)

# Ensure date types match before merging
features_df['filing_date'] = pd.to_datetime(features_df['filing_date'])
raw_df['filing_date'] = pd.to_datetime(raw_df['filing_date'])

# Select only the text column from raw_df to avoid duplicating other data
text_df = raw_df[['ticker', 'filing_date', 'mda_text']]

# Merge on ticker and date
df = pd.merge(features_df, text_df, on=['ticker', 'filing_date'], how='left')


# --- UI Selectors ---
st.header("Select a Filing")

tickers = df['ticker'].unique()
selected_ticker = st.selectbox("1. Select Ticker", tickers)

# Filter by ticker to show relevant filing dates
ticker_df = df[df['ticker'] == selected_ticker].sort_values('filing_date', ascending=False)

# Format date for display
ticker_df['display_date'] = ticker_df['filing_date'].dt.strftime('%Y-%m-%d')
selected_date_str = st.selectbox("2. Select Filing Date", ticker_df['display_date'])

# Get the full row of data for the selected filing
selected_row = ticker_df[ticker_df['display_date'] == selected_date_str].iloc[0]

st.divider()

# --- Display Metrics ---
st.header(f"Analysis for {selected_ticker} on {selected_date_str}")

col1, col2, col3, col4 = st.columns(4)
col1.metric("Sentiment Score", f"{selected_row['sentiment_score']:.2f}")
col2.metric("Sentiment Change", f"{selected_row['sentiment_change']:.2f}")
col3.metric("Revenue Growth", f"{selected_row['revenue_growth']:.2%}")
col4.metric("Actual Next Q Return", f"{selected_row['next_quarter_return']:.2%}")

# --- Display Raw Text ---
with st.expander("📄 View Raw MD&A Text"):
    mda_text = selected_row['mda_text']
    if pd.notna(mda_text) and mda_text:
        st.text(mda_text)
    else:
        st.warning("No MD&A text was found for this filing.")
