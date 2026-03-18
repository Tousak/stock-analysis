import streamlit as st
import pandas as pd
import os

# Import paths from the backend config
from src.config import FEATURES_PATH, get_data_paths

st.set_page_config(
    page_title="Filing Inspector",
    page_icon="🔍",
    layout="wide"
)

# Use the finbert paths for inspection as it's the most complete
paths = get_data_paths("finbert")
PROCESSED_FILINGS_PATH = paths["PROCESSED_FILINGS_PATH"]

st.title("🔍 Filing Inspector")
st.write("A tool to 'trust but verify' the data for any specific filing. Use this to understand why the model might have made a certain prediction.")

# --- Load Data ---
if not os.path.exists(FEATURES_PATH) or not os.path.exists(PROCESSED_FILINGS_PATH):
    st.error("Data files not found. Please run the Data Pipeline first.")
    st.stop()

features_df = pd.read_excel(FEATURES_PATH)
processed_df = pd.read_excel(PROCESSED_FILINGS_PATH)
st.write(PROCESSED_FILINGS_PATH)
# Ensure date types match before merging
features_df['filing_date'] = pd.to_datetime(features_df['filing_date'])
processed_df['filing_date'] = pd.to_datetime(processed_df['filing_date'])

# Select the text and accession from processed_df
text_df = processed_df[['ticker', 'filing_date', 'mda_text']]

# Merge on ticker and date
df = pd.merge(features_df, text_df, on=['ticker', 'filing_date'], how='left')


# --- UI Selectors ---
st.header("Select a Ticker")

tickers = df['ticker'].unique()
selected_ticker = st.selectbox("1. Filter by Ticker", tickers)

# Filter by ticker to show relevant filing dates and plot data
ticker_df = df[df['ticker'] == selected_ticker].sort_values('filing_date', ascending=True)

# --- NEW: Time-Series Plotting ---
st.subheader(f"📈 {selected_ticker} Trends Over Time")

# Prepare data for plotting
plot_df = ticker_df.set_index('filing_date')[['sentiment_score', 'revenue_growth', 'net_margin']]
# Clean up columns for better legend labels
plot_df.columns = ['Sentiment Score', 'Revenue Growth', 'Net Margin']

# Display the chart
st.line_chart(plot_df)

st.divider()

# --- Specific Filing Selection ---
st.header("Inspect Specific Filing")

# Format date for display (reverse sort for selection so newest is first)
select_df = ticker_df.sort_values('filing_date', ascending=False).copy()
select_df['display_date'] = select_df['filing_date'].dt.strftime('%Y-%m-%d')
selected_date_str = st.selectbox("2. Select Filing Date to View Text", select_df['display_date'])

# Get the full row of data for the selected filing
selected_row = select_df[select_df['display_date'] == selected_date_str].iloc[0]

# --- Display Metrics ---
st.header(f"Details for {selected_ticker} on {selected_date_str}")

col1, col2, col3, col4 = st.columns(4)
col1.metric("Sentiment Score", f"{selected_row['sentiment_score']:.2f}")
col2.metric("Sentiment Change", f"{selected_row['sentiment_change']:.2f}" if pd.notna(selected_row['sentiment_change']) else "N/A")
col3.metric("Revenue Growth", f"{selected_row['revenue_growth']:.2%}" if pd.notna(selected_row['revenue_growth']) else "N/A")
col4.metric("Actual Next Q Return", f"{selected_row['next_quarter_return']:.2%}" if pd.notna(selected_row['next_quarter_return']) else "N/A")

# --- Display Raw Text ---
with st.expander("📄 View Raw MD&A Text", expanded=True):
    mda_text = selected_row['mda_text']
    if pd.notna(mda_text) and mda_text:
        # Use a container with a fixed height or just the text
        st.text_area("MD&A Content", value=mda_text, height=600, disabled=True)
    else:
        st.warning("No MD&A text was found for this filing.")

