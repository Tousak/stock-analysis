import streamlit as st
import pandas as pd
import os
from datetime import datetime

# Import paths from the backend config
from src.config import BACKTEST_RESULTS_PATH, PREDICTIONS_PATH, RAW_FILINGS_PATH, PROCESSED_FILINGS_PATH, FEATURES_PATH

st.set_page_config(
    page_title="Stock Analysis Dashboard",
    page_icon="🏠",
    layout="wide"
)

st.title("🏠 Home Dashboard")
st.write("A high-level executive summary of the latest backtest run.")

# --- 1. Load Data ---
# We load the final results of the backtest and the predictions that drove it.
try:
    results_df = pd.read_excel(BACKTEST_RESULTS_PATH)
    predictions_df = pd.read_excel(PREDICTIONS_PATH)
    results_exist = True
except FileNotFoundError:
    st.error("Backtest results not found. Please run the 'Strategy Lab' page to generate results.")
    results_exist = False

if results_exist:
    # --- 2. Display KPIs ---
    st.header("Latest Portfolio Performance")
    
    # Get the most recent portfolio value
    final_value = results_df['portfolio_value'].iloc[-1]
    initial_capital = results_df['portfolio_value'].iloc[0]
    total_return_pct = ((final_value / initial_capital) - 1) * 100

    col1, col2, col3 = st.columns(3)
    col1.metric("Final Portfolio Value", f"${final_value:,.2f}")
    col2.metric("Initial Capital", f"${initial_capital:,.2f}")
    col3.metric("Total Return", f"{total_return_pct:.2f}%")

    # --- 3. Show Latest Picks ---
    st.header("Latest Strategy Decision")
    latest_decision = results_df.iloc[-1]
    latest_quarter = pd.to_datetime(latest_decision['date']).to_period('Q')
    
    st.subheader(f"Top Picks for {latest_quarter}:")
    st.info(f"**Selection:** {latest_decision['selection']}")
    st.write(f"Based on these picks, the portfolio's return for this quarter was **{latest_decision['quarterly_return']:.2%}**.")

    # --- 4. Display Equity Curve ---
    st.header("Equity Curve")
    equity_curve_df = results_df.set_index('date')[['portfolio_value']]
    st.line_chart(equity_curve_df)


# --- 5. Display System Status ---
st.header("System Status")
st.write("Last update times for key data files.")

status_data = []
for path, name in [
    (RAW_FILINGS_PATH, "Raw SEC Filings"),
    (PROCESSED_FILINGS_PATH, "Sentiment Analysis Cache"),
    (FEATURES_PATH, "Engineered Features"),
    (PREDICTIONS_PATH, "Model Predictions"),
    (BACKTEST_RESULTS_PATH, "Backtest Results")
]:
    if os.path.exists(path):
        last_modified_time = datetime.fromtimestamp(os.path.getmtime(path))
        status_data.append({"File": name, "Last Updated": last_modified_time.strftime('%Y-%m-%d %H:%M:%S')})
    else:
        status_data.append({"File": name, "Last Updated": "Not found"})

st.dataframe(pd.DataFrame(status_data), use_container_width=True)

st.info("To re-run the pipeline or backtest with new parameters, navigate to the other pages using the sidebar.")
