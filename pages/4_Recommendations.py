import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import os

from src.config import FEATURES_PATH, PREDICTIONS_PATH, INITIAL_CAPITAL, LATEST_PREDICTIONS_PATH
from src.data_loader import get_current_stock_prices # Now correctly imported

st.set_page_config(
    page_title="Next Quarter Recommendations",
    page_icon="📈",
    layout="wide"
)

st.title("📈 Next Quarter Recommendations")
st.write("Get predictions for the upcoming quarter and manage your virtual portfolio based on our strategy.")

# --- Helper function to get current quarter's predictions ---
@st.cache_data
def get_latest_predictions():
    if not os.path.exists(LATEST_PREDICTIONS_PATH):
        st.error(f"Latest predictions file not found at {LATEST_PREDICTIONS_PATH}. Please run the Data Pipeline and --predict-latest first.")
        return pd.DataFrame(), pd.DataFrame() # Return two empty DataFrames
    
    predictions_df = pd.read_excel(LATEST_PREDICTIONS_PATH)
    predictions_df['filing_date'] = pd.to_datetime(predictions_df['filing_date'])
    predictions_df['predicted_return'] = predictions_df['predicted_return'].fillna(0) # Fill NaN predictions with 0

    # Sort by filing date to get the most recent predictions
    predictions_df = predictions_df.sort_values('filing_date', ascending=False)
    
    # Get the latest quarter for which we have predictions
    latest_quarter_filing_date = predictions_df['filing_date'].max()
    
    # Filter predictions for this latest quarter
    all_latest_predictions = predictions_df[predictions_df['filing_date'] == latest_quarter_filing_date].copy()
    
    # Filter for positive predicted returns only (for rebalancing logic)
    positive_recs = all_latest_predictions[all_latest_predictions['predicted_return'] > 0.0].copy()
    
    if all_latest_predictions.empty:
        st.warning("No predictions found for the latest quarter.")
        return pd.DataFrame(), pd.DataFrame()

    return all_latest_predictions[['ticker', 'predicted_return']].set_index('ticker'), \
           positive_recs[['ticker', 'predicted_return']].set_index('ticker')

# --- Fetch Current Stock Prices ---
@st.cache_data
def fetch_current_prices(tickers):
    if not tickers:
        return pd.Series(dtype='float64')
    # Use the function from data_loader
    return get_current_stock_prices(tickers)


# --- Initialize Session State for Virtual Portfolio ---
if 'portfolio' not in st.session_state:
    st.session_state.portfolio = {
        'cash': INITIAL_CAPITAL,
        'holdings': {} # ticker: shares
    }
    # Flag to prevent adding initial_cash_input on every rerun
    st.session_state.initial_portfolio_inputs_processed = False


def update_portfolio_display():
    st.subheader("Your Virtual Portfolio")
    col_holdings, col_cash = st.columns([0.7, 0.3])
    
    with col_holdings:
        if st.session_state.portfolio['holdings']:
            holdings_display = {
                ticker: f"{shares:.2f} shares" 
                for ticker, shares in st.session_state.portfolio['holdings'].items()
            }
            st.json(holdings_display)
        else:
            st.write("No stock holdings.")
    with col_cash:
        st.metric("Cash", f"${st.session_state.portfolio['cash']:,.2f}")

# --- Portfolio Management Logic ---
def recalculate_portfolio(recommendations_df, current_holding_inputs, additional_cash_input):
    
    # First, update the portfolio session state with current user inputs before rebalancing
    st.session_state.portfolio['holdings'] = {} # Clear current holdings
    for ticker, shares in current_holding_inputs.items():
        if shares > 0:
            st.session_state.portfolio['holdings'][ticker] = shares

    st.session_state.portfolio['cash'] += additional_cash_input # Add new cash for this rebalance

    if recommendations_df.empty:
        st.warning("Cannot recalculate: No positive recommendations available for rebalancing.")
        return

    # 1. Calculate total current value of portfolio (cash + holdings)
    total_available_capital = st.session_state.portfolio['cash']
    holdings_tickers = list(st.session_state.portfolio['holdings'].keys())
    
    all_relevant_tickers = list(set(holdings_tickers + recommendations_df.index.tolist()))
    current_prices = fetch_current_prices(all_relevant_tickers) # Fetch prices for all relevant tickers

    current_holdings_value = 0
    for ticker, shares in st.session_state.portfolio['holdings'].items():
        if ticker in current_prices and not pd.isna(current_prices[ticker]):
            current_holdings_value += shares * current_prices[ticker]
        else:
            st.warning(f"Could not fetch current price for {ticker}. Assuming current value is 0 for recalculation.")
    
    total_available_capital += current_holdings_value # Add value of existing holdings

    if total_available_capital == 0:
        st.error("Total portfolio value is zero. Cannot rebalance.")
        return

    # 2. Distribute total value based on predicted returns
    total_predicted_return_sum = recommendations_df['predicted_return'].sum()
    if total_predicted_return_sum == 0:
        st.info("No positive predicted returns for allocation. Keeping all capital in cash.")
        st.session_state.portfolio['cash'] = total_available_capital
        st.session_state.portfolio['holdings'] = {}
        st.success("Portfolio recalculated based on recommendations.")
        st.rerun()
        return

    # Calculate target allocation (value) for each recommended stock
    target_allocations = {}
    for ticker, row in recommendations_df.iterrows():
        weight = row['predicted_return'] / total_predicted_return_sum
        target_allocations[ticker] = total_available_capital * weight
    
    # 3. Determine trades needed and update holdings
    new_holdings_shares = {}
    st.markdown("### Recommended Trades:")
    
    # Initialize cash to total available capital, then subtract target allocations
    remaining_cash = total_available_capital

    for ticker in recommendations_df.index.tolist(): # Iterate through recommended tickers (only positive ones)
        current_shares = st.session_state.portfolio['holdings'].get(ticker, 0)
        target_value = target_allocations.get(ticker, 0)
        
        current_price = current_prices.get(ticker)
        
        if pd.isna(current_price) or current_price <= 0:
            if target_value > 0:
                st.warning(f"Cannot buy/sell {ticker} (no valid current price). Skipping recommendation for this stock.")
            new_holdings_shares[ticker] = current_shares # Keep current shares if no price
            # We don't deduct target_value from remaining_cash here, it wasn't allocatable
            continue

        target_shares = target_value / current_price if current_price > 0 else 0
        shares_to_trade = target_shares - current_shares
        
        if abs(shares_to_trade * current_price) > 0.01: # Only show significant trades
            if shares_to_trade > 0:
                st.write(f"Buy {shares_to_trade:.2f} shares of {ticker} (Value: ${shares_to_trade * current_price:,.2f})")
            else:
                st.write(f"Sell {-shares_to_trade:.2f} shares of {ticker} (Value: ${-shares_to_trade * current_price:,.2f})")
        
        new_holdings_shares[ticker] = target_shares
        remaining_cash -= target_value # Deduct allocated value from cash
    
    # For any existing holdings not in recommendations, sell them off
    for ticker, shares in st.session_state.portfolio['holdings'].items():
        if ticker not in recommendations_df.index and shares > 0: # Check if holding is not in current positive recommendations
            current_price = current_prices.get(ticker)
            if pd.isna(current_price) or current_price <= 0:
                st.warning(f"Cannot sell {ticker} (no valid current price). Holding existing shares.")
                new_holdings_shares[ticker] = shares
            else:
                st.write(f"Sell {shares:.2f} shares of {ticker} (Value: ${shares * current_price:,.2f})")
                remaining_cash += shares * current_price # Add proceeds to cash
                new_holdings_shares[ticker] = 0 # No longer holding
    
    st.session_state.portfolio['holdings'] = {k: v for k, v in new_holdings_shares.items() if v > 0.01} # Clean up negligible holdings
    st.session_state.portfolio['cash'] = remaining_cash # Final cash balance

    st.success("Portfolio recalculated based on recommendations.")
    st.rerun() # Rerun to update the display


def clear_portfolio():
    st.session_state.portfolio = {
        'cash': INITIAL_CAPITAL,
        'holdings': {}
    }
    st.success("Portfolio cleared to initial cash.")
    st.rerun()

# --- Display Recommendations ---
st.header("Next Quarter Stock Predictions")
all_recs_df, latest_recs_df_positive = get_latest_predictions() # Call function to get all predictions and positive ones

if not all_recs_df.empty:
    st.bar_chart(all_recs_df)
    st.write("---")
    st.subheader("Positive Predicted Returns (used for recommendations)")
    if not latest_recs_df_positive.empty:
        st.dataframe(latest_recs_df_positive.style.format({'predicted_return': '{:.2%}'}), use_container_width=True)
        recommended_tickers = latest_recs_df_positive.index.tolist()
    else:
        st.info("No stocks have positive predicted returns for rebalancing.")
        recommended_tickers = []
else:
    recommended_tickers = []

# --- Virtual Portfolio Input ---
st.header("Virtual Portfolio Management")

# Current Holdings Input
st.subheader("Current Holdings")
current_holding_inputs = {}
# Add input fields for currently recommended tickers and existing holdings
# Combine recommended tickers with current holdings tickers for inputs
all_portfolio_tickers = list(set(recommended_tickers + list(st.session_state.portfolio['holdings'].keys())))

# Ensure input fields maintain state across reruns, pre-filling from session state
for ticker in all_portfolio_tickers:
    default_shares = st.session_state.portfolio['holdings'].get(ticker, 0)
    current_holding_inputs[ticker] = st.number_input(f"Shares of {ticker}", value=float(default_shares), min_value=0.0, format="%.2f", key=f"holdings_input_{ticker}")

# Additional Cash Input
additional_cash_input = st.number_input("Additional Cash to Add for Rebalance (USD)", value=0.0, min_value=0.0, format="%.2f")


update_portfolio_display()

# --- Update Predictions Button ---
def run_full_pipeline_and_predict():
    st.session_state.pipeline_running = True
    with st.spinner("Running full data pipeline and generating latest predictions... This may take several minutes."):
        # Using a direct shell command call here. In a production app,
        # consider offloading long-running tasks to background processes
        # or services to avoid blocking the Streamlit UI.
        try:
            command = ".venv/Scripts/python.exe main.py --all"
            # You might want to capture output for better feedback, but for now, just run it.
            # print(f"Executing: {command}")
            import subprocess
            process = subprocess.run(command, shell=True, capture_output=True, text=True, check=True)
            st.success(f"Pipeline finished successfully! Output:\n{process.stdout}")
        except subprocess.CalledProcessError as e:
            st.error(f"Pipeline failed! Error:\n{e.stderr}")
        except Exception as e:
            st.error(f"An unexpected error occurred: {e}")
    st.session_state.pipeline_running = False
    st.rerun() # Rerun to load fresh predictions

if st.button("Update Latest Predictions (Run Full Pipeline)", use_container_width=True, disabled=st.session_state.get('pipeline_running', False)):
    run_full_pipeline_and_predict()

st.write("---") # Separator

col_buttons1, col_buttons2 = st.columns(2)
with col_buttons1:
    if st.button("Recalculate Portfolio (Based on Recommendations)", use_container_width=True):
        recalculate_portfolio(latest_recs_df_positive, current_holding_inputs, additional_cash_input)
with col_buttons2:
    if st.button("Clear Portfolio (Back to Initial Cash)", use_container_width=True):
        clear_portfolio()

# Display updated portfolio after actions
# This needs to be called again AFTER buttons to reflect changes without full rerun
# (though st.experimental_rerun handles this for the buttons above)
# update_portfolio_display() # Not needed here due to st.experimental_rerun