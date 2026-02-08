import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import os

# Import pipeline functions and config
from src.config import FEATURES_PATH, PREDICTIONS_PATH, BACKTEST_RESULTS_PATH, INITIAL_CAPITAL, TOP_N_STOCKS_TO_INVEST
from src.model import run_walk_forward_predictions
from src.backtester import simulate_portfolio

st.set_page_config(
    page_title="Strategy Lab",
    page_icon="🧪",
    layout="wide"
)

st.title("🧪 Strategy Lab")
st.write("The interactive playground for backtesting. This page reads pre-calculated feature files, so it's fast. Adjust parameters in the sidebar and run a new simulation.")

# --- Helper Functions for Metrics ---
def calculate_sharpe_ratio(returns, risk_free_rate=0.0):
    """Calculates the Sharpe Ratio."""
    # Assuming quarterly returns, annualize them
    annualized_return = returns.mean() * 4
    annualized_std = returns.std() * np.sqrt(4)
    if annualized_std == 0:
        return 0
    return (annualized_return - risk_free_rate) / annualized_std

def calculate_max_drawdown(portfolio_values):
    """Calculates the Maximum Drawdown."""
    peak = portfolio_values.expanding(min_periods=1).max()
    drawdown = (portfolio_values - peak) / peak
    return drawdown.min()

# --- Sidebar for Inputs ---
st.sidebar.header("Simulation Parameters")
initial_capital_input = st.sidebar.number_input("Initial Capital ($)", min_value=100, value=int(INITIAL_CAPITAL), step=100)
top_n_input = st.sidebar.slider("Top N Stocks to Invest In", min_value=1, max_value=10, value=TOP_N_STOCKS_TO_INVEST)
start_year_input = st.sidebar.number_input("Backtest Start Year", min_value=2020, max_value=2025, value=2021, step=1)

# --- Main Page ---
if st.sidebar.button("Run New Simulation", use_container_width=True):
    # Check if features file exists
    if not os.path.exists(FEATURES_PATH):
        st.error(f"Features file not found at {FEATURES_PATH}. Please run the Data Pipeline first.")
    else:
        with st.spinner("Training models and running backtest..."):
            # Load features
            features_df = pd.read_excel(FEATURES_PATH)
            
            # Run model training (walk-forward prediction)
            # We pass the start year to the model training as well to align data
            run_walk_forward_predictions(features_df, start_year=start_year_input)
            
            # Run backtest with UI parameters
            predictions_df = pd.read_excel(PREDICTIONS_PATH)
            simulate_portfolio(predictions_df, initial_capital=initial_capital_input, start_year=start_year_input, top_n=top_n_input)
        
        st.success("Simulation complete!")

# --- Display Results ---
st.header("Backtest Results")

if not os.path.exists(BACKTEST_RESULTS_PATH):
    st.info("No backtest results found. Run a simulation using the sidebar controls.")
else:
    results_df = pd.read_excel(BACKTEST_RESULTS_PATH)
    results_df['date'] = pd.to_datetime(results_df['date'])

    # --- 1. Fetch S&P 500 Benchmark Data ---
    start_date = results_df['date'].min()
    end_date = results_df['date'].max()
    spy_data = yf.download('^GSPC', start=start_date, end=end_date, progress=False)
    
    # Normalize benchmark to the same starting capital
    spy_normalized = (spy_data['Close'] / spy_data['Close'].iloc[0]) * initial_capital_input
    spy_normalized.name = "S&P 500 (SPY)"

    # --- 2. Plot Equity Curve vs. Benchmark ---
    st.subheader("Equity Curve")
    equity_curve_df = results_df.set_index('date')[['portfolio_value']]
    equity_curve_df.rename(columns={'portfolio_value': 'Our Strategy'}, inplace=True)
    
    # Combine strategy and benchmark for plotting
    combined_plot = pd.concat([equity_curve_df, spy_normalized], axis=1).ffill()
    st.line_chart(combined_plot)

    # --- 3. Display Metrics ---
    st.subheader("Performance Metrics")
    
    # Calculate returns for metrics
    returns = results_df['quarterly_return']
    final_value = results_df['portfolio_value'].iloc[-1]
    total_return_pct = ((final_value / initial_capital_input) - 1)
    sharpe = calculate_sharpe_ratio(returns)
    max_dd = calculate_max_drawdown(results_df['portfolio_value'])

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Return", f"{total_return_pct:.2%}")
    col2.metric("Sharpe Ratio", f"{sharpe:.2f}")
    col3.metric("Max Drawdown", f"{max_dd:.2%}")
    
    # --- 4. Display Trade Log ---
    st.subheader("Quarterly Decisions (Trade Log)")
    display_df = results_df.copy()
    display_df['quarterly_return'] = display_df['quarterly_return'].map('{:.2%}'.format)
    st.dataframe(display_df.set_index('date'), use_container_width=True)
