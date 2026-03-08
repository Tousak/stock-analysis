import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import os

# Import pipeline functions and config
from src.config import INITIAL_CAPITAL, get_data_paths
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
start_year_input = st.sidebar.number_input("Backtest Start Year", min_value=2020, max_value=2025, value=2021, step=1)

st.sidebar.header("XGBoost Settings")
use_optuna_input = st.sidebar.checkbox("Use Optuna Auto-Tuning (Slower)", value=False)
max_depth_input = st.sidebar.number_input("Max Depth", min_value=1, max_value=10, value=3, step=1, disabled=use_optuna_input)
learning_rate_input = st.sidebar.number_input("Learning Rate", min_value=0.01, max_value=0.5, value=0.05, step=0.01, disabled=use_optuna_input)
n_estimators_input = st.sidebar.number_input("Number of Estimators", min_value=10, max_value=500, value=100, step=10, disabled=use_optuna_input)
use_triplet_input = st.sidebar.checkbox("Use Triplet Features (Pos/Neg/Neu)", value=False, help="Uses raw FinBERT probabilities instead of a single scalar score. Best for larger datasets.")

st.sidebar.header("Data Source")
nlp_source = st.sidebar.radio("Select NLP Pipeline Data", ["Local FinBERT (Leak-Free)", "OpenAI GPT-4o-mini"])
# Determine method string to get correct paths
nlp_method = "finbert" if "FinBERT" in nlp_source else "openai"
paths = get_data_paths(nlp_method)
FEATURES_PATH = paths["FEATURES_PATH"]
PREDICTIONS_PATH = paths["PREDICTIONS_PATH"]
BACKTEST_RESULTS_PATH = paths["BACKTEST_RESULTS_PATH"]

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
            run_walk_forward_predictions(features_df, output_path=PREDICTIONS_PATH, start_year=start_year_input, 
                                         n_estimators=n_estimators_input, max_depth=max_depth_input,
                                         learning_rate=learning_rate_input, use_optuna=use_optuna_input,
                                         use_triplet=use_triplet_input)
            
            # Run backtest with UI parameters
            predictions_df = pd.read_excel(PREDICTIONS_PATH)
            simulate_portfolio(predictions_df, output_path=BACKTEST_RESULTS_PATH, initial_capital=initial_capital_input, start_year=start_year_input)
        
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
