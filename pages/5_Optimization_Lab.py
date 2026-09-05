import streamlit as st
import pandas as pd
import numpy as np
import os
import yfinance as yf
from datetime import timedelta
import plotly.graph_objects as go

from src.config import INITIAL_CAPITAL, get_data_paths
from src.backtester import simulate_portfolio

st.set_page_config(
    page_title="Optimization Lab",
    page_icon="⚖️",
    layout="wide"
)

st.title("⚖️ Optimization Lab")
st.write("This page helps you find the optimal number of stocks to include in your portfolio. It runs multiple simulations with different 'Top N' settings and plots the total return.")

# --- Sidebar ---
st.sidebar.header("Optimization Settings")
nlp_source = st.sidebar.radio("Select NLP Pipeline Data", ["Local FinBERT (Leak-Free)", "OpenAI GPT-4o-mini"])
nlp_method = "finbert" if "FinBERT" in nlp_source else "openai"

strategy_mode = st.sidebar.radio("Strategy Mode", ["Quarterly Fundamental", "Short-Term Alpha (5-Day + TA)"])
is_alpha = "Alpha" in strategy_mode

start_year = st.sidebar.number_input("Backtest Start Year", min_value=2020, max_value=2025, value=2021)
max_n_input = st.sidebar.slider("Max Stocks to Test", min_value=10, max_value=200, value=50, step=10)

# Get paths
paths = get_data_paths(nlp_method)
FEATURES_PATH = paths["FEATURES_PATH"]
PREDICTIONS_PATH = paths["PREDICTIONS_PATH"]

if is_alpha:
    base, ext = os.path.splitext(PREDICTIONS_PATH)
    PREDICTIONS_PATH = f"{base}_alpha{ext}"

# --- Optimization Logic ---
if st.button(f"Run Portfolio Optimization (1 - {max_n_input} Stocks)", width='stretch'):
    if not os.path.exists(PREDICTIONS_PATH):
        st.error(f"Predictions file not found at {PREDICTIONS_PATH}. Please run the model in Strategy Lab first.")
    else:
        with st.spinner(f"Preparing market data and running {max_n_input} simulations..."):
            preds_df = pd.read_excel(PREDICTIONS_PATH)
            preds_df['filing_date'] = pd.to_datetime(preds_df['filing_date'])
            
            # 1. Pre-fetch Market Data to speed up the loop
            all_tickers = preds_df['ticker'].unique()
            
            # Ensure we don't try to test more stocks than we have in the universe
            actual_max_n = min(max_n_input, len(all_tickers))
            if actual_max_n < max_n_input:
                st.warning(f"Universe only contains {len(all_tickers)} tickers. Testing up to {actual_max_n}.")
            
            # Determine date range (similar to backtester.py)
            lookahead = 5 if is_alpha else 90
            min_date = preds_df[preds_df['filing_date'].dt.year >= start_year]['filing_date'].min()
            max_date = preds_df['filing_date'].max()
            
            start_market_date = (min_date - timedelta(days=5)).strftime('%Y-%m-%d')
            end_market_date = (max_date + timedelta(days=lookahead + 30)).strftime('%Y-%m-%d')
            
            st.write(f"Fetching market data for {len(all_tickers)} tickers...")
            market_prices = yf.download(all_tickers.tolist(), start=start_market_date, end=end_market_date, progress=False)
            
            if isinstance(market_prices.columns, pd.MultiIndex):
                daily_closes = market_prices['Close']
            else:
                daily_closes = pd.DataFrame({all_tickers[0]: market_prices['Close']})
            daily_closes.index = pd.to_datetime(daily_closes.index).tz_localize(None)
            daily_closes = daily_closes[~daily_closes.index.duplicated(keep='first')]

            # 2. Run Simulations
            results = []
            progress_bar = st.progress(0)
            
            # Temporary path for intermediate results
            temp_results_path = "data/fetched/temp_opt_results.xlsx"
            
            # For the Naive benchmark, we need the features to get Revenue
            features_df = pd.read_excel(FEATURES_PATH)
            # Ensure revenue exists in predictions for simpler logic, or merge them
            # Usually predictions only have ticker/date/predicted_return. 
            # We'll merge with features to get the latest revenue.
            merged_df = pd.merge(preds_df, features_df[['ticker', 'filing_date', 'revenue']], on=['ticker', 'filing_date'], how='left')

            for n in range(1, actual_max_n + 1):
                # A. AI Strategy
                res_ai = simulate_portfolio(
                    preds_df.copy(), 
                    output_path=temp_results_path,
                    initial_capital=INITIAL_CAPITAL,
                    start_year=start_year,
                    frequency=('D' if is_alpha else 'Q'),
                    top_n=n,
                    market_data=daily_closes,
                    ranking_col='predicted_return',
                    filter_positive=True
                )
                
                # B. Naive Benchmark (Top N by Revenue)
                res_naive = simulate_portfolio(
                    merged_df.copy(), 
                    output_path=temp_results_path,
                    initial_capital=INITIAL_CAPITAL,
                    start_year=start_year,
                    frequency=('D' if is_alpha else 'Q'),
                    top_n=n,
                    market_data=daily_closes,
                    ranking_col='revenue',
                    filter_positive=False # Just pick the biggest, even if return was negative
                )
                
                ret_ai = (res_ai['portfolio_value'].iloc[-1] / INITIAL_CAPITAL) - 1 if not res_ai.empty else 0
                ret_naive = (res_naive['portfolio_value'].iloc[-1] / INITIAL_CAPITAL) - 1 if not res_naive.empty else 0
                
                results.append({
                    "top_n": n, 
                    "Strategy Return": ret_ai,
                    "Naive Benchmark (Revenue)": ret_naive
                })
                
                progress_bar.progress(n / actual_max_n)

            # 3. Plotting
            opt_df = pd.DataFrame(results)
            
            st.header("Optimization Results")
            
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=opt_df['top_n'], 
                y=opt_df['Strategy Return'],
                mode='lines+markers',
                name='Our AI Strategy',
                line=dict(color='#4CAF50', width=3),
                marker=dict(size=8)
            ))

            fig.add_trace(go.Scatter(
                x=opt_df['top_n'], 
                y=opt_df['Naive Benchmark (Revenue)'],
                mode='lines+markers',
                name='Naive Benchmark (Top N by Revenue)',
                line=dict(color='#757575', width=2, dash='dash'),
                marker=dict(size=6)
            ))
            
            fig.update_layout(
                title=f"Total Return vs. Number of Stocks ({nlp_source})",
                xaxis_title="Top N Stocks Included",
                yaxis_title="Total Return (%)",
                template="plotly_dark",
                hovermode="x unified"
            )
            fig.update_yaxes(tickformat=".2%")
            
            st.plotly_chart(fig, width='stretch')
            
            # Show Raw Data
            st.subheader("Raw Results")
            st.dataframe(opt_df.style.format({
                'Strategy Return': '{:.2%}',
                'Naive Benchmark (Revenue)': '{:.2%}'
            }), width='stretch')

            # Clean up
            if os.path.exists(temp_results_path):
                os.remove(temp_results_path)

st.info("Note: This page uses the existing predictions generated in the Strategy Lab. If you change model parameters (like Max Depth), run a simulation in the Strategy Lab first to update the predictions.")
