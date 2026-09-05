import streamlit as st
import pandas as pd
import numpy as np
import os
import yfinance as yf

from src.config import INITIAL_CAPITAL, get_data_paths
from src.model import generate_next_quarter_prediction
from src.data_loader import get_current_stock_prices
import altair as alt

# --- Caching logic ---
@st.cache_data
def load_predictions(path, mtime):
    if os.path.exists(path):
        return pd.read_excel(path)
    return pd.DataFrame()

@st.cache_data(ttl=3600)
def get_ticker_chart_data(ticker):
    data = yf.download(ticker, period="6mo", progress=False)
    return data['Close'] if not data.empty else None

st.set_page_config(
    page_title="Strategic Recommendations",
    page_icon="🎯",
    layout="wide"
)

# Custom CSS for Premium Look
st.markdown("""
    <style>
    .main {
        background-color: #0e1117;
    }
    .stMetric {
        background-color: #1e2130;
        padding: 15px;
        border-radius: 10px;
        border-left: 5px solid #4CAF50;
    }
    [data-testid="stMetricValue"] > div {
        color: white !important;
    }
    [data-testid="stMetricLabel"], [data-testid="stMetricLabel"] *, .st-emotion-cache-16idsys p {
        color: white !important;
    }
    .buy-header {
        color: #4CAF50;
        font-weight: bold;
    }
    .hold-header {
        color: #757575;
        font-weight: bold;
    }
    </style>
    """, unsafe_allow_html=True)

st.title("🎯 Strategic Recommendations")
st.write("Forward-looking AI forecasts for the next fiscal quarter based on fundamental analysis and executive sentiment.")

st.sidebar.header("Strategy Selection")
strategy_mode = st.sidebar.radio("Strategy Mode", ["Quarterly Fundamental", "Short-Term Alpha (5-Day + TA)"])
is_alpha = "Alpha" in strategy_mode

st.sidebar.header("Model Configuration")
nlp_source = st.sidebar.radio("NLP Intelligence Source", ["Local FinBERT (Leak-Free)", "OpenAI GPT-4o-mini"])
nlp_method = "finbert" if "FinBERT" in nlp_source else "openai"

use_triplet_input = False
use_optuna_input = st.sidebar.checkbox("Use Optuna Auto-Tuning (Slower)", value=False)
if nlp_method == "finbert":
    use_triplet_input = st.sidebar.checkbox("Use Triplet Features (Pos/Neg/Neu)", value=False)

st.sidebar.markdown("---")
st.sidebar.header("Execution")

# Get dynamic paths
paths = get_data_paths(nlp_method)
FEATURES_PATH = paths["FEATURES_PATH"]
LATEST_PREDICTIONS_PATH = paths["LATEST_PREDICTIONS_PATH"]

if is_alpha:
    def get_alpha_path(p):
        base, ext = os.path.splitext(p)
        return f"{base}_alpha{ext}"
    FEATURES_PATH = get_alpha_path(FEATURES_PATH)
    LATEST_PREDICTIONS_PATH = get_alpha_path(LATEST_PREDICTIONS_PATH)

# --- Helper functions ---
def run_prediction_engine():
    if not os.path.exists(FEATURES_PATH):
        st.error(f"Features file not found. Please run the {nlp_source} Data Pipeline first.")
        return
    
    with st.spinner(f"AI Engine crunching numbers using {nlp_source}..."):
        features_df = pd.read_excel(FEATURES_PATH)
        generate_next_quarter_prediction(
            features_df, 
            output_path=LATEST_PREDICTIONS_PATH,
            lookahead_days=(5 if is_alpha else 90),
            use_optuna=use_optuna_input,
            use_triplet=use_triplet_input
        )
        st.success("New recommendations generated!")
        st.rerun()

if st.sidebar.button("Generate Fresh Forecasts", width="stretch"):
    run_prediction_engine()

# --- Load and Display Recommendations ---
# Initialize preds_df to avoid NameError if file doesn't exist
preds_df = pd.DataFrame()

# Load predictions with caching
mtime = os.path.getmtime(LATEST_PREDICTIONS_PATH) if os.path.exists(LATEST_PREDICTIONS_PATH) else 0
preds_df = load_predictions(LATEST_PREDICTIONS_PATH, mtime)

if preds_df.empty:
    st.info("No active forecasts found for this model. Click 'Generate Fresh Forecasts' in the sidebar.")
else:
    # Sort and filter
    preds_df = preds_df.sort_values('predicted_return', ascending=False)
    
    # UI Columns for Summary
    col1, col2, col3 = st.columns(3)
    top_pick = preds_df.iloc[0]
    buy_count = len(preds_df[preds_df['predicted_return'] > 0])
    
    with col1:
        st.metric("Top Conviction Pick", top_pick['ticker'])
    with col2:
        st.metric("Total Tickers Analyzed", len(preds_df))
    with col3:
        st.metric("Positive Opportunities", buy_count)

    st.markdown("---")

    # Split into Buy and Hold lists
    buys_df = preds_df[preds_df['predicted_return'] > 0].copy()
    holds_df = preds_df[preds_df['predicted_return'] <= 0].copy()

    tab1, tab2 = st.tabs(["🚀 Top Buy Recommendations", "⚖️ Hold / Neutral"])

    with tab1:
        if buys_df.empty:
            st.info("No tickers meet the 'Positive Return' criteria at this moment.")
        else:
            target_label = "Predicted 5-Day Return" if is_alpha else "Predicted Qt Forecast"
            st.dataframe(
                buys_df[['ticker', 'predicted_return']].rename(columns={'predicted_return': target_label}).style.format({target_label: '{:.2%}'}),
                width='stretch'
            )
            
            st.markdown("### 🔍 Deep Dive Analysis")
            selected_tickers = st.multiselect("Select Tickers for Detailed Insight", options=buys_df['ticker'].tolist())
            
            for t in selected_tickers:
                row = buys_df[buys_df['ticker'] == t].iloc[0]
                with st.container():
                    st.write(f"#### {t}")
                    c1, c2 = st.columns([1, 2])
                    with c1:
                        st.write(f"**Ticker:** {t}")
                        st.write(f"**Filing Date:** {row['filing_date']}")
                        conf_label = "Short-Term Forecast (5-Day)" if is_alpha else "AI Confidence (Quarterly)"
                        st.write(f"**{conf_label}:** Positive ({row['predicted_return']:.2%})")
                    with c2:
                        hist_data = get_ticker_chart_data(t)
                        if hist_data is not None:
                            st.line_chart(hist_data, height=200)
                        else:
                            st.write("Historical chart unavailable.")
                    st.markdown("---")

    with tab2:
        if holds_df.empty:
            st.info("All analyzed tickers are currently recommended as Buys.")
        else:
            st.dataframe(
                holds_df[['ticker', 'predicted_return']].style.format({'predicted_return': '{:.2%}'}),
                width='stretch'
            )

# --- Virtual Portfolio Section ---
st.markdown("---")
st.header("💼 Virtual Portfolio Manager")

if 'portfolio' not in st.session_state:
    st.session_state.portfolio = {'cash': INITIAL_CAPITAL, 'holdings': {}}

# Display current status in nice metrics
p_col1, p_col2, p_col3 = st.columns(3)

# Calculate total value
total_holdings_value = sum(st.session_state.portfolio['holdings'].values())
total_portfolio_value = st.session_state.portfolio['cash'] + total_holdings_value

with p_col1:
    new_cash = st.number_input("Adjust Portfolio Cash ($)", value=float(st.session_state.portfolio['cash']), min_value=0.0, step=100.0)
    if new_cash != st.session_state.portfolio['cash']:
        st.session_state.portfolio['cash'] = new_cash
        st.rerun()

with p_col2:
    st.metric("Total Portfolio Value", f"${total_portfolio_value:,.2f}")

with p_col3:
    holdings_count = len(st.session_state.portfolio['holdings'])
    st.metric("Active Positions", holdings_count)

# Display Detailed Holdings
if st.session_state.portfolio['holdings']:
    st.write("### Portfolio Composition")
    holdings_data = []
    for ticker, val in st.session_state.portfolio['holdings'].items():
        weight = (val / total_portfolio_value) if total_portfolio_value > 0 else 0
        holdings_data.append({
            "Ticker": ticker,
            "Allocation ($)": val,
            "Weight (%)": weight * 100
        })
    
    holdings_df = pd.DataFrame(holdings_data)
    
    # 2-column layout for visualisation + table
    v_col1, v_col2 = st.columns([1, 1])
    
    with v_col1:
        # Altair Pie Chart
        pie_chart = alt.Chart(holdings_df).mark_arc(innerRadius=50).encode(
            theta=alt.Theta(field="Weight (%)", type="quantitative"),
            color=alt.Color(field="Ticker", type="nominal", scale=alt.Scale(scheme='category20')),
            tooltip=['Ticker', 'Allocation ($)', 'Weight (%)']
        ).properties(width='container', height=300)
        st.altair_chart(pie_chart, width='stretch')
        
    with v_col2:
        st.dataframe(
            holdings_df.style.format({
                "Allocation ($)": "${:,.2f}",
                "Weight (%)": "{:.1f}%"
            }),
            width="stretch",
            hide_index=True
        )

# Rebalance logic
if st.button("Apply AI Rebalance to Portfolio", width="stretch"):
    if not preds_df.empty:
        buys = preds_df[preds_df['predicted_return'] > 0]
        
        if buys.empty:
            st.warning("No Buy recommendations available. Moving all portfolio to Cash.")
            st.session_state.portfolio['cash'] = total_portfolio_value
            st.session_state.portfolio['holdings'] = {}
        else:
            total_pred = buys['predicted_return'].sum()
            # Reallocate the ENTIRE portfolio value (cash + old holdings)
            # This follows the user's requirement to reallocate cash into active positions.
            st.session_state.portfolio['holdings'] = {}
            for _, row in buys.iterrows():
                weight = row['predicted_return'] / total_pred
                allocated_cash = total_portfolio_value * weight
                st.session_state.portfolio['holdings'][row['ticker']] = allocated_cash
            st.session_state.portfolio['cash'] = 0
            st.success("Entire portfolio rebalanced based on latest AI weights!")
            st.rerun()

if st.button("Reset Portfolio", type="secondary"):
    st.session_state.portfolio = {'cash': INITIAL_CAPITAL, 'holdings': {}}
    st.rerun()