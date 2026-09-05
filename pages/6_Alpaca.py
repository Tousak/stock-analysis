import os
from datetime import datetime, timedelta
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from dotenv import load_dotenv

from alpaca.trading.client import TradingClient
from alpaca.trading.requests import GetOrdersRequest
from alpaca.trading.enums import QueryOrderStatus

# Import our backend quantitative rebalance engine
from src.rebalance_engine import (
    SECTOR_MAP, load_master_dataset, check_and_sync_data,
    retrain_tri_horizon_ensemble_and_predict, retrain_single_model_and_predict,
    compute_portfolio_deltas, execute_reallocation_orders
)

# Page Configuration
st.set_page_config(
    page_title="Alpaca Portfolio & Execution Bridge",
    page_icon="🦙",
    layout="wide"
)

st.title("🦙 Alpaca Portfolio & Execution Bridge")
st.write("Automated execution and live monitoring connecting our institutional **Tri-Horizon Multi-Model Ensemble (5d/15d/35d)** to Alpaca.")

# Load Alpaca credentials
load_dotenv()
api_key = os.getenv("ALPACA_API_KEY") or os.getenv("APCA_API_KEY_ID")
secret_key = os.getenv("ALPACA_SECRET_KEY") or os.getenv("APCA_API_SECRET_KEY")

tab_sim, tab_real = st.tabs(["🧪 Virtual Sandbox (Paper Trading)", "💼 Real-Money Brokerage (Live)"])

with tab_sim:
    client = TradingClient(api_key, secret_key, paper=True)
    account = client.get_account()
    total_equity = float(account.portfolio_value)
    cash_balance = float(account.cash)
    buying_power = float(account.buying_power)
    invested_capital = total_equity - cash_balance

    # --- 1. Key Metrics Bar ---
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Portfolio Value", f"${total_equity:,.2f}")
    c2.metric("Cash Balance", f"${cash_balance:,.2f}")
    c3.metric("Invested Capital", f"${invested_capital:,.2f}")
    c4.metric("Buying Power", f"${buying_power:,.2f}")

    st.divider()

    # --- 2. Live Positions & Open Orders Fetching ---
    positions = client.get_all_positions()
    orders = client.get_orders(GetOrdersRequest(status=QueryOrderStatus.OPEN, limit=500))

    # Build Visualization DataFrame
    if positions:
        viz_records = [{
            'Symbol': p.symbol,
            'Sector': SECTOR_MAP.get(p.symbol, 'Other / Diversified'),
            'Dollar Value': float(p.market_value),
            'Return Pct': float(p.unrealized_plpc) * 100.0,
            'Shares': float(p.qty),
            'Price': float(p.current_price)
        } for p in positions]
    elif orders:
        viz_records = [{
            'Symbol': o.symbol,
            'Sector': SECTOR_MAP.get(o.symbol, 'Other / Diversified'),
            'Dollar Value': float(o.notional) if o.notional else 0.0,
            'Return Pct': 0.0,
            'Shares': 0.0,
            'Price': 0.0
        } for o in orders]
    else:
        viz_records = [{
            'Symbol': 'Cash Buffer',
            'Sector': 'Cash',
            'Dollar Value': cash_balance,
            'Return Pct': 0.0,
            'Shares': 1.0,
            'Price': cash_balance
        }]

    df_viz = pd.DataFrame(viz_records)

    # --- 3. Institutional Portfolio Treemap & Top Holdings Breakdown ---
    st.subheader(f"🗺️ Current Portfolio Treemap & Sector Allocation ({len(df_viz)} Positions / Orders)")
    col_treemap, col_top_bars = st.columns([3, 2])

    with col_treemap:
        fig_treemap = px.treemap(
            df_viz,
            path=['Sector', 'Symbol'],
            values='Dollar Value',
            color='Dollar Value',
            color_continuous_scale='Tealgrn',
            title="<b>Current Sector & Stock Asset Allocation Matrix</b>"
        )
        fig_treemap.update_layout(
            template="plotly_dark",
            height=400,
            margin=dict(l=10, r=10, t=40, b=10)
        )
        fig_treemap.update_traces(
            textinfo="label+value+percent parent",
            hovertemplate="<b>%{label}</b><br>Sector: %{parent}<br>Allocation: $%{value:,.2f}<extra></extra>"
        )
        st.plotly_chart(fig_treemap, width="stretch")

    with col_top_bars:
        df_top12 = df_viz.sort_values('Dollar Value', ascending=True).tail(12)
        fig_bars = px.bar(
            df_top12,
            x='Dollar Value',
            y='Symbol',
            orientation='h',
            color='Sector',
            title="<b>Current Top Holdings ($ Allocated)</b>",
            text_auto='.2s'
        )
        fig_bars.update_layout(
            template="plotly_dark",
            height=400,
            margin=dict(l=10, r=10, t=40, b=10),
            showlegend=False,
            xaxis_title="<b>Dollars ($)</b>",
            yaxis_title=""
        )
        st.plotly_chart(fig_bars, width="stretch")

    # --- 4. Positions & Waiting Orders Tables ---
    c_tab1, c_tab2 = st.tabs([f"📋 Open / Waiting Orders Queue ({len(orders)})", f"💼 Active Portfolio Holdings ({len(positions)})"])
    
    with c_tab1:
        if orders:
            col_ord_head, col_ord_act = st.columns([4, 1])
            with col_ord_act:
                if st.button("🚫 Cancel All Open Orders", width="stretch"):
                    client.cancel_orders()
                    st.success("All open orders canceled!")
                    st.rerun()
            order_data = [{
                "Symbol": o.symbol,
                "Sector": SECTOR_MAP.get(o.symbol, "Other"),
                "Side": o.side.value.upper(),
                "Type": o.order_type.value.upper(),
                "Notional": f"${float(o.notional):,.2f}" if o.notional else f"{float(o.qty)} shs",
                "Status": o.status.value.upper(),
                "Submitted (UTC)": o.submitted_at.strftime("%Y-%m-%d %H:%M:%S")
            } for o in orders]
            st.dataframe(pd.DataFrame(order_data), height=250, width="stretch")
        else:
            st.info("No open or waiting orders in queue.")

    with c_tab2:
        if positions:
            pos_df = pd.DataFrame([{
                "Symbol": p.symbol,
                "Sector": SECTOR_MAP.get(p.symbol, "Other"),
                "Shares": f"{float(p.qty):,.4f}",
                "Entry Price": f"${float(p.avg_entry_price):,.2f}",
                "Current Price": f"${float(p.current_price):,.2f}",
                "Market Value": f"${float(p.market_value):,.2f}",
                "Unrealized P&L": f"${float(p.unrealized_pl):,.2f}",
                "Return (%)": f"{float(p.unrealized_plpc)*100:+.2f}%"
            } for p in positions])
            st.dataframe(pos_df, height=250, width="stretch")
        else:
            st.caption("No filled positions yet (market is closed or orders pending in matching engine).")

    st.divider()

    # --- 5. Two-Step Tri-Horizon Retrain, Verification & Reallocation Engine ---
    st.header("🤖 Flagship Tri-Horizon Multi-Model Ensemble Retraining & Execution")

    # Load master Parquet dataset
    master_df = load_master_dataset()
    latest_eval_date, is_synced = check_and_sync_data(master_df)

    col_strat, col_basket, col_freq = st.columns([3, 1, 1])
    with col_strat:
        strategy_choice = st.selectbox(
            "Selected Production Strategy",
            options=[
                "🏆 Tri-Horizon Multi-Model Ensemble (5d + 15d + 35d Confluence, Prop Sizing, T+1 Lag)",
                "⚡ Single Purged XGBoost (15d Rebal, 15d Fwd, Prop, T+1 Lag)",
                "🛡️ Single Purged XGBoost (30d Rebal, 30d Fwd, Prop, T+1 Lag)"
            ],
            index=0
        )
    with col_basket:
        basket_size = st.number_input("Basket Size (Top N)", min_value=10, max_value=100, value=50, step=5)
    with col_freq:
        rebalance_freq = st.number_input("Rebalance Cadence (Days)", min_value=1, max_value=60, value=15, step=1)

    st.info(f"""
    **Strategy Architecture:** `{strategy_choice}`  
    **Multi-Timeframe Horizon Blending:** `30% Fast Catalyst (5d) + 40% Swing Alpha (15d) + 30% Fundamental Drift (35d)`  
    **Strict Zero-Leakage:** `Embargo Purge: T - 5d, T - 15d, T - 35d | Execution Delay: T + 1 Session`  
    **Target Universe:** Top {basket_size} Equities from {len(SECTOR_MAP)} Candidates
    """)

    # Two Step Action Buttons
    col_step1, col_step2 = st.columns(2)
    step1_btn = col_step1.button("🔮 Step 1: Retrain Tri-Horizon Models & Generate Allocation Plan", width="stretch")

    if step1_btn:
        with st.spinner(f"Retraining 3 Purged XGBoost Sub-Models (5d, 15d, 35d) across {len(master_df):,} rows..."):
            if "Tri-Horizon" in strategy_choice:
                new_preds, models_dict = retrain_tri_horizon_ensemble_and_predict(
                    master_df,
                    total_equity=total_equity,
                    top_n=int(basket_size)
                )
            else:
                h_days = 15 if "15d" in strategy_choice else 30
                new_preds, models_dict = retrain_single_model_and_predict(
                    master_df,
                    total_equity=total_equity,
                    top_n=int(basket_size),
                    horizon_days=h_days
                )
                
            df_deltas = compute_portfolio_deltas(positions, orders, new_preds, total_equity=total_equity)
            
            # Store in session state for user review
            st.session_state["pending_rebalance_plan"] = {
                "predictions": new_preds,
                "deltas": df_deltas,
                "strategy": strategy_choice,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            st.success(f"✅ Tri-Horizon Ensemble Retrained Successfully! Review the Top {basket_size} confluence allocation below.")

    # Display Rebalance Plan & Verification Dashboard if generated
    if "pending_rebalance_plan" in st.session_state:
        plan = st.session_state["pending_rebalance_plan"]
        new_preds = plan["predictions"]
        df_deltas = plan["deltas"]

        st.subheader("🔍 Verification: Current Holdings vs. New Tri-Horizon Confluence Allocation")
        
        # Side-by-Side Comparison Plot
        top_deltas_plot = df_deltas.head(15).copy()
        fig_comp = go.Figure()
        fig_comp.add_trace(go.Bar(
            name="Current Value ($)",
            x=top_deltas_plot['Symbol'],
            y=top_deltas_plot['Current Value ($)'],
            marker_color="#636EFA"
        ))
        fig_comp.add_trace(go.Bar(
            name="New Target Value ($)",
            x=top_deltas_plot['Symbol'],
            y=top_deltas_plot['Target Value ($)'],
            marker_color="#00CC96"
        ))
        fig_comp.update_layout(
            template="plotly_dark",
            barmode="group",
            height=380,
            title="<b>Side-by-Side Sizing Comparison: Current Holding vs. Tri-Horizon Confluence Target ($)</b>",
            margin=dict(l=10, r=10, t=40, b=10)
        )
        st.plotly_chart(fig_comp, width="stretch")

        # Full Delta Comparison Table
        st.write("**Top Candidates Confluence Delta Execution Plan:**")
        cols_to_show = ['Symbol', 'Sector', 'Current Value ($)', 'Target Value ($)', 
                        'Current Weight (%)', 'Target Weight (%)', 'Dollar Delta ($)']
        if 'Confluence Score' in df_deltas.columns:
            cols_to_show.append('Confluence Score')
        cols_to_show.extend(['Expected Alpha', 'Action'])
        
        display_deltas = df_deltas[cols_to_show].copy()
        display_deltas['Current Value ($)'] = display_deltas['Current Value ($)'].apply(lambda x: f"${x:,.2f}")
        display_deltas['Target Value ($)'] = display_deltas['Target Value ($)'].apply(lambda x: f"${x:,.2f}")
        display_deltas['Current Weight (%)'] = display_deltas['Current Weight (%)'].apply(lambda x: f"{x:.2f}%")
        display_deltas['Target Weight (%)'] = display_deltas['Target Weight (%)'].apply(lambda x: f"{x:.2f}%")
        display_deltas['Dollar Delta ($)'] = display_deltas['Dollar Delta ($)'].apply(lambda x: f"{x:+,.2f}")
        if 'Confluence Score' in display_deltas.columns:
            display_deltas['Confluence Score'] = display_deltas['Confluence Score'].apply(lambda x: f"{x:.3f}")
        display_deltas['Expected Alpha'] = display_deltas['Expected Alpha'].apply(lambda x: f"{x*100:+.2f}%")
        
        st.dataframe(display_deltas, width="stretch", height=380)

        # Step 2 Button: Execute after checking
        st.divider()
        st.write("### ⚡ Step 2: Confirmation & Live Sandbox Execution")
        st.caption("Once you have inspected the predictions and delta table above, click the button below to cancel any open orders and route the fresh Tri-Horizon portfolio to Alpaca.")
        
        col_exec, col_cancel = st.columns([2, 1])
        exec_btn = col_exec.button("⚡ Execute Tri-Horizon Reallocation on Alpaca Sandbox", type="primary", width="stretch")
        cancel_plan_btn = col_cancel.button("✖️ Dismiss Plan", width="stretch")

        if exec_btn:
            with st.spinner("Canceling old orders and routing fresh Tri-Horizon confluence orders to Alpaca..."):
                submitted_count = execute_reallocation_orders(client, new_preds)
                del st.session_state["pending_rebalance_plan"]
                st.success(f"🚀 Successfully routed {submitted_count} Tri-Horizon confluence orders to Alpaca Paper Trading! Portfolio reallocated.")
                st.rerun()

        if cancel_plan_btn:
            del st.session_state["pending_rebalance_plan"]
            st.rerun()

with tab_real:
    st.header("🔒 Live Real-Money Brokerage Bridge")
    st.warning("Live trading is currently locked for safety. Real capital execution is enabled once paper trading performance validation is complete.")
    st.write("""
    ### Live Trading Checklist:
    - [x] Paper Trading Sandbox Configured
    - [x] Tri-Horizon Multi-Model Ensemble (5d/15d/35d) Integrated (+23,729% 2003–2026)
    - [x] Strict Multi-Horizon Embargo Purge Active
    - [ ] 30-Day Forward Paper P&L Track Record
    """)
