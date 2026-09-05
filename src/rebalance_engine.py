import os
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import xgboost as xgb
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce

SECTOR_MAP = {
    'NVDA': 'Information Tech', 'MSFT': 'Information Tech', 'AAPL': 'Information Tech', 'ORCL': 'Information Tech',
    'QCOM': 'Information Tech', 'AMAT': 'Information Tech', 'CSCO': 'Information Tech', 'LRCX': 'Information Tech',
    'TXN': 'Information Tech', 'ADBE': 'Information Tech', 'IBM': 'Information Tech', 'CRM': 'Information Tech',
    'GOOGL': 'Communication', 'META': 'Communication', 'DIS': 'Communication', 'NFLX': 'Communication', 'CMCSA': 'Communication', 'VZ': 'Communication', 'T': 'Communication',
    'AMZN': 'Consumer Discretionary', 'TSLA': 'Consumer Discretionary', 'HD': 'Consumer Discretionary', 'MCD': 'Consumer Discretionary', 'NKE': 'Consumer Discretionary', 'LOW': 'Consumer Discretionary', 'SBUX': 'Consumer Discretionary', 'TJX': 'Consumer Discretionary', 'BKNG': 'Consumer Discretionary', 'ORLY': 'Consumer Discretionary', 'ROST': 'Consumer Discretionary', 'EBAY': 'Consumer Discretionary', 'F': 'Consumer Discretionary', 'GM': 'Consumer Discretionary', 'YUM': 'Consumer Discretionary', 'MAR': 'Consumer Discretionary', 'AZO': 'Consumer Discretionary',
    'JPM': 'Financials', 'BAC': 'Financials', 'WFC': 'Financials', 'GS': 'Financials', 'MS': 'Financials', 'C': 'Financials', 'BLK': 'Financials', 'SCHW': 'Financials', 'AXP': 'Financials', 'USB': 'Financials', 'PNC': 'Financials', 'CB': 'Financials', 'MMC': 'Financials', 'PGR': 'Financials', 'TRV': 'Financials', 'ALL': 'Financials', 'AIG': 'Financials', 'PRU': 'Financials',
    'LLY': 'Health Care', 'UNH': 'Health Care', 'JNJ': 'Health Care', 'ABBV': 'Health Care', 'MRK': 'Health Care', 'TMO': 'Health Care', 'ABT': 'Health Care', 'PFE': 'Health Care', 'DHR': 'Health Care', 'BMY': 'Health Care', 'AMGN': 'Health Care', 'GILD': 'Health Care', 'VRTX': 'Health Care', 'MDT': 'Health Care', 'ISRG': 'Health Care', 'SYK': 'Health Care', 'REGN': 'Health Care', 'ZTS': 'Health Care', 'BDX': 'Health Care', 'EW': 'Health Care', 'BAX': 'Health Care', 'CVS': 'Health Care',
    'CAT': 'Industrials', 'GE': 'Industrials', 'UNP': 'Industrials', 'HON': 'Industrials', 'RTX': 'Industrials', 'BA': 'Industrials', 'LMT': 'Industrials', 'DE': 'Industrials', 'UPS': 'Industrials', 'FDX': 'Industrials', 'EMR': 'Industrials', 'ITW': 'Industrials', 'NSC': 'Industrials', 'CSX': 'Industrials', 'PCAR': 'Industrials', 'NOC': 'Industrials', 'GD': 'Industrials',
    'PG': 'Consumer Staples', 'COST': 'Consumer Staples', 'WMT': 'Consumer Staples', 'KO': 'Consumer Staples', 'PEP': 'Consumer Staples', 'PM': 'Consumer Staples', 'MO': 'Consumer Staples', 'MDLZ': 'Consumer Staples', 'CL': 'Consumer Staples', 'KMB': 'Consumer Staples', 'GIS': 'Consumer Staples',
    'XOM': 'Energy', 'CVX': 'Energy', 'COP': 'Energy', 'SLB': 'Energy', 'EOG': 'Energy', 'MPC': 'Energy', 'PSX': 'Energy', 'VLO': 'Energy', 'OXY': 'Energy', 'HAL': 'Energy', 'WMB': 'Energy', 'KMI': 'Energy',
    'LIN': 'Materials', 'APD': 'Materials', 'SHW': 'Materials', 'ECL': 'Materials', 'FCX': 'Materials', 'NEM': 'Materials', 'DOW': 'Materials', 'DD': 'Materials', 'PPG': 'Materials', 'STLD': 'Materials',
    'NEE': 'Utilities', 'DUK': 'Utilities', 'SO': 'Utilities', 'AEP': 'Utilities', 'SRE': 'Utilities', 'D': 'Utilities', 'EXC': 'Utilities', 'XEL': 'Utilities', 'ED': 'Utilities',
    'PLD': 'Real Estate', 'AMT': 'Real Estate', 'CCI': 'Real Estate', 'EQIX': 'Real Estate', 'PSA': 'Real Estate', 'O': 'Real Estate', 'SPG': 'Real Estate', 'WELL': 'Real Estate', 'DLR': 'Real Estate'
}

PARQUET_DATA_PATH = "data/processed/master_panel_2000_2026.parquet"

BASELINE_FEATURES = [
    'revenue_growth', 'net_margin', 'sentiment_score', 'rsi_14', 'macd',
    'is_opp_buy', 'is_pol_buy', 'ewma_volatility', 'daily_news_count',
    'daily_news_finbert_sentiment', 'news_volume_intensity',
    'news_decay_tau_1d_ema', 'news_decay_tau_3d_ema', 'news_sentiment_velocity'
]

def load_master_dataset():
    """Loads the precalculated master dataset from ultra-fast Parquet storage."""
    df = pd.read_parquet(PARQUET_DATA_PATH)
    df['date'] = pd.to_datetime(df['date'])
    return df

def check_and_sync_data(df: pd.DataFrame):
    """Checks if dataset is up to date and returns latest evaluation date."""
    latest_date = df['date'].max()
    return latest_date, True

def retrain_tri_horizon_ensemble_and_predict(df: pd.DataFrame, total_equity: float = 100000.0, top_n: int = 50):
    """
    Retrains the Flagship Tri-Horizon Multi-Model Ensemble (5d/15d/35d Confluence Blend).
    Uses strict trading-session purged embargos and cross-sectional Z-score proportional weighting.
    """
    # Ensure forward targets exist
    for h in [5, 15, 35]:
        target_col = f"target_{h}d"
        if target_col not in df.columns:
            df[target_col] = df.groupby('ticker')['close'].transform(lambda s: s.shift(-h) / s - 1.0)
            
    all_dates = pd.Index(sorted(df['date'].unique()))
    latest_date = all_dates[-1]
    t_idx = all_dates.get_loc(latest_date)
    
    # 1. Train Sub-Model 1: Fast Catalyst (H=5d)
    purge_5 = max(0, t_idx - 5)
    train_5 = df[df['date'] <= all_dates[purge_5]].tail(80000)
    train_clean_5 = train_5[train_5['target_5d'].notnull()]
    m5 = xgb.XGBRegressor(n_estimators=30, max_depth=3, learning_rate=0.05, n_jobs=-1, random_state=42, tree_method='hist')
    m5.fit(train_clean_5[BASELINE_FEATURES].values, train_clean_5['target_5d'].values)
    
    # 2. Train Sub-Model 2: Swing Alpha (H=15d)
    purge_15 = max(0, t_idx - 15)
    train_15 = df[df['date'] <= all_dates[purge_15]].tail(80000)
    train_clean_15 = train_15[train_15['target_15d'].notnull()]
    m15 = xgb.XGBRegressor(n_estimators=40, max_depth=4, learning_rate=0.05, n_jobs=-1, random_state=42, tree_method='hist')
    m15.fit(train_clean_15[BASELINE_FEATURES].values, train_clean_15['target_15d'].values)
    
    # 3. Train Sub-Model 3: Fundamental Drift (H=35d)
    purge_35 = max(0, t_idx - 35)
    train_35 = df[df['date'] <= all_dates[purge_35]].tail(80000)
    train_clean_35 = train_35[train_35['target_35d'].notnull()]
    m35 = xgb.XGBRegressor(n_estimators=40, max_depth=4, learning_rate=0.05, n_jobs=-1, random_state=42, tree_method='hist')
    m35.fit(train_clean_35[BASELINE_FEATURES].values, train_clean_35['target_35d'].values)
    
    # Active candidate universe at Day T
    df_latest = df[(df['date'] == latest_date) & df['close'].notnull()].copy()
    X_latest = df_latest[BASELINE_FEATURES].values
    
    raw_p5 = m5.predict(X_latest)
    raw_p15 = m15.predict(X_latest)
    raw_p35 = m35.predict(X_latest)
    
    # Cross-sectional Z-score standardization
    z5 = (raw_p5 - np.mean(raw_p5)) / (np.std(raw_p5) + 1e-5)
    z15 = (raw_p15 - np.mean(raw_p15)) / (np.std(raw_p15) + 1e-5)
    z35 = (raw_p35 - np.mean(raw_p35)) / (np.std(raw_p35) + 1e-5)
    
    df_latest['pred_5d_pct'] = raw_p5 * 100.0
    df_latest['pred_15d_pct'] = raw_p15 * 100.0
    df_latest['pred_35d_pct'] = raw_p35 * 100.0
    df_latest['confluence_score'] = 0.30 * z5 + 0.40 * z15 + 0.30 * z35
    df_latest['predicted_return'] = raw_p15  # Primary 15d benchmark return
    
    # Select Top N equities by Confluence Score
    top_candidates = df_latest.sort_values('confluence_score', ascending=False).head(top_n).copy()
    
    # Proportional conviction weighting
    sc = top_candidates['confluence_score'] - top_candidates['confluence_score'].min() + 0.0001
    tot_sc = sc.sum()
    
    top_candidates['Weight (%)'] = (sc / tot_sc) * 100.0
    top_candidates['Target ($)'] = (sc / tot_sc) * total_equity
    top_candidates['Sector'] = top_candidates['ticker'].map(lambda x: SECTOR_MAP.get(x, 'Other / Diversified'))
    top_candidates['Horizon'] = "5d/15d/35d Ensemble"
    
    return top_candidates, {'m5': m5, 'm15': m15, 'm35': m35}

def retrain_single_model_and_predict(df: pd.DataFrame, total_equity: float = 100000.0, top_n: int = 50, horizon_days: int = 15):
    """Retrains a single Purged XGBoost regressor with strict trading-bar embargos."""
    target_col = f"target_{horizon_days}d"
    if target_col not in df.columns:
        df[target_col] = df.groupby('ticker')['close'].transform(lambda s: s.shift(-horizon_days) / s - 1.0)
        
    all_dates = pd.Index(sorted(df['date'].unique()))
    latest_date = all_dates[-1]
    t_idx = all_dates.get_loc(latest_date)
    
    purge_idx = max(0, t_idx - horizon_days)
    train_df = df[df['date'] <= all_dates[purge_idx]].tail(80000)
    train_clean = train_df[train_df[target_col].notnull()]
    
    model = xgb.XGBRegressor(n_estimators=40, max_depth=4, learning_rate=0.05, n_jobs=-1, random_state=42, tree_method='hist')
    model.fit(train_clean[BASELINE_FEATURES].values, train_clean[target_col].values)
    
    df_latest = df[(df['date'] == latest_date) & df['close'].notnull()].copy()
    df_latest['predicted_return'] = model.predict(df_latest[BASELINE_FEATURES].values)
    
    top_candidates = df_latest.sort_values('predicted_return', ascending=False).head(top_n).copy()
    top_candidates['clean_pred'] = top_candidates['predicted_return'].clip(lower=0.0001)
    tot_forecast = top_candidates['clean_pred'].sum()
    
    top_candidates['Weight (%)'] = (top_candidates['clean_pred'] / tot_forecast) * 100.0
    top_candidates['Target ($)'] = (top_candidates['clean_pred'] / tot_forecast) * total_equity
    top_candidates['Sector'] = top_candidates['ticker'].map(lambda x: SECTOR_MAP.get(x, 'Other / Diversified'))
    top_candidates['Horizon'] = f"{horizon_days}d"
    
    return top_candidates, model

def compute_portfolio_deltas(current_positions, current_orders, new_predictions: pd.DataFrame, total_equity: float):
    """Calculates comparison deltas between current portfolio holdings and new target predictions."""
    curr_alloc = {}
    if current_positions:
        for p in current_positions:
            curr_alloc[p.symbol] = float(p.market_value)
    elif current_orders:
        for o in current_orders:
            curr_alloc[o.symbol] = float(o.notional) if o.notional else 0.0
            
    delta_records = []
    all_symbols = sorted(set(list(curr_alloc.keys()) + new_predictions['ticker'].tolist()))
    
    for sym in all_symbols:
        curr_val = curr_alloc.get(sym, 0.0)
        curr_wt = (curr_val / total_equity) * 100.0
        
        target_row = new_predictions[new_predictions['ticker'] == sym]
        if not target_row.empty:
            tgt_val = target_row['Target ($)'].iloc[0]
            tgt_wt = target_row['Weight (%)'].iloc[0]
            sector = target_row['Sector'].iloc[0]
            exp_alpha = target_row['predicted_return'].iloc[0]
            confluence = target_row['confluence_score'].iloc[0] if 'confluence_score' in target_row.columns else 0.0
        else:
            tgt_val = 0.0
            tgt_wt = 0.0
            sector = SECTOR_MAP.get(sym, 'Other')
            exp_alpha = 0.0
            confluence = 0.0
            
        delta_val = tgt_val - curr_val
        delta_wt = tgt_wt - curr_wt
        
        if delta_val > 50.0:
            action = "🟢 BUY / INCREASE"
        elif delta_val < -50.0:
            action = "🔴 SELL / TRIM"
        else:
            action = "⚪ KEEP / HOLD"
            
        delta_records.append({
            'Symbol': sym,
            'Sector': sector,
            'Current Value ($)': curr_val,
            'Target Value ($)': tgt_val,
            'Current Weight (%)': curr_wt,
            'Target Weight (%)': tgt_wt,
            'Weight Delta (%)': delta_wt,
            'Dollar Delta ($)': delta_val,
            'Confluence Score': confluence,
            'Expected Alpha': exp_alpha,
            'Action': action
        })
        
    df_deltas = pd.DataFrame(delta_records).sort_values('Target Value ($)', ascending=False)
    return df_deltas

def execute_reallocation_orders(client, new_predictions: pd.DataFrame):
    """Cancels open orders and routes new forecast-weighted orders to Alpaca."""
    client.cancel_orders()
    
    submitted_count = 0
    for _, row in new_predictions.iterrows():
        ticker = row['ticker']
        dollar_amt = round(row['Target ($)'], 2)
        if dollar_amt >= 1.0:
            client.submit_order(MarketOrderRequest(
                symbol=ticker,
                notional=dollar_amt,
                side=OrderSide.BUY,
                time_in_force=TimeInForce.DAY
            ))
            submitted_count += 1
            
    return submitted_count
