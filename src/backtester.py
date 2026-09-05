import pandas as pd
import os
from tqdm.auto import tqdm
from datetime import timedelta
import yfinance as yf

from src.config import INITIAL_CAPITAL, DATA_DIR

def simulate_portfolio(predictions_df: pd.DataFrame, output_path: str, initial_capital: float = INITIAL_CAPITAL, start_year: int = 2021, frequency: str = 'Q', top_n: int = 50, rebalance_days: int = 1, market_data: pd.DataFrame = None, ranking_col: str = 'predicted_return', filter_positive: bool = True) -> pd.DataFrame:
    """
    Simulates a portfolio strategy with daily resolution between rebalance points.
    """
    if predictions_df.empty:
        print("Predictions DataFrame is empty. Cannot simulate portfolio.")
        return pd.DataFrame()

    df = predictions_df.copy()
    df['filing_date'] = pd.to_datetime(df['filing_date'])
    
    # Identify the actual return column present in the dataframe
    if 'next_quarter_return' in df.columns:
        return_col = 'next_quarter_return'
        lookahead = 90
    else:
        return_cols = [c for c in df.columns if c.startswith('return_') and c.endswith('d')]
        if return_cols:
            return_col = return_cols[0]
            lookahead = int(return_col.split('_')[1].replace('d', ''))
        else:
            print(f"Error: No return columns found in predictions. Columns: {df.columns.tolist()}")
            return pd.DataFrame()

    if frequency == 'Q':
        df['rebalance_date'] = df['filing_date'].dt.to_period('Q').dt.start_time
        rebalance_dates = sorted([d for d in df['rebalance_date'].unique() if d.year >= start_year])
    else:
        min_date = df[df['filing_date'].dt.year >= start_year]['filing_date'].min()
        max_date = df['filing_date'].max()
        if pd.isna(min_date):
            print(f"No filings found for the selected start year {start_year}.")
            return pd.DataFrame()
        rebalance_dates = pd.date_range(start=min_date, end=max_date, freq=f'{rebalance_days}D')

    # --- Fetch Daily Market Data ---
    all_tickers = df['ticker'].unique()
    start_market_date = (rebalance_dates[0] - timedelta(days=1)).strftime('%Y-%m-%d')
    end_market_date = (rebalance_dates[-1] + timedelta(days=lookahead + 30)).strftime('%Y-%m-%d')
    
    if market_data is not None:
        daily_closes = market_data
    else:
        print(f"Fetching daily price history for {len(all_tickers)} tickers...")
        market_prices = yf.download(all_tickers.tolist(), start=start_market_date, end=end_market_date, progress=False, auto_adjust=True)
        
        if isinstance(market_prices.columns, pd.MultiIndex):
            daily_closes = market_prices['Close']
        else:
            daily_closes = pd.DataFrame({all_tickers[0]: market_prices['Close']})
        
        daily_closes.index = pd.to_datetime(daily_closes.index).tz_localize(None)
        daily_closes = daily_closes[~daily_closes.index.duplicated(keep='first')]

    portfolio_value = initial_capital
    portfolio_history = []
    
    current_selection = []
    current_weights = pd.Series(dtype='float64')
    last_rebalance_value = initial_capital
    last_r_date = rebalance_dates[0] - timedelta(days=1)
    
    print(f"--- Backtest Initializing ({'Quarterly' if frequency == 'Q' else f'Every {rebalance_days} Days'}) ---")
    
    all_market_days = daily_closes.index[daily_closes.index >= pd.to_datetime(rebalance_dates[0])]
    last_processed_period = None
    
    for current_day in tqdm(all_market_days, desc="Simulating Daily"):
        # 1. Determine the period of the current day
        if frequency == 'Q':
            current_period = current_day.to_period('Q')
        else:
            # For daily/interval, we treat every rebalance as a unique period ID
            # based on how many intervals have passed since start
            days_since_start = (current_day - pd.to_datetime(rebalance_dates[0])).days
            current_period = days_since_start // rebalance_days

        # 2. Rebalance Check: Trigger if we enter a new period or have no selection yet
        if current_period != last_processed_period or not current_selection:
            # Set the date to search for in predictions
            if frequency == 'Q':
                r_date_ts = pd.to_datetime(current_day.to_period('Q').start_time)
            else:
                r_date_ts = pd.to_datetime(current_day.date())
            
            # Lock in the previous period's performance
            last_rebalance_value = portfolio_value 
            
            # Find active predictions for this period
            if frequency == 'Q':
                period_predictions = df[df['rebalance_date'] == r_date_ts].sort_values('filing_date').groupby('ticker').tail(1)
            else:
                active_mask = (df['filing_date'] <= r_date_ts) & (df['filing_date'] > r_date_ts - timedelta(days=lookahead))
                period_predictions = df[active_mask].sort_values('filing_date').groupby('ticker').tail(1)

            if filter_positive:
                candidate_predictions = period_predictions[period_predictions[ranking_col] > 0.0]
            else:
                candidate_predictions = period_predictions.dropna(subset=[ranking_col])

            if candidate_predictions.empty:
                current_selection = []
                current_weights = pd.Series(dtype='float64')
            else:
                candidate_predictions = candidate_predictions.sort_values(ranking_col, ascending=False).head(top_n)
                # For weighted allocation, we still need a positive value. 
                # If using revenue, we'll use equal weights for the benchmark to keep it simple and 'naive'.
                if ranking_col == 'predicted_return':
                    total_weight = candidate_predictions[ranking_col].sum()
                    current_weights = candidate_predictions.set_index('ticker')[ranking_col] / total_weight
                else:
                    # Equal weight for naive benchmark
                    current_weights = pd.Series(1.0 / len(candidate_predictions), index=candidate_predictions['ticker'])
                
                current_selection = candidate_predictions['ticker'].tolist()
            
            last_r_date = current_day
            last_processed_period = current_period

        # 3. Daily Valuation
        if not current_selection:
            daily_return = 0.0
        else:
            start_prices = daily_closes.loc[last_r_date, current_selection]
            now_prices = daily_closes.loc[current_day, current_selection]
            
            # Ensure we are dealing with scalars by using .iloc[0] if it returns a Series per ticker
            def get_val(series, t):
                val = series[t]
                return val.iloc[0] if isinstance(val, pd.Series) else val

            valid_tickers = [t for t in current_selection if pd.notna(get_val(start_prices, t)) and pd.notna(get_val(now_prices, t)) and get_val(start_prices, t) > 0]
            
            if not valid_tickers:
                daily_return = 0.0
            else:
                # Use .loc with unique tickers to avoid duplicate column issues
                unique_valid = list(dict.fromkeys(valid_tickers))
                sub_weights = current_weights[unique_valid]
                sub_weights = sub_weights / sub_weights.sum()
                
                s_p = start_prices[unique_valid]
                n_p = now_prices[unique_valid]
                
                growth = (n_p / s_p)
                current_rel_value = (growth * sub_weights).sum()
                portfolio_value = last_rebalance_value * current_rel_value
                
                prev_val = portfolio_history[-1]['portfolio_value'] if portfolio_history else initial_capital
                daily_return = (portfolio_value / prev_val) - 1

        portfolio_history.append({
            'date': current_day,
            'portfolio_value': portfolio_value,
            'quarterly_return': daily_return,
            'selection': ", ".join(current_selection) if current_selection else "CASH"
        })

    # --- Calculate Benchmarks (SPY and Universe Buy & Hold) ---
    print("Calculating benchmarks for comparison...")
    spy_data = yf.download('^GSPC', start=all_market_days[0], end=all_market_days[-1], progress=False, auto_adjust=True)
    
    # Ensure spy_prices is a flat Series even if yfinance returns multi-index
    if isinstance(spy_data.columns, pd.MultiIndex):
        spy_prices = spy_data['Close']['^GSPC']
    else:
        spy_prices = spy_data['Close']
    
    spy_prices = spy_prices.reindex(all_market_days).ffill()
    spy_normalized = (spy_prices / spy_prices.dropna().iloc[0]) * initial_capital
    
    # Universe Buy & Hold (Equal weight all tickers ever present in predictions)
    universe_tickers = df['ticker'].unique()
    u_prices = daily_closes[universe_tickers].reindex(all_market_days).ffill()
    
    # Correct Normalization: Find first available price for each ticker individually
    # to avoid entire rows being dropped by NaNs
    u_start_prices = u_prices.apply(lambda x: x.dropna().iloc[0] if not x.dropna().empty else np.nan)
    u_normalized = (u_prices / u_start_prices).mean(axis=1, skipna=True) * initial_capital
    
    # Average of relative growth across the whole universe


    for entry in portfolio_history:
        current_date = entry['date']
        # Use .item() or float() to ensure we store a scalar, not a Series
        entry['spy_value'] = float(spy_normalized.loc[current_date]) if current_date in spy_normalized.index else initial_capital
        entry['universe_bh_value'] = float(u_normalized.loc[current_date]) if current_date in u_normalized.index else initial_capital


    if not portfolio_history:
        print("No data recorded.")
        return pd.DataFrame()

    final_df = pd.DataFrame(portfolio_history)
    os.makedirs(DATA_DIR, exist_ok=True)
    final_df.to_excel(output_path, index=False)
    print(f"\nDaily Backtest results saved to {output_path}")
    return final_df

if __name__ == "__main__":
    print("Running backtester.py example...")
    from src.config import PROCESSED_FILINGS_PATH
    # We need predictions file, fallback to features if missing but usually predictions exist
    pred_path = os.path.join(DATA_DIR, "predictions_finbert.xlsx")
    if not os.path.exists(pred_path):
        pred_path = os.path.join(DATA_DIR, "features_finbert.xlsx")
        
    predictions_data = pd.read_excel(pred_path)
    # Ensure predicted_return is present (default to sentiment_score if missing for simple test)
    if 'predicted_return' not in predictions_data.columns:
         predictions_data['predicted_return'] = predictions_data.get('sentiment_score', 0)
         
    simulate_portfolio(predictions_data.copy(), output_path=os.path.join(DATA_DIR, "backtest_results_finbert.xlsx"))
