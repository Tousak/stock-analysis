import pandas as pd
import os
from tqdm.auto import tqdm
from datetime import timedelta

from src.config import INITIAL_CAPITAL, DATA_DIR

def simulate_portfolio(predictions_df: pd.DataFrame, output_path: str, initial_capital: float = INITIAL_CAPITAL, start_year: int = 2021, frequency: str = 'Q', top_n: int = 50, rebalance_days: int = 1) -> pd.DataFrame:
    """
    Simulates a portfolio strategy with dynamic rebalancing frequency.
    frequency: 'Q' for Quarterly, 'D' for Daily/Interval-based.
    top_n: Max number of top stocks to include in the portfolio.
    rebalance_days: Interval between rebalances in days (only for frequency='D').
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
        # Look for return_Xd columns
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
        # For Alpha mode, we rebalance at fixed intervals starting from the first filing date
        min_date = df[df['filing_date'].dt.year >= start_year]['filing_date'].min()
        max_date = df['filing_date'].max()
        if pd.isna(min_date):
            print(f"No filings found for the selected start year {start_year}.")
            return pd.DataFrame()
            
        rebalance_dates = pd.date_range(start=min_date, end=max_date, freq=f'{rebalance_days}D')

    portfolio_value = initial_capital
    start_date = rebalance_dates[0] - timedelta(days=1)
    
    portfolio_history = [{
        'date': pd.to_datetime(start_date),
        'portfolio_value': initial_capital,
        'quarterly_return': 0.0,
        'selection': 'Initial Capital'
    }]
    
    print(f"--- Backtest Initializing ({'Quarterly' if frequency == 'Q' else f'Every {rebalance_days} Days'}) ---")
    print(f"Settings: Top {top_n} stocks | Initial Capital: ${initial_capital:,.2f}\n")
    
    for r_date in tqdm(rebalance_dates, desc=f"Backtesting {frequency}"):
        # Find active predictions: 
        # For Quarterly: Filings in this quarter
        # For Alpha: Latest filing for each ticker that is not older than 'lookahead' days from r_date
        
        if frequency == 'Q':
            period_predictions = df[df['rebalance_date'] == r_date]
        else:
            # Alpha mode: Get the latest prediction for each ticker that is available on this rebalance date
            # and is still within its "validity window" (lookahead days)
            # We want filings where: filing_date <= r_date AND filing_date > r_date - lookahead
            active_mask = (df['filing_date'] <= r_date) & (df['filing_date'] > r_date - timedelta(days=lookahead))
            period_predictions = df[active_mask].sort_values('filing_date').groupby('ticker').tail(1)

        if period_predictions.empty:
            period_return = 0.0
            selection = "CASH (No active predictions)"
        else:
            positive_predictions = period_predictions[period_predictions['predicted_return'] > 0.0]

            if positive_predictions.empty:
                period_return = 0.0
                selection = "CASH (No positive predictions)"
            else:
                # Apply Top N
                positive_predictions = positive_predictions.sort_values('predicted_return', ascending=False).head(top_n)
                
                total_weight = positive_predictions['predicted_return'].sum()
                if total_weight == 0:
                    period_return = 0.0
                    selection = "CASH (Zero predicted return sum)"
                else:
                    weights = positive_predictions['predicted_return'] / total_weight
                    
                    # SCALE THE RETURN:
                    # If lookahead is 5 days (Alpha) but we rebalance every 1 day, 
                    # we only take 1/5th of the return (geometrically).
                    # formula: (1 + total_return) ** (days_held / lookahead_days) - 1
                    raw_return = (positive_predictions[return_col] * weights).sum()
                    
                    if frequency == 'Q':
                        period_return = raw_return # 90 days / 90 days = 1
                    else:
                        hold_days = rebalance_days
                        # Safety check: raw_return < -1 is impossible in finance but good for math safety
                        safe_raw = max(raw_return, -0.99)
                        period_return = (1 + safe_raw) ** (hold_days / lookahead) - 1
                        
                    selection = ", ".join(positive_predictions['ticker'].tolist())
        
        # --- Debug prints for the loop ---
        if frequency == 'Q' or (isinstance(r_date, pd.Timestamp) and r_date.day % 10 == 0): # Reduced print frequency for daily
             print(f"\n--- Date: {r_date.date()} ---")
             print(f"Selection: {selection[:100]}..." if len(selection) > 100 else f"Selection: {selection}")
             print(f"Period Return (Scaled): {period_return:.4%}")
        
        portfolio_value *= (1 + period_return)
        
        portfolio_history.append({
            'date': r_date,
            'portfolio_value': portfolio_value,
            'quarterly_return': period_return,
            'selection': selection
        })

    if len(portfolio_history) <= 1:
        print("No trades were made during the backtest period.")
        return pd.DataFrame()

    final_df = pd.DataFrame(portfolio_history)
    os.makedirs(DATA_DIR, exist_ok=True)
    final_df.to_excel(output_path, index=False)
    print(f"\nBacktest results saved to {output_path}")
    return final_df

if __name__ == "__main__":
    print("Running backtester.py example (notebook-style)...")
    try:
        predictions_data = pd.read_excel("data/fetched/predictions_finbert.xlsx")
        portfolio_results = simulate_portfolio(predictions_data.copy(), output_path="data/fetched/backtest_results_finbert.xlsx")
        
        if not portfolio_results.empty:
            print("\nPortfolio Simulation Results:")
            print(portfolio_results.tail())
            final_value = portfolio_results['portfolio_value'].iloc[-1]
            print(f"\nFinal portfolio value: ${final_value:,.2f}")
            print(f"Total return: {((final_value / INITIAL_CAPITAL) - 1):.2%}")

    except FileNotFoundError:
        print("Error: Predictions file not found. Please run --train first.")
    except Exception as e:
        print(f"An error occurred during backtester.py example: {e}")
