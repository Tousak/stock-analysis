import pandas as pd
import os
from tqdm.auto import tqdm
from datetime import timedelta

from src.config import INITIAL_CAPITAL, TOP_N_STOCKS_TO_INVEST, BACKTEST_RESULTS_PATH, PREDICTIONS_PATH, DATA_DIR

def simulate_portfolio(predictions_df: pd.DataFrame, initial_capital: float = INITIAL_CAPITAL, start_year: int = 2021, top_n: int = TOP_N_STOCKS_TO_INVEST) -> pd.DataFrame:
    """
    Simulates a portfolio strategy with detailed logging for debugging.
    """
    if predictions_df.empty:
        print("Predictions DataFrame is empty. Cannot simulate portfolio.")
        return pd.DataFrame()

    df = predictions_df.copy()
    df['filing_date'] = pd.to_datetime(df['filing_date'])
    df['quarter'] = df['filing_date'].dt.to_period('Q')
    df = df.sort_values('filing_date')

    all_quarters = sorted(df['quarter'].unique())
    trading_quarters = [q for q in all_quarters if q.year >= start_year]

    portfolio_value = initial_capital
    
    # --- Log Initial State ---
    # Handle case where there are no trading quarters
    if not trading_quarters:
        print("No trading quarters available for the selected start year.")
        return pd.DataFrame()
    start_date = trading_quarters[0].start_time.date() - timedelta(days=1)
    
    portfolio_history = [{
        'date': pd.to_datetime(start_date),
        'portfolio_value': initial_capital,
        'quarterly_return': 0.0,
        'selection': 'Initial Capital'
    }]
    print(f"--- Backtest Initializing ---")
    print(f"Start Date: {start_date} | Initial Portfolio Value: ${initial_capital:,.2f} | Top N: {top_n}\n")
    
    for current_q in tqdm(trading_quarters, desc="Backtesting Quarters"):
        quarter_predictions = df[df['quarter'] == current_q]
        if quarter_predictions.empty:
            continue

        top_picks = quarter_predictions[quarter_predictions['predicted_return'] > 0.0].nlargest(top_n, 'predicted_return')

        if top_picks.empty:
            period_return = 0.0
            selection = "CASH (Bearish)"
        else:
            period_return = top_picks['next_quarter_return'].mean()
            selection = ", ".join(top_picks['ticker'].tolist())
        
        # --- Debug prints inside the loop ---
        value_before = portfolio_value
        portfolio_value *= (1 + period_return)
        
        print(f"\n--- Quarter: {current_q} ---")
        print(f"Selection: {selection}")
        print(f"Quarterly Return (Mean of Actuals): {period_return:.4%}")
        print(f"Value Before: ${value_before:,.2f}")
        print(f"Value After: ${portfolio_value:,.2f}  <-- {value_before:,.2f} * (1 + {period_return:.4f})")
        
        q_end_date = current_q.end_time
        portfolio_history.append({
            'date': q_end_date,
            'portfolio_value': portfolio_value,
            'quarterly_return': period_return,
            'selection': selection
        })

    if len(portfolio_history) <= 1:
        print("No trades were made during the backtest period.")
        return pd.DataFrame()

    final_df = pd.DataFrame(portfolio_history)
    
    os.makedirs(DATA_DIR, exist_ok=True)
    final_df.to_excel(BACKTEST_RESULTS_PATH, index=False)
    print(f"\nBacktest results saved to {BACKTEST_RESULTS_PATH}")
    
    return final_df

if __name__ == "__main__":
    print("Running backtester.py example (notebook-style)...")
    try:
        predictions_data = pd.read_excel(PREDICTIONS_PATH)
        portfolio_results = simulate_portfolio(predictions_data.copy())
        
        if not portfolio_results.empty:
            print("\nPortfolio Simulation Results:")
            print(portfolio_results.tail())
            final_value = portfolio_results['portfolio_value'].iloc[-1]
            print(f"\nFinal portfolio value: ${final_value:,.2f}")
            print(f"Total return: {((final_value / INITIAL_CAPITAL) - 1):.2%}")

    except FileNotFoundError:
        print(f"Error: {PREDICTIONS_PATH} not found. Please run --train first.")
    except Exception as e:
        print(f"An error occurred during backtester.py example: {e}")
