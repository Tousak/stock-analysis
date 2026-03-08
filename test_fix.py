import pandas as pd
from src.backtester import simulate_portfolio
import os

# Create dummy predictions
df = pd.DataFrame({
    'ticker': ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA'] * 2,
    'filing_date': ['2023-01-01'] * 5 + ['2023-01-06'] * 5,
    'predicted_return': [0.1] * 10,
    'return_5d': [0.05] * 10
})

# Run backtest with rebalance_days=1, lookahead=5
res = simulate_portfolio(df, "test_backtest.xlsx", initial_capital=1000, start_year=2020, frequency='D', rebalance_days=1)
if not res.empty:
    print(f"Initial: 1000")
    print(f"Final: {res['portfolio_value'].iloc[-1]:.2f}")
    # With return_5d = 5% and rebalance=1, lookahead=5
    # Daily return = 1.05^(1/5) - 1 = 0.009806 (approx 0.98%)
    # Over 10 days (two filings overlapping or just stepping through):
    # Actually, filing 1 (Jan 1) is active for Jan 1 to Jan 6. (5 days)
    # Total return over 5 days should be exactly 5%.
    # 1.05^(1/5)^5 = 1.05.
    print(f"Expected approx: 1102.50 (if 5% compound twice)")
else:
    print("Failed.")
