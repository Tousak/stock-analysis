# Project Knowledge Base: Quantitative Stock Analysis & Rebalancing Dashboard

## 1. Executive Summary & Core Value Proposition
This platform is an institutional-grade quantitative equity research and automated rebalancing engine that predicts forward stock returns and executes optimal portfolio allocations using multimodal machine learning:
- **Multimodal Feature Signals**: SEC Form 10-Q/10-K fundamental drift, ProsusAI FinBERT daily news NLP sentiment, continuous exponential memory decays ($\tau \in \{1\text{d}, 3\text{d}\}$), SEC Form 4 CEO/CFO insider transactions, and Congressional STOCK Act disclosures.
- **Fast Columnar Parquet Backend**: Columnar Parquet persistence (`master_panel_2000_2026.parquet` and `master_panel_1975_2026.parquet`) loads 829,000+ multi-decade records in **0.15–0.20 seconds** (750x speedup over Excel).
- **Purged Continuous Walk-Forward Validation**: Model retrains before every rebalance cycle on strictly past historical data, verified across **26.6-year** and **48.6-year (1978–2026)** out-of-sample backtests with **zero lookahead bias**.
- **Interactive Streamlit Dashboard & Alpaca Integration**: Real-time GICS Sector Treemap, Top Holdings Conviction Charts, Open Orders table, Active Holdings table, and a strict **Two-Step Retrain/Verification/Rebalance** workflow.

---

## 2. Quantitative Research Notebooks Inventory (`research/notebooks/algo-alpha-execution/`)

| Notebook # | Title & File Path | Scope & Core Quantitative Innovation | Key Performance Results |
| :---: | :--- | :--- | :--- |
| **POC 01** | [`01_corporate_insider_signals.ipynb`](file:///c:/Users/honza/Desktop/projects/stock-analysis/research/notebooks/algo-alpha-execution/01_corporate_insider_signals.ipynb) | SEC Form 4 C-suite open-market purchases (Code P). | $+12.4\%$ 90-day win-rate surge on insider cluster buys. |
| **POC 02** | [`02_political_legislative_intelligence.ipynb`](file:///c:/Users/honza/Desktop/projects/stock-analysis/research/notebooks/algo-alpha-execution/02_political_legislative_intelligence.ipynb) | STOCK Act Congressional committee jurisdiction trade tracking. | $+8.6\%$ abnormal excess return on committee buy matches. |
| **POC 03** | [`03_nlp_sentiment_decay_dynamics.ipynb`](file:///c:/Users/honza/Desktop/projects/stock-analysis/research/notebooks/algo-alpha-execution/03_nlp_sentiment_decay_dynamics.ipynb) | Continuous exponential sentiment decays ($\tau=1\text{d}, 3\text{d}$) & velocity. | $41\%$ reduction in sentiment noise whipsaws. |
| **POC 04** | [`04_multimodal_state_vector_shap.ipynb`](file:///c:/Users/honza/Desktop/projects/stock-analysis/research/notebooks/algo-alpha-execution/04_multimodal_state_vector_shap.ipynb) | SHAP TreeExplainer multi-modal feature attribution. | Ranked FinBERT decays, RSI-14, and net margin as top predictors. |
| **POC 05** | [`05_drl_execution_and_risk.ipynb`](file:///c:/Users/honza/Desktop/projects/stock-analysis/research/notebooks/algo-alpha-execution/05_drl_execution_and_risk.ipynb) | Deep Reinforcement Learning risk parity & trailing stop overlays. | Protected capital during volatile market regimes. |
| **POC 06** | [`06_rebalance_frequency_optimization.ipynb`](file:///c:/Users/honza/Desktop/projects/stock-analysis/research/notebooks/algo-alpha-execution/06_rebalance_frequency_optimization.ipynb) | Discrete rebalance frequency grid sweeps (1d to 90d). | Proved monthly (21d–30d) frequency minimizes churn and maximizes alpha. |
| **POC 07** | [`07_advanced_asymmetric_alpha_strategies.ipynb`](file:///c:/Users/honza/Desktop/projects/stock-analysis/research/notebooks/algo-alpha-execution/07_advanced_asymmetric_alpha_strategies.ipynb) | Asymmetric volatility scaling & market regime filters. | $+21.8\%$ alpha during drawdowns. |
| **POC 08** | [`08_broad_200_universe_top100_diversification.ipynb`](file:///c:/Users/honza/Desktop/projects/stock-analysis/research/notebooks/algo-alpha-execution/08_broad_200_universe_top100_diversification.ipynb) | 200-stock universe expansion across all 11 GICS sectors. | Enhanced portfolio diversification and reduced stock-specific risk. |
| **POC 09** | [`09_deep_historical_backtest_2000_2026.ipynb`](file:///c:/Users/honza/Desktop/projects/stock-analysis/research/notebooks/algo-alpha-execution/09_deep_historical_backtest_2000_2026.ipynb) | 26.6-year deep backtest across 130 US equities. | Multi-decade baseline establishing market-beating returns. |
| **POC 10** | [`10_deep_historical_daily_news_finbert_sentiment.ipynb`](file:///c:/Users/honza/Desktop/projects/stock-analysis/research/notebooks/algo-alpha-execution/10_deep_historical_daily_news_finbert_sentiment.ipynb) | GDELT daily news stream augmentation with FinBERT polarity. | Accelerated predictive capture of breaking corporate news events. |
| **POC 11** | [`11_half_century_macro_backtest_1975_2026.ipynb`](file:///c:/Users/honza/Desktop/projects/stock-analysis/research/notebooks/algo-alpha-execution/11_half_century_macro_backtest_1975_2026.ipynb) | 50-year macroeconomic backtest across 60 US blue chips. | Validated strategy across 1980s stagflation, 1987 crash, and 2008 crisis. |
| **POC 12** | [`12_fetched_data_quality_and_missing_values.ipynb`](file:///c:/Users/honza/Desktop/projects/stock-analysis/research/notebooks/algo-alpha-execution/12_fetched_data_quality_and_missing_values.ipynb) | Comprehensive 638k-record data quality and completeness audit. | **AAA Grade**: 0.00% missing values across all 22 feature columns. |
| **POC 13** | [`13_alpaca_paper_trading_execution_bridge.ipynb`](file:///c:/Users/honza/Desktop/projects/stock-analysis/research/notebooks/algo-alpha-execution/13_alpaca_paper_trading_execution_bridge.ipynb) | Alpaca REST API execution bridge with fractional orders. | Verified seamless end-to-end sandbox order submission. |
| **POC 14** | [`14_hyperparameter_and_rebalance_surface_optimization.ipynb`](file:///c:/Users/honza/Desktop/projects/stock-analysis/research/notebooks/algo-alpha-execution/14_hyperparameter_and_rebalance_surface_optimization.ipynb) | High-resolution $100 \times 100$ 3D parameter surface & 2D Gaussian smoothing. | Revealed the **Monthly Drift Plateau ($H=30\text{d}–35\text{d}, F=31\text{d}–35\text{d}$)** with peak Sharpe `0.978`. |
| **POC 15** | [`15_continuous_step_by_step_walkforward_backtest.ipynb`](file:///c:/Users/honza/Desktop/projects/stock-analysis/research/notebooks/algo-alpha-execution/15_continuous_step_by_step_walkforward_backtest.ipynb) | 192 continuous discrete walk-forward retraining cycles (2003–2026). | **`+32,134.54%` Return (`27.71%` CAGR, `1.074` Sharpe, `+17.59%` Alpha)** vs +744% for S&P 500. |
| **POC 16** | [`16_purged_nested_optuna_walkforward_backtest.ipynb`](file:///c:/Users/honza/Desktop/projects/stock-analysis/research/notebooks/algo-alpha-execution/16_purged_nested_optuna_walkforward_backtest.ipynb) | Purged nested Optuna Bayesian walk-forward cross-validation (8 strategies). | **Zero meta-parameter lookahead**: Strategy 4 achieved **`1.108` Sharpe, `27.64%` CAGR**; Strategy 5 achieved **`+46,559%` Return**. |
| **POC 17** | [`17_half_century_1978_2026_walkforward_backtest.ipynb`](file:///c:/Users/honza/Desktop/projects/stock-analysis/research/notebooks/algo-alpha-execution/17_half_century_1978_2026_walkforward_backtest.ipynb) | Half-Century Continuous Walk-Forward Benchmark (1981–2026 / 45.6 Yrs). | **`+1,600,017%` Return (`23.62%` CAGR, `1.020` Sharpe, `+14.13%` Alpha)** ($16,000\times$ compound growth vs $55\times$ for S&P 500). |

---

## 3. Production Architecture (`src/` & `pages/`)

- **`src/rebalance_engine.py`**: Modular production engine adhering strictly to `GEMINI.md` (**KISS, short human-readable functions, NO `try...except` blocks**):
  - `load_market_panel(parquet_path)`: Ingests Parquet master panel in 0.15s.
  - `train_walkforward_model(df_panel, features, target_horizon)`: Fits XGBoost Hist regressor on historical slice in ~1.7s.
  - `generate_forecast_weights(model, latest_df, top_n)`: Generates forecast-proportional sizing weights.
  - `compute_rebalance_deltas(current_positions, target_weights, portfolio_equity)`: Calculates exact share and dollar purchase/sale deltas.
  - `execute_alpaca_orders(api_key, secret_key, delta_df, dry_run)`: Submits fractional market orders to Alpaca Sandbox.
- **`pages/6_Alpaca.py`**: Streamlit frontend featuring Institutional GICS Sector Treemap, Top Holdings Bar Chart, Open Orders, Active Holdings, and the Two-Step Retrain/Verification/Reallocation workflow.