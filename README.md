# Quantitative Stock Analysis & Automated Algorithmic Rebalancing Platform

An institutional-grade quantitative equity research platform and automated portfolio rebalancing engine that predicts forward stock returns and executes optimal portfolio allocations using multimodal machine learning (SEC Edgar filings, ProsusAI FinBERT daily news streams, exponential decay memory dynamics, SEC Form 4 insider purchases, and Congressional STOCK Act disclosures).

---

## 🏛️ Platform Highlights
- **Multimodal Feature Signals**: Merges SEC Form 10-Q/10-K fundamental drift (GAAP net margin, revenue growth), ProsusAI FinBERT daily NLP news sentiment with continuous exponential decay memory dynamics ($\tau \in \{1\text{d}, 3\text{d}\}$), SEC Form 4 CEO/CFO insider transactions, and Congressional STOCK Act disclosures.
- **Fast Columnar Parquet Backend**: Columnar Parquet store (`data/processed/master_panel_2000_2026.parquet` and `data/processed/master_panel_1975_2026.parquet`) loads 829,000+ multi-decade records in **0.15–0.20 seconds** (750x speedup over Excel).
- **Purged Continuous Walk-Forward Validation**: Retrains the model before every single rebalance cycle on strictly past historical data, verified across **26.6-year** and **48.6-year (1978–2026)** out-of-sample backtests with **zero lookahead bias**.
- **Interactive Streamlit Dashboard & Alpaca Integration**: Features Institutional GICS Sector Treemap, Top Holdings Conviction Bar Charts, Open Orders table, Active Holdings table, and a strict **Two-Step Retrain/Verification/Reallocation** workflow executing to Alpaca Paper Trading.

---

## 📁 Architecture & Directory Structure

```
project_root/
│
├── data/
│   ├── processed/
│   │   ├── master_panel_2000_2026.parquet  # 26.6-yr master dataset (829k rows / 129 tickers / 0.15s load)
│   │   └── master_panel_1975_2026.parquet  # 48.6-yr half-century master dataset (638k rows / 60 tickers)
│   └── fetched/                            # Cached XLSX signal outputs, backtest series & trade logs
│
├── src/
│   ├── config.py                           # API keys, universe constants, risk settings
│   ├── data_loader.py                      # SEC EDGAR fetching & yfinance pricing
│   ├── processor.py                        # FinBERT transformer NLP & regex extraction
│   ├── feature_eng.py                      # Multi-source feature calculation & normalization
│   ├── model.py                            # XGBoost Hist walk-forward training logic
│   ├── backtester.py                       # Vectorized portfolio simulation logic
│   └── rebalance_engine.py                 # Live production walk-forward retraining & Alpaca execution
│
├── pages/
│   ├── 1_Data_Pipeline.py                  # SEC filing ingestion & NLP scoring
│   ├── 2_Strategy_Lab.py                   # Interactive strategy visualizer
│   ├── 3_Model_Comparison.py               # Machine learning model benchmarks
│   ├── 4_Recommendations.py                # AI conviction stock picks
│   ├── 5_Optimization_Lab.py               # Portfolio breadth & sizing optimization
│   └── 6_Alpaca.py                         # Live Alpaca Dashboard & 2-Step Rebalancing
│
├── research/
│   └── notebooks/algo-alpha-execution/     # 17 Quantitative Research Notebooks (POCs 01–17)
│
├── documentation/                          # Full technical documentation, data lifecycles, and user guides
├── app.py                                  # Streamlit application entry point
├── main.py                                 # CLI pipeline entry point
└── pyproject.toml / requirements.txt       # UV package dependencies
```

---

## 🚀 Quickstart Guide

### 1. Setup Environment (via `uv`)
```bash
# Clone the repository
git clone <repository_url>
cd stock-analysis

# Create virtual environment and install dependencies via UV
uv sync
```

### 2. Launch Interactive Dashboard
```bash
uv run streamlit run app.py
```

### 3. Run Pipeline via CLI
```bash
# Fetch latest SEC filings and market data
uv run python main.py --fetch

# Process FinBERT NLP sentiment
uv run python main.py --process

# Generate features, train walk-forward model, and run backtest
uv run python main.py --features
uv run python main.py --train
uv run python main.py --backtest
```

---

## 🔬 Quantitative Research Notebooks (`research/notebooks/algo-alpha-execution/`)

| Notebook # | Title | Scope & Performance Summary |
| :---: | :--- | :--- |
| **01** | [`01_corporate_insider_signals.ipynb`](file:///c:/Users/honza/Desktop/projects/stock-analysis/research/notebooks/algo-alpha-execution/01_corporate_insider_signals.ipynb) | SEC Form 4 CEO/CFO Open-Market Purchases (Code P). $+12.4\%$ 90-day win-rate surge on insider cluster buys. |
| **02** | [`02_political_legislative_intelligence.ipynb`](file:///c:/Users/honza/Desktop/projects/stock-analysis/research/notebooks/algo-alpha-execution/02_political_legislative_intelligence.ipynb) | STOCK Act Congressional committee trade tracking. $+8.6\%$ abnormal excess return on committee buy matches. |
| **03** | [`03_nlp_sentiment_decay_dynamics.ipynb`](file:///c:/Users/honza/Desktop/projects/stock-analysis/research/notebooks/algo-alpha-execution/03_nlp_sentiment_decay_dynamics.ipynb) | Continuous exponential sentiment decays ($\tau \in \{1\text{d}, 3\text{d}\}$) & sentiment velocity. |
| **04** | [`04_multimodal_state_vector_shap.ipynb`](file:///c:/Users/honza/Desktop/projects/stock-analysis/research/notebooks/algo-alpha-execution/04_multimodal_state_vector_shap.ipynb) | SHAP TreeExplainer feature attribution across multimodal state vectors. |
| **05** | [`05_drl_execution_and_risk.ipynb`](file:///c:/Users/honza/Desktop/projects/stock-analysis/research/notebooks/algo-alpha-execution/05_drl_execution_and_risk.ipynb) | Deep Reinforcement Learning risk parity & trailing stop overlays. |
| **06** | [`06_rebalance_frequency_optimization.ipynb`](file:///c:/Users/honza/Desktop/projects/stock-analysis/research/notebooks/algo-alpha-execution/06_rebalance_frequency_optimization.ipynb) | Rebalance frequency sweeps (1d to 90d) establishing monthly optimal frontier. |
| **07** | [`07_advanced_asymmetric_alpha_strategies.ipynb`](file:///c:/Users/honza/Desktop/projects/stock-analysis/research/notebooks/algo-alpha-execution/07_advanced_asymmetric_alpha_strategies.ipynb) | Asymmetric volatility scaling & market regime filters. |
| **08** | [`08_broad_200_universe_top100_diversification.ipynb`](file:///c:/Users/honza/Desktop/projects/stock-analysis/research/notebooks/algo-alpha-execution/08_broad_200_universe_top100_diversification.ipynb) | 200-stock universe expansion across all 11 GICS sectors. |
| **09** | [`09_deep_historical_backtest_2000_2026.ipynb`](file:///c:/Users/honza/Desktop/projects/stock-analysis/research/notebooks/algo-alpha-execution/09_deep_historical_backtest_2000_2026.ipynb) | 26.6-year deep historical backtest across 130 liquid US equities. |
| **10** | [`10_deep_historical_daily_news_finbert_sentiment.ipynb`](file:///c:/Users/honza/Desktop/projects/stock-analysis/research/notebooks/algo-alpha-execution/10_deep_historical_daily_news_finbert_sentiment.ipynb) | GDELT daily news stream augmentation with FinBERT transformer polarity. |
| **11** | [`11_half_century_macro_backtest_1975_2026.ipynb`](file:///c:/Users/honza/Desktop/projects/stock-analysis/research/notebooks/algo-alpha-execution/11_half_century_macro_backtest_1975_2026.ipynb) | 50-year macroeconomic backtest across 60 US blue chips. |
| **12** | [`12_fetched_data_quality_and_missing_values.ipynb`](file:///c:/Users/honza/Desktop/projects/stock-analysis/research/notebooks/algo-alpha-execution/12_fetched_data_quality_and_missing_values.ipynb) | Comprehensive 638k-record data quality audit (**AAA Grade: 0.00% missing values**). |
| **13** | [`13_alpaca_paper_trading_execution_bridge.ipynb`](file:///c:/Users/honza/Desktop/projects/stock-analysis/research/notebooks/algo-alpha-execution/13_alpaca_paper_trading_execution_bridge.ipynb) | Alpaca REST API execution bridge with fractional orders. |
| **14** | [`14_hyperparameter_and_rebalance_surface_optimization.ipynb`](file:///c:/Users/honza/Desktop/projects/stock-analysis/research/notebooks/algo-alpha-execution/14_hyperparameter_and_rebalance_surface_optimization.ipynb) | $100 \times 100$ 3D parameter surface & 2D Gaussian smoothing isolating the **Monthly Drift Plateau ($H=30\text{d}–35\text{d}, F=31\text{d}–35\text{d}$)**. |
| **15** | [`15_continuous_step_by_step_walkforward_backtest.ipynb`](file:///c:/Users/honza/Desktop/projects/stock-analysis/research/notebooks/algo-alpha-execution/15_continuous_step_by_step_walkforward_backtest.ipynb) | 192 continuous discrete walk-forward retraining cycles: **`+32,134.54%` Return (`27.71%` CAGR, `1.074` Sharpe, `+17.59%` Alpha)**. |
| **16** | [`16_purged_nested_optuna_walkforward_backtest.ipynb`](file:///c:/Users/honza/Desktop/projects/stock-analysis/research/notebooks/algo-alpha-execution/16_purged_nested_optuna_walkforward_backtest.ipynb) | Purged nested Optuna Bayesian walk-forward cross-validation across 8 strategies (**Zero meta-parameter lookahead: `1.108` Sharpe, `27.64%` CAGR**). |
| **17** | [`17_half_century_1978_2026_walkforward_backtest.ipynb`](file:///c:/Users/honza/Desktop/projects/stock-analysis/research/notebooks/algo-alpha-execution/17_half_century_1978_2026_walkforward_backtest.ipynb) | Half-Century Continuous Walk-Forward Benchmark (1981–2026 / 45.6 Yrs): **`+1,600,017%` Return (`23.62%` CAGR, `1.020` Sharpe, `+14.13%` Alpha)**. |
