# 🧪 Algorithmic Alpha Execution: Proof-of-Concept Research Suite

**Directory**: [`research/notebooks/algo-alpha-execution/`](file:///c:/Users/honza/Desktop/projects/stock-analysis/research/notebooks/algo-alpha-execution/)  
**Research Reference**: [`documentation/papers/Daily Algorithmic Trading Research Plan.md`](file:///c:/Users/honza/Desktop/projects/stock-analysis/documentation/papers/Daily%20Algorithmic%20Trading%20Research%20Plan.md)  
**Historical Backtest Horizons**:
- **Point-in-Time Universe POC (2021–2026)**: 5.6 Years / 1,414 Sessions ($M=20 \to N=10$).
- **Modern Post-Sarbanes/Edgar Walk-Forward (2003–2026)**: 23.6 Years / 6,451 Sessions ($M=129$).
- **Half-Century Macro Validation (1978–2026)**: 48.6 Years / 12,264 Sessions ($M=60$).
**Benchmarks**: S&P 500 Index (`^GSPC` / `SPY`) & Point-in-Time Active Universe Buy & Hold.

---

## 1. Business Context & Objective

The legacy stock analysis application relied on quarterly SEC Form 10-Q fundamental filings and static MD&A sentiment, updating only once every 90 days. While effective for long-term valuation, this architecture missed high-frequency alpha catalysts and experienced severe drawdowns during broad market regime shifts.

This research suite establishes an **institutional-grade, multi-domain quantitative pipeline** designed to capture daily and swing alpha by synthesizing:
1. **Corporate Insider Conviction**: SEC Form 4 open-market purchases and cluster buying.
2. **Legislative Intelligence**: Congressional STOCK Act disclosures filtered by committee jurisdictions.
3. **Continuous Sentiment Dynamics**: Local FinBERT NLP news scoring with continuous exponential decay ($S(t)$).
4. **Regime-Conditioned Machine Learning**: Expanding walk-forward XGBoost conditioned on EWMA volatility with TreeSHAP explainability.
5. **Strict Institutional Zero-Leakage Standards**:
   - **Trading-Session Index Purged Embargos** ($T_{\text{train}} \le \text{all\_dates}[t_{\text{idx}} - H]$).
   - **Realistic $T+1$ Execution Lag** (Trades placed at Day $T$ close execute on Day $T+1$).
   - **Dynamic Point-in-Time Active Universes** (Zero pre-IPO `bfill()`).
   - **Survivorship Alpha Decomposition** (Isolating pure machine learning selection & sizing alpha over active universe beta).

---

## 2. Institutional Zero-Leakage Proof-of-Concept Inventory (POC 01–19)

```
research/notebooks/algo-alpha-execution/
├── 01_corporate_insider_signals.ipynb                    # POC 1: Form 4 Routine vs. Opportunistic Classification
├── 02_political_legislative_intelligence.ipynb           # POC 2: Congressional STOCK Act Committee Alpha
├── 03_nlp_sentiment_decay_dynamics.ipynb                 # POC 3: Local FinBERT & Continuous Exponential Decay
├── 04_multimodal_state_vector_shap.ipynb                 # POC 4: Expanding Walk-Forward XGBoost & TreeSHAP
├── 05_drl_execution_and_risk.ipynb                       # POC 5: Two-Tier Active Selection (M=20 -> N=10) & Risk Simulation
├── 06_rebalance_frequency_optimization.ipynb             # POC 6: Frequency Grid Sweep (1-120d) across Broad Universe (M=100)
├── 07_advanced_asymmetric_alpha_strategies.ipynb         # POC 7: Broad Universe (M=100) vs. QQQ/SPY & Unified Alpha Engine
├── 08_broad_200_universe_top100_diversification.ipynb   # POC 8: Broad Universe (M=200) with Dynamic Top 100 Selection
├── 09_deep_historical_backtest_2000_2026.ipynb           # POC 9: Deep 26.6-Year Historical Stress-Testing (2000-2026)
├── 10_deep_historical_daily_news_finbert_sentiment.ipynb # POC 10: Deep Historical Daily News Stream & FinBERT Sentiment
├── 11_half_century_macro_backtest_1975_2026.ipynb        # POC 11: Half-Century Macro Historical Stress-Testing (1978-2026)
├── 12_fetched_data_quality_and_missing_values.ipynb      # POC 12: Fetched Data Quality, Completeness & Missing Value Audit
├── 13_alpaca_paper_trading_execution_bridge.ipynb       # POC 13: Alpaca Paper Trading Automated Execution & Rebalance Bridge
├── 14_hyperparameter_and_rebalance_surface_optimization.ipynb # POC 14: 100x100 3D Parameter Surface Optimization & 2D Smoothing
├── 15_continuous_step_by_step_walkforward_backtest.ipynb # POC 15: Continuous Step-by-Step Walk-Forward (Pre-Rebalance Fit)
├── 16_purged_nested_optuna_walkforward_backtest.ipynb    # POC 16: Purged Nested Optuna 8-Strategy Walk-Forward Benchmark
├── 17_half_century_1978_2026_walkforward_backtest.ipynb # POC 17: Half-Century Continuous Walk-Forward Benchmark (1978-2026)
├── 18_institutional_zero_leakage_purged_walkforward_backtest.ipynb # POC 18: Strict Institutional Zero-Leakage (2003-2026)
├── 19_half_century_institutional_zero_leakage_walkforward_backtest.ipynb # POC 19: Half-Century Strict Zero-Leakage (1981-2026)
├── README.md                                             # Quantitative architecture & notebook inventory
└── RESEARCH_POC_REPORT.md                                # Comprehensive executive research report
```

---

## 3. Flagship Institutional Benchmarks

### 🛡️ POC 18: Institutional Zero-Leakage Walk-Forward Benchmark (2003–2026 / 23.6 Years)
- **File**: [`18_institutional_zero_leakage_purged_walkforward_backtest.ipynb`](file:///c:/Users/honza/Desktop/projects/stock-analysis/research/notebooks/algo-alpha-execution/18_institutional_zero_leakage_purged_walkforward_backtest.ipynb)
- **Standards**: Exact trading-session index purge ($T_{\text{train}} \le \text{all\_dates}[t_{\text{idx}} - H]$), $T+1$ execution lag, dynamic active universe (zero `bfill()`), and active universe survivorship decomposition.
- **Results (6,451 Daily Trading Sessions)**:

| Strategy / Model | Total Return (%) | CAGR (%) | Sharpe Ratio | Sortino Ratio | Max Drawdown (%) | Jensen Alpha (α) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Purged XGBoost (15d/15d Prop, T+1)** | **+11,326.88%** | **22.22%** | **0.809** | **1.089** | **-69.32%** | **+11.55%** |
| **Purged XGBoost (30d/30d Prop, T+1)** 🏆 | **+8,361.94%** | **20.68%** | **0.868** | **1.140** | **-53.20%** | **+10.71%** |
| **Purged XGBoost (30d/5d Eq, T+1)** | **+6,769.95%** | **19.62%** | **0.872** | **1.113** | **-50.46%** | **+9.83%** |
| **Point-in-Time Active Universe B&H** | +3,727.09% | 16.69% | 0.799 | 0.986 | -50.24% | +7.30% |
| **S&P 500 Index (`^GSPC` Benchmark)** | +744.38% | 9.46% | 0.471 | 0.579 | -56.78% | +0.00% |

- **Excess ML Alpha**: The model adds **+4,634.85% excess return (+3.99% annual CAGR alpha)** above the active universe benchmark.

---

### 🏛️ POC 19: Half-Century Institutional Zero-Leakage Benchmark (1981–2026 / 45.6 Years)
- **File**: [`19_half_century_institutional_zero_leakage_walkforward_backtest.ipynb`](file:///c:/Users/honza/Desktop/projects/stock-analysis/research/notebooks/algo-alpha-execution/19_half_century_institutional_zero_leakage_walkforward_backtest.ipynb)
- **Standards**: Exact trading-session index purge, $T+1$ execution lag, zero pre-IPO backfills, active universe benchmark.
- **Results (12,264 Daily Trading Sessions)**:

| Strategy / Model | Total Return (%) | CAGR (%) | Sharpe Ratio | Sortino Ratio | Max Drawdown (%) | Jensen Alpha (α) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Purged XGBoost (15d/15d Prop, T+1)** | **+934,181.00%** | **22.17%** | **0.940** | **1.224** | **-47.06%** | **+12.54%** |
| **Purged XGBoost (30d/30d Prop, T+1)** 🏆 | **+859,228.24%** | **21.95%** | **0.950** | **1.235** | **-41.15%** | **+12.44%** |
| **Purged XGBoost (30d/5d Eq, T+1)** | **+321,974.44%** | **19.35%** | **0.914** | **1.175** | **-46.67%** | **+10.33%** |
| **Point-in-Time Active Universe B&H** | +299,723.34% | 19.17% | 0.909 | 1.167 | -45.71% | +10.17% |
| **S&P 500 Index (`^GSPC` Benchmark)** | +5,529.82% | 9.23% | 0.417 | 0.524 | -56.78% | +0.00% |

- **Excess ML Alpha**: Generated **+559,505% pure selection and forecast-proportional conviction excess return (+2.78% CAGR alpha)** over the active universe, while maintaining the lowest maximum drawdown in the benchmark (**-41.15%**).

---

## 4. Production Integration

The production rebalance pipeline is implemented in [`src/rebalance_engine.py`](file:///c:/Users/honza/Desktop/projects/stock-analysis/src/rebalance_engine.py) and surfaced via the Streamlit Alpaca execution page ([`pages/6_Alpaca.py`](file:///c:/Users/honza/Desktop/projects/stock-analysis/pages/6_Alpaca.py)).

### 🔬 [20_temporal_evolution_of_rebalance_surface_plateaus.ipynb](file:///c:/Users/honza/Desktop/projects/stock-analysis/research/notebooks/algo-alpha-execution/20_temporal_evolution_of_rebalance_surface_plateaus.ipynb)
- **Scope**: Temporal Evolution of Rebalance ($F \in [1, 50]$) vs. Horizon ($H \in [1, 50]$) Surface Plateaus across **9 Distinct 5-Year Historical Epochs (1981–2026 / 45.6 Years / 22,500 Grid Points)**.
- **Key Findings**:
  - **1981–2015 Eras**: Dominated by high-frequency fast price momentum plateaus ($H \in [1, 6]\text{d}, F \in [1, 3]\text{d}$).
  - **2016–2026 Modern Regimes**: Plateaus shifted dramatically toward **Longer Fundamental Drift ($H=42\text{d}\dots 48\text{d}, F=8\text{d}\dots 15\text{d}$)**, confirming the secular emergence of quarterly fundamental themes and post-earnings drift in modern algorithmic markets.

---

### 🔬 [21_annual_and_rolling_plateau_dynamics.ipynb](file:///c:/Users/honza/Desktop/projects/stock-analysis/research/notebooks/algo-alpha-execution/21_annual_and_rolling_plateau_dynamics.ipynb)
- **Scope**: Annual & Rolling Plateau Dynamics across **45 Individual Years (1981–2026 / 112,500 Grid Points)**.
- **Empirical Confirmation of Horizon Compression Law**:
  - **Bear / Crisis Regimes (1987, 2008, 2020, 2022)**: Optimal Horizon $H^*$ collapses to **1.0 Day** (mean 6.7d) as fast momentum and rapid capital preservation dominate.
  - **Low-Volatility Secular Bull Regimes (1995, 2017, 2021, 2024)**: Optimal Horizon $H^*$ expands up to **35d–50d** as post-earnings fundamental drift and thematic trends persist.

---

### 🏆 [22_regime_adaptive_dynamic_horizon_walkforward_backtest.ipynb](file:///c:/Users/honza/Desktop/projects/stock-analysis/research/notebooks/algo-alpha-execution/22_regime_adaptive_dynamic_horizon_walkforward_backtest.ipynb)
- **Scope**: Multi-Decade (1981–2026 / 45.6 Years) Walk-Forward Backtest of **9 Strategies**, including Naive Default, Optuna-Tuned 15d, Regime-Adaptive Dynamic Horizon, and Optuna-Tuned Tri-Horizon Confluence Ensemble.
- **Results (12,264 Daily Sessions / 45.6 Years)**:

| Strategy / Model | Total Return (%) | CAGR (%) | Sharpe Ratio | Sortino Ratio | Max Drawdown (%) | Jensen Alpha (α) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **9. Optuna-Tuned Tri-Horizon Ensemble** 🏆 | **+2,734,948.33%** | **25.08%** | **0.965** | **1.294** | **-40.48%** | **+15.01%** |
| **8. Tri-Horizon Multi-Model Ensemble (5d/15d/35d)** | **+2,560,012.39%** | **24.90%** | **0.959** | **1.281** | **-41.86%** | **+14.82%** |
| **7. Regime-Adaptive Dynamic Horizon XGBoost** | **+1,242,846.51%** | **22.94%** | **0.918** | **1.188** | **-51.50%** | **+13.13%** |
| **5. Static Purged XGBoost (15d/15d Prop, T+1)** | **+934,181.00%** | **22.17%** | **0.940** | **1.224** | **-47.06%** | **+12.54%** |
| **4. Static Purged XGBoost (30d/30d Prop, T+1)** | **+859,228.24%** | **21.95%** | **0.950** | **1.235** | **-41.15%** | **+12.44%** |
| **6. Optuna-Tuned Purged XGBoost (15d/15d Prop)** | **+603,264.71%** | **21.01%** | **0.920** | **1.190** | **-47.21%** | **+11.53%** |
| **3. Naive XGBoost (Default Params, 30d/30d Eq)** | **+301,307.82%** | **19.18%** | **0.914** | **1.176** | **-41.25%** | **+10.23%** |
| **2. Point-in-Time Active Universe B&H** | +299,723.34% | 19.17% | 0.909 | 1.167 | -45.71% | +10.17% |
| **1. S&P 500 Index (`^GSPC` Benchmark)** | +5,529.82% | 9.23% | 0.417 | 0.524 | -56.78% | +0.00% |

- **Empirical Superiority**: The **Optuna-Tuned Tri-Horizon Ensemble** achieves **`-40.48%` Max Drawdown** (lowest drawdown of all ML models) while compounding **`+2.73M%`** ($27,349\times$) through continuous 3-timeframe confluence.

---

### 🔬 [23_tri_horizon_advanced_tuning_methods_2000_2026.ipynb](file:///c:/Users/honza/Desktop/projects/stock-analysis/research/notebooks/algo-alpha-execution/23_tri_horizon_advanced_tuning_methods_2000_2026.ipynb)
- **Scope**: Modern Era (2000–2026 / 23.6 Years Out-of-Sample / 5,950 Sessions / 129 Equities) Comparative Benchmark of **Advanced Hyperparameter Optimization Methodologies** for the Tri-Horizon Ensemble (Rank-IC Maximization, Multi-Objective Pareto, Regime-Conditioned Volatility Scaling, Standard Bayesian MSE).
- **Results (2003–2026 / 23.6 Years)**:

| Optimization Method / Strategy | Total Return (%) | CAGR (%) | Sharpe Ratio | Sortino Ratio | Max Drawdown (%) | Jensen Alpha (α) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **5. Multi-Objective Pareto Tri-Horizon Ensemble** 🏆 | **+23,729.27%** | **26.09%** | **0.844** | **1.208** | **-63.66%** | **+15.26%** |
| **4. Rank-IC Maximized Tri-Horizon Ensemble** | **+21,972.73%** | **25.68%** | **0.835** | **1.196** | **-64.02%** | **+14.84%** |
| **7. Standard Bayesian MSE Optuna Tri-Horizon** | **+20,254.91%** | **25.25%** | **0.828** | **1.169** | **-64.12%** | **+14.42%** |
| **3. Baseline Tri-Horizon (Fixed Default Params)** | **+19,026.37%** | **24.92%** | **0.816** | **1.162** | **-65.48%** | **+14.05%** |
| **6. Regime-Conditioned Volatility-Scaled Tri-Horizon** | **+18,687.42%** | **24.83%** | **0.816** | **1.163** | **-64.18%** | **+13.99%** |
| **2. Point-in-Time Active Universe B&H** | +3,421.59% | 16.28% | 0.727 | 0.900 | -50.24% | +6.86% |
| **1. S&P 500 Index (`^GSPC` Benchmark)** | +744.38% | 9.46% | 0.417 | 0.513 | -56.78% | +0.00% |

- **Key Finding**: Optimizing hyperparameters for **Rank Information Coefficient (Stock Ranking)** and **Multi-Objective Pareto (Sharpe vs. Drawdown)** outperforms traditional MSE loss minimization by **`+3,474%` excess terminal wealth**, proving that ranking-aligned loss functions are superior for quantitative equity selection.


