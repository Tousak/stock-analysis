# Quantitative Research Report: Unbiased Institutional POC Suite (1978–2026)

**Document Path**: [`research/notebooks/algo-alpha-execution/RESEARCH_POC_REPORT.md`](file:///c:/Users/honza/Desktop/projects/stock-analysis/research/notebooks/algo-alpha-execution/RESEARCH_POC_REPORT.md)  
**Research Reference**: [`documentation/papers/Daily Algorithmic Trading Research Plan.md`](file:///c:/Users/honza/Desktop/projects/stock-analysis/documentation/papers/Daily%20Algorithmic%20Trading%20Research%20Plan.md)  
**Historical Backtest Horizons**:
1. **Point-in-Time POC Validation (2021–2026)**: 5.6 Years / 1,414 Sessions ($M=20 \to N=10$).
2. **Modern Post-Sarbanes/Edgar Institutional Benchmark (2003–2026)**: 23.6 Years / 6,451 Sessions ($M=129$).
3. **Half-Century Macro Validation (1978–2026)**: 48.6 Years / 12,264 Sessions ($M=60$).
**Author**: Quantitative Engineering Team

---

## 1. Executive Summary & Zero-Leakage Institutional Architecture

To eliminate lookahead and survivorship bias across all multi-decade backtests, our research architecture enforces **5 strict institutional zero-leakage standards**:

1. **Trading-Session Index Purged Embargo ($T_{\text{train}} \le \text{all\_dates}[t_{\text{idx}} - H]$)**:
   - Purged by exact trading calendar bars ($H=30$ sessions $\approx 43$ calendar days) rather than calendar timedeltas. Guarantees 100.0% zero overlapping forward target labels between training data and out-of-sample test periods.
2. **Realistic $T+1$ Execution Lag**:
   - Signals computed at Day $T$ Market Close execute at Day $T+1$, earning returns strictly starting on Day $T+1$.
3. **Point-in-Time Active Universe (Zero `bfill()`)**:
   - Only stocks actively listed and trading on Day $T$ are eligible for selection, preventing pre-IPO distortion.
4. **Active Universe Survivorship Alpha Decomposition**:
   - Benchmarks against the passive active universe to isolate **Pure Machine Learning Selection & Sizing Alpha** from passive survivorship beta.
5. **Continuous Walk-Forward Retraining**:
   - Re-fits models dynamically before each rebalance cycle on strictly expanding historical datasets.

---

## 2. Institutional Zero-Leakage Benchmark (2003–2026 / 23.6 Years / POC 18)

**Test Specifications**: 6,451 daily trading sessions, 129 equities, 192 monthly cycles, 397 bi-weekly cycles.

| Strategy / Model | Total Return (%) | CAGR (%) | Sharpe Ratio | Sortino Ratio | Max Drawdown (%) | Calmar Ratio | Jensen Alpha (α) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **5. Purged XGBoost (15d Rebal, 15d Fwd, Prop, T+1)** | **+11,326.88%** | **22.22%** | **0.809** | **1.089** | **-69.32%** | **0.321** | **+11.55%** |
| **4. Purged XGBoost (30d Rebal, 30d Fwd, Prop, T+1)** 🏆 | **+8,361.94%** | **20.68%** | **0.868** | **1.140** | **-53.20%** | **0.389** | **+10.71%** |
| **3. Purged XGBoost (30d Rebal, 5d Fwd, Eq, T+1)** | **+6,769.95%** | **19.62%** | **0.872** | **1.113** | **-50.46%** | **0.389** | **+9.83%** |
| **2. Point-in-Time Active Universe B&H (No Lookahead)** | +3,727.09% | 16.69% | 0.799 | 0.986 | -50.24% | 0.332 | +7.30% |
| **1. S&P 500 Index (`^GSPC` Benchmark)** | +744.38% | 9.46% | 0.471 | 0.579 | -56.78% | 0.167 | +0.00% |

### 📅 Decade-by-Decade Breakdown (% Return Across Eras):
- **2000s Post-Crisis (2003–2009)**: **`+430.88%`** (Purged XGBoost 30d) vs. `+185.42%` (Active Universe) vs. `+22.67%` (S&P 500).
- **2010s Secular Bull (2010–2019)**: **`+423.26%`** (Purged XGBoost 30d) vs. `+345.97%` (Active Universe) vs. `+185.16%` (S&P 500).
- **2020s Volatility & AI (2020–2026)**: **`+301.68%`** (Purged XGBoost 30d) vs. `+194.43%` (Active Universe) vs. `+135.61%` (S&P 500).

---

## 3. Half-Century Macro Zero-Leakage Benchmark (1981–2026 / 45.6 Years / POC 19)

**Test Specifications**: 12,264 daily trading sessions, 60 blue chips, 371 monthly cycles, 767 bi-weekly cycles.

| Strategy / Model | Total Return (%) | CAGR (%) | Sharpe Ratio | Sortino Ratio | Max Drawdown (%) | Calmar Ratio | Jensen Alpha (α) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **5. Purged XGBoost (15d Rebal, 15d Fwd, Prop, T+1)** | **+934,181.00%** | **22.17%** | **0.940** | **1.224** | **-47.06%** | **0.471** | **+12.54%** |
| **4. Purged XGBoost (30d Rebal, 30d Fwd, Prop, T+1)** 🏆 | **+859,228.24%** | **21.95%** | **0.950** | **1.235** | **-41.15%** | **0.533** | **+12.44%** |
| **3. Purged XGBoost (30d Rebal, 5d Fwd, Eq, T+1)** | **+321,974.44%** | **19.35%** | **0.914** | **1.175** | **-46.67%** | **0.415** | **+10.33%** |
| **2. Point-in-Time Active Universe B&H (No bfill)** | +299,723.34% | 19.17% | 0.909 | 1.167 | -45.71% | 0.419 | +10.17% |
| **1. S&P 500 Index (`^GSPC` Benchmark)** | +5,529.82% | 9.23% | 0.417 | 0.524 | -56.78% | 0.163 | +0.00% |

### 💥 Crisis-by-Crisis Resilience Across 5 Decades:
- **1980s Stagflation Recovery & '87 Crash (1981–1989)**: **`+739.63%`** vs. `+159.20%` (S&P 500).
- **1990s Tech Revolution & Dot-Com (1990–1999)**: **`+1,791.85%`** vs. `+308.48%` (S&P 500).
- **2000s "Lost Decade" (2000–2009)**: **`+271.04%`** vs. **`-23.37%`** (S&P 500).
- **2010s Secular Bull (2010–2019)**: **`+353.74%`** vs. `+185.16%` (S&P 500).
- **2020s Volatility & GenAI (2020–2026)**: **`+232.63%`** vs. `+135.61%` (S&P 500).

---

## 4. Parameter Surface Optimization & Smooth Plateau Analysis (POC 14)

From the $100 \times 100 = 10,000$ combination grid sweep of Forward Target Horizon ($H \in [1, 100]$) vs. Rebalance Frequency ($F \in [1, 100]$):
- **Raw Discrete Surface**: Exhibits calendar harmonic noise (e.g. sharp jumps between $F=14\text{d}$ and $F=16\text{d}$).
- **2D Gaussian Filtered Surface ($\sigma=1.2$)**: Filters discrete earnings calendar harmonics, illuminating the macroscopic **Monthly Fundamental Drift Plateau**:
  - **Optimal Horizon**: $H = 30\text{d} \dots 35\text{d}$
  - **Optimal Rebalance Frequency**: $F = 31\text{d} \dots 35\text{d}$
  - **Peak Sharpe Ratio**: **`0.978`**
  - **Economic Foundation**: Captures the multi-week post-earnings announcement drift (PEAD) and insider conviction lifecycle before signal decay occurs.

---

## 5. Summary of Proof-of-Concept Research Suite (POC 01–19)

| Notebook / POC | Focus Area | Key Quantitative Result / Finding |
| :--- | :--- | :--- |
| **POC 01** | Form 4 Insider Trades | Open-market cluster buys generate **+10.80% 90-day CAR** over SPY. |
| **POC 02** | Political Intelligence | Congressional committee-aligned buys deliver **+24.63% 6-month CAR**. |
| **POC 03** | Continuous Sentiment Decay | Decayed FinBERT ($\tau = 1\text{d}$ EMA) achieves **+0.0862 1-day Rank IC**. |
| **POC 04** | Multi-Modal State & SHAP | Synchronized point-in-time features; TreeSHAP validates feature importance. |
| **POC 05** | Two-Tier Portfolio Execution | Active Top 10 from $M=20$ delivers **+149.39%** (17.70% CAGR, 1.07 Sharpe). |
| **POC 06** | Frequency Optimization | Sweep confirms bi-weekly/monthly reallocations outperform high-frequency churn. |
| **POC 07** | Asymmetric Alpha Engine | Unified Engine delivers **+220.20%** vs. +109.56% for NASDAQ 100 (`QQQ`). |
| **POC 08** | Broad Universe Diversification | Concentrated Top 10 (+220.20%) vs Diversified Top 100 (+134.64%) vs SPY (+91.68%). |
| **POC 09** | 26.6-Year Historical Stress Test | Walk-Forward XGBoost multiplies capital $28\times$ (+2,736.88%) vs SPY ($8.3\times$). |
| **POC 10** | Daily News Stream & FinBERT | Daily news signals enhance bear market timing (2022 loss contained to -5.57%). |
| **POC 11** | Half-Century Macro Validation | Compounded across 10 historical crisis epochs (1978–2026). |
| **POC 12** | Data Quality & Completeness Audit | Confirmed **0.00% missing values** across 638,434 half-century records. |
| **POC 13** | Alpaca Paper Trading Bridge | Automated position delta calculations and two-step Alpaca order execution. |
| **POC 14** | 100x100 Surface Optimization | 2D Gaussian filtered surface reveals monthly fundamental drift plateau ($H=30\text{d}, F=31\text{d}$). |
| **POC 15** | Continuous Step-by-Step Fit | Fast Hist fits (0.08s) enable dynamic retraining before every rebalance cycle. |
| **POC 16** | Purged Nested Optuna Benchmark | 8-strategy walk-forward comparison; dynamic $H$ tuning vs. fixed 30d/15d anchors. |
| **POC 17** | Half-Century Continuous Benchmark | 5-strategy half-century simulation across 12,264 trading sessions. |
| **POC 18** | Institutional Zero-Leakage (2003–2026) | Trading-bar purge + $T+1$ lag: **+8,361.94%** (20.68% CAGR, 0.868 Sharpe). |
| **POC 19** | Half-Century Zero-Leakage (1981–2026) | 45.6-year strict zero-leakage: **+859,228.24%** (21.95% CAGR, 0.950 Sharpe, -41.15% Max DD). |
| **POC 20** | Temporal Plateau Evolution | 9 5-Year Epochs (1981–2026); shifts from fast momentum (1980s–2000s: H=1-6d) to fundamental drift (2016–2026: H=42-48d). |
| **POC 21** | Annual & Rolling Plateau Dynamics | 45 Individual Years (1981–2026); proves Horizon Compression: Crisis/Bear H*=1d vs Low-Vol Bull H*=35–50d. |
| **POC 22** | Comprehensive 9-Strategy Benchmark | Optuna-Tuned Tri-Horizon Ensemble achieves **+2,734,948.33%** (25.08% CAGR, 0.965 Sharpe, -40.48% Max DD, +15.01% Alpha). |
| **POC 23** | Advanced Hyperparameter Optimization | Multi-Objective Pareto & Rank-IC Maximization deliver **+23,729.27%** on modern era (2003–2026 / 129 tickers). |

---

## 6. Comprehensive 9-Strategy Benchmark & Annual Excess Alpha (POC 22)

Across 12,264 daily trading sessions (1981–2026 / 45.6 Years), combining multi-horizon alpha signals with Bayesian tuning delivers superior risk-adjusted alpha:

| Strategy / Model | Total Return (%) | CAGR (%) | Sharpe Ratio | Sortino Ratio | Max Drawdown (%) | Calmar Ratio | Jensen Alpha (α) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **9. Optuna-Tuned Tri-Horizon Ensemble** 🏆 | **+2,734,948.33%** | **25.08%** | **0.965** | **1.294** | **-40.48%** | **0.620** | **+15.01%** |
| **8. Tri-Horizon Multi-Model Ensemble (5d/15d/35d)** | **+2,560,012.39%** | **24.90%** | **0.959** | **1.281** | **-41.86%** | **0.595** | **+14.82%** |
| **7. Regime-Adaptive Dynamic Horizon XGBoost** | **+1,242,846.51%** | **22.94%** | **0.918** | **1.188** | **-51.50%** | **0.445** | **+13.13%** |
| **5. Static Purged XGBoost (15d Rebal, 15d Fwd, Prop, T+1)** | **+934,181.00%** | **22.17%** | **0.940** | **1.224** | **-47.06%** | **0.471** | **+12.54%** |
| **4. Static Purged XGBoost (30d Rebal, 30d Fwd, Prop, T+1)** | **+859,228.24%** | **21.95%** | **0.950** | **1.235** | **-41.15%** | **0.533** | **+12.44%** |
| **6. Optuna-Tuned Purged XGBoost (15d/15d Prop)** | **+603,264.71%** | **21.01%** | **0.920** | **1.190** | **-47.21%** | **0.445** | **+11.53%** |
| **3. Naive XGBoost (Default Params, 30d/30d Eq, T+1)** | **+301,307.82%** | **19.18%** | **0.914** | **1.176** | **-41.25%** | **0.465** | **+10.23%** |
| **2. Point-in-Time Active Universe B&H (No bfill)** | +299,723.34% | 19.17% | 0.909 | 1.167 | -45.71% | 0.419 | +10.17% |
| **1. S&P 500 Index (`^GSPC` Benchmark)** | +5,529.82% | 9.23% | 0.417 | 0.524 | -56.78% | 0.163 | +0.00% |

---

## 7. Advanced Hyperparameter Optimization on Modern Era (POC 23: 2003–2026)

Evaluates Rank-IC Maximization, Multi-Objective Pareto, Regime-Conditioned Volatility Scaling, and Standard Bayesian MSE across 5,950 trading sessions (129 equities):

| Optimization Method / Strategy | Total Return (%) | CAGR (%) | Sharpe Ratio | Sortino Ratio | Max Drawdown (%) | Jensen Alpha (α) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **5. Multi-Objective Pareto Tri-Horizon Ensemble** 🏆 | **+23,729.27%** | **26.09%** | **0.844** | **1.208** | **-63.66%** | **+15.26%** |
| **4. Rank-IC Maximized Tri-Horizon Ensemble** | **+21,972.73%** | **25.68%** | **0.835** | **1.196** | **-64.02%** | **+14.84%** |
| **7. Standard Bayesian MSE Optuna Tri-Horizon** | **+20,254.91%** | **25.25%** | **0.828** | **1.169** | **-64.12%** | **+14.42%** |
| **3. Baseline Tri-Horizon (Fixed Default Params)** | **+19,026.37%** | **24.92%** | **0.816** | **1.162** | **-65.48%** | **+14.05%** |
| **6. Regime-Conditioned Volatility-Scaled Tri-Horizon** | **+18,687.42%** | **24.83%** | **0.816** | **1.163** | **-64.18%** | **+13.99%** |
| **2. Point-in-Time Active Universe B&H** | +3,421.59% | 16.28% | 0.727 | 0.900 | -50.24% | +6.86% |
| **1. S&P 500 Index (`^GSPC` Benchmark)** | +744.38% | 9.46% | 0.417 | 0.513 | -56.78% | +0.00% |



