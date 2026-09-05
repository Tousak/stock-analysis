# Roadmap: Multi-Modal Feature Synthesis, EWMA Volatility & SHAP Attribution

**Notebook Target**: `research/notebooks/04_multimodal_state_vector_shap.ipynb`  
**Research Paper Pillar**: Section 4 — *Multi-Modal Strategy Synthesis, Machine Learning Integration, and Local Execution Architecture*  
**Primary References**: Lundberg & Lee (SHAP); ResearchGate 394994117 (2026), *Stock Price Prediction Using FinBERT-Enhanced Sentiment with SHAP Explainability*

---

## 1. Business & Quantitative Context

### The Core Problem
Individual alpha signals (insider buys, political disclosures, sentiment spikes, and technical indicators) have distinct decay horizons and perform differently across market volatility regimes. Training a model on raw concatenated features without rigorous regime conditioning or interpretability can lead to overfitting, black-box failure, and beta drift during market sell-offs.

### The Value Proposition (Alpha Thesis)
1. **Multi-Domain State Matrix**: Unifies low-frequency fundamentals (growth, margins), medium-frequency alternative data (opportunistic insider cluster buys, political committee conviction), and high-frequency market data (decayed sentiment, Wilder RSI, MACD).
2. **EWMA Dynamic Volatility Conditioning**:
   $$\sigma_t^2 = \lambda \sigma_{t-1}^2 + (1-\lambda) r_t^2, \quad \lambda = 1 - \frac{2}{N+1}$$
   Adapts model weighting dynamically to high vs. low volatility regimes.
3. **SHAP (SHapley Additive exPlanations) Attribution**: Disentangles the exact marginal contribution of each feature category, ensuring the model's predictions are fundamentally driven by true idiosyncratic alpha rather than transient noise.

---

## 2. Technical Implementation & Workflow

```
 ┌────────────────┐ ┌────────────────┐ ┌────────────────┐ ┌────────────────┐
 │  Fundamentals  │ │ Form 4 Insider │ │ STOCK Act PTRs │ │ Decayed FinBERT│
 │ (Growth/Margin)│ │(Opportunistic) │ │(Committee Buy) │ │(Daily Smoothed)│
 └───────┬────────┘ └───────┬────────┘ └───────┬────────┘ └───────┬────────┘
         │                  │                  │                  │
         └──────────────────┼──────────────────┼──────────────────┘
                            ▼
              ┌───────────────────────────┐
              │ Unified Multi-Modal Matrix│
              │ + EWMA Volatility + TA    │
              └─────────────┬─────────────┘
                            │
                            ▼
              ┌───────────────────────────┐
              │ Purged Walk-Forward CV    │
              │ XGBoost Regressor         │
              └─────────────┬─────────────┘
                            │
                            ▼
              ┌───────────────────────────┐
              │ SHAP Explainability &     │
              │ Feature Attribution       │
              └─────────────┬─────────────┘
                            │
                            ▼
              ┌───────────────────────────┐
              │ Feature Ablation Study    │
              │ (Baseline vs. Full Model) │
              └───────────────────────────┘
```

### Step 1: Multi-Modal Feature Alignment
- Build a point-in-time merged daily feature matrix with backward `merge_asof` joins:
  - **Fundamental**: `revenue_growth`, `net_margin`.
  - **Insider**: `is_opportunistic_buy`, `cluster_count_10d`, `insider_seniority_index`.
  - **Political**: `congress_buy_flag`, `committee_match_flag`, `log_transaction_value`.
  - **Sentiment**: `decayed_sentiment_score`, `sentiment_ema_delta`.
  - **Technical**: `rsi_14`, `macd`, `ewma_volatility_20d`, `volume_ratio_5d_20d`.
- Define target: Forward $k$-day return $r_{t, t+k}$ (e.g. $k=5$ for swing or $k=30$ for intermediate).

### Step 2: Purged Walk-Forward Model Training
- Train `XGBRegressor` using chronological walk-forward splits.
- Enforce strict lookahead purging: remove any training observation where event horizon overlaps with the decision date.
- Perform Optuna hyperparameter optimization strictly inside training folds (80/20 chronological split).

### Step 3: SHAP Explainability & Feature Ablation
- Compute TreeSHAP values (`shap.TreeExplainer(model)`) for all test predictions.
- Generate:
  - **SHAP Global Summary Bar/Beeswarm Plots**: Rank top features by mean absolute SHAP value $|\text{SHAP}|$.
  - **Regime Dependence Interaction Plots**: Plot SHAP interaction between `decayed_sentiment` / `insider_buy` and `ewma_volatility`.
- Conduct **Feature Ablation Experiments**:
  1. *Model A (Baseline)*: Fundamentals + Basic TA only (Current App baseline).
  2. *Model B*: Baseline + Form 4 Insider features.
  3. *Model C*: Baseline + Decayed Sentiment features.
  4. *Model D (Full Multi-Modal)*: All features combined.

---

## 3. Deliverables & Evaluation Metrics

| Metric | Target / Benchmark | Purpose |
| :--- | :--- | :--- |
| **Purged CV MSE Reduction** | $\ge 15\%$ reduction vs. baseline | Proves alternative features improve forecasting accuracy |
| **SHAP Top-5 Ranking** | $\ge 2$ alternative features in Top 5 | Validates that model relies on proposed alpha signals |
| **Ablation Sharpe Spread** | Model D Sharpe $> \text{Model A Sharpe} + 0.40$ | Demonstrates cumulative power of multi-modal fusion |
| **Output File** | `data/fetched/multimodal_predictions_poc.xlsx` | Out-of-sample prediction series for backtesting |

---

## 4. Go / No-Go Decision Gate
- **Proceed**: If Model D achieves lower Purged CV MSE and a significantly higher Sharpe ratio than the fundamental-only baseline, with SHAP confirming non-trivial contribution from insider/sentiment signals.
- **Prune**: If certain features (e.g. naive sentiment or unmatched political trades) show near-zero or negative mean SHAP values, prune them to prevent dimensionality bloat.
