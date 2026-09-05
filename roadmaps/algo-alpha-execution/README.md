# Algorithmic Alpha Execution: POC Master Roadmap

This roadmap orchestrates the proof-of-concept (POC) validation strategy for the research framework defined in [`documentation/papers/Daily Algorithmic Trading Research Plan.md`](file:///c:/Users/honza/Desktop/projects/stock-analysis/documentation/papers/Daily%20Algorithmic%20Trading%20Research%20Plan.md).

The goal is to rigorously validate each individual alpha driver in an isolated notebook before committing code changes to the production backend (`src/`) or Streamlit frontend (`pages/`).

---

## 🗺️ Notebook Roadmaps Index

| Notebook | Title | Core Focus & Alpha Hypothesis | Roadmap Document |
| :--- | :--- | :--- | :--- |
| **01** | **SEC Form 4 Insider Alpha** | Cohen-Malloy routine vs. opportunistic insider classification, cluster buys, local director asymmetry (+0.82% to +1.80%/mo alpha). | [01_corporate_insider_signals.md](file:///c:/Users/honza/Desktop/projects/stock-analysis/roadmaps/algo-alpha-execution.md/notebooks/01_corporate_insider_signals.md) |
| **02** | **Congressional Trade Intelligence** | STOCK Act PTR filings, purchases >$50k, committee-jurisdiction matching, and late-disclosure exhaustion filter (+4% to +8% annualized alpha). | [02_political_legislative_intelligence.md](file:///c:/Users/honza/Desktop/projects/stock-analysis/roadmaps/algo-alpha-execution.md/notebooks/02_political_legislative_intelligence.md) |
| **03** | **NLP Sentiment Decay Dynamics** | FinBERT transformer inference on news streams, continuous exponential decay $S(t)$, and causal EMA smoothing for 1–5 day swing signals. | [03_nlp_sentiment_decay_dynamics.md](file:///c:/Users/honza/Desktop/projects/stock-analysis/roadmaps/algo-alpha-execution.md/notebooks/03_nlp_sentiment_decay_dynamics.md) |
| **04** | **Multi-Modal Synthesis & SHAP** | Unified feature matrix (fundamentals + insider + political + sentiment + EWMA volatility), purged walk-forward XGBoost, and SHAP explainability. | [04_multimodal_state_vector_shap.md](file:///c:/Users/honza/Desktop/projects/stock-analysis/roadmaps/algo-alpha-execution.md/notebooks/04_multimodal_state_vector_shap.md) |
| **05** | **DRL Execution & Risk Simulation** | Alpha reward formulation ($R_t = r_i - r_{\text{SPY}}$), 2-5% position caps, 10-15% hard stop-losses, sentiment decay exits, and MOC execution. | [05_drl_execution_and_risk.md](file:///c:/Users/honza/Desktop/projects/stock-analysis/roadmaps/algo-alpha-execution.md/notebooks/05_drl_execution_and_risk.md) |

---

## 🔄 Phased Execution & Integration Pipeline

```
  ┌─────────────────────────────────────────────────────────────┐
  │                 PHASE 1: SIGNAL ISOLATION                   │
  │  Notebook 01 (Form 4)   Notebook 02 (STOCK Act)   NB 03 (NLP)│
  └──────────────────────────────┬──────────────────────────────┘
                                 │
                                 ▼
  ┌─────────────────────────────────────────────────────────────┐
  │                 PHASE 2: MULTI-MODAL ML                     │
  │  Notebook 04: Purged Walk-Forward XGBoost + SHAP Explain    │
  └──────────────────────────────┬──────────────────────────────┘
                                 │
                                 ▼
  ┌─────────────────────────────────────────────────────────────┐
  │                 PHASE 3: SIMULATION & RISK                  │
  │  Notebook 05: Risk-Constrained Alpha Backtest (Stop-Losses) │
  └──────────────────────────────┬──────────────────────────────┘
                                 │ (Validation Gates Passed)
                                 ▼
  ┌─────────────────────────────────────────────────────────────┐
  │                 PHASE 4: PRODUCTION MERGE                   │
  │  • Modularize into src/ (loaders, processors, features, ML) │
  │  • Expose in Streamlit UI (pages/ & app.py)                │
  └─────────────────────────────────────────────────────────────┘
```

---

## 🎯 Global Go / No-Go Decision Criteria

To qualify for integration into the main application, the simulated strategy across Notebooks 01–05 must meet all of the following baseline hurdles:

1. **Portfolio Sharpe Ratio** $\ge 2.0$ (vs. S&P 500 $\approx 1.1$).
2. **Max Drawdown** $< -18\%$ during the 2021–2026 backtest window.
3. **Purged CV MSE Reduction** $\ge 15\%$ over the existing fundamental-only baseline.
4. **SHAP Confirmation**: At least two alternative signal categories (insider or sentiment decay) must rank in the top 5 global feature importance.
