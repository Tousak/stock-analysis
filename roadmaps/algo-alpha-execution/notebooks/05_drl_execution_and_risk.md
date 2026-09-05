# Roadmap: Alpha-Reward Reinforcement Learning & Risk-Constrained Execution Simulation

**Notebook Target**: `research/notebooks/05_drl_execution_and_risk.ipynb`  
**Research Paper Pillar**: Section 4 & 5 — *Reinforcement Learning, Alpha Rewards, and Local Execution Architecture*  
**Primary References**: ArXiv 2607.16028 (2026), *Sentiment-Augmented Deep Reinforcement Learning for Trading*; TraderCongress Execution Framework (2026)

---

## 1. Business & Quantitative Context

### The Core Problem
Generating directional predictions is only half the battle. A high-accuracy model will still lose money if:
1. Position sizing is naive (e.g. concentrated over-allocation in volatile micro-caps).
2. It suffers from macro beta drift (making money solely when the broader market goes up, but failing to produce true alpha during downturns).
3. It lacks explicit risk limits (no hard stop-losses, holding through sentiment decay, or incurring high slippage).

### The Value Proposition (Alpha Thesis)
1. **Alpha-Based Reward Formulation**:
   $$R_t = r_{i,t} - r_{\text{SPY},t}$$
   Forces Deep Reinforcement Learning (DRL) policies (DDPG / PPO) or dynamic allocation rules to optimize purely for idiosyncratic benchmark outperformance rather than riding market beta.
2. **Institutional Risk Management Engine**:
   - **Position Sizing**: Strictly capped at 2% to 5% of portfolio equity.
   - **Stop-Loss Safeguard**: Hard stop-loss bounds at 10% to 15% below entry price.
   - **Sentiment Decay Exit**: Automatic position liquidation or reduction as continuous sentiment $S(t)$ decays back to neutral ($S_{\text{baseline}} \approx 0$).
   - **Transaction Cost & Slippage**: Realistic modeling (10 to 20 bps per round trip).

---

## 2. Technical Implementation & Workflow

```
       ┌──────────────────────────┐
       │ Multi-Modal State Vector │
       │ (Predictions + Volatility)│
       └────────────┬─────────────┘
                    │
                    ▼
       ┌──────────────────────────┐
       │ Policy / Sizing Engine:  │
       │ • DRL (DDPG/PPO) or      │
       │ • Risk-Parity Allocation │
       └────────────┬─────────────┘
                    │
                    ▼
       ┌──────────────────────────┐
       │ Hard Risk Guardrails:    │
       │ • Max 2-5% per Position  │
       │ • 10-15% Hard Stop-Loss  │
       │ • Sentiment Decay Exit   │
       └────────────┬─────────────┘
                    │
                    ▼
       ┌──────────────────────────┐
       │ Realistic Market Sim:    │
       │ • Slippage + 15bps Fees  │
       │ • Market-On-Close (MOC)  │
       └────────────┬─────────────┘
                    │
                    ▼
       ┌──────────────────────────┐
       │ Performance Analytics:   │
       │ Sharpe, Sortino, Max DD, │
       │ Beta, Jensen's Alpha     │
       └──────────────────────────┘
```

### Step 1: Environment & State-Action Definition
- Implement a lightweight trading simulation environment:
  - **State $s_t$**: Multi-modal predictions vector, current portfolio weights, EWMA volatility, 14-day RSI, and active trade holding durations.
  - **Action $a_t$**: Target portfolio allocation weights $w_t \in [0, 0.05]$ for all universe assets, with cash buffer $\ge 0$.
  - **Reward Function**: Alpha reward $R_t = \sum_i w_{i,t} r_{i,t} - r_{\text{SPY},t} - \text{Penalty}_{\text{turnover}}$.

### Step 2: Policy Optimization & Benchmark Allocation Rules
- Compare two execution engines:
  1. **Rule-Based Dynamic Sizing (Baseline)**: Inverse-volatility weighted allocation of top-N positive predicted assets, subject to 5% individual max cap and stop-loss rules.
  2. **DRL Policy Agent (Advanced)**: PPO / DDPG continuous actor-critic network trained over historical walk-forward windows.

### Step 3: Comprehensive Stress Testing & Risk Analytics
- Backtest over diverse market regimes (2020 COVID shock, 2022 rate-hike bear market, 2023–2026 bull/swing regimes).
- Calculate full performance suite:
  - **Sharpe Ratio** & **Sortino Ratio**
  - **Maximum Drawdown (MDD)** and **Calmar Ratio**
  - **Annualized Jensen's Alpha ($\alpha$)** and **Market Beta ($\beta$)**
  - **Win Rate, Profit Factor, and Average Trade Duration**

---

## 3. Deliverables & Evaluation Metrics

| Metric | Target / Benchmark | Purpose |
| :--- | :--- | :--- |
| **Strategy Sharpe Ratio** | $\ge 2.0$ (vs. S&P 500 $\approx 1.1$) | Proves institutional-grade risk-adjusted return |
| **Max Drawdown (MDD)** | $< -18\%$ (vs. S&P 500 $\sim -25\%$) | Confirms stop-loss and decay rules limit downside tail risk |
| **Market Beta ($\beta$)** | $0.40 \le \beta \le 0.75$ | Confirms low systemic market dependency / true alpha |
| **Output File** | `data/fetched/final_backtest_simulation_poc.xlsx` | Daily portfolio equity curve and trade logs |

---

## 4. Go / No-Go Decision Gate
- **Production Green Light**: If the simulated strategy beats SPY on both Total Return and Sharpe ratio while maintaining Max Drawdown $< 20\%$ after realistic transaction costs.
- **Merge into App**: Once approved, modularize the tested components into `src/` ([`data_loader.py`](file:///c:/Users/honza/Desktop/projects/stock-analysis/src/data_loader.py), [`processor.py`](file:///c:/Users/honza/Desktop/projects/stock-analysis/src/processor.py), [`feature_eng.py`](file:///c:/Users/honza/Desktop/projects/stock-analysis/src/feature_eng.py), [`model.py`](file:///c:/Users/honza/Desktop/projects/stock-analysis/src/model.py), [`backtester.py`](file:///c:/Users/honza/Desktop/projects/stock-analysis/src/backtester.py)) and update the Streamlit UI.
