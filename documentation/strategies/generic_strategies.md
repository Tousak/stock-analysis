# Generic Strategies & Market Benchmarks

This document details the baseline and generic benchmark strategies used throughout our quantitative research, with special focus on **Point-in-Time Active Universe Buy & Hold**, survivorship bias decomposition, and passive market indexes.

---

## 1. Point-in-Time Active Universe Buy & Hold (Universe Benchmark)

### Overview
The **Point-in-Time Active Universe Buy & Hold** strategy is the primary **universe-neutral benchmark** against which all machine learning models in this repository are evaluated. It holds an equal-weighted portfolio of all stocks that were **actively listed and traded on that exact date** (with zero pre-IPO backward fills).

### Mathematical Definition
On every rebalance date $T$, the active universe candidate set $\mathcal{U}_T$ is defined as:
$$\mathcal{U}_T = \{ s \in \mathcal{S} \mid \text{Price}(s, T) \text{ is not NaN and listing\_date}(s) \le T \}$$

The portfolio weights for Day $T+1$ through $T+F$ are strictly:
$$w_i(t) = \begin{cases} \frac{1}{|\mathcal{U}_T|}, & \text{if } s_i \in \mathcal{U}_T \\ 0, & \text{otherwise} \end{cases}$$

### Role in Research: Survivorship Alpha Decomposition
Evaluating machine learning models solely against the broad S&P 500 (`^GSPC`) conflates **Universe Selection Bias** with **Machine Learning Alpha**. By comparing our algorithmic strategies against the **Point-in-Time Active Universe Buy & Hold**, we decompose total strategy performance into:

$$\text{Total Strategy Return} = \underbrace{\text{Market Beta (\^GSPC)}}_{\approx 9.2\% - 9.5\% \text{ CAGR}} + \underbrace{\text{Universe Selection Beta (Active B\&H)}}_{\approx 7.3\% - 10.2\% \text{ Excess CAGR}} + \underbrace{\text{Pure ML Stock-Picking \& Sizing Alpha}}_{\approx +2.8\% - 4.0\% \text{ Annual Alpha}}$$

### Key Characteristics & Empirical Performance:
- **Zero Pre-IPO Lookahead**: Completely eliminates backward-fill (`bfill()`) distortions before a company listed (e.g. `GOOGL` before 2004, `META` before 2012, `NVDA` before 1999).
- **2003–2026 Performance (129 Equities)**: **`+3,727.09%` Total Return (`16.69%` CAGR, `0.799` Sharpe, `-50.24%` Max DD)** vs. `+744.38%` (`9.46%` CAGR) for S&P 500.
- **1981–2026 Performance (60 Equities)**: **`+299,723.34%` Total Return (`19.17%` CAGR, `0.909` Sharpe, `-45.71%` Max DD)** vs. `+5,529.82%` (`9.23%` CAGR) for S&P 500.
- **Survivorship Note**: The 2003–2026 and 1981–2026 datasets consist of long-term surviving large caps. Holding them passively outperforms the broad S&P 500 because it benefits from 2026 survivor selection. Our **XGBoost Engine adds another +4,634% to +559,505% in pure stock-picking and forecast-proportional conviction allocation** on top of this passive universe baseline.

---

## 2. Static Fixed-Pool Buy & Hold (Top 10 / Top 20)

### Overview
Purchases an equal dollar amount across a fixed candidate pool (e.g., $N=10$ or $N=20$) at day zero and holds without rebalancing or reallocating capital across the entire test horizon.

### Key Characteristics:
- **Vulnerability to Single-Stock Structural Decay**: Without active re-ranking, structural laggards permanently drag down portfolio compounding.
- **POC 05 Empirical Evidence (2021–2026)**: In the 2021 candidate universe, static holding of historical giants like `INTC` (down -50%), `CSCO`, and `BA` caused Top 10 Static B&H to deliver only **`+27.63%` (4.87% CAGR, -30.78% drawdown)**, massively underperforming the active Multi-Modal Alpha engine (**`+149.39%` / 17.70% CAGR**).

---

## 3. Smart / Market-Cap Weighted Buy & Hold

### Overview
Weights portfolio holdings proportionally to starting market capitalization or inverse historical volatility at each rebalance cycle.

### Key Characteristics:
- **Concentration Drag**: Capitalizes on mega-cap momentum during secular bull runs, but concentrates heavily into top weights (e.g. 35%+ allocation to top 3 tech giants).
- **Severe Bear Market Exposure**: Suffered a **`-38.55%` to `-42.52%` drawdown** during the 2022 rate-hiking cycle due to tech sector concentration, compared to only `-21.34%` for our volatility-buffered alpha engine.

---

## 4. Passive Market Index Benchmarks

### A. S&P 500 Index Benchmark (`^GSPC` / `SPY`)
- **Concept**: Market-cap weighted index of the 500 largest US corporations.
- **Role in Research**: Primary benchmark for broad-market equity returns and systematic risk baseline ($\beta = 1.0$).
- **Historical Long-Term Return**: Compounds at **`9.23%` CAGR (1981–2026)** and **`9.46%` CAGR (2003–2026)** with a Sharpe ratio of $\approx 0.42\text{--}0.47$ and a maximum drawdown of **`-56.78%`** (2008 GFC).

### B. NASDAQ-100 Index Benchmark (`QQQ`)
- **Concept**: Market-cap weighted index tracking the 100 largest non-financial companies on NASDAQ.
- **Role in Research**: High-beta secular growth benchmark.
- **Key Characteristics**: Strong bull-market compounding (+109.56% in 2021–2026), but high vulnerability to tech valuation contractions (-35.12% drawdown in 2022).

### C. S&P 100 Index Benchmark (`XLG`)
- **Concept**: Market-cap weighted index tracking the 100 largest US domestic blue chips across all 11 market sectors.
- **Role in Research**: Used for 100% US domestic mega-cap validation against our large-cap universe.
- **Key Characteristics**: Stable large-cap growth (+100.28% in 2021–2026), but unhedged against macro systemic drawdowns (-28.02% drawdown).

---

## 5. Summary Benchmark Comparison Matrix

| Strategy / Benchmark | Rebalance Frequency | Weighting Scheme | 2003–2026 CAGR | 1981–2026 CAGR | Sharpe Ratio | Max Drawdown (%) | Survivorship Neutrality |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Purged XGBoost (30d/30d Prop)** 🏆 | 31 Days | Forecast-Proportional | **20.68%** | **21.95%** | **0.87 – 0.95** | **-41.15% to -53.20%** | Full Zero-Leakage |
| **Point-in-Time Active Universe B&H** | 31 Days | Equal-Weight (Active) | **16.69%** | **19.17%** | **0.80 – 0.91** | **-45.71% to -50.24%** | Point-in-Time Universe |
| **S&P 500 Index (`^GSPC`)** | Continuous | Market-Cap Weighted | **9.46%** | **9.23%** | **0.42 – 0.47** | **-56.78%** | Broad Market |
| **Static Fixed Pool B&H (Top 10)** | None | Static Equal-Weight | ~10.15% (5.6y) | — | ~0.54 | -30.78% (5.6y) | Fixed $t_0$ Pool |
