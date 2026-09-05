# Roadmap: High-Frequency FinBERT Sentiment & Exponential Signal Decay Dynamics

**Notebook Target**: `research/notebooks/03_nlp_sentiment_decay_dynamics.ipynb`  
**Research Paper Pillar**: Section 1 — *Contextual Financial NLP and Sentiment Signal Engineering*  
**Primary References**: ArXiv 2605.30652 (2026), *Bridging the Gap Between Natural Language and Market Dynamics*; ArXiv 2603.23568 (2026), *Causal Reconstruction of Sentiment Signals from Sparse News Data*

---

## 1. Business & Quantitative Context

### The Core Problem
Financial markets process textual information rapidly. While 10-Q MD&A filings capture quarterly macro trends, swing trading (1-to-5-day holding periods) requires streaming news headlines. However, unstructured news feeds suffer from **temporal sparsity, irregular arrival intervals, and rapid sentiment decay**. A naive static sentiment score becomes stale within hours to days, causing models to trade on obsolete news sentiment.

### The Value Proposition (Alpha Thesis)
1. **Domain-Specific Transformer (FinBERT)**: Significantly outperforms general LLMs and Loughran-McDonald lexicons in financial nomenclature parsing without heavy compute latency.
2. **Continuous Exponential Sentiment Decay Function**:
   $$S_i(t) = S_{0,i} \cdot e^{-\lambda_i (t - t_0)} + S_{\text{baseline}} \cdot \left(1 - e^{-\lambda_i (t - t_0)}\right)$$
   Where $S_{0,i}$ is initial news sentiment intensity, $S_{\text{baseline}} = 0.0$ (neutral on [-1, 1] scale), and $\lambda = \frac{\ln(2)}{\tau_{1/2}}$ represents the asset-specific decay rate (half-life $\tau_{1/2} \in [1, 5]$ days).
3. **Causal Temporal Smoothing**: Exponential Moving Averages (EMA) and Kalman filtering project discrete news events onto continuous daily trading grids, filtering out microstructural noise while preserving regime shifts.

---

## 2. Technical Implementation & Workflow

```
       ┌────────────────────────┐
       │ Streaming News & RSS   │
       │ (Tickers: NVDA, AAPL...)│
       └───────────┬────────────┘
                   │
                   ▼
       ┌────────────────────────┐
       │ FinBERT Inference:     │
       │ P(pos), P(neg), P(neu) │
       │ Scalar Score: pos - neg│
       └───────────┬────────────┘
                   │
                   ▼
       ┌────────────────────────┐
       │ Exponential Decay      │
       │ Transform:             │
       │ S(t) with Half-Life τ  │
       └───────────┬────────────┘
                   │
                   ▼
       ┌────────────────────────┐
       │ Causal EMA Smoothing:  │
       │ Filter Noise & Map to  │
       │ Daily Trading Grid     │
       └───────────┬────────────┘
                   │
                   ▼
       ┌────────────────────────┐
       │ Return Correlation &   │
       │ Rank Information Coeff │
       │ (1-day & 5-day IC)     │
       └────────────────────────┘
```

### Step 1: High-Frequency News Ingestion & Scoring
- Ingest historical news headlines and summaries for selected focus tickers (e.g. `AAPL`, `NVDA`, `TSLA`, `MSFT`, `AMZN`) across a 1–2 year historical period.
- Run batch inference using HuggingFace `ProsusAI/finbert` (or class-weighted/market fine-tuned variants).
- Extract probability vectors: $[P(\text{pos}), P(\text{neg}), P(\text{neu})]$ and raw sentiment intensity $S_{\text{raw}} = P(\text{pos}) - P(\text{neg}) \in [-1.0, 1.0]$.

### Step 2: Continuous Decay Grid & Causal Smoothing
- Map discrete timestamped articles $k$ at time $t_k$ to a continuous daily market timestamp $t$.
- Implement the recursive exponential decay equation across non-event days:
  $$S(t) = S(t-1) \cdot e^{-\lambda \Delta t} + \sum_{k \in \text{Day } t} S_{\text{raw}, k}$$
- Benchmark multiple half-life decay parameters: $\tau_{1/2} \in \{1\text{ day}, 2\text{ days}, 3\text{ days}, 5\text{ days}\}$.
- Apply causal smoothing:
  $$\tilde{S}(t) = \alpha \cdot S(t) + (1-\alpha) \cdot \tilde{S}(t-1)$$

### Step 3: Information Coefficient & Predictive Validity
- Calculate **Rank Information Coefficient (Rank IC)** against forward 1-day ($r_{t+1}$) and 5-day ($r_{t+1:t+5}$) returns:
  $$\text{IC}_t = \text{SpearmanCorr}(\tilde{S}_t, r_{t+1})$$
- Test a simple sign-based long/short strategy:
  $$\text{Position}_t = \begin{cases} +1 & \text{if } \tilde{S}_t > +\theta \\ -1 & \text{if } \tilde{S}_t < -\theta \\ 0 & \text{otherwise} \end{cases}$$

---

## 3. Deliverables & Evaluation Metrics

| Metric | Target / Benchmark | Purpose |
| :--- | :--- | :--- |
| **Rank Information Coefficient (1d)** | $\text{Rank IC} \ge 0.015$ | Proves linear/monotonic rank correlation with forward price returns |
| **Rank IC Information Ratio (ICIR)** | $\text{ICIR} \ge 0.50$ | Validates statistical consistency of the sentiment signal over time |
| **Decayed vs. Raw Sentiment Spread** | $\ge +25\%$ Sharpe increase | Confirms that exponential decay removes obsolete headline noise |
| **Output File** | `data/fetched/decayed_sentiment_poc.xlsx` | Smoothed daily sentiment feature series for training |

---

## 4. Go / No-Go Decision Gate
- **Proceed**: If exponentially decayed sentiment achieves $\text{Rank IC} \ge 0.015$ and improves 1-day/5-day return prediction over un-decayed raw sentiment.
- **Iterate**: If general FinBERT has low signal-to-noise ratio on headline-only feeds, test market-reaction fine-tuning or add headline relevance filtering.
