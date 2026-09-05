# Roadmap: Congressional STOCK Act Disclosures & Committee Alpha Extraction

**Notebook Target**: `research/notebooks/02_political_legislative_intelligence.ipynb`  
**Research Paper Pillar**: Section 2 — *Political Trade Intelligence and Legislative Signal Extraction*  
**Primary References**: Stop Trading on Congressional Knowledge (STOCK) Act 2012; TraderCongress empirical studies (2025/2026)

---

## 1. Business & Quantitative Context

### The Core Problem
Under the STOCK Act, US Senators and Representatives must disclose equity transactions within 30 to 45 days via Periodic Transaction Reports (PTRs). However, in practice, there is a **median filing delay of 28 days and a mean delay of 52 days**. Naive copy-trading on raw public disclosure dates incurs execution drag and buys into exhausted short-term price momentum.

### The Value Proposition (Alpha Thesis)
Longitudinal research demonstrates that legislative insider knowledge has a **multi-month persistence horizon (3 to 12 months)** that generates **4% to 8% annualized excess return** over broad market benchmarks when filtered systematically:
1. **Direction & Size Filter**: Congressional sales are noise (liquidity/political optics). Open-market purchases $> \$50\text{k}$ (and especially $> \$100\text{k}$) represent high-conviction directional bets.
2. **Committee Jurisdiction Overlap**: Lawmakers transacting in sectors under their direct legislative purview (e.g. Armed Services $\rightarrow$ Defense, Energy & Commerce $\rightarrow$ Utilities/Healthcare) generate 5% to 7% annualized alpha.
3. **Late-Disclosure Momentum Guard**: If a stock has already appreciated by $> 20\%$ between lawmaker trade date and public disclosure date, the signal is invalidated to prevent buying at top of range.

---

## 2. Technical Implementation & Workflow

```
       ┌────────────────────────┐
       │   Congressional PTRs   │
       │   (House & Senate APIs)│
       └───────────┬────────────┘
                   │
                   ▼
       ┌────────────────────────┐
       │ Structural Filtering:  │
       │ • Purchases Only       │
       │ • Size > $50k          │
       └───────────┬────────────┘
                   │
                   ▼
       ┌────────────────────────┐
       │ Committee Cross-Ref:   │
       │ Member Committee Data  │
       │ ↔ Ticker GICS Sector   │
       └───────────┬────────────┘
                   │
                   ▼
       ┌────────────────────────┐
       │ Exhaustion Filter:     │
       │ If Return(Trade→Pub)   │
       │ > +20% → Invalidate    │
       └───────────┬────────────┘
                   │
                   ▼
       ┌────────────────────────┐
       │ Multi-Horizon Tracking │
       │ (30d, 90d, 180d, 360d) │
       └────────────────────────┘
```

### Step 1: Legislative Disclosure Ingestion
- Ingest historical House and Senate PTR disclosure tables (via public House/Senate Clerk disclosure APIs or open repositories such as CapitolTrades/HouseStockWatcher).
- Standardize schema: `politician_name`, `chamber` (House/Senate), `party`, `state`, `committee_assignments`, `ticker`, `asset_description`, `transaction_type` (Buy/Sale/Exchange), `amount_bracket` (e.g. $15k-$50k, $50k-$100k, $100k-$250k, $500k+), `trade_date`, `disclosure_date`.

### Step 2: Committee-Jurisdiction Mapping Engine
- Map each lawmaker's active committee assignments to corresponding GICS sectors / industry groups:
  - *Armed Services / Intelligence* $\rightarrow$ Aerospace & Defense (`RTX`, `LMT`, `GD`, `NOC`, `BA`).
  - *Energy & Commerce / Natural Resources* $\rightarrow$ Energy & Healthcare (`XOM`, `CVX`, `UNH`, `LLY`).
  - *Financial Services / Banking* $\rightarrow$ Financials (`JPM`, `BAC`, `GS`, `MS`, `V`, `MA`).
  - *Science, Space & Technology* $\rightarrow$ Semiconductors & Cloud (`NVDA`, `MSFT`, `GOOGL`, `AMD`).
- Construct binary indicator: $\text{CommitteeJurisdictionFlag}_{i,t} \in \{0, 1\}$.

### Step 3: Anti-Chasing Momentum Filter & Portfolio Tracking
- Compute price drift between transaction date $t_{\text{trade}}$ and public disclosure date $t_{\text{pub}}$:
  $$\Delta P_{\text{lag}} = \frac{P(t_{\text{pub}}) - P(t_{\text{trade}})}{P(t_{\text{trade}})}$$
  If $\Delta P_{\text{lag}} > 0.20$, mark signal as `INVALID_EXHAUSTED`.
- Measure calendar-time holding period returns: 3 months, 6 months, and 12 months post-$t_{\text{pub}}$ against SPY.

---

## 3. Deliverables & Evaluation Metrics

| Metric | Target / Benchmark | Purpose |
| :--- | :--- | :--- |
| **Filtered Annualized Alpha** | $\ge +4.0\%$ vs. SPY | Proves filtered disclosures beat naive copy-trading |
| **Committee Overlap Win Rate** | $\ge 60\%$ (6-month horizon) | Validates information advantage from legislative oversight |
| **Exhaustion Guard Efficacy** | $> 1.5\%$ drawdown reduction | Confirms protection against chasing late filings |
| **Output File** | `data/fetched/political_signals_poc.xlsx` | Feature table for multi-modal model training |

---

## 4. Go / No-Go Decision Gate
- **Proceed**: If filtered purchases (> $50k + committee match - momentum filter) produce positive annualized alpha ($\ge +4\%$) over 3–12 month holding periods.
- **Halt/Refine**: If public legislative disclosures after 2024 lack sufficient trade frequency or show statistically insignificant alpha, deprioritize or use solely as a secondary confirmation flag.
