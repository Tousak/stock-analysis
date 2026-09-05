# Roadmap: SEC Form 4 Insider Alpha & Routine vs. Opportunistic Disambiguation

**Notebook Target**: `research/notebooks/01_corporate_insider_signals.ipynb`  
**Research Paper Pillar**: Section 3 — *Corporate Insider Strategy and Form 4 Signal Disambiguation*  
**Primary Reference**: Cohen, Malloy, and Pomorski (2012), *Decoding Inside Information* (NBER/HBS)

---

## 1. Business & Quantitative Context

### The Core Problem
Corporate insiders (executives, board members, 10%+ owners) possess asymmetrical operational information regarding corporate health, order flow, and product margins. While they are mandated to file SEC Form 4 within two business days under Section 16(a), raw aggregate Form 4 filings have low predictive power because ~80% of trades are non-informational (pre-scheduled 10b5-1 plans, tax withholding liquidations, stock option exercises, and routine bonuses).

### The Value Proposition (Alpha Thesis)
By partitioning the insider universe into **routine** and **opportunistic** traders:
1. **Routine trades** (placed in the same calendar month for $\ge 3$ consecutive years) show **zero** return predictability.
2. **Opportunistic trades** (non-periodic open-market purchases) generate statistically significant abnormal returns of **+0.82% per month (value-weighted)** and **+1.80% per month (equal-weighted)**.
3. **Local Non-Senior Insiders** (regional directors, operating officers) exhibit the strongest information asymmetry (+2.10% to +2.45%/month) as they observe ground-level operations without senior C-suite regulatory blackout constraints.
4. **Cluster Buys** (multiple opportunistic insiders purchasing shares within a rolling 10-day window) yield the highest conviction signal (+2.80%+ abnormal return).

---

## 2. Technical Implementation & Workflow

```
       ┌────────────────────────┐
       │   SEC EDGAR Form 4     │
       │   XML / XBRL Ingestion │
       └───────────┬────────────┘
                   │
                   ▼
       ┌────────────────────────┐
       │ Transaction Filter:    │
       │ Code 'P' (Open Market) │
       │ Filter out 10b5-1/Grants│
       └───────────┬────────────┘
                   │
                   ▼
       ┌────────────────────────┐
       │  Trader Classification │
       │  (3-Year Calendar Mask)│
       ├───────────┴────────────┤
       │ Routine     → Discard  │
       │ Opportunistic → Keep   │
       └───────────┬────────────┘
                   │
                   ▼
       ┌────────────────────────┐
       │ Signal Feature Engine: │
       │ • Opportunistic Dummy  │
       │ • Cluster Count (10d)  │
       │ • Insider Seniority    │
       └───────────┬────────────┘
                   │
                   ▼
       ┌────────────────────────┐
       │ Event Study & CAR      │
       │ (10d, 30d, 60d, 90d)   │
       └────────────────────────┘
```

### Step 1: Form 4 Ingestion & Filtering
- Ingest Form 4 historical XML/tables via `edgar-tools` or SEC EDGAR submissions API for the 50 universe tickers over a 5-year historical window.
- Extract fields: `reportingOwnerName`, `isDirector`, `isOfficer`, `isTenPercentOwner`, `officerTitle`, `transactionDate`, `transactionCode`, `sharesTransacted`, `pricePerShare`, `sharesOwnedFollowingTransaction`, `is10b51Plan`.
- Filter strictly for **Open Market Purchases** (`transactionCode == 'P'`). Filter out gifts, stock option exercises (`M`), and sales (`S`).

### Step 2: Routine vs. Opportunistic Disambiguation Algorithm
- Build a historical trade calendar for every unique insider (`reportingOwnerName` + `issuerTicker`):
  $$\text{CalendarMonth}(t) \in \{1, 2, \dots, 12\}$$
- Mark an insider as **Routine** if they transacted in month $M$ in years $Y, Y-1, Y-2$.
- Mark all non-periodic insiders as **Opportunistic**.
- Flag **Cluster Purchases**: Detect whenever $\ge 2$ distinct opportunistic insiders execute open-market buys within any 10-day rolling window for ticker $i$.

### Step 3: Event Study & Cumulative Abnormal Returns (CAR)
- Define event date $t_0 = \text{Form 4 Public Filing Date}$ (or next market open if filed after 16:00 EST).
- Calculate Cumulative Abnormal Returns against S&P 500 benchmark:
  $$\text{AR}_{i,t} = r_{i,t} - r_{\text{SPY},t}, \quad \text{CAR}_{i}[t_0, t_0+T] = \sum_{t=t_0}^{t_0+T} \text{AR}_{i,t}$$
  Evaluate over $T \in \{10, 30, 60, 90\}$ trading days.

---

## 3. Deliverables & Evaluation Metrics

| Metric | Target / Benchmark | Purpose |
| :--- | :--- | :--- |
| **Opportunistic vs. Routine Spread** | $\ge +1.00\%$ per month | Validates Cohen-Malloy hypothesis on the active universe |
| **Cluster Buy Win Rate (60d)** | $\ge 62\%$ win rate | Validates multi-insider high conviction catalyst |
| **Information Coefficient (Rank IC)** | $\ge 0.020$ | Confirms quantitative viability for downstream ML integration |
| **Output File** | `data/fetched/insider_signals_poc.xlsx` | Feature table ready for multi-modal synthesis |

---

## 4. Go / No-Go Decision Gate
- **Proceed**: If opportunistic purchases outperform routine purchases by $> 1.0\%$ CAR over a 60-day horizon with a positive Information Coefficient.
- **Pivot/Halt**: If Form 4 filing volume in the current 50-ticker large-cap universe is too sparse to yield statistically meaningful signals, expand universe to Russell 1000/2000.
