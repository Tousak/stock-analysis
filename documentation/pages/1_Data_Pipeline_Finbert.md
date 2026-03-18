# Page 1: Data Pipeline (FinBERT)

## Overview
This page serves as the local extraction engine for the stock analysis system. It utilizes a local transformer model to process SEC filings without relying on external APIs (like OpenAI), ensuring data privacy and preventing look-ahead bias by using a static pre-2020 sentiment model.

## Data Workflow

### 1. Data Acquisition (Step 1)
- **Method**: Fetches SEC 10-Q filings using the `edgar` library and historical price data via `yfinance`.
- **Logic**: 
    - Downloads the last 48 quarters of filings for the specified tickers.
    - Extracts the "Management's Discussion and Analysis" (MD&A) section using a regex-based pattern matcher defined in `src/config.py`.
    - Incremental Loading: Skips filings already present in the local cache to save bandwidth.
- **Storage**: Results are saved to `data/fetched/raw_filings.xlsx`.

### 2. Sentiment Analysis (Step 2)
- **Model**: **ProsusAI/finbert** (HuggingFace Transformers).
- **Processing**:
    - MD&A text is split into chunks (approx. 400 words each) to stay within the model's 512-token limit.
    - Sentiment probabilities (Positive, Negative, Neutral) are averaged across all chunks for a single filing.
    - A scalar `sentiment_score` is calculated as `pos - neg`.
- **Storage**: Results are stored in `data/fetched/processed_filings_finbert.xlsx`. Redundant textual data is dropped from this file to optimize storage.

### 3. Feature Engineering (Step 3)
The feature engineering module (`src/feature_eng.py`) transforms raw sentiment and financial data into a structured dataset ready for machine learning. It employs a multi-stage process of calculation, market data integration, and differentiated cleaning.

#### Feature Calculation Formulas
- **Financial Dynamics**:
    - `revenue_growth`: Calculated as the quarter-over-quarter percentage change in revenue per ticker (`df.groupby('ticker')['revenue'].pct_change()`).
    - `net_margin`: Calculated as `net_income / revenue`, representing the company's profitability efficiency.
- **Sentiment Dynamics**:
    - `sentiment_change`: The absolute difference between the current quarter's sentiment score and the previous quarter's score.
    - **Triplet Changes**: If using FinBERT, the system also calculates QoQ changes for the raw `pos`, `neg`, and `neu` probabilities (e.g., `sentiment_pos_change`).
- **Technical Indicators (Selective)**:
    - **RSI (Relative Strength Index)**: A 14-day momentum oscillator that measures the speed and change of price movements.
    - **MACD (Moving Average Convergence Divergence)**: Calculated using the difference between the 12-day and 26-day Exponential Moving Averages (EMAs).
    - **Volatility**: A relative measure calculated as `(High - Low) / (Open + 1e-9)`.
- **Target Variable (Label)**:
    - `next_quarter_return`: The percentage change in stock price between the filing date and 90 days (or 5 days for Alpha) into the future. It uses a "price on-or-after" lookup to handle weekends and market holidays.

#### Market Data Integration
The system automatically fetches historical price data from `yfinance` to support feature calculation. It fetches a broad window (120 days before the first filing to 210 days after the last filing) to ensure all lookbacks and future return horizons are covered without missing data points.

#### Differentiated Data Cleaning
To maximize model quality while preserving live signal availability, the system applies two distinct cleaning regimes:

1.  **Granular Training Clean**: For historical samples where the outcome is known, the system is aggressive. It drops any row missing critical metrics (Revenue, Margin, Sentiment Change). This ensures the XGBoost model is trained on "high-fidelity" complete records.
2.  **Lenient Prediction Clean**: For the latest filings where we want to forecast the future, the system is more permissive. It only requires a valid `sentiment_score` and `mda_text`. This ensures we can still generate a buy/hold recommendation even for companies that may have slight gaps in their reported historical metadata.

#### Storage & Output
- **File**: `data/fetched/features_finbert.xlsx`
- **Output Schema**: A standardized 15-20 column matrix including Ticker, Date, Financial ratios, Sentiment deltas, and the Target return.

## Technology & Libraries
- **Libraries**: `pandas`, `streamlit`, `transformers` (PyTorch backend), `edgar-tools`, `yfinance`.
- **Models**: `ProsusAI/finbert` (Specialized BERT model for financial sentiment).
