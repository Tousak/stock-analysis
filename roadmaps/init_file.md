# Task: Generate Quarterly Prediction Model Notebook

**Goal:** Create a Jupyter Notebook that builds a "Multimodal" stock prediction model by combining SEC financial data, text sentiment from 10-Q reports, and historical price data.

**Output File:** `research/notebooks/quarterly_prediction.ipynb`
**Reference File:** `research/notebooks/data_fetch.ipynb` (Use this for correct `edgartools` usage patterns)

**Critical Rules:** Final code should be as simple stupid (KISS principle), do not use error handling, do not use try blocks, code must be human readable 


## 1. Notebook Configuration
* **Kernel:** Python 3
* **Libraries:** `edgartools`, `pandas`, `yfinance`, `matplotlib`, `textblob` (or `vaderSentiment`), `sklearn` (if another libraries are needed - use `uv add` command).
* **Visual Style:** Use `matplotlib.pyplot` with a clean style (e.g., `plt.style.use('ggplot')`).

## 2. Step-by-Step Implementation Instructions

### Step 1: Imports & Setup
* Import necessary libraries.
* **Crucial:** Include the SEC identity setup using `set_identity()` as seen in `data_fetch.ipynb`.
* *Hint:* Use `from edgartools import set_identity, Company`.

### Step 2: Data Extraction Function (The Feature Engine)
Create a function (or class) that accepts a `ticker` and `num_quarters` and performs the following:
1.  **Fetch Filings:** Use `edgartools` to get the last `N` "10-Q" filings.
2.  **Iterate & Extract:**
    * **Date:** Extract `filing_date`.
    * **Financials:** Use `filing.financials` to safely extract "Revenue" and "Net Income" (handle missing keys gracefully).
    * **Text:** Extract "Item 2" (MD&A) using the bracket notation `tenq['Item 2']`.
    * **Sentiment:** Calculate a "Polarity Score" for the MD&A text using `TextBlob` or `Vader`.
3.  **Return:** A clean Pandas DataFrame with columns: `['date', 'revenue', 'net_income', 'sentiment_score']`.

### Step 3: Market Data Alignment (The Target)
* Use `yfinance` to fetch daily price history for the same ticker.
* **Target Creation:** For each filing date in the DataFrame:
    * Find the stock price on the `filing_date`.
    * Find the stock price 90 days *after* the filing (Next Quarter).
    * Calculate the **Percentage Return** `(Price_90d - Price_Filing) / Price_Filing`.
* **Merge:** Join this target variable onto your main features DataFrame. Drop rows where future data is not yet available (NaN targets).

### Step 4: Feature Engineering
* Create a `revenue_growth` column (Current Quarter vs Previous Quarter).
* Create a `net_margin` column (`net_income / revenue`).
* Create a `sentiment_change` column (Current Sentiment - Previous Sentiment).

### Step 5: Visualization (EDA)
Use `matplotlib` to generate two specific plots:
1.  **Dual-Axis Plot:** Overlay **Stock Price** (Line) vs. **Sentiment Score** (Bar/Scatter) over time to visualize correlation.
2.  **Scatter Plot:** X-axis = `Sentiment Score`, Y-axis = `Next Quarter Return`. Add a trendline to see if higher sentiment correlates with higher returns.

### Step 6: Predictive Modeling
* **Data Split:** Split data into Train/Test sets based on *time* (do not shuffle randomly, to respect time-series causality). Train on the first 80%, Test on the last 20%.
* **Model:** Initialize a `RandomForestRegressor` from `sklearn`.
* **Features:** `['revenue_growth', 'net_margin', 'sentiment_score', 'sentiment_change']`.
* **Target:** `['next_quarter_return']`.
* **Training:** Fit the model on the training set.

### Step 7: Evaluation & Conclusion
* Predict on the Test set.
* Calculate and print **Mean Squared Error (MSE)**.
* **Feature Importance:** Plot the feature importances from the Random Forest to see which factor (Sentiment vs. Financials) drove the predictions most.

---

## Technical Constraints & Notes
* **Error Handling:** The extraction step must handle cases where `Item 2` is missing or `yfinance` returns empty data.
* **EdgarTools Syntax:** Strictly follow the syntax `filing.obj()['Item 2']` or `filing.markdown()` as established in the reference notebook.
* **Comments:** Add markdown cells explaining the logic of "Data Alignment" (why we use Filing Date and not Quarter End Date).