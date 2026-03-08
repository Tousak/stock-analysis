# Gemini DeepSearch Prompt: Machine Learning for Stock Price Prediction

**Instructions:** Please analyze the current state-of-the-art in applying machine learning to predict stock prices based on the provided context of our existing project and our broader goals. Please provide a detailed breakdown of the best models, features, and overall strategies to adopt.

### Project Context: Current State
We are currently building a stock analysis pipeline and Streamlit dashboard. Here is what we have implemented and what we are experimenting with:

1.  **Current Implementation (The Baseline):**
    *   **Data Source:** SEC 10-Q (Quarterly) filings fetched using the `edgar` library.
    *   **Features:** Raw financial numbers (Revenue, Net Income) and basic engineered features (`revenue_growth`, `net_margin`).
    *   **Sentiment Analysis:** Extracting the Management's Discussion and Analysis (MD&A) section from 10-Q filings and sending it to OpenAI's `gpt-4o-mini` API to get a sentiment score (-1.0 to 1.0) and justification. We also track the `sentiment_change` quarter-over-quarter.
    *   **Model:** A simple `RandomForestRegressor` trained on (`sentiment_score`, `sentiment_change`, `revenue_growth`, `net_margin`) to predict the `next_quarter_return` (the stock's return 90 days post-filing date). No hyperparameter tuning.
    *   **Evaluation:** Robust walk-forward validation (purging overlapping time windows) and simulating a portfolio holding the top *N* positive predictions.

2.  **Experimental Work (Jupyter Notebooks):**
    *   **Alternative Data Sources:** Scraping live news articles via NewsAPI and the `newspaper` library to get full article text.
    *   **Alternative Models (NLP):** Experimenting with local, open-source models like `ProsusAI/finbert` (via HuggingFace) to classify financial text sentiment, avoiding paid API costs.
    *   **Additional Quantitative Features:** Experimenting with broader `yfinance` fundamentals (e.g., `trailingPE`, `debtToEquity`, `profitMargins`).

### Broader Project Goal
Our ultimate goal is to build a highly effective, robust machine learning system for predicting stock prices. 
*   **Time Horizon:** We are not strictly limited to predicting quarterly (90-day) returns based on 10-Q filings. We are open to shorter-term (e.g., weekly/monthly using news) or longer-term predictions.
*   **Flexibility:** We are open to adopting new data pipelines or replacing the current `RandomForest` / `gpt-4o-mini` architecture entirely if the literature and current state-of-the-art suggest a better path. 

### DeepSearch Research Questions

Given the context above, I need you to perform a deep search on the current best practices for predicting stock prices using machine learning. Please structure your response around these core areas:

**1. What are the best predictive models currently used in the industry/academia?**
*   How do lightweight gradient boosting models (like XGBoost, LightGBM, CatBoost) compare to simpler models (Random Forest, SVM) or deep learning approaches (LSTM, Transformers) for pure timeseries/structured financial data?
*   What is the current consensus on the role of Large Language Models (LLMs) vs. specialized smaller models (like FinBERT) in stock prediction? Are LLMs better used as feature extractors (generating sentiment scores) or as end-to-end predictors?

**2. Which features actually hold predictive power (Alpha)?**
*   What is the hierarchy of feature importance? Do quantitative metrics (P/E ratios, moving averages, momentum indicators) generally outperform qualitative metrics (News sentiment, 10-Q MD&A sentiment), or is the combination where the real edge lies?
*   Are there specific technical or fundamental indicators that consistently show higher feature importance across modern models?

**3. What is the optimal time horizon for predictions?**
*   Our current baseline predicts 90 days out based on quarterly filings. Is this horizon too noisy? Does the current state-of-the-art lean towards shorter horizons (e.g., predicting 1-5 days out using daily news/price action) or longer horizons?

**4. Based on your findings, what specific architectural changes should we prioritize for our project?**
*   Should we stick with 10-Q filings as the core data, or pivot immediately to higher-frequency news data?
*   Should we transition our `RandomForestRegressor` to an `XGBoost` model, and if so, what are the most critical hyperparameters or cross-validation techniques (e.g., Purged K-Fold) to implement to avoid overfitting?
