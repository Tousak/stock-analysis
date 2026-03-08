# Roadmap: Transition to High-Frequency News Data

## 🎯 Objective
Pivot from relying exclusively on sparse quarterly 10-Q SEC filings to a dense, continuous stream of high-frequency alternative data. Scrape daily financial news articles using the NewsAPI and `newspaper3k` libraries.

## ⚠️ Critical Coding Guidelines (MANDATORY)
*   **KISS (Keep It Simple, Stupid):** No over-engineering. Write bare-bones implementations.
*   **Human-Readable & Short:** Keep the code as concise as absolutely possible.
*   **NO Error Blocks / NO Try Blocks:** The `try...except` pattern is strictly forbidden. Assume data is correct or let the program fail loudly.

## 📂 Architecture
*   **Backend (`src/`):** Update `src/config.py` with new API keys and URLs. Create/update a new data pipeline script `src/data_loader_news.py`.
*   **Frontend (`pages/`):** Update `pages/1_Data_Pipeline.py` to allow the user to select the data source.

## 🚀 Implementation Steps

1.  **Integrate NewsAPI:**
    *   In `src/config.py`, add the `NEWSAPI_KEY` environmental variable definition.
    *   In `src/data_loader_news.py`, write a function to hit the `https://newsapi.org/v2/everything` endpoint for a given `TICKER` between `start_date` and `end_date`.
2.  **Scrape Full Articles:**
    *   Initialize the `newspaper3k` library inside the news loader script.
    *   For each URL returned by the NewsAPI, fetch the full HTML and extract the raw text content to form a local corpus of news text per ticker, per day.
3.  **Sanitize Output:**
    *   Filter out empty or blocked URLs, and keep the text structured in a simple pandas DataFrame (`ticker`, `date`, `full_text`, `url`).
4.  **UI Data Options:**
    *   Modify `pages/1_Data_Pipeline.py` to add a new "Fetch News Data" button to run the `data_loader_news.py` script.
