import os
import requests
import pandas as pd
import trafilatura
from datetime import datetime, timedelta
from dotenv import load_dotenv

class NewsFetcher:
    def __init__(self):
        load_dotenv()
        self.api_key = os.getenv("API_KEY_CURRENTS_NEWS_API")
        self.url = "https://api.currentsapi.services/v2/search"

    def fetch_news(self, tickers: list[str], days: int = 1):
        """Fetches news for multiple tickers for multiple days."""
        all_articles = []
        for ticker in tickers:
            for day_offset in range(days):
                target_date = datetime.now() - timedelta(days=day_offset)
                start_date = target_date.strftime("%Y-%m-%dT00:00:00+00:00")
                end_date = target_date.strftime("%Y-%m-%dT23:59:59+00:00")

                params = {
                    "query": ticker,
                    "language": "en",
                    "page_size": 20,
                    "start_date": start_date,
                    "end_date": end_date,
                    "apiKey": self.api_key
                }

                response = requests.get(self.url, params=params)
                articles = response.json().get("news", [])

                for article in articles:
                    url = article.get("url")
                    html = trafilatura.fetch_url(url)
                    article["full_text"] = trafilatura.extract(html) if html else "DOWNLOAD_FAILED"
                    article["ticker"] = ticker
                    all_articles.append(article)
        
        return all_articles

    def save_to_ods(self, articles: list[dict], filename: str):
        """Saves article data to an ODS file in data/fetched/news/."""
        df = pd.DataFrame(articles)
        os.makedirs("data/fetched/news", exist_ok=True)
        path = os.path.join("data", "fetched", "news", filename)
        df.to_excel(path, engine="odf", index=False)
        return path
