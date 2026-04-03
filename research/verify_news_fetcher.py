import sys
import os

# Ensure the project root is in the path
sys.path.append(os.getcwd())

from src.data_fetch.news_fetcher import NewsFetcher

def main():
    fetcher = NewsFetcher()
    print("Fetching news for AAPL for 1 day...")
    # Fetching for 1 ticker and 1 day to be extremely quick
    articles = fetcher.fetch_news(["AAPL"], days=1)
    print(f"Fetched {len(articles)} articles.")
    
    if articles:
        output_file = "test_news.ods"
        path = fetcher.save_to_ods(articles, output_file)
        print(f"\n✅ Success! Saved {len(articles)} articles to {path}")
    else:
        print("\n⚠️ No articles found. Check your API key and connection.")

if __name__ == "__main__":
    main()
