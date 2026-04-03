import os
import requests
import json
import trafilatura
from datetime import datetime, timedelta
from dotenv import load_dotenv

# 1. Load environment variables
load_dotenv()
api_key = os.getenv("API_KEY_CURRENTS_NEWS_API")

if not api_key:
    print("❌ Error: API_KEY_CURRENTS_NEWS_API not found in the .env file.")
    exit(1)

# 2. Configure the API Request
url = "https://api.currentsapi.services/v2/search"
target_date = datetime.now() - timedelta(days=30)
start_date = target_date.strftime("%Y-%m-%dT00:00:00+00:00")
end_date = target_date.strftime("%Y-%m-%dT23:59:59+00:00")

params = {
    "query": '"AAPL" OR ("Apple" AND "AAPL") OR ("Apple" AND "iPhone")',
    "category": "economy_business_finance", 
    "language": "en",
    "page_size": 50,
    "start_date": start_date,
    "end_date": end_date,
    "apiKey": api_key
}

try:
    # 3. Fetch the URLs and metadata from Currents API
    print("Fetching article URLs from Currents API...")
    response = requests.get(url, params=params)
    response.raise_for_status() 
    
    data = response.json()
    articles = data.get("news", [])
    
    if not articles:
        print("⚠️ API request succeeded, but no articles matched your query.")
        exit(0)
        
    print(f"✅ Found {len(articles)} articles. Starting full-text extraction...\n")
    
    # 4. Extract the full text for each article URL
    for i, article in enumerate(articles):
        article_url = article.get("url")
        print(f"[{i+1}/{len(articles)}] Scraping: {article_url}")
        
        try:
            downloaded_html = trafilatura.fetch_url(article_url)
            
            if downloaded_html:
                full_text = trafilatura.extract(downloaded_html)
                # If extraction succeeds, add the text; otherwise flag it
                article["full_text"] = full_text if full_text else "EXTRACTION_FAILED"
            else:
                article["full_text"] = "DOWNLOAD_FAILED"
                
        except Exception as e:
            print(f"  ↳ Error scraping {article_url}: {e}")
            article["full_text"] = "ERROR"

    # 5. Save the final enriched dataset
    output_dir = "research/data/"
    os.makedirs(output_dir, exist_ok=True)
    
    file_name = f"apple_news_full_dataset_{target_date.strftime('%Y%m%d')}.json"
    file_path = os.path.join(output_dir, file_name)
    
    with open(file_path, "w", encoding="utf-8") as json_file:
        json.dump(articles, json_file, indent=4, ensure_ascii=False)
        
    print(f"\n✅ Pipeline complete! Enriched dataset saved to {os.path.abspath(file_path)}")
        
except requests.exceptions.RequestException as e:
    print(f"❌ API Request Failed: {e}")
    if 'response' in locals() and response is not None:
        print(f"Raw Response: {response.text}")