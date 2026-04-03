import os
import requests
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("API_KEY_CURRENTS_NEWS_API")
url = "https://api.currentsapi.services/v2/search"

params = {
    "query": "AAPL",
    "language": "en",
    "apiKey": api_key
}

print(f"Testing API with minimum params...")
response = requests.get(url, params=params)
print(f"Status Code: {response.status_code}")
data = response.json()
news = data.get("news", [])
print(f"Number of articles found: {len(news)}")
if news:
    print(f"First article title: {news[0].get('title')}")
else:
    print(f"Full response: {data}")
