import requests
from utils.config import TIVALY_API_KEY


def search_web(query, num_results=5):
    try:
        res = requests.post(
            "https://api.tavily.com/search",
            headers={"Authorization": f"Bearer {TIVALY_API_KEY}"},
            json={"query": query, "num_results": num_results},
            timeout=10
        )
        return res.json()["results"][0]["content"]
    except Exception as e:
        print(f"Search API error: {e}")
        return "Sorry, I couldn't fetch any results right now."


# Testing 
# print(search_web("What is Tivaly API?", num_results=1)) 
