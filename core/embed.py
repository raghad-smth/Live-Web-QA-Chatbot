import requests
from utils.config import OPEN_ROUTER_API


EMBED_MODEL = "text-embedding-3-small"  
EMBED_URL = "https://openrouter.ai/api/v1/embeddings"


def get_embedding(text: str):
    """
    Sends text to OpenRouter to get an embedding.
    Returns a list of floats (vector).
    """
    payload = {
        "model": EMBED_MODEL,
        "input": text
    }

    headers = {
        "Authorization": f"Bearer {OPEN_ROUTER_API}",
        "Content-Type": "application/json"
    }

    response = requests.post(
        EMBED_URL,
        json=payload,
        headers=headers,
        timeout=10
    )

    response.raise_for_status()
    data = response.json()

    return data["data"][0]["embedding"]

# Testing
# print("Embedding length: ", get_embedding("Hey cuite ;)!").__len__())
# print("Embedding : " , get_embedding("Hey cuite ;)!"))