import faiss
import numpy as np
import os
import pickle
from core.embed import get_embedding


FAISS_PATH = "faiss_index.index"
TEXTS_PATH = "faiss_texts.pkl"

# Load FAISS index
def load_faiss():
    if os.path.exists(FAISS_PATH):
        idx = faiss.read_index(FAISS_PATH)
    else:
        idx = faiss.IndexFlatL2(1536)  # embedding dimension
    return idx

# Load stored texts
def load_texts():
    if os.path.exists(TEXTS_PATH):
        with open(TEXTS_PATH, "rb") as f:
            return pickle.load(f)
    return []

# Initialize
index = load_faiss()
stored_texts = load_texts()

def store_in_faiss(text: str):
    """
    Embeds a piece of text and stores both the vector in FAISS and text in a list.
    """
    global index, stored_texts

    vector = get_embedding(text)
    vector = np.array(vector).astype("float32").reshape(1, -1)

    index.add(vector)
    stored_texts.append(text)

    # Persist
    faiss.write_index(index, FAISS_PATH)
    with open(TEXTS_PATH, "wb") as f:
        pickle.dump(stored_texts, f)

    print("✓ Stored in FAISS with original text")

def retrieve_from_faiss(query: str, top_k=3):
    """
    Performs similarity search and returns the closest stored texts.
    """
    global index, stored_texts

    query_vec = get_embedding(query)
    query_vec = np.array(query_vec).astype("float32").reshape(1, -1)

    distances, indices = index.search(query_vec, top_k)

    results = [stored_texts[i] for i in indices[0] if i < len(stored_texts)]
    return results, distances



#  Testing 
# texts_to_store = [
#     "Tivaly API allows you to search web content easily.",
#     "FAISS is a library for efficient similarity search.",
#     "Python is a versatile programming language."
# ]

# # Store the texts
# for text in texts_to_store:
#     store_in_faiss(text)

# # Test retrieval
# query = "What is FAISS?"
# results, distances = retrieve_from_faiss(query, top_k=2)

# print(f"Query: {query}")
# print(f"Top Results: {results}")
# print(f"Distances: {distances}")
