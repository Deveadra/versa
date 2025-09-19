
import openai
import numpy as np

def get_embedding(text: str) -> np.ndarray:
    resp = openai.Embedding.create(
        input=text,
        model="text-embedding-3-small"
    )
    return np.array(resp["data"][0]["embedding"], dtype=np.float32)

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
