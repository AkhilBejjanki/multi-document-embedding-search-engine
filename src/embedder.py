# src/embedder.py
from sentence_transformers import SentenceTransformer
import numpy as np

class Embedder:
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)

    def embed_text(self, text: str) -> np.ndarray:
        emb = self.model.encode(text, show_progress_bar=False)
        emb = np.array(emb, dtype=np.float32)
        return emb

    def embed_texts(self, texts: list) -> np.ndarray:
        embs = self.model.encode(texts, show_progress_bar=False)
        return np.array(embs, dtype=np.float32)

    @staticmethod
    def normalize_embeddings(embs: np.ndarray) -> np.ndarray:
        # Normalize rows to unit length for cosine similarity with IndexFlatIP
        norms = np.linalg.norm(embs, axis=1, keepdims=True)
        norms[norms == 0] = 1e-8
        return embs / norms
