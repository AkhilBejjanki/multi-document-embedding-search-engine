import numpy as np
from sentence_transformers import SentenceTransformer

class Embedder:
    def __init__(self):
        self.model = SentenceTransformer("thenlper/gte-small", device="cpu")

    def embed_text(self, text):
        return self.model.encode([text], convert_to_numpy=True)[0].astype(np.float32)

    def embed_texts(self, texts):
        return self.model.encode(texts, convert_to_numpy=True).astype(np.float32)

    def normalize_embeddings(self, embs):
        norms = np.linalg.norm(embs, axis=1, keepdims=True) + 1e-10
        return embs / norms
