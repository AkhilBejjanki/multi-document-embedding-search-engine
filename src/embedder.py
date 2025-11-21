import numpy as np
from sentence_transformers import SentenceTransformer

class Embedder:
    def __init__(self):
        # Best lightweight model for deployment
        self.model = SentenceTransformer(
            "thenlper/gte-small",
            device="cpu"
        )

    def embed_text(self, text: str) -> np.ndarray:
        emb = self.model.encode(
            [text],
            convert_to_numpy=True
        )
        return emb[0].astype(np.float32)

    def embed_texts(self, texts: list) -> np.ndarray:
        embs = self.model.encode(
            texts,
            convert_to_numpy=True
        )
        return embs.astype(np.float32)

    def normalize_embeddings(self, embs: np.ndarray):
        norms = np.linalg.norm(embs, axis=1, keepdims=True) + 1e-10
        return embs / norms
