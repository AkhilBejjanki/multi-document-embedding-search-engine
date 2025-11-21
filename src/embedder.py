import numpy as np
from sentence_transformers import SentenceTransformer

class Embedder:
    def __init__(self):
        # Ultra-small model (renders instantly)
        self.model = SentenceTransformer(
            "sentence-transformers/paraphrase-MiniLM-L3-v2",
            device="cpu"
        )

    def embed_text(self, text: str) -> np.ndarray:
        emb = self.model.encode(
            [text],
            convert_to_numpy=True,
            normalize_embeddings=False
        )
        return emb[0].astype(np.float32)

    def embed_texts(self, texts: list) -> np.ndarray:
        embs = self.model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=False
        )
        return embs.astype(np.float32)

    def normalize_embeddings(self, embs: np.ndarray):
        norms = np.linalg.norm(embs, axis=1, keepdims=True) + 1e-10
        return embs / norms
