import os
import json
import hashlib
import numpy as np

class CacheManager:
    def __init__(self, cache_dir="cached"):
        self.cache_dir = cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)

    def _path(self, doc_id):
        return os.path.join(self.cache_dir, f"{doc_id}.npy")

    def _meta_path(self, doc_id):
        return os.path.join(self.cache_dir, f"{doc_id}.json")

    def save(self, doc_id, hash_value, embedding):
        """Save embedding + metadata."""
        np.save(self._path(doc_id), embedding)

        meta = {"hash": hash_value}
        with open(self._meta_path(doc_id), "w") as f:
            json.dump(meta, f)

    def get_embedding_if_valid(self, doc_id, hash_value):
        """Check cached hash. If match → return embedding."""
        meta_path = self._meta_path(doc_id)
        emb_path = self._path(doc_id)

        if not os.path.exists(meta_path) or not os.path.exists(emb_path):
            return None

        try:
            with open(meta_path, "r") as f:
                meta = json.load(f)
        except:
            return None

        if meta.get("hash") != hash_value:
            return None

        try:
            emb = np.load(emb_path)
            return emb
        except:
            return None


# Helper hashing function used in SearchEngine
def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()
