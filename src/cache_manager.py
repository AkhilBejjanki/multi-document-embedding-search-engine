# src/cache_manager.py
import os
import json
import hashlib
import time
from typing import Optional, Dict
import numpy as np

CACHE_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "..", "cached")
os.makedirs(CACHE_DIR, exist_ok=True)

def sha256_text(text: str) -> str:
    h = hashlib.sha256()
    h.update(text.encode("utf-8"))
    return h.hexdigest()

class CacheManager:
    """
    JSON per-document cache. Each doc will have a file cached/<doc_id>.json
    Structure:
    {
      "doc_id": "doc_001",
      "hash": "sha256_of_text",
      "embedding": [...],
      "updated_at": 1234567890.0
    }
    """
    def __init__(self, cache_dir: str = CACHE_DIR):
        self.cache_dir = os.path.abspath(cache_dir)
        os.makedirs(self.cache_dir, exist_ok=True)

    def get_cache_path(self, doc_id: str) -> str:
        return os.path.join(self.cache_dir, f"{doc_id}.json")

    def load(self, doc_id: str) -> Optional[Dict]:
        p = self.get_cache_path(doc_id)
        if not os.path.exists(p):
            return None
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)

    def save(self, doc_id: str, text_hash: str, embedding: np.ndarray):
        p = self.get_cache_path(doc_id)
        payload = {
            "doc_id": doc_id,
            "hash": text_hash,
            "embedding": embedding.tolist(),
            "updated_at": time.time()
        }
        with open(p, "w", encoding="utf-8") as f:
            json.dump(payload, f)

    def get_embedding_if_valid(self, doc_id: str, text_hash: str) -> Optional[np.ndarray]:
        rec = self.load(doc_id)
        if not rec:
            return None
        if rec.get("hash") != text_hash:
            return None
        emb = np.array(rec["embedding"], dtype=np.float32)
        return emb
