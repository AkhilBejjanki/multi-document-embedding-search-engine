# src/search_engine.py
import os
import numpy as np
from typing import List, Dict, Optional
from src.cache_manager import CacheManager, sha256_text
from src.embedder import Embedder
import glob
import re

# Try importing FAISS, otherwise set to None
try:
    import faiss
except ImportError:
    faiss = None

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")

class Document:
    def __init__(self, doc_id: str, text: str):
        self.doc_id = doc_id
        self.text = text
        self.length = len(text.split())

    def preview(self, n_chars: int = 200) -> str:
        return self.text[:n_chars].replace("\n", " ") + ("..." if len(self.text) > n_chars else "")

def simple_clean(text: str) -> str:
    text = text.lower()
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

class SearchEngine:
    def __init__(self, use_faiss: bool = False):
        # Force FAISS only if available
        self.use_faiss = use_faiss and (faiss is not None)

        self.embedder = Embedder()
        self.cache = CacheManager()
        self.docs: List[Document] = []
        self.doc_ids: List[str] = []
        self.embeddings: Optional[np.ndarray] = None
        self.index = None

    def load_documents(self, data_dir: str = DATA_DIR, pattern: str = "*.txt"):
        files = sorted(glob.glob(os.path.join(data_dir, pattern)))
        docs = []
        for fp in files:
            doc_id = os.path.splitext(os.path.basename(fp))[0]
            with open(fp, "r", encoding="utf-8", errors="ignore") as f:
                raw = f.read()
            cleaned = simple_clean(raw)
            docs.append((doc_id, cleaned, raw))
        self.docs = [Document(doc_id, raw) for doc_id, cleaned, raw in docs]
        self.doc_ids = [d.doc_id for d in self.docs]

    def build_embeddings_from_cache_or_model(self):
        embeddings = []
        to_embed_texts = []
        to_embed_indices = []

        for idx, doc in enumerate(self.docs):
            cleaned = simple_clean(doc.text)
            h = sha256_text(cleaned)
            cached = self.cache.get_embedding_if_valid(doc.doc_id, h)
            if cached is not None:
                embeddings.append(cached)
            else:
                embeddings.append(None)
                to_embed_texts.append(cleaned)
                to_embed_indices.append(idx)

        if to_embed_texts:
            new_embs = self.embedder.embed_texts(to_embed_texts)
            for i, emb in enumerate(new_embs):
                idx = to_embed_indices[i]
                cleaned = simple_clean(self.docs[idx].text)
                h = sha256_text(cleaned)
                self.cache.save(self.docs[idx].doc_id, h, emb)
                embeddings[idx] = emb

        emb_matrix = np.vstack(embeddings).astype(np.float32)
        emb_matrix = self.embedder.normalize_embeddings(emb_matrix)
        self.embeddings = emb_matrix
        return emb_matrix

    def build_faiss_index(self):
        if self.embeddings is None or faiss is None:
            return None
        dim = self.embeddings.shape[1]
        index = faiss.IndexFlatIP(dim)
        index.add(self.embeddings)
        self.index = index

    def build(self):
        if not self.docs:
            self.load_documents()
        self.build_embeddings_from_cache_or_model()
        if self.use_faiss:
            self.build_faiss_index()

    def search(self, query: str, top_k: int = 5) -> List[Dict]:
        q_emb = self.embedder.embed_text(query)
        q_emb = q_emb.reshape(1, -1).astype(np.float32)
        q_emb = self.embedder.normalize_embeddings(q_emb)[0]

        if self.use_faiss and self.index is not None:
            D, I = self.index.search(np.array([q_emb]), top_k)
            scores = D[0].tolist()
            idxs = I[0].tolist()
        else:
            sims = (self.embeddings @ q_emb)
            idxs = np.argsort(-sims)[:top_k]
            scores = sims[idxs].tolist()

        results = []
        for score, idx in zip(scores, idxs):
            doc = self.docs[idx]
            results.append({
                "doc_id": doc.doc_id,
                "score": float(score),
                "preview": doc.preview(),
            })
        return results

    def explain(self, query: str, doc_text: str) -> Dict:
        q_words = set(w for w in query.lower().split() if len(w) > 2)
        d_words = set(w for w in simple_clean(doc_text).split() if len(w) > 2)
        overlapped = sorted(list(q_words & d_words))
        overlap_ratio = len(overlapped) / max(1, len(q_words))
        doc_len = len(doc_text.split())
        length_norm = 1.0 / (1 + doc_len / 500.0)
        return {
            "matched_keywords": overlapped,
            "overlap_ratio": overlap_ratio,
            "length_norm": length_norm,
        }

    def search_with_explanation(self, query: str, top_k: int = 5):
        results = self.search(query, top_k=top_k)
        for r in results:
            doc = next(d for d in self.docs if d.doc_id == r["doc_id"])
            r["explanation"] = self.explain(query, doc.text)
        return results
