# Multi-document Embedding Search Engine with Caching

## Overview
Lightweight embedding-based search engine over text files with:
- Embeddings via `sentence-transformers/all-MiniLM-L6-v2`
- Local JSON cache (per-document)
- FAISS Index (IndexFlatIP) with normalized embeddings
- FastAPI `/search` endpoint
- Simple ranking explanation (keyword overlap + length normalization)

## Folder structure
project/
├─ src/ # core code files
├─ data/ # put your .txt documents here (gitignored)
├─ cached/ # JSON cache created automatically
├─ requirements.txt
└─ README.md


## How caching works
For each `doc_id` a JSON file is stored in `cached/<doc_id>.json`:
```json
{
 "doc_id": "doc_001",
 "hash": "sha256_of_cleaned_text",
 "embedding": [...],
 "updated_at": 1234567890.0
}

On build, the system computes the SHA-256 of the cleaned text and checks cache. If unchanged, cached embedding is reused; if changed, a new embedding is generated and cached.

How to run

Create virtualenv and install dependencies:
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

Put your .txt files under data/ (filenames like doc_001.txt).

Start the API:

python -m src.api


The server runs at http://0.0.0.0:8000.

Example request

POST http://localhost:8000/search
Body:

{"query":"quantum physics basics", "top_k":5}

Design choices

FAISS IndexFlatIP with normalized embeddings for fast cosine-similarity retrieval.

JSON per-doc cache for simplicity and easy debugging.

Sentence-Transformers model provides a good speed/accuracy tradeoff.

Notes

For many documents or production, consider persistent FAISS index file and SQLite cache.

Optionally add Streamlit UI, batch embedding, or query expansion for improvements.


---

# Quick usage tips & troubleshooting
- If `faiss-cpu` fails to install, run with `use_faiss=False` in `src/api.py` and `src/search_engine.py`; code falls back to numpy cosine similarity.
- Ensure `data/` contains `.txt` files and `cached/` is writable.
- If embeddings seem slow, pre-generate embeddings by running a small script that calls `SearchEngine.build()` once — it caches to `cached/`.

---

# Next steps I can do for you (pick any, I’ll do it now)
1. Generate these files as downloadable files for you.  
2. Add a small Streamlit UI (`app.py`) to interactively search.  
3. Swap JSON cache to SQLite and add migrations.  
4. Add a script to pre-populate embeddings (batch + multiprocessing).  
5. Provide a short demo script showing curl requests and example outputs.

Which one do you want me to do **right now**? (I’ll produce the code/files in this chat immediately.)