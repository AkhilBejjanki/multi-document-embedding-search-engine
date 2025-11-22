# Multi-Document Embedding Search Engine

A lightweight semantic search engine that loads multiple text documents, generates embeddings, caches them, and performs similarity-based search with explanations.


---

## 1. How Caching Works

The project uses a simple but effective caching system stored inside:

```
cached/
├── <doc_id>.npy
└── <doc_id>.json
```

### How it works:

1. Every document is cleaned and hashed using **SHA-256**
2. Before generating embeddings, the system checks:
   - If an embedding file exists for that document
   - If the stored hash matches the current hash
3. **If both match** → cached embedding is reused (fast, no recomputation)
4. **If not** → a new embedding is generated, stored in `cached/`, and used

### Benefits:

- ⚡ Faster startup
- 🔄 Only changed/new documents get re-embedded
- 💾 Ideal for local development and large datasets

---

## 2. How to Run Embedding Generation

Embeddings are generated automatically when the search engine is built.

### Steps:

1. Place your `.txt` documents inside:
   ```
   data/
   ```

2. Run the following command:
   ```bash
   python -m src.api
   ```

### On first run:

- The system loads all documents
- Generates embeddings for uncached documents
- Saves them in `cached/`

### On next runs:

- Cached embeddings load instantly
- No model call needed unless a document changed

---

## 3. How to Start the API

Make sure you are in the project's root folder and run:

```bash
uvicorn src.api:app --reload
```

**OR:**

```bash
python -m src.api
```

This starts a FastAPI server at:

```
http://127.0.0.1:8000
```

### API Documentation

Open the interactive Swagger UI:

```
http://127.0.0.1:8000/docs
```

You can test:

- `/search`
- `/explain`
- `/reload-docs`
- `/status`

---

## 4. Folder Structure

```
multi-document-embedding-search-engine/
│
├── data/                     # All text documents (.txt)
├── cached/                   # Cached embeddings + metadata
│
├── src/
│   ├── api.py                # FastAPI routes
│   ├── search_engine.py      # Core search engine logic
│   ├── embedder.py           # Embedding model (gte-small)
│   ├── cache_manager.py      # Hashing + caching logic
│   └── __init__.py
│
├── requirements.txt          # Project dependencies
├── Procfile                  # Start command (Railway/Render)
└── README.md                 # Project documentation
```

---

## 5. Design Choices

### 🟢 1. Lightweight Embedding Model

**Used:** `thenlper/gte-small`

**Reason:**
- Very fast on CPU
- No GPU required
- Small memory footprint
- Great performance for semantic search

### 🟢 2. Manual Caching Layer

A simple JSON + NumPy caching system:

- No database required
- Easy to reset
- Works offline
- Perfect for assignment scale

### 🟢 3. Cleaned & Normalized Embeddings

All embeddings are **L2-normalized** so cosine similarity becomes a simple dot product (fast and accurate).

### 🟢 4. Optional FAISS Index

FAISS support is included but disabled by default:

- Python-only version can run without installation issues
- For large datasets, FAISS gives instant nearest-neighbors search

### 🟢 5. Explanations Included

Every search result includes:

- Overlapping keywords
- Overlap ratio
- Length normalization factor

This helps understand why a document was returned.

---

## 6. How to Run the Project Locally (Quick Start)

```bash
git clone <your_repo>
cd multi-document-embedding-search-engine
pip install -r requirements.txt
python -m src.api
```

Then open:

```
http://127.0.0.1:8000/docs
```

Upload your documents into `data/` and the system handles everything.

---

## 📦 Requirements

See `requirements.txt` for dependencies. Key packages:

- `fastapi`
- `uvicorn`
- `sentence-transformers`
- `numpy`
- `torch`

---

## 📝 License

This project is for educational purposes as part of an AI Engineer Intern Assignment.

---


## 📧 Contact

For questions or feedback, please reach out via GitHub issues.
