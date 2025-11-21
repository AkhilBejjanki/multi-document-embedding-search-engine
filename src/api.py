from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os

from src.search_engine import SearchEngine

app = FastAPI(title="Embedding Search Engine")

# Lazy initialization
ENGINE = None
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")

class SearchRequest(BaseModel):
    query: str
    top_k: int = 5

def get_engine():
    global ENGINE
    if ENGINE is None:
        ENGINE = SearchEngine(use_faiss=False)
        ENGINE.load_documents(data_dir=DATA_DIR)
        ENGINE.build()
    return ENGINE

@app.post("/search")
def search(req: SearchRequest):
    engine = get_engine()

    if not req.query or req.top_k <= 0:
        raise HTTPException(status_code=400, detail="Invalid query or top_k")

    results = engine.search_with_explanation(req.query, top_k=req.top_k)
    return {"results": results}
