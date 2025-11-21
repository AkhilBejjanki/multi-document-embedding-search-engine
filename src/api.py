# src/api.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os
import uvicorn

from src.search_engine import SearchEngine

app = FastAPI(title="Embedding Search Engine")

# Force FAISS = OFF (macOS compatible)
ENGINE = SearchEngine(use_faiss=False)

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")

class SearchRequest(BaseModel):
    query: str
    top_k: int = 5

@app.on_event("startup")
def startup_event():
    ENGINE.load_documents(data_dir=DATA_DIR)
    ENGINE.build()

@app.post("/search")
def search(req: SearchRequest):
    if not req.query or req.top_k <= 0:
        raise HTTPException(status_code=400, detail="Invalid query or top_k")
    results = ENGINE.search_with_explanation(req.query, top_k=req.top_k)
    return {"results": results}

if __name__ == "__main__":
    uvicorn.run("src.api:app", host="0.0.0.0", port=8000, reload=True)
