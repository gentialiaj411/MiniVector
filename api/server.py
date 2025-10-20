from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
from contextlib import asynccontextmanager
import sys
import json
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from minivector.embedder import Embedder
from minivector.vector_store import VectorStore

embedder = None
store = None
metadata = None  

@asynccontextmanager
async def lifespan(app: FastAPI):
    global embedder, store, metadata
    embedder = Embedder()
    store = VectorStore()
    store.load_index()

    meta_path = Path("data/processed/metadata.json")
    if meta_path.exists():
        with open(meta_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)
        print(f"[INFO] Loaded {len(metadata)} metadata entries.")
    else:
        metadata = []
        print("[WARN] Metadata file not found.")
    yield

app = FastAPI(title="MiniVector API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class SearchRequest(BaseModel):
    query: str
    k: Optional[int] = 10

class SearchResult(BaseModel):
    id: str
    title: str
    category: str
    text_preview: str
    score: float
    url: Optional[str] = None

class SearchResponse(BaseModel):
    query: str
    results: List[SearchResult]
    took_ms: float

@app.post("/search", response_model=SearchResponse)
async def search(request: SearchRequest):
    import time
    start = time.time()
    query_vector = embedder.embed_query(request.query)
    results = store.search(query_vector, k=request.k)
    took_ms = (time.time() - start) * 1000
    return SearchResponse(query=request.query, results=results, took_ms=took_ms)

@app.get("/article/{doc_id}")
async def get_article(doc_id: str):
    if doc_id.startswith("doc_"):
        try:
            idx = int(doc_id.replace("doc_", ""))
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid document ID")
    else:
        try:
            idx = int(doc_id)
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid document ID")

    if metadata is None or idx < 0 or idx >= len(metadata):
        raise HTTPException(status_code=404, detail="Article not found")

    return metadata[idx]

@app.get("/stats")
async def stats():
    return {"num_vectors": store.index.ntotal if store else 0, 
            "dimension": store.dimension if store else 0,}

@app.get("/")
async def root():
    return {"message": "MiniVector API"}
