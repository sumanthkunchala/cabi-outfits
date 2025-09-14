# src/cabi_outfits/api.py
from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Literal
from datetime import datetime
import uuid
import json

import faiss
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from rich.console import Console

from .recommend import assemble_outfits, Catalog
from .taxonomy import Taxonomy

# ---------
# Paths
# ---------
APP_ROOT = Path(__file__).resolve().parents[2]  # project root
ITEMS_PARQUET = APP_ROOT / "data" / "tmp" / "items_clean.parquet"
IMG_IDS = APP_ROOT / "embeddings" / "image_ids.json"
IMG_EMBS = APP_ROOT / "embeddings" / "image_clip_vitb32.npy"
IMG_INDEX = APP_ROOT / "embeddings" / "image_clip_vitb32.faiss"
EVENTS_PATH = APP_ROOT / "data" / "events" / "feedback.jsonl"

app = FastAPI(title="Cabi Outfits API", version="0.2.0")
console = Console()

# Allow local front-ends
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://127.0.0.1:8080",
        "http://localhost:8080",
        "http://127.0.0.1:5500",  # VSCode Live Server (optional)
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve images so the front-end can render them
app.mount(
    "/images",
    StaticFiles(directory=str(APP_ROOT / "data" / "images")),
    name="images",
)

g_tx: Taxonomy | None = None
g_cat: Catalog | None = None
g_index: faiss.Index | None = None

# ---------
# Models
# ---------
class OutfitItem(BaseModel):
    product_id: str
    name: Optional[str] = None
    subtype: Optional[str] = None
    category: Optional[str] = None
    colors: List[str] = []
    patterns: List[str] = []
    occasions: List[str] = []

class Outfit(BaseModel):
    score: float
    palette: List[str]
    base: str
    items: List[OutfitItem]

class RecommendRequest(BaseModel):
    prompt: str
    must_include: Optional[List[str]] = None
    k: int = 8

class RecommendResponse(BaseModel):
    prompt: str
    outfits: List[Outfit]

class FeedbackEvent(BaseModel):
    # minimal schema; extend later when you add auth/user ids
    prompt: Optional[str] = None
    action: Literal["view", "like", "save", "add_to_cart", "dismiss"]
    product_ids: List[str] = []          # ids of items in the outfit
    outfit_score: Optional[float] = None # score you showed to the user
    user_id: Optional[str] = None        # optional if you have one
    session_id: Optional[str] = None     # optional client-side session id
    extra: dict = {}                     # free-form (e.g., UI placement, variant)

# ---------
# Startup
# ---------
@app.on_event("startup")
def _startup():
    global g_tx, g_cat, g_index
    console.rule("[bold]API Startup")
    if not ITEMS_PARQUET.exists():
        raise RuntimeError(f"Missing catalog: {ITEMS_PARQUET}")
    if not IMG_IDS.exists() or not IMG_EMBS.exists() or not IMG_INDEX.exists():
        raise RuntimeError("Missing embeddings or FAISS index. Run steps 13–14 first.")

    g_tx = Taxonomy.load()
    g_cat = Catalog.load(ITEMS_PARQUET, IMG_IDS, IMG_EMBS)
    g_index = faiss.read_index(str(IMG_INDEX))

    EVENTS_PATH.parent.mkdir(parents=True, exist_ok=True)

    console.print(f"Loaded taxonomy ✓")
    console.print(f"Loaded catalog items: {len(g_cat.items)}")
    console.print(f"Loaded image vectors: {len(g_cat.img_ids)}")
    console.print(f"Loaded FAISS index: {g_index.ntotal}")
    console.print(f"Feedback sink: {EVENTS_PATH}")
    console.rule("[bold green]Ready")

# ---------
# Routes
# ---------
@app.get("/healthz")
def healthz():
    return {
        "ok": True,
        "items": int(len(g_cat.items) if g_cat else 0),
        "vectors": int(len(g_cat.img_ids) if g_cat else 0),
        "indexed": int(g_index.ntotal if g_index else 0),
        "taxonomy": bool(g_tx is not None),
    }

@app.post("/recommend/outfit", response_model=RecommendResponse)
def recommend(req: RecommendRequest):
    if not (g_tx and g_cat and g_index):
        raise RuntimeError("Service not initialized")
    outfits = assemble_outfits(
        prompt=req.prompt,
        cat=g_cat,
        faiss_index=g_index,
        img_ids=g_cat.img_ids,
        tx=g_tx,
        must_include=req.must_include or None,
        k=req.k,
    )
    # Cast dicts into Pydantic models
    hydrated = []
    for o in outfits:
        hydrated.append(Outfit(
            score=o["score"],
            palette=o["palette"],
            base=o["base"],
            items=[OutfitItem(**it) for it in o["items"]],
        ))
    return RecommendResponse(prompt=req.prompt, outfits=hydrated)

@app.post("/feedback")
def feedback(ev: FeedbackEvent):
    """
    Append a single feedback event as one JSON line to data/events/feedback.jsonl.
    """
    record = ev.model_dump()
    record["event_id"] = str(uuid.uuid4())
    record["ts"] = datetime.utcnow().isoformat(timespec="seconds") + "Z"
    EVENTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(EVENTS_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
    return {"ok": True, "event_id": record["event_id"]}
