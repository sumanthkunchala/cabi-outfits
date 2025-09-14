# src/cabi_outfits/server.py
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

import faiss
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field
from contextlib import asynccontextmanager

try:
    from .recommend import Catalog, assemble_outfits
    from .taxonomy import Taxonomy
except ImportError:
    from recommend import Catalog, assemble_outfits
    from taxonomy import Taxonomy

# Initialize FastAPI app
app = FastAPI(
    title="CABI Outfit Recommender API",
    description="AI-powered outfit recommendations using CLIP embeddings and rule-based scoring",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Global state for loaded models and data
_catalog: Optional[Catalog] = None
_index: Optional[faiss.Index] = None
_taxonomy: Optional[Taxonomy] = None

# Pydantic models for API
class OutfitItem(BaseModel):
    product_id: str
    name: Optional[str] = None
    subtype: Optional[str] = None
    category: Optional[str] = None
    colors: List[str] = Field(default_factory=list)
    patterns: List[str] = Field(default_factory=list)
    occasions: List[str] = Field(default_factory=list)

class Outfit(BaseModel):
    score: float
    palette: List[str] = Field(default_factory=list)
    base: str  # "dress" or "top+bottom"
    items: List[OutfitItem]

class RecommendationRequest(BaseModel):
    prompt: str = Field(..., description="Natural language description of desired outfit", example="smart casual dinner outfit in fall tones")
    k: int = Field(default=5, ge=1, le=20, description="Number of outfit recommendations to return")
    must_include: Optional[List[str]] = Field(default=None, description="Product IDs that must be included in recommendations")

class RecommendationResponse(BaseModel):
    query: str
    num_results: int
    outfits: List[Outfit]

class HealthResponse(BaseModel):
    status: str
    message: str
    models_loaded: bool

# Startup event to load models and data
@app.on_event("startup")
async def startup_event():
    """Load models and data on server startup"""
    global _catalog, _index, _taxonomy
    
    try:
        # Define paths
        items_parquet = Path("data/tmp/items_clean.parquet")
        img_ids_path = Path("embeddings/image_ids.json")
        img_embs_path = Path("embeddings/image_clip_vitb32.npy")
        img_index_path = Path("embeddings/image_clip_vitb32.faiss")
        
        # Check if all required files exist
        missing_files = []
        for path in [items_parquet, img_ids_path, img_embs_path, img_index_path]:
            if not path.exists():
                missing_files.append(str(path))
        
        if missing_files:
            raise FileNotFoundError(f"Missing required files: {', '.join(missing_files)}")
        
        # Load models and data
        print("🔄 Loading catalog...")
        _catalog = Catalog.load(items_parquet, img_ids_path, img_embs_path)
        
        print("🔄 Loading FAISS index...")
        _index = faiss.read_index(str(img_index_path))
        
        print("🔄 Loading taxonomy...")
        _taxonomy = Taxonomy.load()
        
        print("✅ Server startup complete! Models loaded successfully.")
        
    except Exception as e:
        print(f"❌ Error during startup: {e}")
        raise e

# Health check endpoint
@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint to verify server and model status"""
    models_loaded = all([_catalog is not None, _index is not None, _taxonomy is not None])
    
    return HealthResponse(
        status="healthy" if models_loaded else "unhealthy",
        message="All systems operational" if models_loaded else "Models not loaded",
        models_loaded=models_loaded
    )

# Main recommendation endpoint
@app.post("/recommend", response_model=RecommendationResponse)
async def recommend_outfits(request: RecommendationRequest):
    """
    Generate outfit recommendations based on natural language prompt
    
    **Example requests:**
    - "smart casual dinner outfit in fall tones"
    - "black slim trousers for work with a camel blazer"  
    - "floral summer dress for weekend brunch"
    - "elegant work outfit in navy and cream"
    """
    # Check if models are loaded
    if not all([_catalog, _index, _taxonomy]):
        raise HTTPException(
            status_code=503, 
            detail="Models not loaded. Please check server startup logs."
        )
    
    try:
        # Call the recommendation function
        outfits = assemble_outfits(
            prompt=request.prompt,
            cat=_catalog,
            faiss_index=_index,
            img_ids=_catalog.img_ids,
            tx=_taxonomy,
            must_include=request.must_include,
            k=request.k
        )
        
        # Convert to response format
        outfit_models = []
        for outfit_data in outfits:
            items = [
                OutfitItem(
                    product_id=item["product_id"],
                    name=item.get("name"),
                    subtype=item.get("subtype"),
                    category=item.get("category"),
                    colors=item.get("colors", []),
                    patterns=item.get("patterns", []),
                    occasions=item.get("occasions", [])
                )
                for item in outfit_data["items"]
            ]
            
            outfit_models.append(Outfit(
                score=outfit_data["score"],
                palette=outfit_data.get("palette", []),
                base=outfit_data.get("base", ""),
                items=items
            ))
        
        return RecommendationResponse(
            query=request.prompt,
            num_results=len(outfit_models),
            outfits=outfit_models
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error generating recommendations: {str(e)}"
        )

# GET version of recommendation endpoint for easier testing
@app.get("/recommend", response_model=RecommendationResponse)
async def recommend_outfits_get(
    prompt: str = Query(..., description="Natural language description of desired outfit"),
    k: int = Query(default=5, ge=1, le=20, description="Number of recommendations to return"),
    must_include: Optional[str] = Query(default=None, description="Comma-separated product IDs that must be included")
):
    """
    GET version of recommendations endpoint for easier testing with curl/browser
    
    **Example:**
    ```
    GET /recommend?prompt=smart%20casual%20dinner%20outfit&k=3
    ```
    """
    # Parse must_include if provided
    must_include_list = None
    if must_include:
        must_include_list = [pid.strip() for pid in must_include.split(",") if pid.strip()]
    
    # Create request object and call POST handler
    request = RecommendationRequest(
        prompt=prompt,
        k=k,
        must_include=must_include_list
    )
    
    return await recommend_outfits(request)

# Info endpoint
@app.get("/info")
async def get_info():
    """Get information about the loaded catalog"""
    if not _catalog:
        raise HTTPException(status_code=503, detail="Catalog not loaded")
    
    return {
        "catalog_items": len(_catalog.items),
        "image_embeddings": len(_catalog.img_ids),
        "categories": _catalog.items["category"].value_counts().to_dict(),
        "color_palette_size": len(_taxonomy.palette) if _taxonomy else 0,
        "occasions": list(_taxonomy.allowed_occasions) if _taxonomy else []
    }

# Root endpoint
@app.get("/")
async def root():
    """API root endpoint with basic information"""
    return {
        "name": "CABI Outfit Recommender API",
        "version": "1.0.0",
        "description": "AI-powered outfit recommendations using CLIP embeddings",
        "endpoints": {
            "health": "/health",
            "recommend_post": "/recommend (POST)",
            "recommend_get": "/recommend (GET)",
            "info": "/info",
            "docs": "/docs"
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
