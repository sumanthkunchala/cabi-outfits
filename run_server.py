#!/usr/bin/env python3
"""
Start the CABI Outfit Recommender API server
"""

import sys
from pathlib import Path

# Add src to Python path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

import uvicorn

if __name__ == "__main__":
    print("🚀 Starting CABI Outfit Recommender API server...")
    print("📚 Interactive API docs will be available at: http://localhost:8000/docs")
    print("🔗 API root: http://localhost:8000")
    print("❤️  Health check: http://localhost:8000/health")
    print()
    
    uvicorn.run(
        "cabi_outfits.server:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        access_log=True,
        app_dir="src"
    )
