# src/cabi_outfits/embed_text.py
from __future__ import annotations

import json
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import typer
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

app = typer.Typer(help="Encode product text with MiniLM (384-d)")


def build_text(row: dict) -> str:
    # Construct a compact, information-dense text for embedding
    name = str(row.get("name") or "").strip()
    subtype = str(row.get("subtype") or "").strip()
    category = str(row.get("category") or "").strip()

    # Handle numpy arrays and lists safely
    colors_data = row.get("colors")
    if colors_data is not None and hasattr(colors_data, '__len__') and len(colors_data) > 0:
        colors = ", ".join([str(c) for c in colors_data])
    else:
        colors = ""
    
    patterns_data = row.get("patterns")
    if patterns_data is not None and hasattr(patterns_data, '__len__') and len(patterns_data) > 0:
        patterns = ", ".join([str(p) for p in patterns_data])
    else:
        patterns = ""
    
    occasions_data = row.get("occasions")
    if occasions_data is not None and hasattr(occasions_data, '__len__') and len(occasions_data) > 0:
        occasions = ", ".join([str(o) for o in occasions_data])
    else:
        occasions = ""
    
    tags_data = row.get("tags")
    if tags_data is not None and hasattr(tags_data, '__len__') and len(tags_data) > 0:
        tags = ", ".join([str(t) for t in tags_data])
    else:
        tags = ""

    parts = [name]
    if subtype: parts.append(subtype)
    if category: parts.append(f"category: {category}")
    if colors: parts.append(f"colors: {colors}")
    if patterns: parts.append(f"patterns: {patterns}")
    if occasions: parts.append(f"occasions: {occasions}")
    if tags: parts.append(f"tags: {tags}")

    return " | ".join(parts)


@app.command()
def run(
    items_parquet: str = typer.Argument(..., help="data/tmp/items_clean.parquet"),
    out_npy: str = typer.Option("embeddings/text_minilm.npy", help="Output .npy path"),
    out_ids: str = typer.Option("embeddings/text_ids.json", help="Output JSON list of product_ids"),
    out_texts: str = typer.Option("embeddings/text_texts.json", help="Output JSON list of text used per row (debug)"),
    batch_size: int = typer.Option(256, help="Batch size"),
):
    items_parquet = Path(items_parquet)
    out_npy = Path(out_npy)
    out_ids = Path(out_ids)
    out_texts = Path(out_texts)

    out_npy.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(items_parquet)
    if "product_id" not in df.columns:
        raise SystemExit("items_clean.parquet missing 'product_id' column")

    rows = df.to_dict(orient="records")
    ids: List[str] = [str(r["product_id"]) for r in rows]
    texts: List[str] = [build_text(r) for r in rows]

    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    vecs: List[np.ndarray] = []
    for i in tqdm(range(0, len(texts), batch_size), desc="Embedding text"):
        batch = texts[i:i+batch_size]
        emb = model.encode(batch, normalize_embeddings=True, convert_to_numpy=True)
        vecs.append(emb.astype("float32"))

    embs = np.vstack(vecs) if vecs else np.zeros((0, 384), dtype="float32")

    # Save
    np.save(out_npy, embs)
    out_ids.write_text(json.dumps(ids), encoding="utf-8")
    out_texts.write_text(json.dumps(texts, ensure_ascii=False), encoding="utf-8")

    print(f"Saved text embeddings: {out_npy} shape={embs.shape}")
    print(f"Saved ids:            {out_ids} count={len(ids)}")
    print(f"Saved texts:          {out_texts}")
    print("Done.")


if __name__ == "__main__":
    app()
